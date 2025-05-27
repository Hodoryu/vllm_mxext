# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import asyncio
import os
import time
from typing import AsyncIterator, List, Optional, Type

import vllm
import vllm_mxext
from vllm import AsyncEngineArgs
from vllm.config import (
    CacheConfig,
    DeviceConfig,
    LoadConfig,
    LoRAConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    SpeculativeConfig,
    VisionLanguageConfig,
)
from vllm.engine.arg_utils import EngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine, AsyncStream
from vllm.engine.llm_engine import LLMEngine, _load_generation_config_dict
from vllm.executor.executor_base import ExecutorBase
from vllm.executor.gpu_executor import GPUExecutorAsync
from vllm.lora.request import LoRARequest
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams
from vllm.sequence import MultiModalData
from vllm.transformers_utils.detokenizer import Detokenizer
from vllm.usage.usage_lib import UsageContext
from vllm.utils import Counter, make_async
from vllm_mxext.engine.metrics import NimMetrics
from vllm_mxext.engine.trtllm_model_runner import TrtllmModelRunner
from vllm_mxext.hub.hardware_inspect import GPUUnit
from vllm_mxext.hub.mx_injector import is_trt_llm_model
from vllm_mxext.logger import init_logger
from vllm_mxext.lora import TrtllmLoraSource
from vllm_mxext.trtllm.request import TrtRequest

logger = init_logger(__name__)
ENGINE_ITERATION_TIMEOUT_S = int(os.environ.get("VLLM_ENGINE_ITERATION_TIMEOUT_S", "60"))


class _TRTLLMEngine(LLMEngine):
    def __init__(
        self,
        model_config: ModelConfig,
        cache_config: CacheConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        device_config: DeviceConfig,
        load_config: LoadConfig,
        lora_config: Optional[LoRAConfig],
        vision_language_config: Optional[VisionLanguageConfig],
        speculative_config: Optional[SpeculativeConfig],
        executor_class: Type[ExecutorBase],
        log_stats: bool,
        selected_gpus: List[GPUUnit],
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        **kwargs,
    ) -> None:
        logger.info(
            f"Initializing an LLM engine (v{vllm_mxext.__version__}) with config: "
            f"model={model_config.model!r}, "
            f"speculative_config={speculative_config!r}, "
            f"tokenizer={model_config.tokenizer!r}, "
            f"tokenizer_mode={model_config.tokenizer_mode}, "
            f"revision={model_config.revision}, "
            f"tokenizer_revision={model_config.tokenizer_revision}, "
            f"trust_remote_code={model_config.trust_remote_code}, "
            f"dtype={model_config.dtype}, "
            f"max_seq_len={model_config.max_model_len}, "
            f"download_dir={load_config.download_dir!r}, "
            f"load_format={load_config.load_format}, "
            f"tensor_parallel_size={parallel_config.tensor_parallel_size}, "
            f"disable_custom_all_reduce="
            f"{parallel_config.disable_custom_all_reduce}, "
            f"quantization={model_config.quantization}, "
            f"enforce_eager={model_config.enforce_eager}, "
            f"kv_cache_dtype={cache_config.cache_dtype}, "
            f"quantization_param_path={model_config.quantization_param_path}, "
            f"device_config={device_config.device}, "
            f"seed={model_config.seed})"
        )
        self.model_config = model_config
        self.cache_config = cache_config
        self.lora_config = lora_config
        self.vision_language_config = vision_language_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.device_config = device_config
        self.speculative_config = speculative_config
        self.load_config = load_config
        self.log_stats = log_stats

        self._init_tokenizer()
        self.detokenizer = Detokenizer(self.tokenizer)
        self.seq_counter = Counter()
        self.generation_config_fields = _load_generation_config_dict(model_config)

        # Ping the tokenizer to ensure liveness if it runs in a
        # different process.
        self.tokenizer.ping()

        num_post_proc_threads = 1
        if parallel_config.tokenizer_pool_config:
            if parallel_config.tokenizer_pool_config.pool_size > 0:
                num_post_proc_threads = parallel_config.tokenizer_pool_config.pool_size

        self._tllm_engine = TrtllmModelRunner(
            model_config,
            parallel_config,
            scheduler_config,
            device_config,
            lora_config,
            cache_config,
            log_stats,
            selected_gpus,
            num_postprocessor_threads=num_post_proc_threads,
            lora_source=TrtllmLoraSource(),
            num_lora_workers=2,  # TODO set properly
        )

    @classmethod
    def from_engine_args(
        cls,
        engine_args: EngineArgs,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
    ) -> "_TRTLLMEngine":
        """Creates an LLM engine from the engine arguments."""
        # Create the engine configs.
        engine_config = engine_args.create_engine_config()

        from vllm.executor.gpu_executor import GPUExecutor

        # Create the LLM engine.
        engine = cls(
            **engine_config.to_dict(),
            executor_class=GPUExecutor,
            log_stats=not engine_args.disable_log_stats,
            selected_gpus=engine_args.selected_gpus,
            usage_context=usage_context,
        )
        return engine

    def step(self) -> List[RequestOutput]:
        raise NotImplementedError("Sync trtllm engine is not implemented. Please use AsyncTRTLLMEngine")

    def do_log_stats(self) -> None:
        """Forced log when no requests active."""
        pass

    def add_lora(self, lora_request: LoRARequest) -> bool:
        return False

    def remove_lora(self, lora_id: int) -> bool:
        return False

    def list_loras(self) -> List[int]:
        return []

    def check_health(self) -> None:
        return self._tllm_engine.health()


class _AsyncTRTLLMEngine(_TRTLLMEngine):
    """Extension of LLMEngine to add async methods."""

    async def encode_request_async(
        self,
        request_id: str,  # pylint: disable=unused-argument
        prompt: Optional[str],
        prompt_token_ids: Optional[List[int]] = None,
        lora_request: Optional[LoRARequest] = None,
    ):
        if prompt_token_ids is None:
            assert prompt is not None
            prompt_token_ids = await self.tokenizer.encode_async(
                request_id=request_id, prompt=prompt, lora_request=lora_request
            )
        return prompt_token_ids

    async def enqueue_request(
        self,
        request_id: str,
        prompt: Optional[str],
        sampling_params: SamplingParams,
        prompt_token_ids: Optional[List[int]] = None,
        arrival_time: Optional[float] = None,
        lora_request: Optional[LoRARequest] = None,
        multi_modal_data: Optional[MultiModalData] = None,
    ) -> TrtRequest:
        if lora_request is not None and not self.lora_config:
            raise ValueError(f"Got lora_request {lora_request} but LoRA is not enabled!")
        max_logprobs = self.get_model_config().max_logprobs
        if (sampling_params.logprobs and sampling_params.logprobs > max_logprobs) or (
            sampling_params.prompt_logprobs and sampling_params.prompt_logprobs > max_logprobs
        ):
            raise ValueError(f"Cannot request more than " f"{max_logprobs} logprobs.")
        if arrival_time is None:
            arrival_time = time.time()
        if prompt_token_ids is None:
            if prompt is None:
                raise ValueError("prompt must not be None")
        prompt_token_ids = await self.encode_request_async(
            request_id=request_id, prompt=prompt, prompt_token_ids=prompt_token_ids, lora_request=lora_request
        )

        eos_token_id: Optional[int] = None
        if self.tokenizer:
            eos_token_id = self.tokenizer.get_lora_tokenizer(lora_request).eos_token_id

        if eos_token_id is not None:
            sampling_params.eos_token_id = eos_token_id
        sampling_params.update_from_generation_config(self.generation_config_fields)

        req_wrap = TrtRequest(
            request_id,
            prompt,
            prompt_token_ids,
            self.scheduler_config.max_model_len,
            self.tokenizer.get_lora_tokenizer(lora_request),
            sampling_params,
            lora_request,
            arrival_time=arrival_time,
        )
        return req_wrap

    async def check_health_async(self) -> None:
        return await make_async(self._tllm_engine.health)()


class AsyncTRTLLMEngine(AsyncLLMEngine):
    _engine_class = _AsyncTRTLLMEngine

    def __init__(
        self,
        worker_use_ray: bool,
        engine_use_ray: bool,
        *args,
        log_requests: bool = True,
        max_log_len: Optional[int] = None,
        start_engine_loop: bool = True,
        image_feature_size: Optional[str] = None,
        **kwargs,
    ) -> None:
        # disable ray: trt backend does not support ray
        self.worker_use_ray = False
        self.engine_use_ray = False
        self.log_requests = log_requests
        self.max_log_len = max_log_len
        self.image_feature_size = image_feature_size
        self.engine: _AsyncTRTLLMEngine = self._init_engine(*args, **kwargs)
        self.image_feature_size = image_feature_size
        self._tllm_engine: TrtllmModelRunner = self.engine._tllm_engine

        self._engine_loop_started = False
        self._errored_with = None
        self._shutdown = False

    def _start_engine_loop(self):
        logger.debug("Starting engine loop")
        self._tllm_engine.start(asyncio.get_running_loop())
        self._engine_loop_started = True

    @classmethod
    def from_engine_args(
        cls,
        engine_args: AsyncEngineArgs,
        start_engine_loop: bool = True,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
    ) -> "AsyncLLMEngine":
        """Creates an async LLM engine from the engine arguments."""
        engine_config = engine_args.create_engine_config()
        executor_class = GPUExecutorAsync
        return cls(
            # we are passing in worker_use_ray, engine_use_ray for
            # compatibility but trtllm ignores this and does not use ray
            engine_config.parallel_config.worker_use_ray,
            engine_args.engine_use_ray,
            **engine_config.to_dict(),
            executor_class=executor_class,
            log_requests=not engine_args.disable_log_requests,
            log_stats=not engine_args.disable_log_stats,
            max_log_len=engine_args.max_log_len,
            start_engine_loop=start_engine_loop,
            usage_context=usage_context,
            image_feature_size=engine_args.image_feature_size,
            selected_gpus=engine_args.selected_gpus,
        )

    @property
    def is_running(self) -> bool:
        return self._engine_loop_started

    @property
    def is_stopped(self) -> bool:
        return self._tllm_engine.is_shutdown()

    @property
    def errored(self) -> bool:
        return super().errored

    def _error_callback(self, exc: Exception) -> None:
        return self._tllm_engine.propagate_exception(exc)

    async def add_request(
        self,
        request_id: str,
        prompt: Optional[str],
        sampling_params: SamplingParams,
        prompt_token_ids: Optional[List[int]] = None,
        arrival_time: Optional[float] = None,
        lora_request: Optional[LoRARequest] = None,
        multi_modal_data: Optional[MultiModalData] = None,
    ) -> AsyncStream:
        if not self._engine_loop_started:
            self._start_engine_loop()
        trt_req = await self.engine.enqueue_request(
            request_id=request_id,
            prompt=prompt,
            sampling_params=sampling_params,
            prompt_token_ids=prompt_token_ids,
            arrival_time=arrival_time,
            lora_request=lora_request,
            multi_modal_data=multi_modal_data,
        )
        # init_logger
        if self.log_requests:
            shortened_prompt = prompt
            shortened_token_ids = prompt_token_ids
            if self.max_log_len is not None:
                if shortened_prompt is not None:
                    shortened_prompt = shortened_prompt[: self.max_log_len]
                if shortened_token_ids is not None:
                    shortened_token_ids = shortened_token_ids[: self.max_log_len]
            logger.info(
                f"Received request {request_id}: "
                f"prompt: {shortened_prompt!r}, "
                f"sampling_params: {sampling_params}, "
                f"prompt_token_ids: {shortened_token_ids}, "
                f"lora_request: {lora_request}."
            )

        return self._tllm_engine.enqueue_request(trt_req)

    async def generate(
        self,
        prompt: Optional[str],
        sampling_params: SamplingParams,
        request_id: str,
        prompt_token_ids: Optional[List[int]] = None,
        lora_request: Optional[LoRARequest] = None,
        multi_modal_data: Optional[MultiModalData] = None,
    ) -> AsyncIterator[RequestOutput]:
        if self._shutdown:
            raise Exception("Cannot except requests engine shutting down")
        arrival_time = time.time()

        try:
            stream = await self.add_request(
                request_id,
                prompt,
                sampling_params,
                prompt_token_ids=prompt_token_ids,
                arrival_time=arrival_time,
                lora_request=lora_request,
                multi_modal_data=multi_modal_data,
            )

            async for request_output in stream:
                yield request_output
        except (Exception, asyncio.CancelledError) as e:
            # If there is an exception or coroutine is cancelled, abort the
            # request.
            self._abort(request_id)
            raise e

    def _abort(self, request_id: str) -> None:
        self._tllm_engine.abort(request_id)

    def shutdown(self):
        self._shutdown = True
        self._tllm_engine.shutdown()


class AsyncLLMEngineFactory:
    @classmethod
    def from_engine_args(
        cls,
        engine_args: AsyncEngineArgs,
        start_engine_loop: bool = True,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
    ):
        if not hasattr(engine_args, "selected_gpus"):
            engine_args.selected_gpus = None
        engine_cls = AsyncTRTLLMEngine if is_trt_llm_model(engine_args.model) else AsyncLLMEngine
        engine = engine_cls.from_engine_args(engine_args, start_engine_loop, usage_context)
        # Metrics are registered while creating the engine, so unify after that
        if not is_trt_llm_model(engine_args.model):
            NimMetrics.unify_vllm_metrics(engine_args.max_num_seqs, engine.engine.stat_loggers)
        return engine
