# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import concurrent.futures
import re
import time
from asyncio import wait
from collections import deque
from concurrent.futures import CancelledError, ThreadPoolExecutor
from datetime import timedelta
from functools import partial
from pathlib import Path
from threading import Condition, Lock
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import tensorrt_llm.bindings.executor as trtllm
from fsspec.exceptions import asyncio
from transformers.dynamic_module_utils import typing
from vllm import RequestOutput
from vllm.config import CacheConfig, DeviceConfig, LoRAConfig, ModelConfig, ParallelConfig, SchedulerConfig
from vllm.engine.async_llm_engine import AsyncStream
from vllm.lora.request import LoRARequest
from vllm.sequence import SequenceStatus
from vllm_mxext.engine.metrics import CompletionStats, IterationStats, StatLogger
from vllm_mxext.hub.hardware_inspect import GPUUnit
from vllm_mxext.logger import init_logger
from vllm_mxext.lora import LoraSource, TrtllmLoraSource
from vllm_mxext.trtllm.request import TrtRequest
from vllm_mxext.trtllm.utils import TrtllmConfig, create_trt_executor

logger = init_logger(__name__)


class TrtllmInferenceError(RuntimeError):
    pass


class LoraCacheMissError(TrtllmInferenceError):
    pass


class LoraCacheFullError(TrtllmInferenceError):
    pass


_TRTLLM_ERROR_PATTERNS: List[Tuple[re.Pattern, Type[RuntimeError]]] = [
    (
        re.compile(r".*LoRA task [0-9]+ not found in cache. Please send LoRA weights with request"),
        LoraCacheMissError,
    ),
    (
        re.compile(r".*Cache is full. There are no done tasks to evict"),
        LoraCacheFullError,
    ),
]


def _trtllm_raise_for_error(req_id: str, response: trtllm.Response):
    if not response.has_error():
        return
    error_cls = TrtllmInferenceError
    for pattern, e_cls in _TRTLLM_ERROR_PATTERNS:
        m = pattern.match(response.error_msg)
        if m is not None:
            error_cls = e_cls
    raise error_cls(f"Error during inference of request {req_id} -- {response.error_msg}")


class QClosedError(Exception):
    """Raised on get or put if a DQueue is close"""

    pass


class DQueue:
    """
    Simple syncronized multiple producer consumer FIFO queue backed be a deque.
    This is more performant that using the heavier queue.Queue.
    deque's append and pop operations are thread safe, so the class just implements
    blocking to wait for next available item
    """

    def __init__(self):
        self._q = deque()
        self._cv = Condition()
        self._closed = False

    def put(self, item) -> None:
        """
        put an item on the queue
        """
        if self._closed:
            raise QClosedError()
        with self._cv:
            self._q.append(item)
            self._cv.notify()

    def get(self, block=True) -> Any:
        """
        get an item from the queue. if block is False this will return imediately, which may return None
        """
        if self._closed:
            raise QClosedError()
        # popleft is thread safe
        has, item = self._try_get_item()
        if has or not block:
            return item
        with self._cv:
            while not has:
                self._cv.wait()
                has, item = self._try_get_item()
                if self._closed:
                    raise QClosedError()
        return item

    def _try_get_item(self) -> Tuple[bool, Any]:
        try:
            item = self._q.popleft()
            return True, item
        except IndexError:
            return False, None

    def close(self):
        """
        Close the queue.  All future gets and puts will fail raise QClosedError
        """
        with self._cv:
            self._closed = True
            self._cv.notify_all()


class RequestStore:
    """
    Synchronized Request store. Stores TrtRequest object associated with trt_id and vllm_id and the output stream
    """

    def __init__(self):
        self._requests: Dict[str, Tuple[AsyncStream, TrtRequest]] = {}
        self._trt_to_vllm_id: Dict[int, str] = {}
        # self._active_loras: Dict[int, int] = {}  # TODO is this needed
        self._loaded_loras: Dict[int, bool] = {}
        self._lock = Lock()

    def new_request(self, stream: AsyncStream, request: TrtRequest, engine_enqueue_func: Callable[[], int]) -> None:
        with self._lock:
            if request.request_id in self._requests:
                raise Exception(f"request id {request.request_id} already exists")
            request.trt_request_id = engine_enqueue_func()
            self._requests[request.request_id] = (stream, request)
            self._trt_to_vllm_id[request.trt_request_id] = request.request_id

    def requeue_request(self, request: TrtRequest, engine_enqueue_fn: Callable[[], int]) -> None:
        with self._lock:
            if request.trt_request_id in self._trt_to_vllm_id:
                del self._trt_to_vllm_id[request.trt_request_id]
            request.trt_request_id = engine_enqueue_fn()
            self._trt_to_vllm_id[request.trt_request_id] = request.request_id

    def _append_to_stream(self, stream: AsyncStream, item: Exception | RequestOutput):
        stream.put(item)
        if isinstance(item, RequestOutput) and item.finished:
            stream.finish()

    def append_to_stream(
        self, request_id: str, item: Union[RequestOutput, Exception], loop: asyncio.AbstractEventLoop
    ) -> None:
        with self._lock:
            if request_id in self._requests:
                logger.debug(f"Writing to stream {request_id}")
                if loop is None:
                    self._append_to_stream(self._requests[request_id][0], item)
                else:
                    loop.call_soon_threadsafe(self._append_to_stream, self._requests[request_id][0], item)

    def _finish_stream_async(self, stream: AsyncStream):
        return stream.finish()

    def abort_stream(self, request_id: str, loop: asyncio.AbstractEventLoop) -> Tuple[Optional[str], Optional[int]]:
        with self._lock:
            if request_id in self._requests:
                trt_id = self._requests[request_id][1].trt_request_id
                loop.call_soon_threadsafe(self._finish_stream_async, self._requests[request_id][0])
                del self._requests[request_id]
                del self._trt_to_vllm_id[trt_id]
                return request_id, trt_id
            return None, None

    def get_trt_request(self, trt_req_id: int) -> Optional[TrtRequest]:
        logger.debug(self._trt_to_vllm_id)
        logger.debug(trt_req_id)
        with self._lock:
            if trt_req_id in self._trt_to_vllm_id:
                req_id = self._trt_to_vllm_id[trt_req_id]
                return self._requests[req_id][1]
            return None

    def propagate_exception(self, e: Exception, loop: asyncio.AbstractEventLoop):
        with self._lock:
            for _, (stream, _) in self._requests.items():
                loop.call_soon_threadsafe(self._append_to_stream, stream, e)

    def record_lora_cache_miss(self, request_id: str, task_id: int):
        with self._lock:
            if request_id in self._requests:
                _, req = self._requests[request_id]
                req.record_lora_cache_miss()
            self._loaded_loras[task_id] = False

    def record_lora_load(self, task_id: int):
        with self._lock:
            self._loaded_loras[task_id] = True

    def is_lora_loaded(self, task_id: int):
        with self._lock:
            return self._loaded_loras.get(task_id, False)

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._requests)


class TrtllmModelRunner:
    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        device_config: DeviceConfig,
        lora_config: Optional[LoRAConfig],
        cache_config: Optional[CacheConfig],
        log_stats: bool,
        selected_gpus: List[GPUUnit],
        num_postprocessor_threads: int = 1,
        lora_source: Optional[TrtllmLoraSource] = None,
        num_lora_workers: int = 1,
    ):

        self._loop: asyncio.AbstractEventLoop = None

        self._lora_source = lora_source

        self._requests_q = DQueue()
        self._tllm_responses_q = DQueue()
        self._abort_q = DQueue()
        self._output_stats_q = DQueue()
        self._request_dict = RequestStore()
        self._shutdown = False
        self._started = False
        self._error: Optional[Exception] = None

        self._num_post_proc_threads = num_postprocessor_threads
        self._num_lora_workers = num_lora_workers
        self._post_proc_qs: List[DQueue] = [DQueue() for _ in range(self._num_post_proc_threads)]
        self._lora_requeue_qs: List[DQueue] = [DQueue() for _ in range(self._num_lora_workers)]
        # abort_loop
        # enqueue loop
        # await loop
        # post proc fan loop
        # post proc loops
        # lora loops
        # stats loop
        self._num_threads = 5 + self._num_post_proc_threads + self._num_lora_workers
        self._thread_pool = ThreadPoolExecutor(max_workers=self._num_threads)
        self._futures = []

        self._stat_logger = None
        if log_stats:
            self._stat_logger = StatLogger(
                labels=dict(model_name=model_config.model), max_model_len=model_config.max_model_len
            )

        self._tllm_exec, self._cfg = self._create_engine(
            model_config,
            parallel_config,
            scheduler_config,
            device_config,
            cache_config,
            lora_config,
            log_stats,
            selected_gpus,
        )

        if self._lora_source is not None:
            self._lora_source.dtype = self._cfg.dtype

    def shutdown(self):
        if self._shutdown:
            return
        self._shutdown = True
        self._shutdown_engine()
        self._requests_q.close()
        self._tllm_responses_q.close()
        self._abort_q.close()
        self._output_stats_q.close()
        for q in self._post_proc_qs:
            q.close()
        for q in self._lora_requeue_qs:
            q.close()
        self._thread_pool.shutdown()

    def start(self, loop: asyncio.AbstractEventLoop):
        self._start(loop)

    def _start(self, loop: asyncio.AbstractEventLoop, start_await=True):
        if self._loop is not None:
            raise Exception("Engine already started")

        self._loop = loop
        self._futures.append(self._thread_pool.submit(self._abort_loop, input_q=self._abort_q))
        self._futures.append(self._thread_pool.submit(self._engine_enqueue_loop, input_q=self._requests_q))
        if self._stat_logger:
            self._futures.append(self._thread_pool.submit(self._stats_loop, input_q=self._output_stats_q))
        self._futures.append(
            self._thread_pool.submit(
                self._post_process_fan_out_loop, input_q=self._tllm_responses_q, output_qs=self._post_proc_qs
            )
        )
        for idx in range(self._num_post_proc_threads):
            self._futures.append(
                self._thread_pool.submit(
                    self._post_process_loop, input_q=self._post_proc_qs[idx], output_q=self._output_stats_q
                )
            )
        for idx in range(self._num_lora_workers):
            self._futures.append(self._thread_pool.submit(self._requeue_lora_loop, input_q=self._lora_requeue_qs[idx]))

        if start_await:
            self._start_await_loop()

        self._started = True

    def _start_await_loop(self):
        self._futures.append(self._thread_pool.submit(self._await_requests_loop, output_q=self._tllm_responses_q))

    def abort(self, request_id: str) -> None:
        if self._loop is None:
            raise RuntimeError("Must call start")
        self._abort_q.put(request_id)

    def _abort_loop(self, input_q: DQueue):
        try:
            while not self._shutdown:
                req_id = input_q.get()
                self._abort(req_id)
        except QClosedError:
            return
        except CancelledError:
            return

    def _abort(self, req_id: str):
        try:
            _, trt_id = self._request_dict.abort_stream(req_id, self._loop)
            if trt_id is not None:
                self._cancel_request(trt_id)
        except Exception:
            logger.exception(f"Error aborting request {req_id}")

    def enqueue_request(self, trt_request: TrtRequest) -> AsyncStream:
        if self._loop is None:
            raise RuntimeError("Must call start")
        stream = AsyncStream(trt_request.request_id)
        self._requests_q.put((stream, trt_request))
        return stream

    def _check_request(self, req: TrtRequest):
        if req.lora_request is not None and self._lora_source is None:
            raise Exception("Recieved lora request without setting up lora source")

    def _engine_enqueue_loop(self, input_q: DQueue):
        try:
            while not self._shutdown:
                stream, trt_req_wrap = input_q.get()
                self._engine_enqueue(stream, trt_req_wrap)
        except QClosedError:
            return
        except CancelledError:
            return

    def _engine_enqueue(self, stream: AsyncStream, trt_req_wrap: TrtRequest) -> None:
        try:
            self._check_request(trt_req_wrap)
            trt_req = trt_req_wrap.to_trt_request()
            self._request_dict.new_request(stream, trt_req_wrap, partial(self._enqueue_tllm_request, trt_req))
        except Exception as e:
            logger.exception("Error in enqueue loop")
            self._loop.call_soon_threadsafe(stream.put, e)
            self._loop.call_soon_threadsafe(stream.finish)

    def _await_requests_loop(self, output_q: DQueue):
        try:
            while not self._shutdown:
                responses = self._await_responses()
                logger.debug(f"Got responses")
                output_q.put(responses)
        except QClosedError:
            return
        except CancelledError:
            return
        except Exception as e:
            logger.exception("Error while awaiting responses")
            self.propagate_exception(e)

    def _post_process_fan_out_loop(self, input_q: DQueue, output_qs: List[DQueue]):
        try:
            while not self._shutdown:
                responses: List[trtllm.Response] = input_q.get()
                logger.debug(f"Got responses fan")
                for r in responses:
                    req_partition = r.request_id % len(output_qs)
                    output_qs[req_partition].put(r)
        except QClosedError:
            return
        except CancelledError:
            return
        except Exception as e:
            logger.exception("Error while faning responses")
            self.propagate_exception(e)

    def _post_process_loop(self, input_q: DQueue, output_q: DQueue):
        try:
            while not self._shutdown:
                tllm_response: trtllm.Response = input_q.get()
                logger.debug(f"Got response post {tllm_response.request_id}")
                self._post_proc(tllm_response, output_q, self._lora_requeue_qs)
        except QClosedError:
            return
        except CancelledError:
            return

    def _post_proc(self, tllm_response: trtllm.Response, output_q: DQueue, lora_requeue_qs: List[DQueue]):
        request_id = None
        try:
            trt_id = tllm_response.request_id
            trt_req_wrap = self._request_dict.get_trt_request(trt_id)
            if trt_req_wrap is None:
                return
            request_id = trt_req_wrap.request_id
            self._maybe_requeue_lora_miss(trt_req_wrap, tllm_response, lora_requeue_qs)

            output, num_token = trt_req_wrap.postprocess(tllm_response)
            finished = output.finished
            self._request_dict.append_to_stream(request_id, output, self._loop)
            if self._stat_logger:
                # Passing current generation info along with TrtRequest
                # to avoid read / write TrtRequest member in different threads.
                output_q.put((time.time(), num_token, finished, trt_req_wrap))
            if finished:
                self.abort(trt_req_wrap.request_id)
        except LoraCacheMissError:
            return
        except LoraCacheFullError as e:
            logger.warning("PEFT cache is full.  Reduce the number of concurrent PEFT / LoRA requests")
            if request_id is not None:
                self._request_dict.append_to_stream(request_id, e, self._loop)
                self._request_dict.abort_stream(request_id, self._loop)
        except BaseException as e:
            logger.exception("Error while postprocessing request")
            if request_id is not None:
                # stream excepts Exception so we'll wrap exceptions not extending Exception
                exc = Exception(e) if not isinstance(e, Exception) else e
                self._request_dict.append_to_stream(request_id, exc, self._loop)
                self._request_dict.abort_stream(request_id, self._loop)

    def _maybe_requeue_lora_miss(self, req: TrtRequest, response: trtllm.Response, lora_requeue_qs: List[DQueue]):
        try:
            _trtllm_raise_for_error(req.request_id, response)
        except LoraCacheMissError:
            lora_req = req.lora_request
            if lora_req is None:
                raise Exception(
                    "LoraRequest is None while processing LoRA request. This should not happen. Please file a bug."
                )
            self._request_dict.record_lora_cache_miss(req.request_id, lora_req.lora_int_id)
            q_idx = lora_req.lora_int_id % len(lora_requeue_qs)
            lora_requeue_qs[q_idx].put(req)
            raise

    def _requeue_lora_loop(self, input_q: DQueue):
        try:
            while not self._shutdown:
                req: TrtRequest = input_q.get()
                self._requeue_lora(req)
        except QClosedError:
            return
        except CancelledError:
            return

    def _requeue_lora(self, req: TrtRequest) -> None:
        try:
            if req.lora_cache_miss_count > 1 or not self._request_dict.is_lora_loaded(req.lora_request.lora_int_id):
                lora_req = typing.cast(LoRARequest, req.lora_request)
                trtllm_lora = typing.cast(LoraSource, self._lora_source).get_lora(
                    lora_req.lora_name, Path(lora_req.lora_local_path)
                )
                # check that the request has not been aborted in the time it took to load the lora
                check_req = self._request_dict.get_trt_request(req.trt_request_id)
                if check_req is None:
                    return None
                config, weights = trtllm_lora
                lora_config = trtllm.LoraConfig(task_id=lora_req.lora_int_id, config=config, weights=weights)
            else:
                lora_config = None
            tllm_req = req.to_trt_request(lora_config)
            self._request_dict.requeue_request(req, partial(self._enqueue_tllm_request, tllm_req))
            self._request_dict.record_lora_load(req.lora_request.lora_int_id)
        except Exception as e:
            logger.exception("Error while postprocessing request")
            self._request_dict.append_to_stream(req.request_id, Exception(e), self._loop)
            self._request_dict.abort_stream(req.request_id, self._loop)

    def _stats_loop(self, input_q: DQueue):
        loop_timeout = 0.1
        while not self._shutdown:
            try:
                output_stats = []
                # Only block on first item to avoid busy looping in the thread,
                # then exhausts the queue to get all the completion stats to be
                # reported together.
                # The loop may never exit under load so a timeout is set
                pr = input_q.get()
                loop_start = time.time()
                while pr is not None and (time.time() - loop_start) < loop_timeout:
                    output_stats.append(pr)
                    pr = input_q.get(block=False)
                trtllm_iter_stats = self._get_trtllm_stats()
                self._log_iteration(trtllm_iter_stats)
                self._log_completion(output_stats)
            except QClosedError:
                return
            except CancelledError:
                return
            except Exception:
                logger.exception(f"Error collecting stats")

    def _await_responses(self) -> List[trtllm.Response]:
        return self._tllm_exec.await_responses(timeout=timedelta(seconds=1))

    def _get_trtllm_stats(self) -> List[trtllm.IterationStats]:
        return self._tllm_exec.get_latest_iteration_stats()

    def _enqueue_tllm_request(self, req: trtllm.Request) -> int:
        return self._tllm_exec.enqueue_request(req)

    def _cancel_request(self, req_id: int) -> None:
        return self._tllm_exec.cancel_request(req_id)

    def _shutdown_engine(self) -> None:
        return self._tllm_exec.shutdown()

    def _create_engine(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        device_config: DeviceConfig,
        cache_config: Optional[CacheConfig],
        lora_config: Optional[LoRAConfig],
        log_stats: bool,
        selected_gpus: List[GPUUnit],
    ) -> Tuple[trtllm.Executor, TrtllmConfig]:
        return create_trt_executor(
            model_config,
            parallel_config,
            scheduler_config,
            device_config,
            cache_config,
            lora_config,
            log_stats,
            selected_gpus,
        )

    def _log_iteration(self, trtllm_iter_stats: List[trtllm.IterationStats]):
        """
        Helper function to consume the iteration stats from TRT-LLM.
        The TRT-LLM stats will be used to report system-wise stats such as:
         - number of request at different stages
         - resource utilization
        """
        # TRT-LLM returns list of stats for iterations happened since last retrieval,
        # so should iterate the list even though it is likely that only the current
        # iteration stats are in the list, because stats is collected whenever the
        # responses are produced (per iteration)
        for iter_stats in trtllm_iter_stats:
            gpu_cache_usage_sys = 0
            max_num_blocks = iter_stats.kv_cache_stats.max_num_blocks
            if max_num_blocks > 0:
                gpu_cache_usage_sys = iter_stats.kv_cache_stats.used_num_blocks / max_num_blocks

            num_running_sys = iter_stats.inflight_batching_stats.num_scheduled_requests
            # Total number of waiting request in the system is calculated by
            # "# of requests tracked - # of scheduled request".
            # NOTE: 'max' is used to ensure # of waiting requests reported is
            # in valid range (>=0).
            # The # of scheduled requests from TRT-LLM stats also contains
            # requests completed in the iteration, so there will be miscalculation
            # in the iteration that completes requests, which soon will be corrected
            # in the next iteration.
            # Always accurate result can be obtained by going over request stats
            # from TRT-LLM, but it introduces noticeable overhead due to frequent
            # access of pybind objects.
            num_waiting_sys = max((self._request_dict.size - num_running_sys), 0)

            self._stat_logger.log_iteration(
                IterationStats(
                    num_running_sys=num_running_sys,
                    num_waiting_sys=num_waiting_sys,
                    num_request_max_sys=iter_stats.max_num_active_requests,
                    gpu_cache_usage_sys=gpu_cache_usage_sys,
                )
            )

    def _log_completion(self, output_stats: List[Tuple[float, int, bool, TrtRequest]]):
        """
        Helper function to report request stats as the request makes progress,
        stats includes
         - latency: token / e2e
         - request completion state
        """
        # Additonal iteration stats
        num_prompt_tokens_iter = 0
        num_generation_tokens_iter = 0
        # per request
        time_e2e_requests: List[float] = []
        num_prompt_tokens_requests: List[int] = []
        num_generation_tokens_requests: List[int] = []
        finished_reason_requests: List[str] = []
        # per token
        time_to_first_tokens_iter: List[float] = []
        time_per_output_tokens_iter: List[float] = []
        for timestamp, num_token, finished, trt_req_wrap in output_stats:
            if num_token > 0:
                # First token
                if trt_req_wrap.last_token_time == 0:
                    time_to_first_tokens_iter.append(timestamp - trt_req_wrap.arrival_time)
                    num_prompt_tokens_iter += trt_req_wrap.num_prompt_tokens
                else:
                    time_per_output_tokens_iter.append((timestamp - trt_req_wrap.last_token_time) / num_token)
                trt_req_wrap.last_token_time = timestamp
                num_generation_tokens_iter += num_token
            if finished:
                num_prompt_tokens_requests.append(trt_req_wrap.num_prompt_tokens)
                num_generation_tokens_requests.append(trt_req_wrap.num_generated_tokens)
                finished_reason_requests.append(SequenceStatus.get_finished_reason(trt_req_wrap.seq_status))
                time_e2e_requests.append(timestamp - trt_req_wrap.arrival_time)

        self._stat_logger.log_completion(
            CompletionStats(
                num_prompt_tokens_iter=num_prompt_tokens_iter,
                num_generation_tokens_iter=num_generation_tokens_iter,
                time_e2e_requests=time_e2e_requests,
                num_prompt_tokens_requests=num_prompt_tokens_requests,
                num_generation_tokens_requests=num_generation_tokens_requests,
                finished_reason_requests=finished_reason_requests,
                time_to_first_tokens_iter=time_to_first_tokens_iter,
                time_per_output_tokens_iter=time_per_output_tokens_iter,
            )
        )

    def propagate_exception(self, e: Exception):
        self._error = e
        self._request_dict.propagate_exception(e, self._loop)
        self.shutdown()

    def health(self):
        if self._error is not None:
            raise self._error
        elif self._shutdown:
            raise Exception("Engine has shutdown")
        # TODO need to check the actual engine
        if not self._tllm_exec.can_enqueue_requests():
            raise Exception("Engine cant queue requests")

    def ready(self):
        self.health()
        if not self._started:
            raise Exception("Engine has not started")

    def is_shutdown(self):
        return self._shutdown
