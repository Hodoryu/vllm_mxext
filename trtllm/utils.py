# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import ctypes
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import tensorrt_llm
import tensorrt_llm.bindings.executor as trtllm
from vllm import AsyncEngineArgs
from vllm.config import CacheConfig, DeviceConfig, LoRAConfig, ModelConfig, ParallelConfig, SchedulerConfig
from vllm.executor.executor_base import ExecutorBase
from vllm.lora.request import LoRARequest
from vllm.sampling_params import SamplingParams
from vllm_mxext.hub.hardware_inspect import GPUInspect, GPUUnit, get_hardware_spec
from vllm_mxext.logger import init_logger

logger = init_logger(__name__)


@dataclass
class TrtllmLoraConfig:
    enable: bool = False
    modules: List[str] = field(default_factory=list)
    max_rank: int = 32


@dataclass
class TrtllmConfig:
    dtype: str
    num_layers: int
    pp_size: int
    tp_size: int
    num_heads: int
    num_kv_heads: int
    head_size: int
    inter_size: int
    hidden_size: int
    batch_size: int
    max_input_len: int
    lora: TrtllmLoraConfig | None = None


def parse_trtllm_config(profile_or_config_path: Path):
    if profile_or_config_path.is_file():
        config_path = profile_or_config_path
    elif profile_or_config_path.is_dir():
        config_path = profile_or_config_path / "config.json"
    else:
        raise ValueError(f"Invalid profile dir or config path {profile_or_config_path}")

    with open(config_path, 'r') as fh:
        config = json.load(fh)
    build_config = config["build_config"]

    pretrained_config = config["pretrained_config"]
    num_layers = pretrained_config["num_hidden_layers"]
    pp_size = pretrained_config["mapping"]["pp_size"]
    tp_size = pretrained_config["mapping"]["tp_size"]
    num_heads = pretrained_config["num_attention_heads"]
    head_size = pretrained_config["head_size"]
    num_kv_heads = pretrained_config["num_key_value_heads"]
    dtype_str = pretrained_config["dtype"]
    inter_size = pretrained_config.get("intermediate_size", 0)  # some models won't have this
    hidden_size = pretrained_config["hidden_size"]
    max_batch_size = build_config["max_batch_size"]
    max_input_size = build_config["max_input_len"]

    lora_cfg = None
    plugin_config = build_config["plugin_config"]
    lora_enabled = plugin_config["lora_plugin"] is not None
    if lora_enabled:
        lora_cfg = build_config["lora_config"]
        max_lora_rank = lora_cfg["max_lora_rank"]
        target_modules = lora_cfg["lora_target_modules"]
        lora_cfg = TrtllmLoraConfig(enable=True, modules=target_modules, max_rank=max_lora_rank)

    return TrtllmConfig(
        dtype=dtype_str,
        batch_size=max_batch_size,
        max_input_len=max_input_size,
        num_layers=num_layers,
        pp_size=pp_size,
        tp_size=tp_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        inter_size=inter_size,
        hidden_size=hidden_size,
        lora=lora_cfg,
    )


class TrtEngineInitError(Exception):
    pass


def select_trt_profile(model_path: Path) -> Path:
    profile_dir = model_path / "trtllm_engine"

    if not (profile_dir.exists() and profile_dir.is_dir()):
        raise RuntimeError(f"Did not find valid trtllm profile.")

    return profile_dir


def tensorrt_llm_executor_worker_path() -> str:
    worker_path = Path(tensorrt_llm.__file__).parent / 'bin' / 'executorWorker'
    if not worker_path.exists():
        raise Exception("TensorRT-LLM executor worker not found")
    return str(worker_path)


def get_trt_parallel_config(tp_size: int, pp_size: int, selected_gpus: List[GPUUnit]):
    world_size = tp_size * pp_size
    if world_size > 1:
        executor_worker_path = tensorrt_llm_executor_worker_path()
        orchestrator_config = trtllm.OrchestratorConfig(True, executor_worker_path)
        return trtllm.ParallelConfig(
            trtllm.CommunicationType.MPI,
            trtllm.CommunicationMode.ORCHESTRATOR,
            orchestrator_config=orchestrator_config,
            device_ids=get_device_ids(selected_gpus, world_size),
        )
    else:
        return trtllm.ParallelConfig(trtllm.CommunicationType.MPI, trtllm.CommunicationMode.LEADER)


def get_device_ids(selected_gpus: List[GPUUnit], world_size: int) -> List[int]:
    if selected_gpus:
        device_ids = [int(gpu.device_index) for gpu in selected_gpus]
        logger.info(f"Using provided selected GPUs list {device_ids}")
        return device_ids
    else:
        device_ids = list(range(world_size))
        logger.info(f"Using default GPUs list {device_ids}")
        return device_ids


def _get_rank_engine_file_size_bytes(profile_dir: Path):
    rank0_engine = (profile_dir / "rank0.engine").resolve()
    engine_size_bytes = rank0_engine.stat().st_size
    return engine_size_bytes


def _get_kvcache_free_fraction(
    engine_size_bytes: int, lora_device_mem: int, lora_act_mem: int, avail_device_mem: int
) -> float:
    min_head_space = min(avail_device_mem * 0.1, 8 * 2**30)
    mem_after_engine_load = avail_device_mem - (engine_size_bytes + lora_act_mem)
    logger.debug(f"estimate mem_after_engine_load {mem_after_engine_load}")
    mem_after_lora = mem_after_engine_load - lora_device_mem
    logger.debug(f"estimate mem_after_lora {mem_after_lora}")
    free_fraction = (mem_after_lora - min_head_space) / mem_after_engine_load
    if free_fraction <= 0:
        raise RuntimeError("Insuficient memory to run model. Not enough memory to have kvcahce and lora cache")
    elif free_fraction <= 0.1:
        logger.warning(f"kvcache allocation is low {free_fraction}% of free memory")
    return free_fraction


def create_trt_executor(
    model_config: ModelConfig,
    parallel_config: ParallelConfig,
    scheduler_config: SchedulerConfig,
    device_config: DeviceConfig,
    cache_config: Optional[CacheConfig],
    lora_config: Optional[LoRAConfig],
    log_stats: bool,
    selected_gpus: List[GPUUnit],
    max_iteration_logged: int = 1000,
) -> Tuple[trtllm.Executor, TrtllmConfig]:
    profile_dir = select_trt_profile(Path(model_config.model))
    cfg = parse_trtllm_config(profile_dir)

    if parallel_config.pipeline_parallel_size != cfg.pp_size:
        logger.warning(
            f"Overriding pp size {parallel_config.pipeline_parallel_size} with pp size from model {cfg.pp_size}"
        )
    if parallel_config.tensor_parallel_size != cfg.tp_size:
        logger.warning(
            f"Overriding tp size {parallel_config.tensor_parallel_size} with tp size from model {cfg.tp_size }"
        )

    trt_parallel_config = get_trt_parallel_config(cfg.tp_size, cfg.pp_size, selected_gpus)
    trt_scheduler_config = trtllm.SchedulerConfig(trtllm.CapacitySchedulerPolicy.GUARANTEED_NO_EVICT)
    # set max iteration to 0 to turn off TRT-LLM stats collection
    max_iteration = max_iteration_logged if log_stats else 0
    peft_cache_config, lora_device_mem, lora_activation_mem = get_peft_cache_config(cfg, lora_config)

    logger.info(f"Using {lora_device_mem} bytes of gpu memory for PEFT cache")

    devices = trt_parallel_config.device_ids
    if devices is None:
        devices = [0]
    engine_size_bytes = _get_rank_engine_file_size_bytes(profile_dir)
    logger.info(f"Engine size in bytes {engine_size_bytes}")

    gpus = GPUInspect()
    _, avail_device_mem = gpus.device_mem(devices[0])
    avail_device_mem_MiBs = avail_device_mem/1024/1024
    logger.info(f"available device memory {avail_device_mem_MiBs} MiBs")

    kvcache_free_fraction = 0.9  # default
    if lora_device_mem > 0:
        kvcache_free_fraction = _get_kvcache_free_fraction(
            engine_size_bytes, lora_device_mem, lora_activation_mem, avail_device_mem
        )
    logger.info(f"Setting free_gpu_memory_fraction to {kvcache_free_fraction}")

    kv_cache_config = trtllm.KvCacheConfig(
        free_gpu_memory_fraction=kvcache_free_fraction,
    )

    trtllm_exec = trtllm.Executor(
        profile_dir,
        trtllm.ModelType.DECODER_ONLY,
        trtllm.ExecutorConfig(
            1,
            iter_stats_max_iterations=max_iteration,
            # nvbugs/4662826
            request_stats_max_iterations=0,
            parallel_config=trt_parallel_config,
            normalize_log_probs=False,
            batching_type=trtllm.BatchingType.INFLIGHT,
            scheduler_config=trt_scheduler_config,
            peft_cache_config=peft_cache_config,
            kv_cache_config=kv_cache_config,
        ),
    )

    return trtllm_exec, cfg


def align16(a):
    return ((a + 16 - 1) // 16) * 16


def dtype_bytes(dtype: str) -> int:
    try:
        dtype_size = {
            "float16": 2,
            "bfloat16": 2,
            "fp16": 2,
            "float8": 1,
            "fp8": 1,
            "float32": 4,
            "fp32": 4,
        }[dtype]
        return dtype_size
    except KeyError:
        raise ValueError(f"Invalid dtype {dtype}")


class LoraModule:
    def __init__(self, name: str, in_size: int, out_size: int, tp_split_in: bool, tp_split_out: bool):
        self._name = name
        self._in_size = in_size
        self._out_size = out_size
        self._tp_split_in = tp_split_in
        self._tp_split_out = tp_split_out

    def __eq__(self, o):
        return (
            self._name == o._name
            and self._in_size == o._in_size
            and self._out_size == o._out_size
            and self._tp_split_in == o._tp_split_in
            and self._tp_split_out == o._tp_split_out
        )

    @classmethod
    def create_lora_modules(
        cls, modules: List[str], num_heads: int, head_size: int, num_kv_heads: int, inter_size: int, hidden_size: int
    ) -> List["LoraModule"]:
        mod_list = []
        for mod in modules:
            if mod in ["attn_qkv", "cross_attn_qkv"]:
                mod_list.append(
                    cls(mod, hidden_size, num_heads * head_size + 2 * num_kv_heads * head_size, False, True)
                )
            elif mod in ["attn_q", "cross_attn_q", "attn_k", "cross_attn_k", "attn_v", "cross_attn_v"]:
                mod_list.append(cls(mod, hidden_size, hidden_size, False, True))
            elif mod in ["attn_dense", "cross_attn_dense"]:
                mod_list.append(cls(mod, hidden_size, hidden_size, True, False))
            elif mod in ["mlp_h_to_4h"]:
                mod_list.append(cls(mod, hidden_size, inter_size, False, True))
            elif mod in ["mlp_gate"]:
                mod_list.append(cls(mod, hidden_size, inter_size, False, True))
            elif mod in ["mlp_4h_to_h"]:
                mod_list.append(cls(mod, inter_size, hidden_size, True, False))
            else:
                raise ValueError(f"Invalid LoRA module {mod}")
        return mod_list

    def total_1d_bytes(self, tp_size: int = 1, dtype: str = "float16") -> int:
        dtype_size = {
            "float16": 2,
            "bfloat16": 2,
            "fp16": 2,
            "float8": 1,
            "fp8": 1,
            "float32": 4,
            "fp32": 4,
        }[dtype]
        in_tp_size = tp_size if self._tp_split_in else 1
        out_tp_size = tp_size if self._tp_split_out else 1

        in_size = dtype_size * (self._in_size // in_tp_size)
        out_size = dtype_size * (self._out_size // out_tp_size)

        return int(in_size + out_size)

    @staticmethod
    def compute_num_lora_tokens(
        num_loras: int, num_modules: int, lora_rank: int, num_layers: int, pp_size: int
    ) -> int:
        local_layers = num_layers // pp_size
        return lora_rank * local_layers * num_modules * num_loras

    @staticmethod
    def max_1d_mod_size(lora_mods: List["LoraModule"], tp_size: int = 1, dtype: str = "float16") -> int:
        return max([m.total_1d_bytes(tp_size, dtype) for m in lora_mods])

    @staticmethod
    def estimate_activate_bytes(
        lora_mods: List["LoraModule"], batch_size: int, max_context: int, adapter_size: int, dtype: str
    ) -> int:
        splitk = 16
        sizeof_float = 4
        param_bytes = 84
        dtype_size = dtype_bytes(dtype)
        mod_set = {m._name for m in lora_mods}
        max_single_layer_mods = 1
        if {"attn_q", "attn_k", "attn_v"}.issubset(mod_set):
            max_single_layer_mods = 3

        size1 = align16(batch_size * max_context * max_single_layer_mods * adapter_size * sizeof_float * splitk)
        size2 = align16(batch_size * max_context * max_single_layer_mods * adapter_size * dtype_size)
        size3 = param_bytes * batch_size * max_single_layer_mods
        return size1 + size2 + size3


def get_peft_cache_config(
    cfg: TrtllmConfig, lora_config: Optional[LoRAConfig]
) -> Tuple[trtllm.PeftCacheConfig, int, int]:
    if cfg.lora is None or not cfg.lora.enable:
        if lora_config:
            raise Exception("you are attempting to use LoRA with an engine not built with LoRA support")
        return trtllm.PeftCacheConfig(), 0, 0

    lora_mods = LoraModule.create_lora_modules(
        cfg.lora.modules, cfg.num_heads, cfg.head_size, cfg.num_kv_heads, cfg.inter_size, cfg.hidden_size
    )

    if lora_config is not None:
        if cfg.lora.max_rank < lora_config.max_lora_rank:
            raise Exception(
                f"model supports lora rank of {cfg.lora.max_rank}, but configured for greater rank {lora_config.max_lora_rank}"
            )
        lora_rank = min(cfg.lora.max_rank, lora_config.max_lora_rank)

        num_host_loras = lora_config.max_cpu_loras if lora_config.max_cpu_loras is not None else lora_config.max_loras
        num_device_loras = lora_config.max_loras
    else:
        lora_rank = cfg.lora.max_rank
        num_host_loras = 1
        num_device_loras = 1

    host_lora_tokens = LoraModule.compute_num_lora_tokens(
        num_host_loras, len(lora_mods), lora_rank, cfg.num_layers, cfg.pp_size
    )
    device_lora_tokens = LoraModule.compute_num_lora_tokens(
        num_device_loras, len(lora_mods), lora_rank, cfg.num_layers, cfg.pp_size
    )

    lora_device_cache_mem = LoraModule.max_1d_mod_size(lora_mods, cfg.tp_size, cfg.dtype) * device_lora_tokens
    lora_device_activation = LoraModule.estimate_activate_bytes(
        lora_mods, cfg.batch_size, cfg.max_input_len, cfg.lora.max_rank, cfg.dtype
    )

    peft_cache_config = trtllm.PeftCacheConfig(
        num_host_module_layer=host_lora_tokens,
        num_device_module_layer=device_lora_tokens,
        optimal_adapter_size=lora_rank,
        max_adapter_size=lora_rank,
    )
    return peft_cache_config, lora_device_cache_mem, lora_device_activation


def greedy_trtllm_sampling_params(sampling_params):
    # TODO: (INFE-2236) Revert this WAR
    # def greedy_trtllm_sampling_params():
    #     return trtllm.SamplingConfig(beam_width=1, top_k=1)

    return trtllm.SamplingConfig(
        beam_width=1,  # beam_width=1 for inflight batching
        top_k=1,  # SizeType topK
        top_p=1.0,
        top_p_min=None,
        top_p_reset_ids=None,  # SizeType topPResetIds
        top_p_decay=None,  # FloatType topPDecay
        random_seed=ctypes.c_uint64(sampling_params.seed).value if sampling_params.seed is not None else None,
        temperature=1,
        min_length=1,  # SizeType minLength
        beam_search_diversity_rate=None,  # FloatType beamSearchDiversityRate
        repetition_penalty=1,  # FloatType repetitionPenalty
        presence_penalty=0,  # FloatType presencePenalty
        frequency_penalty=0,  # FloatType frequencyPenalty
        length_penalty=1,  # FloatType lengthPenalty
        early_stopping=None,  # SizeType earlyStopping. Controls beam search, so irrelevant until we have beam_width > 1
    )


def vllm_sampling_to_trtllm_sampling(sampling_params: SamplingParams) -> trtllm.SamplingConfig:
    temperature = None
    ## temperature
    ### We are considering temperature first because it triggers greedy decoding
    # vllm default: 1.0 (lower value makes model more predictable)
    # trtllm: > 0.0 (default 1.0)
    # TODO: (INFE-2236) Revert this WAR
    # if sampling_params.temperature == 0:
    #     return greedy_trtllm_sampling_params()
    # elif sampling_params.temperature > 0.0:
    if sampling_params.temperature == 0:
        return greedy_trtllm_sampling_params(sampling_params)
    elif sampling_params.temperature > 0.0:
        temperature = sampling_params.temperature
    else:
        raise ValueError(f"Invalid temperature value {temperature}")

    ## top_k
    # vllm default: -1 (all logits), not metioned what 0 means, but we're treating it the same a -1 (all logits)
    # trtlm default: 0 (None) (all logits)
    top_k = None
    if sampling_params.top_k == -1 or sampling_params.top_k == 0:
        top_k = None
    elif sampling_params.top_k > 0:
        top_k = sampling_params.top_k
    else:
        raise ValueError(f"Invalid top_k value {sampling_params.top_k}")

    top_p = None
    ## top_p
    # vllm (0, 1] (default 1.0)
    # trtllm [0, 1] (default 0)
    if 0 <= sampling_params.top_p <= 1.0:
        top_p = sampling_params.top_p
    else:
        raise ValueError(f"Invalid top_p value {sampling_params.min_p}")

    top_p_min = None
    ## top_p_min
    # vllm (min_p) [0, 1] (default 0.0 disable)
    # trtllm (0.0, 1.0] (default 1e-6)
    if sampling_params.min_p == 0.0:
        top_p_min = None
    elif 0 < sampling_params.min_p <= 1:
        top_p_min = sampling_params.min_p
    else:
        raise ValueError(f"Invalid top_p_min value {sampling_params.min_p}")

    seed = ctypes.c_uint64(sampling_params.seed).value if sampling_params.seed is not None else None

    repetition_penalty = None
    ## repetition_penalty = None
    # vllm Values > 1 encourage the model to use new tokens, while values < 1 encourage the model to repeat tokens. (default 1.0)
    # trtllm It can have any value > 0.f. Values < 1.f encourages repetition, values > 1.f discourages it. Default is 1.f (default 1.0)
    if sampling_params.repetition_penalty is None:
        repetition_penalty = 1.0
    elif 0 < sampling_params.repetition_penalty:
        repetition_penalty = sampling_params.repetition_penalty
    else:
        raise ValueError(f"Invalid repetition_penalty value {sampling_params.repetition_penalty}")

    presence_penalty = None
    ## presence_penalty
    # vllm Values > 0 encourage the model to use new tokens, while values < 0 encourage the model to repeat tokens. (default 0.0)
    # trtllm It can have any values. Values < 0.f encourage repetition, values > 0.f discourage it. Default is 0.f
    presence_penalty = sampling_params.presence_penalty

    frequency_penalty = None
    ## frequency_penalty
    # vllm   Values > 0 encourage the model to use new tokens, while values < 0 encourage the model to repeat tokens. (default 0.0)
    # trtllm It can have any values. Values < 0.f encourage repetition, values > 0.f discourage it. Default is 0.f
    frequency_penalty = sampling_params.frequency_penalty

    length_penalty = None
    ## length_penalty
    # vllm default 1.0
    # trtllm Default is 0.f
    length_penalty = sampling_params.length_penalty

    min_length = None
    ## min_length
    # vllm min_tokens (default 0) no effect
    # trtllm (default 1) values < 1 have not effect
    if sampling_params.min_tokens <= 0:
        min_length = None
    else:
        min_length = sampling_params.min_tokens

    ret = trtllm.SamplingConfig(
        beam_width=1,  # beam_width=1 for inflight batching
        top_k=top_k,  # SizeType topK
        top_p=top_p,
        top_p_min=top_p_min,
        top_p_reset_ids=None,  # SizeType topPResetIds
        top_p_decay=None,  # FloatType topPDecay
        random_seed=seed,
        temperature=temperature,
        min_length=min_length,  # SizeType minLength
        beam_search_diversity_rate=None,  # FloatType beamSearchDiversityRate
        repetition_penalty=repetition_penalty,  # FloatType repetitionPenalty
        presence_penalty=presence_penalty,  # FloatType presencePenalty
        frequency_penalty=frequency_penalty,  # FloatType frequencyPenalty
        length_penalty=length_penalty,  # FloatType lengthPenalty
        early_stopping=None,  # SizeType earlyStopping. Controls beam search, so irrelevant until we have beam_width > 1
    )
    return ret


def to_trt_req(
    token_ids: List[int],
    sampling_params: SamplingParams,
    lora_request: LoRARequest | None = None,
    trtllm_lora_config: trtllm.LoraConfig | None = None,
) -> trtllm.Request:
    output_config = trtllm.OutputConfig()
    output_config.exclude_input_from_output = True
    if sampling_params.logprobs is not None and sampling_params.logprobs > 0:
        output_config.return_log_probs = True
    sampling_config = vllm_sampling_to_trtllm_sampling(sampling_params)
    lora_config = None
    if trtllm_lora_config is not None:
        lora_config = trtllm_lora_config
    elif lora_request is not None:
        lora_config = trtllm.LoraConfig(task_id=lora_request.lora_int_id)
    return trtllm.Request(
        input_token_ids=token_ids,
        max_new_tokens=sampling_params.max_tokens,
        streaming=True,
        output_config=output_config,
        sampling_config=sampling_config,
        end_id=-1 if sampling_params.ignore_eos else sampling_params.eos_token_id,
        lora_config=lora_config,
    )
