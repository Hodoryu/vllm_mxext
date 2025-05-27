# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#!/usr/bin/env python

import os
import pathlib
import time
from pathlib import Path
from tempfile import mkdtemp
from typing import List, Optional, Tuple

import yaml
from nim_hub import Manifest, Repo
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm_mxext.hub.hardware_inspect import GPUUnit, HwSystem, get_hardware_spec
from vllm_mxext.hub.mx_download import MIM_MANIFEST_UNSAFE_ENV_NAME, get_manifest_path
from vllm_mxext.hub.mx_profile import filter_manifest_configs, get_profile_description
from vllm_mxext.hub.utils import error_and_exit
from vllm_mxext.logger import init_logger
from vllm_mxext.utils.minio_get import download_models
import torch

# this should not be set by default, and is a power tool that can be wielded carefully
# to pull an arbitrary model
MIM_MODEL_ENV_NAME = "MIM_MODEL_NAME"
MIM_MODEL_PROFILE_ENV_NAME = "MIM_MODEL_PROFILE"
MIM_MANIFEST_UNSAFE_ENV_NAME = "MIM_MANIFEST_ALLOW_UNSAFE"

MIM_TENSOR_PARALLEL_ENV_NAME = "MIM_TENSOR_PARALLEL_SIZE"

TAG_TENSOR_PARALLEL_SIZE_KEY = "tp"
TAG_PIPELINE_PARALLEL_SIZE_KEY = "pp"
TAG_SWAP_SPACE_KEY = "swap_space"
GPU_FAMILIES_REQUIRING_EAGER = ["A10G", "L40S"]

logger = init_logger(__name__)


def is_trt_llm_model(model_path: os.PathLike | str):
    model_path = Path(model_path)
    if not model_path.is_dir():
        return False

    trt_profiles_dir = model_path / "trtllm_engine"

    if trt_profiles_dir.is_dir():
        return True

    return False
def get_precision_dtype(precision):
    if precision == 'fp16':
        return 'float16'
    elif precision == 'fp32':
        return 'float32'
    elif precision == 'half':
        return 'half'
    elif precision == 'bfloat16':
        return 'bfloat16'
    elif precision == 'float':
        return 'float'
    elif precision == 'auto':
        return 'auto'
    else:
        logger.warning(f"unsupported model precision:{precision}, so set to torch.fp16 default.")  
        return 'auto'
def _maybe_get_tp_size(tp_size: Optional[str], source: str) -> Optional[int]:
    # returns tensor parallel size if set as env variable
    if tp_size is None:
        return None
    if tp_size.isdigit() and int(tp_size) > 0:
        return int(tp_size)
    error_message = (
        f"Found invalid value for tensor_parallel_size set in {source}."
        f"It is currently set to '{tp_size}' instead of a positive integer"
    )
    error_and_exit(error_message)

def _maybe_get_pp_size(pp_size: Optional[str], source: str) -> Optional[int]:
    # returns tensor parallel size if set as env variable
    if pp_size is None:
        return None
    if pp_size.isdigit() and int(pp_size) > 0:
        return int(pp_size)
    error_message = (
        f"Found invalid value for tensor_parallel_size set in {source}."
        f"It is currently set to '{pp_size}' instead of a positive integer"
    )
    error_and_exit(error_message)

def _maybe_get_swap_space(swap_space: Optional[str], source: str) -> Optional[int]:
    # returns tensor parallel size if set as env variable
    if swap_space is None:
        return None
    if swap_space.isdigit() and int(swap_space) > 0:
        return int(swap_space)
    error_message = (
        f"Found invalid value for tensor_parallel_size set in {source}."
        f"It is currently set to '{swap_space}' instead of a positive integer"
    )
    error_and_exit(error_message)

def _maybe_set_tp_size_from_env(engine_args: AsyncEngineArgs):
    # sets tp size if a valid value is available in env vars
    tp_size_env_value = os.environ.get(MIM_TENSOR_PARALLEL_ENV_NAME, None)
    tp_size_from_env = _maybe_get_tp_size(tp_size_env_value, f"env {MIM_TENSOR_PARALLEL_ENV_NAME}")
    if tp_size_from_env:
        engine_args.tensor_parallel_size = tp_size_from_env


def _should_enforce_eager_vllm(engine_args: AsyncEngineArgs, system: HwSystem):
    # Workaround: enable enforce eager flag in vllm engines for a subset of GPUs (L40s and A10G) for multi-gpu setup
    #  it disables CUDA graph and always execute the model in eager mode
    #  this can cause performance issues
    #  another alternative is to reduce gpu_memory_utilization
    has_eligible_gpus = any([gpu.family in GPU_FAMILIES_REQUIRING_EAGER for gpu in system.get_free_gpus()])
    return engine_args.tensor_parallel_size > 1 and not is_trt_llm_model(engine_args.model) and has_eligible_gpus


def get_optimal_manifest_config(
    manifest_path: Path, system: HwSystem, enable_lora: bool, override: str = None, force: bool = False
) -> Tuple[str, Manifest, List[GPUUnit]]:
    config_results = filter_manifest_configs(manifest_path, system, enable_lora, override=override, force=force)
    if len(config_results) == 0:
        error_message = (
            "Could not find a profile that is currently runnable with the detected hardware. "
            "Please check the system information below and make sure you have enough free GPUs."
        )
        error_message += "\n"
        # Print the detected system configuration to help the user debug
        if len(system.total_gpu):
            error_message += str(system)
        else:
            error_message += "No GPUs found - did you forget `--runtime=nvidia --gpus='\"device=gpuids\"']` ?"
        error_and_exit(error_message)
    if len(config_results) > 0:
        for config_id, config, selected_gpus in config_results:
            desc = get_profile_description(config)
            selected_gpu_ids = [gpu.device_index for gpu in selected_gpus]
            logger.info(f"Valid profile: {config_id} ({desc}) on GPUs {selected_gpu_ids}")
    optimal_config_id, optimal_config, selected_gpus = config_results[0]
    return optimal_config_id, optimal_config, selected_gpus


def inject_mx_hub(engine_args: AsyncEngineArgs) -> Tuple[AsyncEngineArgs, str]:
    engine_args.selected_gpus = None
    # if not ngc path, return local path
    engine_args.selected_gpus = None
    if os.path.isdir(engine_args.model) and os.path.isabs(engine_args.model):
        return engine_args, engine_args.model

    # sets tensor_parallel_size from env before exiting this function
    should_set_tp_size_from_env = True
    should_set_pp_size_from_env = True

    # disable HF
    os.environ["HF_HUB_OFFLINE"] = "1"

    # get hw system
    system = get_hardware_spec()

    model_name = None

    model_uri = os.environ.get(MIM_MODEL_ENV_NAME, engine_args.model)
    model_profile = os.environ.get(MIM_MODEL_PROFILE_ENV_NAME, None)
    model_profile_force = os.environ.get(MIM_MANIFEST_UNSAFE_ENV_NAME, False)

    # default is to try to load repo from manifest
    manifest_path = get_manifest_path()
    repo = None
    if manifest_path is not None:
        optimal_config_id, optimal_config, selected_gpus = get_optimal_manifest_config(
            manifest_path, system, engine_args.enable_lora, override=model_profile, force=model_profile_force
        )
        optimal_config_desc = get_profile_description(optimal_config)
        logger.info(f"Selected profile: {optimal_config_id} ({optimal_config_desc})")
        logger.debug(f" optimal_config: {optimal_config.model()} {optimal_config.tags()}")
        model_name = optimal_config.model()
        tags = optimal_config.tags()
        for k, v in tags.items():
            logger.info(f"Profile metadata: {k}: {v}")
        engine_args.selected_gpus = selected_gpus
        tp_size_from_tags = _maybe_get_tp_size(tags.get(TAG_TENSOR_PARALLEL_SIZE_KEY), "manifest config")
        pp_size_from_tags = _maybe_get_pp_size(tags.get(TAG_PIPELINE_PARALLEL_SIZE_KEY), "manifest config")
        swap_space_from_tags = _maybe_get_swap_space(tags.get(TAG_SWAP_SPACE_KEY), "manifest config")
        if tp_size_from_tags is not None:
            engine_args.tensor_parallel_size = tp_size_from_tags
            should_set_tp_size_from_env = False
        if pp_size_from_tags is not None:
            engine_args.pipeline_parallel_size = pp_size_from_tags
            should_set_pp_size_from_env = False
        if swap_space_from_tags is not None:
            engine_args.swap_space = swap_space_from_tags
        #change dtype
        precision = tags.get('precision')
        logger.info("selected model profile set precision:%s", precision)
        engine_args.dtype = get_precision_dtype(precision)
    elif os.path.exists(model_uri) and os.path.isdir(model_uri):
        # load local mapped path
        logger.info("model exist at dir:%s no need download.", model_uri)
        engine_args.model = model_uri
        engine_args.tokenizer = model_uri
        model_name = model_uri
        return engine_args, engine_args.model
        
    logger.info("starting to  download model:%s ", model_uri)
    download_models(model_uri)
    logger.info("downloaded models :%s ", model_uri)
    #downloaded model modify model and tokenizer
    model_cache_dir = os.getenv("MIM_CACHE_PATH", "/opt/mim/.cache")
    model_dir = os.path.join(model_cache_dir, model_name)
    engine_args.model = model_dir
    engine_args.tokenizer = model_dir
    

    if should_set_tp_size_from_env:
        _maybe_set_tp_size_from_env(engine_args)

    if _should_enforce_eager_vllm(engine_args, system):
        # Workaround: enable enforce eager flag to load 70B in vllm engines
        #  it disables CUDA graph and always execute the model in eager mode
        #  this can cause performance issues
        #  another alternative is to reduce gpu_memory_utilization
        logger.info('Adding --enforce_eager flag...')
        engine_args.enforce_eager = True

    return engine_args, model_name
