# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

__version__ = "2.0.0"
import os
from pathlib import Path
from vllm_mxext.logger import init_logger


def get_cache_dir() -> Path:
    return Path(os.getenv("MIM_CACHE_PATH", "/opt/mim/.cache"))


def check_cache_dir(cache_path: Path = get_cache_dir()):
    logger = init_logger(__name__)
    logger.debug(f"Checking permissions of {cache_path} ...")
    if cache_path.is_dir() and os.access(cache_path, os.R_OK):
        logger.debug(f"Permissions of {cache_path} are valid")
        if not os.access(cache_path, os.W_OK):
            logger.warn(f"{cache_path} is read-only, application may fail if model is not already present in cache")
    else:
        raise RuntimeError(f"Unable to write to MIM_CACHE_PATH ({cache_path})")


def get_cache_dir_mapping(cache_path: Path):
    return {
        "NUMBA_CACHE_DIR": str(os.path.join("/tmp", "numba")),
        "NGC_HOME": str(cache_path.joinpath("ngc").absolute()),
        "HF_HOME": str(cache_path.joinpath("huggingface").absolute()),
        "VLLM_CONFIG_ROOT": str(cache_path.joinpath("vllm").absolute()),
    }


def set_cache_dirs():
    cache_path = get_cache_dir()
    if cache_path is not None:
        for cache_env, path in get_cache_dir_mapping(cache_path).items():
            os.environ[cache_env] = path


def set_cache_dirs_script():
    cache_path = get_cache_dir()
    if cache_path is not None:
        for cache_env, path in get_cache_dir_mapping(cache_path).items():
            print(f"export {cache_env}={path}")
