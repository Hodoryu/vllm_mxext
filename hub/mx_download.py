# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import logging
import os
import sys
from pathlib import Path
from typing import List, Mapping, Set, Tuple, Union

import yaml
from nim_hub import Manifest, Repo
from vllm_mxext.logger import init_logger

nim_manifest_default = Path("/etc/mim/config/model_manifest.yaml")
nim_manifest_override = Path("/mim/config/model_manifest.yaml")

nim_config_default = Path("/etc/mim/config/config.yaml")
nim_config_override = Path("/mim/config/config.yaml")

# override the selected profile in the config file
MIM_MANIFEST_PATH_ENV_NAME = "MIM_MANIFEST_PATH"
MIM_MANIFEST_UNSAFE_ENV_NAME = "MIM_MANIFEST_ALLOW_UNSAFE"

logger = init_logger(__name__)


def get_manifest_path(manifest_path: Path = None) -> Path:
    # path passed into this function trumps all
    if manifest_path is not None:
        return manifest_path

    # default behavior is to load a manifest of possible models from nim_config_default
    # this can be overridden with nim_config_override file
    if nim_manifest_default.exists():
        manifest_path = nim_manifest_default

    if nim_manifest_override.exists():
        manifest_path = nim_manifest_override

    if (x := os.environ.get(MIM_MANIFEST_PATH_ENV_NAME, None)) is not None:
        new_path = Path(x)
        if new_path.exists():
            manifest_path = new_path

    if manifest_path is None:
        if os.environ.get(NIM_MANIFEST_UNSAFE_ENV_NAME):
            logger.warn(f"No model manifest file found, but {NIM_MANIFEST_UNSAFE_ENV_NAME} is set. Proceeding.")
        else:
            raise RuntimeError("Error: no model manifest found")

    return manifest_path


def get_config_from_manifest(manifest_path: Path = None, opt_profile_name: str = None) -> Repo:
    if manifest_path is not None and manifest_path.exists():
        manifest = Manifest.load_from_file(manifest_path.as_posix())

        logger.debug(f"Loading profile '{opt_profile_name}' from manifest at {manifest_path}")

        config = None
        if opt_profile_name in manifest.configs():
            config = manifest.configs()[opt_profile_name]
        else:
            raise ValueError("invalid profile requested")
        return config
    return None


def check_mx_model_path(model_path):
    prefix_removed = model_path.removeprefix("ngc://")
    if len(model_path) == len(prefix_removed):
        raise ValueError(f"model uri '{model_path}' contains no ngc:// prefix. is this an ngc model?")

    model_path_segments = prefix_removed.split("/")
    if len(model_path_segments) > 3 or len(model_path_segments) < 2:
        raise ValueError("invalid ngc model path provided.")

    # try to extract a version
    model_name = model_path_segments[-1]
    model_version = None
    if ":" in model_name:
        model_name_segments = model_name.split(":")
        model_name = model_name_segments[0]
        model_version = model_name_segments[1]

    return {
        "org": model_path_segments[0],
        "model": model_name,
        "team": None if len(model_path_segments) == 2 else model_path_segments[1],
        "version": model_version,
    }
