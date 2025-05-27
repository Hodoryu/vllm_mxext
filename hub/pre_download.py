#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
import argparse
import json
import os
from pathlib import Path
from nim_hub import Manifest

from vllm_mxext.hub.hardware_inspect import get_hardware_spec
from vllm_mxext.hub.mx_download import get_manifest_path
from vllm_mxext.hub.mx_injector import get_optimal_manifest_config
from vllm_mxext.hub.mx_profile import get_profile_description
from vllm_mxext.logger import init_logger

logger = init_logger(__name__)


def get_optimal_profile_id(manifest_path: Path, enable_lora: bool):
    system = get_hardware_spec()
    optimal_config_id, optimal_config, _ = get_optimal_manifest_config(manifest_path, system, enable_lora)
    optimal_config_desc = get_profile_description(optimal_config)
    logger.info(f"Selected profile: {optimal_config_id} ({optimal_config_desc})")
    return optimal_config_id


def download_to_cache():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profiles",
        "-p",
        help="Profile hashes to download. If none are provided, the optimal profile is downloaded",
        nargs="*",
        type=str,
    )
    parser.add_argument("--all", help="Set this to download all profiles to cache", default=False, action="store_true")
    parser.add_argument(
        "--lora",
        help="Set this to download default lora profile. "
        "This expects --profiles and --all arguments are not specified.",
        default=False,
        action="store_true",
    )
    manifest_path = get_manifest_path()
    args = parser.parse_args()
    manifest = Manifest.load_from_file(manifest_path.as_posix())

    configs = manifest.configs()
    if args.all:
        profiles = configs.keys()
    elif args.profiles:
        profiles = args.profiles
    else:
        profiles = [get_optimal_profile_id(manifest_path=manifest_path, enable_lora=args.lora)]

    for profile in profiles:
        logger.info(f"Downloading contents for profile {profile}")
        try:
            config = configs[profile]
        except KeyError:
            logger.info(f"The profile {profile} doesn't exist, skipping to the next profile")
            continue
        logger.info(json.dumps(config.tags(), indent=2))

        repo = config.workspace()

        cached_files = repo.download_all()


def create_model_store():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", "-p", help="Profile hash to create a model directory of", type=str, required=True)
    parser.add_argument(
        "--model-store", "-m", help="Directory path which where model will be copied", type=str, required=True
    )

    args = parser.parse_args()
    manifest = Manifest.load_from_file("/etc/mim/config/model_manifest.yaml")

    configs = manifest.configs()

    # should throw a key error if key is not present
    try:
        config = configs[args.profile]
    except KeyError:
        raise Error(f"This profile doesn't exist. Please input a correct profile ID.")

    repo = config.workspace()

    cached_files = repo.download_all()

    if not os.path.isdir(args.model_store):
        os.makedirs(args.model_store)
    # The nim_hub is a caching API. Thus, we dont want to move files/blob out of the cache
    # and should always copy if we want non-linked model directories
    cached_files.copy_all(args.model_store)
