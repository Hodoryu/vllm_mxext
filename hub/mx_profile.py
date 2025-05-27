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
import re
import sys
from functools import cmp_to_key
from pathlib import Path
from typing import List, Mapping, Set, Tuple, Union

import yaml
from nim_hub import Manifest, Repo
from vllm_mxext.hub.hardware_inspect import (
    GPUUnit,
    HwSystem,
    RunnableState,
    get_hardware_spec,
    retrieve_compatible_hardware,
)
from vllm_mxext.hub.utils import error_and_exit
from vllm_mxext.logger import init_logger

logger = init_logger(__name__)


def sort_profiles(profiles: Mapping[str, Tuple[Manifest, List[GPUUnit]]]) -> List[Tuple[str, Manifest, List[GPUUnit]]]:
    """Sort hardware compatible manifest configs
    We use penalty points: higher numbers is worse
    """

    def string_to_int(s):
        # arbitrary str to int for deterministic reasons
        # since hash is not deterministic across runs
        return int.from_bytes(s.encode(), 'big')

    def engine_penalty(profile: Manifest) -> int:
        engine = profile.tags().get("llm_engine", None)
        match engine:
            case "vllm":
                return 0
            case "tensorrt_llm":
                return 2
            case _:
                return 2

    def precision_digit_penalty(profile: Manifest) -> int:
        precision = profile.tags().get("precision", None)
        if not precision:
            return 64

        pdigit = re.findall(r'\d+$', precision)
        return int(pdigit[0]) if len(pdigit) else 64

    def precision_type_penalty(profile: Manifest) -> int:
        precision = profile.tags().get("precision", None)
        if not precision:
            return 2

        ptype = re.findall(r'^[a-zA-Z]+', precision)
        if not len(ptype):
            return 2

        match ptype[0]:
            case "fp":
                return 0
            case "bf":
                return 1
            case _:
                return 2

    def profile_penalty(profile: Manifest) -> int:
        match profile.tags().get("profile", None):
            case "latency":
                return 0
            case "balanced":
                return 1
            case "throughput":
                return 2
            case _:
                return 3

    def tp_penalty(profile: Manifest) -> int:
        tp = profile.tags().get("tp", 1)
        return -int(tp)  # higher tp is better when possible, so negate for penalty

    def gpu_name_penalty(profile: Manifest) -> int:
        # fully arbitrary, but need something for determinism
        # Note: ideally would sort based on most performant
        gpu = profile.tags().get("gpu", None)
        return string_to_int(gpu) if gpu else 0

    # TODO: Refactor INFE-2257
    def compare_profiles(profile1: Tuple[str, Manifest], profile2: Tuple[str, Manifest]) -> int:
        # return a negative value when profile1 should be sorted before profile2
        # return a positive value when profile1 should be sorted after profile2
        # return 0 when both profile1 and profile2 have the same weight - we don't want that for determinism
        # We use penalty points below: larger penalty is worse, so if we do penalty1 - penalty2:
        #  if profile1 is better, it has a lower penalty than profile2, and (penalty1 - penalty2) < 0 so profile1 is sorted before - ok!
        #  if profile1 is worse, it has a higher penalty than profile2, and (penalty1 - penalty2) > 0 so profile2 is sorted before - ok!

        # Progressively check tags in this order
        prioritized_tag_checks = [
            engine_penalty,
            precision_digit_penalty,
            precision_type_penalty,
            profile_penalty,
            tp_penalty,
            gpu_name_penalty,
        ]
        for penalty_func in prioritized_tag_checks:
            penalty1 = penalty_func(profile1[1])
            penalty2 = penalty_func(profile2[1])
            # If we have different scores, we can stop comparing right away
            # If the score is the same, check the next tag in priority list
            if penalty1 != penalty2:
                return penalty1 - penalty2

        # Can't sort based on tags? Use profile name for abritrary but deterministic order
        return string_to_int(profile1[0]) - string_to_int(profile2[0])

    # Go from map to list of tuples so we can order it
    sorted_profiles = [(k, v[0], v[1]) for k, v in profiles.items()]

    # Sort
    sorted_profiles.sort(key=cmp_to_key(compare_profiles))

    # return name of most optimal
    return sorted_profiles


def get_profile_description(profile: Manifest) -> str:

    tags = profile.tags()
    components = []
    if "llm_engine" in tags:
        components.append(tags["llm_engine"])
    if "gpu" in tags:
        components.append(tags["gpu"])
    if "precision" in tags:
        components.append(tags["precision"])
    if "tp" in tags:
        components.append(f"tp{tags['tp']}")
    if "pp" in tags:
        components.append(f"pp{tags['pp']}")
    if "profile" in tags:
        components.append(tags['profile'])
    if uses_lora(profile):
        components.append("lora")
    return "-".join(components).lower()


def uses_lora(profile: Manifest) -> bool:
    lora_str = profile.tags().get("feat_lora", "false")
    return lora_str.lower() in ("1", "true", "yes")


def get_all_profiles(manifest_path: Path) -> Mapping[str, Manifest]:
    manifest = Manifest.load_from_file(manifest_path.as_posix())
    return manifest.configs()


# This function returns all profiles matching `enable_lora` and splits them into three buckets
# as defined by the RunnableStatus enum - RUNNABLE, NOT_RUNNABLE_LOW_FREE_GPU_MEMORY, and NOT_COMPATIBLE
def categorize_profiles(
    profiles: Mapping[str, Manifest], system: HwSystem, enable_lora: bool
) -> Mapping[RunnableState, Mapping[str, Tuple[Manifest, List[GPUUnit]]]]:
    groups = {
        RunnableState.RUNNABLE: {},
        RunnableState.NOT_RUNNABLE_LOW_FREE_GPU_MEMORY: {},
        RunnableState.NOT_COMPATIBLE: {},
    }
    for profile_name, profile in profiles.items():
        #if "tensorrt_llm" in profile.tags().get("llm_engine") :
        #    continue
        runnable_state, compatible_gpus = retrieve_compatible_hardware(system, profile)
        if enable_lora == uses_lora(profile):
            groups[runnable_state][profile_name] = (profile, compatible_gpus)
    return groups


def filter_manifest_configs(
    manifest_path: Path,
    system: HwSystem,
    enable_lora: bool,
    override: str = None,
    force: bool = False,
) -> List[Tuple[str, Manifest, List[GPUUnit]]]:
    # parse our local manifest and select a config
    profiles = get_all_profiles(manifest_path)
    profile_descriptions_to_id_map = {
        get_profile_description(config): profile_id for profile_id, config in profiles.items()
    }
    grouped_profiles = categorize_profiles(profiles, system, enable_lora)
    runnable_profiles = grouped_profiles[RunnableState.RUNNABLE]
    not_runnable_low_free_gpu_memory_profiles = grouped_profiles[RunnableState.NOT_RUNNABLE_LOW_FREE_GPU_MEMORY]

    # This function tests whether the service args are incompatible with
    # a given profile because the profile doesnt have lora enabled (or vice versa)
    # This function assumes that `profile_name` is a valid profile in the manifest
    def is_lora_mismatch(profile_name: str) -> bool:
        # Get profiles with the opposite enable_lora flag
        grouped_profiles_mismatch_lora = categorize_profiles(profiles, system, not enable_lora)
        if (
            profile_name in grouped_profiles_mismatch_lora[RunnableState.RUNNABLE]
            or profile_name in grouped_profiles_mismatch_lora[RunnableState.NOT_RUNNABLE_LOW_FREE_GPU_MEMORY]
        ):
            return True
        return False

    if enable_lora:
        logger.info(f"Running MIM with LoRA enabled. Only looking for compatible profiles that support LoRA.")
    else:
        logger.info(f"Running MIM without LoRA. Only looking for compatible profiles that do not support LoRA.")

    logger.info(f"Detected {len(runnable_profiles)} compatible profile(s).")
    if not_runnable_low_free_gpu_memory_profiles:
        logger.info(
            f"Detected additional {len(not_runnable_low_free_gpu_memory_profiles)} compatible profile(s)"
            " that are currently not runnable due to low free GPU memory."
        )

    sorted_profiles = sort_profiles(runnable_profiles)

    if override:
        # Check if override is provided in the human-readable format
        maybe_human_readable = override.lower()
        if maybe_human_readable in profile_descriptions_to_id_map:
            profile_id = profile_descriptions_to_id_map[maybe_human_readable]
        else:
            # override is provided as the hash string
            profile_id = override

        if profile_id == "default":
            # return only the last entry in sorted_profiles because it should be most generic
            return [sorted_profiles[-1]]
        if profile_id in profiles:
            if profile_id in runnable_profiles:
                # a compatible selection was made
                profile, gpu_list = runnable_profiles[profile_id]
                return [(profile_id, profile, gpu_list)]
            elif force:
                return [(profile_id, profiles[profile_id], [])]
            elif profile_id in not_runnable_low_free_gpu_memory_profiles:
                # a compatible but non runnable selection was made
                error_message = (
                    f"Profile '{override}' is currently not runnable due to low free GPU memory."
                    " Please free up the GPUs currently in use and try again."
                )
                error_and_exit(error_message)
            elif is_lora_mismatch(override):
                # there is a mismatch between the lora args in MIM and the profile selected
                if enable_lora:
                    error_message = (
                        f"You are running MIM with LoRA enabled, but the selected profile '{override}'"
                        " does not support LoRA. Please select a profile that supports LoRA, or alternatively,"
                        " run MIM without LoRA."
                    )
                    error_and_exit(error_message)
                else:
                    error_message = (
                        f"You are running MIM without LoRA, but the selected profile '{override}'"
                        " has LoRA enabled. Please select a profile that does not have LoRA enabled,"
                        " or alternatively, run MIM with LoRA enabled."
                    )
                    error_and_exit(error_message)
            else:
                # an incompatible selection was made
                error_message = (
                    f"Profile '{override}' is incompatible with detected hardware."
                    " Please check the system information below and select a compatible profile."
                )
                error_message += "\n"
                # Print the detected system configuration to help the user debug
                if len(system.total_gpu):
                    error_message += str(system)
                else:
                    error_message += "No GPUs found - did you forget `--runtime=nvidia --gpus='\"device=gpuids\"']` ?"
                error_and_exit(error_message)
        else:
            error_message = f"profile '{override}' does not exist"
            error_and_exit(error_message)
    else:
        return sorted_profiles


def profiles_summary(manifest_path: Path, system: HwSystem = get_hardware_spec()) -> str:
    result = ["MODEL PROFILES"]

    all_profiles = get_all_profiles(manifest_path)
    logger.info(f"manifest_path:{manifest_path} all_profiles count:{len(all_profiles)}")
    grouped_profiles_without_lora = categorize_profiles(all_profiles, system, enable_lora=False)
    # 暂时不支持lora
    grouped_profiles_with_lora = categorize_profiles(all_profiles, system, enable_lora=True)

    profile_descriptions = {id: get_profile_description(config) for id, config in all_profiles.items()}

    def _profiles_summary_helper(profiles: Mapping[str, Manifest], indent_level=0):
        spaces = "  " * indent_level  # indent with two spaces
        for id, _, _ in sort_profiles(profiles):
            desc = profile_descriptions[id]
            result.append(f"{spaces}- {id} ({desc})")

    if len(grouped_profiles_without_lora[RunnableState.RUNNABLE]) or len(
        grouped_profiles_with_lora[RunnableState.RUNNABLE]
    ):
        result.append("- Compatible with system and runnable:")
        _profiles_summary_helper(grouped_profiles_without_lora[RunnableState.RUNNABLE], indent_level=1)
        result.append("  - With LoRA support:")
        _profiles_summary_helper(grouped_profiles_with_lora[RunnableState.RUNNABLE], indent_level=2)
    else:
        result.append("- Compatible with system and runnable: <None>")

    if len(grouped_profiles_without_lora[RunnableState.NOT_RUNNABLE_LOW_FREE_GPU_MEMORY]) or len(
        grouped_profiles_with_lora[RunnableState.NOT_RUNNABLE_LOW_FREE_GPU_MEMORY]
    ):
        result.append("- Compatible with system but not runnable due to low GPU free memory")
        _profiles_summary_helper(
            grouped_profiles_without_lora[RunnableState.NOT_RUNNABLE_LOW_FREE_GPU_MEMORY], indent_level=1
        )
        result.append("  - With LoRA support:")
        _profiles_summary_helper(
            grouped_profiles_with_lora[RunnableState.NOT_RUNNABLE_LOW_FREE_GPU_MEMORY], indent_level=2
        )

    if len(grouped_profiles_without_lora[RunnableState.NOT_COMPATIBLE]) or len(
        grouped_profiles_with_lora[RunnableState.NOT_COMPATIBLE]
    ):
        result.append("- Incompatible with system:")
        _profiles_summary_helper(grouped_profiles_without_lora[RunnableState.NOT_COMPATIBLE], indent_level=1)
        _profiles_summary_helper(grouped_profiles_with_lora[RunnableState.NOT_COMPATIBLE], indent_level=1)

    return "\n".join(result)
