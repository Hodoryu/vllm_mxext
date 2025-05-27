# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
from vllm_mxext.hub.hardware_inspect import get_hardware_spec
from vllm_mxext.hub.mx_download import get_manifest_path
from vllm_mxext.hub.mx_profile import profiles_summary


def print_system_and_profile_summaries() -> None:
    system = get_hardware_spec()
    if len(system.total_gpu):
        print(system)
    else:
        print("No GPUs found - did you forget `--runtime=nvidia --gpus=[all|gpu ids]` ?")

    try:
        manifest_path = get_manifest_path()
        print(profiles_summary(manifest_path, system))
    except:
        print("Unable to find manifest.")
