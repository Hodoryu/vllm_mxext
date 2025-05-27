# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

INJECT_NGC = "inject_mx_hub"
_MODULES = [
    "hardware_inspect",
    "mx_download",
    "mx_injector",
    "mx_profile",
]

__all__ = [INJECT_NGC]
__all__.extend(_MODULES)


# Autocomplete
def __dir__():
    return __all__


# Lazily load extra modules
def __getattr__(name):
    import importlib
    import sys

    if name in _MODULES:
        module_name = f"{__name__}.{name}"
        module = importlib.import_module(module_name)  # import
        sys.modules[module_name] = module  # cache
        return module
    elif name is INJECT_NGC:
        from .mx_injector import inject_mx_hub

        return inject_mx_hub
    elif name == "print_system_and_profile_summaries":
        from .info import print_system_and_profile_summaries

        return print_system_and_profile_summaries
    else:
        raise AttributeError(f"module {__name__} has no attribute {name}")
