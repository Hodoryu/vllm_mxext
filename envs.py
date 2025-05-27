# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# This file is derived from: https://github.com/vllm-project/vllm/blob/main/vllm/envs.py
#
#   Copyright 2024,2024 vLLM Team
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import os
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Dict, Literal, NewType, Optional

_supported_log_levels = ["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"]
LoggingLevel = NewType("LoggingLevel", Literal["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"])


if TYPE_CHECKING:
    VLLM_MXEXT_LOG_LEVEL: LoggingLevel = "INFO"
    VLLM_MXEXT_CONFIGURE_LOGGING: bool = True
    VLLM_MXEXT_LOGGING_CONFIG_PATH: Optional[str] = None
    VLLM_MXEXT_JSONL_LOGGING: bool = False


def _get_logging_level(env_var_name: str, default: str = "INFO") -> LoggingLevel:
    level = os.getenv(env_var_name, default=default).upper()
    if level not in _supported_log_levels:
        raise ValueError(
            f"The value {repr(level)} of {env_var_name} environment "
            f"variable is not supported. Allowed values: {_supported_log_levels} (case doesn't matter). "
            f"If the variable is an empty string or not set, then the default {default} value is used."
        )
    return level


def is_int(value: Any) -> bool:
    try:
        int(value)
    except Exception:
        return False
    return True


def _get_bool_env_variable(env_var_name: str, default: bool) -> bool:
    value = os.getenv(env_var_name)
    if value is None:
        return default
    if not is_int(value) or value not in ("0", "1"):
        raise ValueError(
            f"Supported values of boolean env variable {env_var_name} are '0' and '1', "
            f"whereas the actual value is {repr(value)}"
        )
    return bool(int(value))


environment_variables: Dict[str, Callable[[], Any]] = {
    # ================== Runtime Env Vars ==================
    # Logging configuration
    # If set to 0, vllm_mxext will not configure logging
    # If set to 1, vllm_mxext will configure logging using the default configuration
    #    or the configuration file specified by VLLM_MXEXT_LOGGING_CONFIG_PATH
    "VLLM_MXEXT_CONFIGURE_LOGGING": partial(
        _get_bool_env_variable, env_var_name="VLLM_MXEXT_CONFIGURE_LOGGING", default=True
    ),
    "VLLM_MXEXT_LOGGING_CONFIG_PATH": lambda: os.getenv("VLLM_MXEXT_LOGGING_CONFIG_PATH"),
    # Overrides logging level of the logger of the top of vllm_mxext library
    "VLLM_MXEXT_LOG_LEVEL": partial(_get_logging_level, env_var_name="VLLM_MXEXT_LOG_LEVEL"),
    # By default a vLLM style (readable) logging is used. If the following variable is set to 1, then
    # jsonl format is used
    "VLLM_MXEXT_JSONL_LOGGING": partial(
        _get_bool_env_variable, env_var_name="VLLM_MXEXT_JSONL_LOGGING", default=False
    ),
}

# end-env-vars-definition


def __getattr__(name):
    # lazy evaluation of environment variables
    if name in environment_variables:
        return environment_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(environment_variables.keys())
