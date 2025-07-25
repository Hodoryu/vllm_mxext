# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# This file is derived from: https://github.com/vllm-project/vllm/blob/main/vllm/logger.py
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

"""Logging configuration for MIM."""
import datetime
import json
import logging
import os
import sys
from copy import deepcopy
from functools import partial
from logging import Logger, StreamHandler, getLogger
from logging.config import dictConfig
from os import path
from typing import Any, Dict, List, Optional

import vllm_mxext.envs as envs
from vllm_mxext.mx_logging.const import JSONL_LOGGING_CONFIG, READABLE_LOGGING_CONFIG, VLLM_MXEXT_PACKAGE_NAME


VLLM_MXEXT_LOG_LEVEL = envs.VLLM_MXEXT_LOG_LEVEL
VLLM_MXEXT_JSONL_LOGGING = envs.VLLM_MXEXT_JSONL_LOGGING
VLLM_MXEXT_CONFIGURE_LOGGING = envs.VLLM_MXEXT_CONFIGURE_LOGGING
VLLM_MXEXT_LOGGING_CONFIG_PATH = envs.VLLM_MXEXT_LOGGING_CONFIG_PATH


def _rename_logger_in_config(logger_config: Dict[str, Any], logger_name: str, new_name: str) -> Dict[str, Any]:
    logger_config = deepcopy(logger_config)
    if logger_name in logger_config["loggers"]:
        d = logger_config["loggers"][logger_name]
        del logger_config["loggers"][logger_name]
        logger_config["loggers"][new_name] = d
    else:
        raise ValueError(
            f"The logger '{logger_name}' which you are trying to rename doesn't exist "
            f"in logger config:\n{logger_config}"
        )
    return logger_config


def get_handler_tty_stream_level(logger: logging.Logger) -> Optional[int]:
    min_level = None
    while logger is not None:
        current_level = min(
            [
                h.level
                for h in logger.handlers
                if isinstance(h, StreamHandler) and h.stream in [sys.stdout, sys.stderr]
            ],
            default=None,
        )
        if current_level is not None:
            if min_level is None:
                min_level = current_level
            elif min_level is not None:
                min_level = min(min_level, current_level)
        if logger.propagate:
            logger = logger.parent
        else:
            logger = None
    return min_level


def set_level_in_handlers(logging_config: Dict[str, Any], logger_name: str, level: int) -> None:
    for h_name in logging_config["loggers"][logger_name]["handlers"]:
        logging_config["handlers"][h_name]["level"] = level


def do_handlers_stream_into_std(logging_config: Dict[str, Any]) -> bool:
    return all(
        [
            logging_config["handlers"][h_name]["class"] == "logging.StreamHandler"
            and logging_config["handlers"][h_name]["stream"] in ["ext://sys.stdout", "ext://sys.stderr"]
            for h_name in logging_config["loggers"][VLLM_MXEXT_PACKAGE_NAME]["handlers"]
        ]
    )


def raise_on_faulty_logging_config(logging_config: Dict[str, Any], error_prefix: str) -> None:
    if "loggers" not in logging_config or set(logging_config["loggers"]) != {VLLM_MXEXT_PACKAGE_NAME}:
        raise ValueError(
            f"{error_prefix} must contain fields \"loggers\" and there should be only one logger ('vllm_mxext')"
        )
    if "handlers" not in logging_config or not do_handlers_stream_into_std(logging_config):
        raise ValueError(
            f"{error_prefix} must be of class `logging.StreamHandler` " f"and stream into sys.stdout or sys.stderr."
        )


def get_logging_config_for_package(package: str, keep_original_log_level: bool = False) -> Dict[str, Any] | None:
    logging_config: Optional[Dict] = None

    if not VLLM_MXEXT_CONFIGURE_LOGGING and VLLM_MXEXT_LOGGING_CONFIG_PATH:
        raise RuntimeError(
            "VLLM_MXEXT_CONFIGURE_LOGGING evaluated to false, but "
            "VLLM_MXEXT_LOGGING_CONFIG_PATH was given. VLLM_MXEXT_LOGGING_CONFIG_PATH "
            "implies VLLM_MXEXT_CONFIGURE_LOGGING. Please enable "
            "VLLM_MXEXT_CONFIGURE_LOGGING or unset VLLM_MXEXT_LOGGING_CONFIG_PATH."
        )

    if VLLM_MXEXT_CONFIGURE_LOGGING:
        if VLLM_MXEXT_JSONL_LOGGING:
            logging_config = JSONL_LOGGING_CONFIG
        else:
            logging_config = READABLE_LOGGING_CONFIG
    if VLLM_MXEXT_LOGGING_CONFIG_PATH:
        if not path.exists(VLLM_MXEXT_LOGGING_CONFIG_PATH):
            raise RuntimeError(
                "Could not load logging config. File does not exist: %s", VLLM_MXEXT_LOGGING_CONFIG_PATH
            )
        with open(VLLM_MXEXT_LOGGING_CONFIG_PATH, encoding="utf-8", mode="r") as file:
            custom_config = json.loads(file.read())

        if not isinstance(custom_config, dict):
            raise ValueError("Invalid logging config. Expected Dict, got %s.", type(custom_config).__name__)
        logging_config = custom_config
        raise_on_faulty_logging_config(
            logging_config,
            f"The config provided in the file passed in environment variable "
            f"`VLLM_MXEXT_LOGGING_CONFIG_PATH` ({VLLM_MXEXT_LOGGING_CONFIG_PATH})",
        )
    elif logging_config is not None:
        raise_on_faulty_logging_config(
            logging_config,
            f"A default logging config contains an error. "
            f"VLLM_MXEXT_JSONL_LOGGING={VLLM_MXEXT_JSONL_LOGGING} The logging config",
        )
    if logging_config is not None:
        logging_config["loggers"][VLLM_MXEXT_PACKAGE_NAME]["level"] = VLLM_MXEXT_LOG_LEVEL
        logging_config = _rename_logger_in_config(logging_config, VLLM_MXEXT_PACKAGE_NAME, package)
        if (
            package != VLLM_MXEXT_PACKAGE_NAME
            and package in logging.root.manager.loggerDict
            and keep_original_log_level
        ):
            existing_logger = getLogger(package)
            existing_logger_level = existing_logger.level
            handler_level = get_handler_tty_stream_level(existing_logger)
            logging_config["loggers"][package]["level"] = existing_logger_level
            if handler_level is None:
                logging_config["loggers"][package]["handlers"] = []
            else:
                set_level_in_handlers(logging_config, package, handler_level)
    return logging_config


def configure_logger(package: str, keep_original_log_level: bool = False) -> None:
    logging_config = get_logging_config_for_package(package, keep_original_log_level)
    if logging_config is not None:
        dictConfig(logging_config)


def configure_all_loggers_with_handlers_except(
    not_configured_loggers: List[str], keep_original_log_level: bool
) -> None:
    for name in logging.root.manager.loggerDict:
        if (
            not any([logger_name.startswith(name) for logger_name in not_configured_loggers])
            and getLogger(name).handlers
        ):
            configure_logger(name, keep_original_log_level)


def init_logger(name: str) -> Logger:
    """The main purpose of this function is to ensure that loggers are
    retrieved in such a way that we can be sure the root MIM logger has
    already been configured."""

    return logging.getLogger(name)


# The root logger is initialized when the module is imported.
# This is thread-safe as the module is only imported once,
# guaranteed by the Python GIL.
configure_logger(VLLM_MXEXT_PACKAGE_NAME)

logger = init_logger(__name__)
