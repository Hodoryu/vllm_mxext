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

# JSON values must be strings. Floats, ints, arrays, booleans, objects are not allowed
_JSON_FORMAT = (
    '{"level": "%(levelname)s", "time": "%(asctime)s.%(msecs)03d", "file_path": "%(pathname)s", '
    '"line_number": "%(lineno)d", "message": "%(message)s", "exc_info": "%(exc_text)s", '
    '"stack_info": "%(stack_info)s"}'
)
_READABLE_FORMAT = "%(levelname)s %(asctime)s.%(msecs)d %(filename)s:%(lineno)d] %(message)s"
_DATE_FORMAT = "%m-%d %H:%M:%S"

VLLM_MXEXT_PACKAGE_NAME = "vllm_mxext"

JSONL_LOGGING_CONFIG = {
    "disable_existing_loggers": False,
    "formatters": {
        VLLM_MXEXT_PACKAGE_NAME: {
            "class": "vllm_mxext.logging.JsonFormatter",
            "datefmt": _DATE_FORMAT,
            "format": _JSON_FORMAT,
        },
    },
    "handlers": {
        VLLM_MXEXT_PACKAGE_NAME: {
            "class": "logging.StreamHandler",
            "formatter": VLLM_MXEXT_PACKAGE_NAME,
            "level": "DEBUG",
            "stream": "ext://sys.stdout",
        },
    },
    "loggers": {
        VLLM_MXEXT_PACKAGE_NAME: {
            "handlers": [VLLM_MXEXT_PACKAGE_NAME],
            "level": "INFO",
            "propagate": False,
        },
    },
    "version": 1,
}

READABLE_LOGGING_CONFIG = {
    "disable_existing_loggers": False,
    "formatters": {
        VLLM_MXEXT_PACKAGE_NAME: {
            "class": "logging.Formatter",
            "datefmt": _DATE_FORMAT,
            "format": _READABLE_FORMAT,
        },
    },
    "handlers": {
        VLLM_MXEXT_PACKAGE_NAME: {
            "class": "logging.StreamHandler",
            "formatter": VLLM_MXEXT_PACKAGE_NAME,
            "level": "DEBUG",
            "stream": "ext://sys.stdout",
        },
    },
    "loggers": {
        VLLM_MXEXT_PACKAGE_NAME: {
            "handlers": [VLLM_MXEXT_PACKAGE_NAME],
            "level": "INFO",
            "propagate": False,
        },
    },
    "version": 1,
}
