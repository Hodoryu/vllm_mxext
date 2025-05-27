# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import asyncio
import enum
import io
import json
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union, cast

import numpy as np
import torch
from nvidia.lora_conversions.convert_lora import convert_nemo_to_canonical, open_nemo_lora
from nvidia.lora_conversions.huggingface_to_trtllm import (
    convert_hf_model_to_trtllm,
    get_all_lora_weights,
    str_dtype_to_torch,
)
from nvidia.lora_conversions.unfused_to_huggingface import (
    nemo_model_config_to_adapter_config,
    reformat_module_names_to_hf,
)
from nvidia.lora_conversions.utils import np_bfloat16, np_float8


class LoraFormat(enum.Enum):
    NEMO = 0
    HUGGINGFACE = 1
    TRTLLM = 2


RawLora = Dict[str, io.BytesIO]
NemoLora = Tuple[Dict[str, Any], Dict[str, torch.Tensor]]
HuggingfaceLora = Tuple[Dict[str, Any], Dict[str, torch.Tensor]]
TrtllmLora = Tuple[torch.Tensor, torch.Tensor]

AnyLora = Union[NemoLora, HuggingfaceLora, TrtllmLora]


class LoraSource:
    """
    Provides LoRAs from a variety of sources
    """

    def __init__(self, source: str | Path | None = None, dtype: str = 'float16'):
        self._source = Path(source) if source is not None else None
        self._dtype = dtype
        if self._source is not None and not self._source.is_dir():
            raise ValueError("source must be a directory")

    @property
    def dtype(self):
        return self._dtype

    @dtype.setter
    def dtype(self, t: str):
        self._dtype = t

    @staticmethod
    def _get_raw_lora_bytes(lora_path: Path) -> Dict[str, io.BytesIO]:
        if not lora_path.is_dir():
            raise ValueError(f"lora not found {lora_path}")

        file_list = lora_path.glob('*')
        return {p.name: LoraSource._load_file_bytes(p) for p in file_list}

    @staticmethod
    def _load_file_bytes(path: Path) -> io.BytesIO:
        with open(path, 'rb') as fh:
            return io.BytesIO(fh.read())

    @staticmethod
    def _detect_format(lora_bytes: Dict[str, io.BytesIO]) -> LoraFormat:
        is_nemo = False
        has_hf_config, has_hf_weights = False, False
        has_tllm_config, has_tllm_weights = False, False
        for k in lora_bytes.keys():
            if k.endswith(".nemo"):
                is_nemo = True
            elif k == "adapter_config.json":
                has_hf_config = True
            elif k == "adapter_model.bin":
                has_hf_weights = True
            elif k == "model.lora_config.npy":
                has_tllm_config = True
            elif k == "model.lora_weights.npy":
                has_tllm_weights = True

        if is_nemo:
            return LoraFormat.NEMO
        elif has_hf_config and has_hf_weights:
            return LoraFormat.HUGGINGFACE
        elif has_tllm_config and has_tllm_weights:
            return LoraFormat.TRTLLM

        raise Exception("lora format could not be determined")

    @staticmethod
    def _load_nemo(lora_bytes: RawLora) -> NemoLora:
        nemo_bytes: Optional[bytes] = None
        for k, v in lora_bytes.items():
            if k.endswith(".nemo"):
                nemo_bytes = v.read()
        return open_nemo_lora(lora_path=None, nemo_bytes=nemo_bytes)

    @staticmethod
    def _load_hugging_face(raw_lora: RawLora) -> HuggingfaceLora:
        config = json.load(raw_lora["adapter_config.json"])
        model = torch.load(raw_lora["adapter_model.bin"])
        return config, model

    @staticmethod
    def _load_trtllm(raw_lora: RawLora) -> TrtllmLora:
        config = LoraSource._numpy_to_torch(np.load(raw_lora["model.lora_config.npy"])).squeeze(0)
        weights = LoraSource._numpy_to_torch(np.load(raw_lora["model.lora_weights.npy"])).squeeze(0)
        return config, weights

    @staticmethod
    def _load_lora(raw_lora: RawLora) -> Tuple[AnyLora, LoraFormat]:
        format = LoraSource._detect_format(raw_lora)
        load_fn: Callable[[RawLora], AnyLora] = {
            LoraFormat.NEMO: LoraSource._load_nemo,
            LoraFormat.HUGGINGFACE: LoraSource._load_hugging_face,
            LoraFormat.TRTLLM: LoraSource._load_trtllm,
        }[format]
        return load_fn(raw_lora), format

    @staticmethod
    def _convert_to_huggingface(lora: AnyLora, format: LoraFormat) -> HuggingfaceLora:
        if format == LoraFormat.HUGGINGFACE:
            return cast(HuggingfaceLora, lora)
        if format != LoraFormat.NEMO:
            raise NotImplementedError()
        return LoraSource._convert_nemo_to_hf(cast(NemoLora, lora))

    @staticmethod
    def _convert_nemo_to_hf(lora: NemoLora) -> HuggingfaceLora:
        config, lora_weights = lora
        lora_config = config["peft"]["lora_tuning"]
        if "variant" not in lora_config or lora_config["variant"] == "nemo":
            config, lora_weights = convert_nemo_to_canonical(config, lora_weights)

        tensors = reformat_module_names_to_hf(lora_weights)
        lora_config = nemo_model_config_to_adapter_config(lora_config)
        return lora_config, tensors

    @staticmethod
    def _convert_hf_to_trtllm(lora: HuggingfaceLora, dtype: str) -> TrtllmLora:
        config, weights = lora
        weights_obj = get_all_lora_weights(weights)
        # not all the ops needed fro conversion are implemented for fp8 so do the conversion in fp16
        if dtype in ["fp8", "float8"]:
            convert_dtype = "float16"
        else:
            convert_dtype = dtype
        np_config, np_weights = convert_hf_model_to_trtllm(config, weights_obj, convert_dtype)
        return (
            LoraSource._numpy_to_torch(np_config).squeeze(0),
            LoraSource._numpy_to_torch(np_weights).squeeze(0).to(str_dtype_to_torch(dtype)),
        )

    @staticmethod
    def _numpy_to_torch(numpy_tensor: np.ndarray) -> torch.Tensor:
        if numpy_tensor.dtype == np_float8:
            return torch.from_numpy(numpy_tensor.view(np.int8)).view(torch.float8_e4m3fn)
        elif numpy_tensor.dtype == np_bfloat16:
            return torch.from_numpy(numpy_tensor.view(np.int16)).view(torch.bfloat16)
        else:
            return torch.from_numpy(numpy_tensor)

    @staticmethod
    def _convert_to_trtllm(lora: AnyLora, format: LoraFormat, dtype: str) -> TrtllmLora:
        if format == LoraFormat.TRTLLM:
            return cast(TrtllmLora, lora)
        if format == LoraFormat.NEMO:
            hf_lora = LoraSource._convert_nemo_to_hf(cast(NemoLora, lora))
            return LoraSource._convert_hf_to_trtllm(hf_lora, dtype)
        if format == LoraFormat.HUGGINGFACE:
            return LoraSource._convert_hf_to_trtllm(cast(HuggingfaceLora, lora), dtype)
        raise NotImplementedError()

    def get_lora(self, lora_id: str, local_path: Optional[Path] = None) -> Any:
        """
        Returns lora specific to backend
        """
        if local_path is not None:
            lora_path = local_path
        elif self._source is not None:
            lora_path = self._source / lora_id
        else:
            raise ValueError("lora_path must be specified as the LoraSource has no source dir")

        lora_bytes = self._get_raw_lora_bytes(lora_path)
        lora, format = self._load_lora(lora_bytes)
        return self._convert_to_backend_inputs(lora, format)

    def _convert_to_backend_inputs(self, lora: AnyLora, format: LoraFormat) -> AnyLora:
        """
        convert raw lora to backend specific type
        """
        raise NotImplementedError()


class TrtllmLoraSource(LoraSource):
    def _convert_to_backend_inputs(self, lora: AnyLora, format: LoraFormat) -> TrtllmLora:
        return self._convert_to_trtllm(lora, format, self._dtype)
