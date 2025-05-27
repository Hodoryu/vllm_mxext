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
import uuid
from enum import Enum
from typing import List, Mapping, Tuple, TypeVar
import torch
import subprocess


from vllm_mxext.logger import init_logger
from vllm_mxext.hub.utils import error_and_exit

logger = init_logger(__name__)

TENSOR_PARALLEL_SIZE = "tp"
TAG_GPU_ID_KEY = "gpu_device"


# Enum to indicate whether a profile can be run on the avaiable hardware or not
class RunnableState(Enum):
    # Profile is compatible and can run on the current hardware in the current state
    RUNNABLE = 1

    # Profile is compatible, but one or more GPUs has insufficient free memory
    NOT_RUNNABLE_LOW_FREE_GPU_MEMORY = 2

    # Profile is not compatible and cannot run on the current hardware
    NOT_COMPATIBLE = 3


# note sxm engines are not coming up cleanly for all PCIe configs
gpu_families = {
    "C500 64GB": set(
        [
            "9999:4001",  
        ]
    ),
    "C280 64GB": set(
        [
            "9999:4081",
        ]
    ),
    "N260 64GB": set(
        [
            "9999:4083",  
        ]
    ),
    "C550 64GB": set(
        [
            "9999:4000",  
        ]
    ),
}
gpu_family_lookup = {did: family for family, devices in gpu_families.items() for did in devices}

R = TypeVar('R')

def gpu_used_memory(device_id: int):
            cmd = "mx-smi |grep ' MiB'  | awk '{print $7}'"
            result = subprocess.run("mx-smi | grep ' MiB' | awk '{print $7}'", shell=True, capture_output=True, text=True)
            # 从 result 中获取标准输出
            out = result.stdout  # result 是一个包含 stdout 和 stderr 的对象
            # 将输出内容存放在 key-value 容器中，其中 key 为输出内容的行号，value 为 MiB 值
            output_lines = out.strip().split('\n')  # 按行分割输出
            result = {}
            for i, line in enumerate(output_lines, 0):  # 从 1 开始计数行号
                value = line.split('/')[0]  
                result[i] = (int(value) * 1024 * 1024)
            return result[device_id]
        
def gpu_pcie_id():
            cmd = "mx-smi | grep ' MetaX' | grep -v 'MetaX System Management Interface Log' | awk '{print $4}'"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            # 从 result 中获取标准输出
            out = result.stdout  # result 是一个包含 stdout 和 stderr 的对象
            # 将输出内容存放在 key-value 容器中，其中 key 为输出内容的行号，value 为 MiB 值
            output_lines = out.strip().split('\n')  # 按行分割输出
            result = {}
            if "C280" in output_lines[0]:
                return "9999:4081"
            elif "C500" in  output_lines[0]:
                return "9999:4001"
            elif "N260"  in output_lines[0]:
                return "9999:4083"
            elif "C550" in output_lines[0]:
                return "9999:4000"
            else:
                logger.error(f"get gpu_gpu_pcie_id fail. C280 or C500 chars not contained in mx-smi output")
                error_message = (
                    f"Get gpu_gpu_pcie_id fail. C280 or C500 chars not contained in mx-smi output."
                )   
                error_and_exit(error_message)
                return ""
            
            

class GPUInspect:
    def __init__(self):
        if not torch.cuda.is_available():
            raise RuntimeError("No GPU available.")

    def device_count(self):
        return torch.cuda.device_count()

    def device_name(self, device_id: int) -> str:
        return torch.cuda.get_device_name(device_id)

    def device_uuid(self, device_id: int) -> str:
        # Torch does not provide direct access to UUID, using pynvml
        """Generate a mock UUID for the GPU."""
        # PyTorch doesn't expose GPU UUIDs, so generating a random UUID.
        # Replace this if you can access the real UUID via another method.
        return f"GPU-{uuid.uuid4()}"

    def device_mem(self, device_id: int) -> Tuple[int, int]:
        """Return the total and free memory (in bytes) for the GPU."""
        total_memory = torch.cuda.get_device_properties(device_id).total_memory
        free_memory = total_memory - gpu_used_memory(device_id)
        return total_memory, free_memory

    def device_compute_capability(self, device_id: int) -> Tuple[int, int]:
        capability = torch.cuda.get_device_capability(device_id)
        return capability

    def device_pcie_device_id(self, device_id: int) -> str:
        # PCIe device ID needs to be obtained using pynvml
        # PyTorch doesn't expose PCIe device IDs, using major and minor compute capability as a placeholder
        #major, minor = self.device_compute_capability(device_id)
        #device_vendor_id = f"{major:02x}{minor:02x}"
        device_vendor_id = gpu_pcie_id()
        return device_vendor_id
# class GPUInspect:
#     def __init__(self):
#         GPUInspect._safe_exec(cuda.cuInit(0))
#         pynvml.nvmlInit()

#     @staticmethod
#     def _safe_exec(result: Tuple) -> R:
#         status = result[0]
#         if status == cuda.CUresult.CUDA_SUCCESS:
#             return result
#         raise RuntimeError(f"Unexpected error: {status.name}")

#     def device_count(self):
#         return GPUInspect._safe_exec(cuda.cuDeviceGetCount())[1]

#     def device_name(self, device_id: int) -> str:
#         _, device = GPUInspect._safe_exec(cuda.cuDeviceGet(device_id))
#         _, name_bytes = GPUInspect._safe_exec(cuda.cuDeviceGetName(255, device))
#         return name_bytes.split(b'\x00', 1)[0].decode("utf-8")

#     def device_uuid(self, device_id: int) -> str:
#         _, device = GPUInspect._safe_exec(cuda.cuDeviceGet(device_id))
#         _, uuid_data = GPUInspect._safe_exec(cuda.cuDeviceGetUuid(device))
#         return f"GPU-{uuid.UUID(int=int.from_bytes(uuid_data.bytes, 'big'))}"

#     def device_mem(self, device_id: int) -> Tuple[int, int]:
#         device_uuid = self.device_uuid(device_id)
#         handle = pynvml.nvmlDeviceGetHandleByUUID(device_uuid)
#         mem_data = pynvml.nvmlDeviceGetMemoryInfo(handle)
#         return mem_data.total, mem_data.free

#     def device_compute_capability(self, device_id: int) -> Tuple[int, int]:
#         _, device = GPUInspect._safe_exec(cuda.cuDeviceGet(device_id))
#         _, major = GPUInspect._safe_exec(
#             cuda.cuDeviceGetAttribute(cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device)
#         )
#         _, minor = GPUInspect._safe_exec(
#             cuda.cuDeviceGetAttribute(cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device)
#         )
#         return major, minor

#     def device_pcie_device_id(self, device_id: int) -> str:
#         device_uuid = self.device_uuid(device_id)
#         handle = pynvml.nvmlDeviceGetHandleByUUID(device_uuid)
#         pcie_data = pynvml.nvmlDeviceGetPciInfo_v3(handle)
#         device_vendor_id = f"{pcie_data.pciDeviceId:0x}"
#         return f"{device_vendor_id[:4]}:{device_vendor_id[4:]}"


class GPUUnit:
    """A single GPU device"""

    def __init__(self, name: str, device_index: int, device_id: str, total_memory: float, free_memory: float) -> None:
        self.name = name
        self.device_index = device_index
        self.device_id = device_id
        self.total_memory = total_memory  # bytes
        self.free_memory = free_memory  # bytes
        self.family = None

        # look up family
        for family, devices in gpu_families.items():
            if device_id in devices:
                self.family = family

    def __str__(self) -> str:
        #family_str ="Metax C500 64GB PCIe"
        mem = int(self.current_memory_utilization * 100)
        return f"[{self.device_id}] ({self.device_index}) {self.name} {self.family}[current utilization: {mem}%]"

    @property
    def current_memory_utilization(self) -> int:
        return 1 - self.free_memory / self.total_memory

    def is_free(self) -> bool:
        return self.current_memory_utilization <= 0.05


class HwSystem:
    """System artifacts containing `GPUUnit`"""

    def __init__(self) -> None:
        # mapping GPU families to list of GPUs
        self.total_gpu: Mapping[str, List[GPUUnit]] = {}
        self.free_gpu: Mapping[str, List[GPUUnit]] = {}

    def __str__(self) -> str:
        result = ["SYSTEM INFO"]
        total_gpus = [gpu for family in self.total_gpu.values() for gpu in family]
        free_gpus = [gpu for family in self.free_gpu.values() for gpu in family]
        if len(free_gpus):
            result.append("- Free GPUs:")
            result.extend([f"  -  {gpu}" for gpu in free_gpus])
        else:
            result.append("- Free GPUs: <None>")

        if len(total_gpus) > len(free_gpus):
            result.append("- Non-free GPUs:")
            result.extend([f"  -  {gpu}" for gpu in total_gpus if gpu not in free_gpus])

        return "\n".join(result)

    def num_total_gpus(self, family: str = None) -> int:
        if family:
            return len(self.total_gpu.get(family, []))
        return sum(len(all_gpus) for all_gpus in self.total_gpu.values())

    def num_free_gpus(self, family: str = None) -> int:
        if family:
            return len(self.free_gpu.get(family, []))
        return sum(len(all_gpus) for all_gpus in self.free_gpu.values())

    def add_gpu(self, gpu: GPUUnit) -> None:
        family = gpu.family if gpu.family else "unknown"
        if not self.total_gpu.get(family):
            self.total_gpu[family] = []
        self.total_gpu.get(family).append(gpu)
        if gpu.is_free():
            if not self.free_gpu.get(family):
                self.free_gpu[family] = []
            self.free_gpu.get(family).append(gpu)

    def get_free_gpus(self, requested_number_of_gpus: int = 0, family: str = None) -> List[GPUUnit]:
        if family:
            if self.free_gpu.get(family) and len(self.free_gpu.get(family)) >= requested_number_of_gpus:
                return self.free_gpu.get(family)[:requested_number_of_gpus]
            return []
        else:
            # if no family is specified, we just return the first n first available GPUs of the same family
            for _, sublist in self.free_gpu.items():
                if len(sublist) >= requested_number_of_gpus:
                    return sublist
            return []


def get_hardware_spec() -> HwSystem:
    """GPU specifications for the current system"""
    # Get the GPU list
    gpus = GPUInspect()
    system = HwSystem()
    # Iterate through GPUs and detect system information
    for device_id in range(gpus.device_count()):
        device_mem_total, device_mem_free = gpus.device_mem(device_id)
        device_name = gpus.device_name(device_id)
        device_pcie_id = gpus.device_pcie_device_id(device_id)

        gpu_device = GPUUnit(device_name, device_id, device_pcie_id, device_mem_total, device_mem_free)
        logging.info(
            f"Device {device_name} device_id:{device_id} device_pcie_id:{device_pcie_id} -- Total memory: {device_mem_total}, Total free memory: {device_mem_free} gpu famliy:{gpu_device.family}"
        )
        system.add_gpu(gpu_device)
    return system


def retrieve_compatible_hardware(system: HwSystem, config) -> Tuple[RunnableState, List[GPUUnit]]:
    """retrieve the list of compatible gpus"""
    tags = config.tags()
    #logger.info(f"retrieve_compatible_hardware====> config:{config} tags:{tags}")
    if "tensorrt_llm" in tags.get("llm_engine") :
        logger.warning(f"Not support tensorrt_llm inference engine. tag:{tags}")
        return RunnableState.NOT_COMPATIBLE, []
    if tags:
        profile_gpu_family = gpu_family_lookup.get(tags.get(TAG_GPU_ID_KEY), None)
        requested_number_of_gpus = int(tags.get(TENSOR_PARALLEL_SIZE, 1))
        num_total_gpus = system.num_total_gpus(profile_gpu_family)
        free_gpus = system.get_free_gpus(requested_number_of_gpus, profile_gpu_family)
        #logger.info(f"retrieve_compatible_hardware ===> profile_gpu_family:{profile_gpu_family} requested_number_of_gpus:{requested_number_of_gpus} num_total_gpus:{num_total_gpus} free_gpus:{free_gpus}")
        if free_gpus:
            logger.debug(
                f"Profile matched with 'tp': {tags.get(TENSOR_PARALLEL_SIZE)}, 'gpu_arch': {tags.get(TAG_GPU_ID_KEY)}"
            )
            return RunnableState.RUNNABLE, free_gpus
        elif requested_number_of_gpus <= num_total_gpus:
            logger.debug(
                f"Profile matched with 'tp': {tags.get(TENSOR_PARALLEL_SIZE)}, 'gpu_arch': {tags.get(TAG_GPU_ID_KEY)}"
                " however there is insufficient free GPU memory to use this profile."
            )
            return RunnableState.NOT_RUNNABLE_LOW_FREE_GPU_MEMORY, []
    return RunnableState.NOT_COMPATIBLE, []
