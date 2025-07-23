import asyncio
import json
import os
import tempfile
import torch
import numpy as np
from typing import Dict, Optional, Tuple, Union, Any
from pathlib import Path

from vllm.lora.request import LoRARequest
from vllm_mxext.logger import init_logger

logger = init_logger(__name__)

# Type definitions
RawLora = Dict[str, Any]
HuggingfaceLora = Tuple[Dict, torch.Tensor]
TrtllmLora = Tuple[torch.Tensor, torch.Tensor]
NemoLora = Any

class LoraSource:
    """Base class for LoRA model loading and management."""
    
    def __init__(self):
        self.loaded_loras: Dict[str, Any] = {}
        self.lora_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    
    @staticmethod
    def _numpy_to_torch(arr: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(arr)
    
    def load_lora_from_path(self, lora_path: str, lora_format: str = "auto") -> Optional[Any]:
        """Load LoRA from local path."""
        try:
            if not os.path.exists(lora_path):
                logger.error(f"LoRA path does not exist: {lora_path}")
                return None
            
            if lora_format == "auto":
                lora_format = self._detect_lora_format(lora_path)
            
            if lora_format == "huggingface":
                return self._load_huggingface_from_path(lora_path)
            elif lora_format == "nemo":
                return self._load_nemo_from_path(lora_path)
            elif lora_format == "trtllm":
                return self._load_trtllm_from_path(lora_path)
            else:
                logger.error(f"Unsupported LoRA format: {lora_format}")
                return None
                
        except Exception as e:
            logger.error(f"Error loading LoRA from {lora_path}: {e}")
            return None
    
    def _detect_lora_format(self, lora_path: str) -> str:
        """Auto-detect LoRA format based on files present."""
        path = Path(lora_path)
        
        if (path / "adapter_config.json").exists() and (path / "adapter_model.bin").exists():
            return "huggingface"
        elif any(f.suffix == ".nemo" for f in path.iterdir() if f.is_file()):
            return "nemo"
        elif (path / "model.lora_config.npy").exists() and (path / "model.lora_weights.npy").exists():
            return "trtllm"
        else:
            logger.warning(f"Could not detect LoRA format for {lora_path}, defaulting to huggingface")
            return "huggingface"
    
    def _load_huggingface_from_path(self, lora_path: str) -> HuggingfaceLora:
        """Load HuggingFace format LoRA from path."""
        config_path = os.path.join(lora_path, "adapter_config.json")
        model_path = os.path.join(lora_path, "adapter_model.bin")
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        model = torch.load(model_path, map_location='cpu')
        return config, model
    
    def _load_nemo_from_path(self, lora_path: str) -> NemoLora:
        """Load NEMO format LoRA from path."""
        nemo_files = list(Path(lora_path).glob("*.nemo"))
        if not nemo_files:
            raise FileNotFoundError(f"No .nemo files found in {lora_path}")
        
        nemo_file = nemo_files[0]
        with open(nemo_file, 'rb') as f:
            nemo_bytes = f.read()
        
        return self.open_nemo_lora(lora_path=str(nemo_file), nemo_bytes=nemo_bytes)
    
    def _load_trtllm_from_path(self, lora_path: str) -> TrtllmLora:
        """Load TensorRT-LLM format LoRA from path."""
        config_path = os.path.join(lora_path, "model.lora_config.npy")
        weights_path = os.path.join(lora_path, "model.lora_weights.npy")
        
        config = self._numpy_to_torch(np.load(config_path)).squeeze(0)
        weights = self._numpy_to_torch(np.load(weights_path)).squeeze(0)
        return config, weights
    
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

    def open_nemo_lora(self, lora_path: Optional[str] = None, nemo_bytes: Optional[bytes] = None):
        """Open NEMO LoRA - placeholder for actual NEMO implementation."""
        # This would need actual NEMO library integration
        logger.warning("NEMO LoRA support requires additional dependencies")
        return None


class TrtllmLoraSource(LoraSource):
    """TensorRT-LLM specific LoRA source implementation."""
    
    def __init__(self):
        super().__init__()
        self.dtype = torch.float16
        self._lora_registry: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
    
    async def load_lora_async(self, lora_request: LoRARequest) -> Optional[TrtllmLora]:
        """Asynchronously load LoRA adapter."""
        try:
            # Check cache first
            if lora_request.lora_int_id in self._lora_registry:
                logger.info(f"Using cached LoRA {lora_request.lora_int_id}")
                return self._lora_registry[lora_request.lora_int_id]
            
            # Load from path
            if lora_request.lora_local_path:
                lora_data = await asyncio.get_event_loop().run_in_executor(
                    None, self.load_lora_from_path, lora_request.lora_local_path, "trtllm"
                )
                
                if lora_data:
                    # Convert to appropriate dtype
                    config, weights = lora_data
                    if self.dtype != config.dtype:
                        config = config.to(self.dtype)
                        weights = weights.to(self.dtype)
                    
                    # Cache the loaded LoRA
                    self._lora_registry[lora_request.lora_int_id] = (config, weights)
                    logger.info(f"Successfully loaded LoRA {lora_request.lora_int_id} from {lora_request.lora_local_path}")
                    return config, weights
            
            logger.error(f"Failed to load LoRA {lora_request.lora_int_id}")
            return None
            
        except Exception as e:
            logger.error(f"Error loading LoRA {lora_request.lora_int_id}: {e}")
            return None
    
    def unload_lora(self, lora_id: int):
        """Unload LoRA from cache."""
        if lora_id in self._lora_registry:
            del self._lora_registry[lora_id]
            logger.info(f"Unloaded LoRA {lora_id}")
    
    def list_loaded_loras(self) -> Dict[int, str]:
        """List currently loaded LoRAs."""
        return {lora_id: f"LoRA_{lora_id}" for lora_id in self._lora_registry.keys()}


def open_nemo_lora(lora_path: Optional[str] = None, nemo_bytes: Optional[bytes] = None):
    """Placeholder for NEMO LoRA loading."""
    logger.warning("NEMO LoRA support not implemented")
    return None