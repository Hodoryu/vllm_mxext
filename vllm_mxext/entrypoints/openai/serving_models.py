"""Enhanced LoRA support for model serving."""

import asyncio
from typing import Dict, List, Optional
from vllm.lora.request import LoRARequest
from vllm.entrypoints.openai.protocol import LoadLoRAAdapterRequest, UnloadLoRAAdapterRequest
from vllm_mxext.logger import init_logger

logger = init_logger(__name__)

class LoRAModelManager:
    """Manages LoRA adapters for the model serving."""
    
    def __init__(self, engine_client):
        self.engine_client = engine_client
        self.loaded_adapters: Dict[str, LoRARequest] = {}
        self._next_lora_id = 1
    
    async def load_lora_adapter(self, request: LoadLoRAAdapterRequest) -> Dict:
        """Load a LoRA adapter."""
        try:
            # Create LoRA request
            lora_request = LoRARequest(
                lora_name=request.lora_name,
                lora_int_id=self._next_lora_id,
                lora_local_path=request.lora_path
            )
            
            # Load through engine
            if hasattr(self.engine_client, 'add_lora'):
                await self.engine_client.add_lora(lora_request)
            
            # Track loaded adapter
            self.loaded_adapters[request.lora_name] = lora_request
            self._next_lora_id += 1
            
            logger.info(f"Successfully loaded LoRA adapter: {request.lora_name}")
            return {"status": "success", "lora_id": lora_request.lora_int_id}
            
        except Exception as e:
            logger.error(f"Failed to load LoRA adapter {request.lora_name}: {e}")
            raise
    
    async def unload_lora_adapter(self, request: UnloadLoRAAdapterRequest) -> Dict:
        """Unload a LoRA adapter."""
        try:
            if request.lora_name not in self.loaded_adapters:
                raise ValueError(f"LoRA adapter {request.lora_name} not found")
            
            lora_request = self.loaded_adapters[request.lora_name]
            
            # Unload through engine
            if hasattr(self.engine_client, 'remove_lora'):
                await self.engine_client.remove_lora(lora_request.lora_int_id)
            
            # Remove from tracking
            del self.loaded_adapters[request.lora_name]
            
            logger.info(f"Successfully unloaded LoRA adapter: {request.lora_name}")
            return {"status": "success"}
            
        except Exception as e:
            logger.error(f"Failed to unload LoRA adapter {request.lora_name}: {e}")
            raise
    
    def list_adapters(self) -> List[Dict]:
        """List currently loaded adapters."""
        return [
            {
                "name": name,
                "lora_id": lora_req.lora_int_id,
                "path": lora_req.lora_local_path
            }
            for name, lora_req in self.loaded_adapters.items()
        ]
    
    def get_lora_request(self, lora_name: str) -> Optional[LoRARequest]:
        """Get LoRA request by name."""
        return self.loaded_adapters.get(lora_name)