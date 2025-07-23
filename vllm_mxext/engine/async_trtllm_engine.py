# Add LoRA management methods to AsyncTRTLLMEngine

    async def add_lora(self, lora_request: LoRARequest) -> bool:
        """Add a LoRA adapter to the engine."""
        try:
            if self._tllm_engine and hasattr(self._tllm_engine, '_lora_source'):
                # Load LoRA through the TrtllmModelRunner
                success = await self._tllm_engine.load_lora_adapter(lora_request)
                if success:
                    logger.info(f"Successfully added LoRA {lora_request.lora_name} (ID: {lora_request.lora_int_id})")
                    return True
                else:
                    logger.error(f"Failed to add LoRA {lora_request.lora_name}")
                    return False
            else:
                logger.error("LoRA source not available in TrtllmModelRunner")
                return False
        except Exception as e:
            logger.error(f"Error adding LoRA {lora_request.lora_name}: {e}")
            return False
    
    async def remove_lora(self, lora_id: int) -> bool:
        """Remove a LoRA adapter from the engine."""
        try:
            if self._tllm_engine and hasattr(self._tllm_engine, '_lora_source'):
                success = await self._tllm_engine.unload_lora_adapter(lora_id)
                if success:
                    logger.info(f"Successfully removed LoRA ID: {lora_id}")
                    return True
                else:
                    logger.error(f"Failed to remove LoRA ID: {lora_id}")
                    return False
            else:
                logger.error("LoRA source not available in TrtllmModelRunner")
                return False
        except Exception as e:
            logger.error(f"Error removing LoRA ID {lora_id}: {e}")
            return False
    
    async def list_loras(self) -> Dict[int, str]:
        """List currently loaded LoRA adapters."""
        try:
            if self._tllm_engine and hasattr(self._tllm_engine, '_lora_source'):
                return self._tllm_engine._lora_source.list_loaded_loras()
            return {}
        except Exception as e:
            logger.error(f"Error listing LoRAs: {e}")
            return {}