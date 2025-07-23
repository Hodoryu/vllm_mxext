# Add LoRA management methods to TrtllmModelRunner

    async def load_lora_adapter(self, lora_request: LoRARequest) -> bool:
        """Load a LoRA adapter."""
        try:
            if self._lora_source is None:
                logger.error("LoRA source not initialized")
                return False
            
            # Load LoRA asynchronously
            lora_data = await self._lora_source.load_lora_async(lora_request)
            if lora_data is None:
                logger.error(f"Failed to load LoRA data for {lora_request.lora_name}")
                return False
            
            logger.info(f"Successfully loaded LoRA adapter: {lora_request.lora_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading LoRA adapter {lora_request.lora_name}: {e}")
            return False
    
    async def unload_lora_adapter(self, lora_id: int) -> bool:
        """Unload a LoRA adapter."""
        try:
            if self._lora_source is None:
                logger.error("LoRA source not initialized")
                return False
            
            self._lora_source.unload_lora(lora_id)
            logger.info(f"Successfully unloaded LoRA adapter ID: {lora_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error unloading LoRA adapter ID {lora_id}: {e}")
            return False
    
    def get_loaded_loras(self) -> Dict[int, str]:
        """Get list of currently loaded LoRA adapters."""
        if self._lora_source:
            return self._lora_source.list_loaded_loras()
        return {}