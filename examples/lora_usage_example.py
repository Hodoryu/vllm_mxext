#!/usr/bin/env python3
"""Example usage of LoRA functionality in vLLM MxExt."""

import asyncio
import json
import requests
from typing import Dict, List

class LoRAUsageExample:
    """Example class demonstrating LoRA usage."""
    
    def __init__(self, server_url: str = "http://localhost:8000"):
        self.server_url = server_url.rstrip('/')
    
    def load_lora_adapter(self, lora_name: str, lora_path: str) -> Dict:
        """Load a LoRA adapter."""
        url = f"{self.server_url}/v1/load_lora_adapter"
        payload = {
            "lora_name": lora_name,
            "lora_path": lora_path
        }
        
        response = requests.post(url, json=payload)
        return response.json()
    
    def chat_with_lora(self, lora_name: str, messages: List[Dict]) -> Dict:
        """Chat using a specific LoRA adapter."""
        url = f"{self.server_url}/v1/chat/completions"
        payload = {
            "model": f"base_model:{lora_name}",  # Format: base_model:lora_name
            "messages": messages,
            "max_tokens": 150,
            "temperature": 0.7
        }
        
        response = requests.post(url, json=payload)
        return response.json()
    
    def run_example(self):
        """Run the complete example."""
        print("🚀 LoRA Usage Example for vLLM MxExt")
        print("=" * 50)
        
        # Example 1: Load a math-tuned LoRA
        print("\n1. Loading math-tuned LoRA adapter...")
        lora_path = "/path/to/math_lora_adapter"  # Replace with actual path
        result = self.load_lora_adapter("math_tuned", lora_path)
        print(f"Load result: {json.dumps(result, indent=2)}")
        
        # Example 2: Use the LoRA for math problems
        print("\n2. Testing math reasoning with LoRA...")
        math_messages = [
            {"role": "user", "content": "Solve this equation: 2x + 5 = 13"}
        ]
        
        response = self.chat_with_lora("math_tuned", math_messages)
        print(f"Math response: {json.dumps(response, indent=2)}")
        
        # Example 3: Load a code-tuned LoRA
        print("\n3. Loading code-tuned LoRA adapter...")
        code_lora_path = "/path/to/code_lora_adapter"  # Replace with actual path
        result = self.load_lora_adapter("code_tuned", code_lora_path)
        print(f"Load result: {json.dumps(result, indent=2)}")
        
        # Example 4: Use the LoRA for code generation
        print("\n4. Testing code generation with LoRA...")
        code_messages = [
            {"role": "user", "content": "Write a Python function to calculate fibonacci numbers"}
        ]
        
        response = self.chat_with_lora("code_tuned", code_messages)
        print(f"Code response: {json.dumps(response, indent=2)}")
        
        # Example 5: List all loaded LoRAs
        print("\n5. Listing all loaded LoRA adapters...")
        list_url = f"{self.server_url}/v1/lora_adapters"
        response = requests.get(list_url)
        print(f"Loaded LoRAs: {json.dumps(response.json(), indent=2)}")


if __name__ == "__main__":
    example = LoRAUsageExample()
    example.run_example()