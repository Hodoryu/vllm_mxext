#!/usr/bin/env python3
"""LoRA management CLI tool for vLLM MxExt."""

import argparse
import asyncio
import json
import requests
from typing import Dict, List
from pathlib import Path

from vllm_mxext.logger import init_logger

logger = init_logger(__name__)

class LoRAManagerCLI:
    """Command-line interface for managing LoRA adapters."""
    
    def __init__(self, server_url: str = "http://0.0.0.0:8000"):
        self.server_url = server_url.rstrip('/')
    
    def load_lora(self, lora_name: str, lora_path: str) -> Dict:
        """Load a LoRA adapter."""
        url = f"{self.server_url}/v1/load_lora_adapter"
        payload = {
            "lora_name": lora_name,
            "lora_path": lora_path
        }
        
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Failed to load LoRA: {e}")
            return {"error": str(e)}
    
    def unload_lora(self, lora_name: str) -> Dict:
        """Unload a LoRA adapter."""
        url = f"{self.server_url}/v1/unload_lora_adapter"
        payload = {"lora_name": lora_name}
        
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Failed to unload LoRA: {e}")
            return {"error": str(e)}
    
    def list_loras(self) -> Dict:
        """List currently loaded LoRA adapters."""
        url = f"{self.server_url}/v1/lora_adapters"
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Failed to list LoRAs: {e}")
            return {"error": str(e)}
    
    def test_lora_inference(self, lora_name: str, prompt: str) -> Dict:
        """Test inference with a specific LoRA adapter."""
        url = f"{self.server_url}/v1/chat/completions"
        payload = {
            "model": lora_name,  # Use LoRA name as model
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 100
        }
        
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Failed to test LoRA inference: {e}")
            return {"error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="LoRA Management CLI for vLLM MxExt")
    parser.add_argument("--server-url", default="http://0.0.0.0:8000", 
                       help="Server URL (default: http://0.0.0.0:8000)")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Load LoRA command
    load_parser = subparsers.add_parser("load", help="Load a LoRA adapter")
    load_parser.add_argument("lora_name", help="Name for the LoRA adapter")
    load_parser.add_argument("lora_path", help="Path to the LoRA adapter files")
    
    # Unload LoRA command
    unload_parser = subparsers.add_parser("unload", help="Unload a LoRA adapter")
    unload_parser.add_argument("lora_name", help="Name of the LoRA adapter to unload")
    
    # List LoRAs command
    list_parser = subparsers.add_parser("list", help="List loaded LoRA adapters")
    
    # Test LoRA command
    test_parser = subparsers.add_parser("test", help="Test LoRA inference")
    test_parser.add_argument("lora_name", help="Name of the LoRA adapter to test")
    test_parser.add_argument("prompt", help="Test prompt")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    manager = LoRAManagerCLI(args.server_url)
    
    if args.command == "load":
        if not Path(args.lora_path).exists():
            print(f"Error: LoRA path does not exist: {args.lora_path}")
            return
        
        result = manager.load_lora(args.lora_name, args.lora_path)
        print(json.dumps(result, indent=2))
    
    elif args.command == "unload":
        result = manager.unload_lora(args.lora_name)
        print(json.dumps(result, indent=2))
    
    elif args.command == "list":
        result = manager.list_loras()
        print(json.dumps(result, indent=2))
    
    elif args.command == "test":
        result = manager.test_lora_inference(args.lora_name, args.prompt)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()