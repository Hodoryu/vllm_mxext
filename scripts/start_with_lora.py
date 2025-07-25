#!/usr/bin/env python3
"""
启动脚本：支持LoRA的vLLM MxExt服务器
"""

import os
import sys
import argparse
import yaml
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from vllm_mxext.entrypoints.openai.api_server import main

def load_lora_config(config_path: str):
    """加载LoRA配置文件"""
    if not os.path.exists(config_path):
        print(f"Warning: LoRA config file not found: {config_path}")
        return None
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

def setup_lora_environment(lora_config):
    """设置LoRA环境变量"""
    if not lora_config:
        return
    
    # 设置LoRA相关环境变量
    if 'lora_settings' in lora_config:
        settings = lora_config['lora_settings']
        
        if 'max_lora_rank' in settings:
            os.environ['VLLM_MAX_LORA_RANK'] = str(settings['max_lora_rank'])
        
        if 'max_loras' in settings:
            os.environ['VLLM_MAX_LORAS'] = str(settings['max_loras'])
        
        if 'enable_dynamic_loading' in settings:
            os.environ['VLLM_ALLOW_RUNTIME_LORA_UPDATING'] = str(settings['enable_dynamic_loading']).lower()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start vLLM MxExt with LoRA support")
    parser.add_argument("--lora-config", type=str, 
                       default="examples/lora_config_example.yaml",
                       help="Path to LoRA configuration file")
    
    args, unknown_args = parser.parse_known_args()
    
    print("🚀 Starting vLLM MxExt with LoRA support...")
    print(f"📁 LoRA config: {args.lora_config}")
    
    # 加载LoRA配置
    lora_config = load_lora_config(args.lora_config)
    setup_lora_environment(lora_config)
    
    # 启用LoRA运行时更新
    os.environ['VLLM_ALLOW_RUNTIME_LORA_UPDATING'] = 'true'
    
    print("✅ LoRA support enabled")
    print("📊 Dashboard available at: http://0.0.0.0:8000/dashboard")
    print("🔗 API server available at: http://0.0.0.0:8000")
    print("🎯 LoRA management endpoints:")
    print("   - POST /v1/load_lora_adapter")
    print("   - POST /v1/unload_lora_adapter") 
    print("   - GET /v1/lora_adapters")
    
    # 调用原始的main函数
    sys.argv = [sys.argv[0]] + unknown_args
    main()