#!/usr/bin/env python3
"""
启动脚本示例：同时启动API服务器和监控仪表板
"""

import os
import sys
import argparse
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 导入必要的函数
from vllm_mxext.entrypoints.openai.api_server import run_server, complete_args_with_env_vars
import uvloop

if __name__ == "__main__":
    print("🚀 Starting vLLM MxExt with integrated monitoring dashboard...")
    print("📊 Dashboard will be available at: http://0.0.0.0:8000/dashboard")
    print("🔗 API server will be available at: http://0.0.0.0:8000")
    
    # 获取完整的参数配置
    args = complete_args_with_env_vars()
    
    # 启动服务器
    uvloop.run(run_server(args))
