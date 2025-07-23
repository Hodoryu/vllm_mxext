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

from vllm_mxext.entrypoints.openai.api_server import main

if __name__ == "__main__":
    print("🚀 Starting vLLM MxExt with integrated monitoring dashboard...")
    print("📊 Dashboard will be available at: http://localhost:8000/dashboard")
    print("🔗 API server will be available at: http://localhost:8000")
    
    # 调用原始的main函数，仪表板会自动集成
    main()