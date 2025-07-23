#!/usr/bin/env python3
"""
Convenience script for running performance tests.
"""

import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vllm_mxext.tools.performance_tester import main
import asyncio

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))