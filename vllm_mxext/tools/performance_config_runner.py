#!/usr/bin/env python3
"""
Performance test runner that uses YAML configuration files.
"""

import asyncio
import argparse
import yaml
from typing import Dict, Any
from vllm_mxext.tools.performance_tester import PerformanceTester
from vllm_mxext.logger import init_logger

logger = init_logger(__name__)


class ConfigBasedPerformanceTester:
    """Performance tester that uses YAML configuration."""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load config file {self.config_path}: {e}")
            raise
    
    async def run_test(self) -> None:
        """Run performance test based on configuration."""
        server_config = self.config.get('server', {})
        test_params = self.config.get('test_params', {})
        output_config = self.config.get('output', {})
        
        # Initialize tester
        tester = PerformanceTester(
            server_url=server_config.get('url', 'http://localhost:8000'),
            log_dir=output_config.get('log_dir', '/opt/mim/log')
        )
        
        # Get prompts
        prompts = self.config.get('prompts', [])
        if not prompts:
            logger.error("No prompts specified in configuration")
            return
        
        # Run test
        try:
            metrics = await tester.run_performance_test(
                model_name=server_config.get('model', 'default'),
                prompts=prompts,
                max_tokens=test_params.get('max_tokens', 100),
                temperature=test_params.get('temperature', 0.7),
                concurrent_requests=test_params.get('concurrent_requests', 1),
                stream=test_params.get('stream', True)
            )
            
            # Output results
            if output_config.get('console_output', True):
                tester.print_results(metrics)
            
            # Save results
            tester.save_results(
                metrics, 
                output_config.get('save_individual_metrics', False)
            )
            
        except Exception as e:
            logger.error(f"Performance test failed: {e}")
            raise


async def main():
    parser = argparse.ArgumentParser(
        description="Run performance test using YAML configuration"
    )
    parser.add_argument("config", help="Path to YAML configuration file")
    
    args = parser.parse_args()
    
    tester = ConfigBasedPerformanceTester(args.config)
    await tester.run_test()


if __name__ == "__main__":
    asyncio.run(main())