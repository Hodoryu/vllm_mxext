#!/usr/bin/env python3
"""
Performance testing tool for vLLM MxExt LLM inference.
Measures tokens throughput, TTFT, TPOT, and End-to-End Latency.
"""

import asyncio
import argparse
import json
import time
import statistics
import os
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
import requests
import aiohttp
import logging

from vllm_mxext.logger import init_logger

logger = init_logger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for a single request."""
    request_id: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    ttft: float  # Time to First Token (seconds)
    tpot: float  # Time Per Output Token (seconds)
    e2e_latency: float  # End-to-End Latency (seconds)
    throughput: float  # Tokens per second
    timestamp: float


@dataclass
class AggregatedMetrics:
    """Aggregated performance metrics across multiple requests."""
    total_requests: int
    total_prompt_tokens: int
    total_completion_tokens: int
    total_tokens: int
    
    # TTFT metrics
    ttft_mean: float
    ttft_median: float
    ttft_p95: float
    ttft_p99: float
    ttft_min: float
    ttft_max: float
    
    # TPOT metrics
    tpot_mean: float
    tpot_median: float
    tpot_p95: float
    tpot_p99: float
    tpot_min: float
    tpot_max: float
    
    # E2E Latency metrics
    e2e_mean: float
    e2e_median: float
    e2e_p95: float
    e2e_p99: float
    e2e_min: float
    e2e_max: float
    
    # Throughput metrics
    throughput_mean: float
    throughput_median: float
    throughput_p95: float
    throughput_p99: float
    throughput_min: float
    throughput_max: float
    
    # Overall throughput
    overall_throughput: float  # Total tokens / Total time
    test_duration: float


class PerformanceTester:
    """Performance testing tool for LLM inference."""
    
    def __init__(self, server_url: str = "http://localhost:8000", 
                 log_dir: str = "/opt/mim/log"):
        self.server_url = server_url.rstrip('/')
        self.log_dir = log_dir
        self.metrics: List[PerformanceMetrics] = []
        self.test_start_time = None
        self.test_end_time = None
        
        # Ensure log directory exists
        os.makedirs(self.log_dir, exist_ok=True)
    
    def _calculate_percentile(self, values: List[float], percentile: int) -> float:
        """Calculate percentile from a list of values."""
        if not values:
            return 0.0
        sorted_values = sorted(values)
        index = int((percentile / 100) * len(sorted_values))
        return sorted_values[min(index, len(sorted_values) - 1)]
    
    def _aggregate_metrics(self) -> AggregatedMetrics:
        """Aggregate performance metrics from all requests."""
        if not self.metrics:
            raise ValueError("No metrics to aggregate")
        
        ttft_values = [m.ttft for m in self.metrics]
        tpot_values = [m.tpot for m in self.metrics]
        e2e_values = [m.e2e_latency for m in self.metrics]
        throughput_values = [m.throughput for m in self.metrics]
        
        total_tokens = sum(m.total_tokens for m in self.metrics)
        test_duration = self.test_end_time - self.test_start_time
        overall_throughput = total_tokens / test_duration if test_duration > 0 else 0
        
        return AggregatedMetrics(
            total_requests=len(self.metrics),
            total_prompt_tokens=sum(m.prompt_tokens for m in self.metrics),
            total_completion_tokens=sum(m.completion_tokens for m in self.metrics),
            total_tokens=total_tokens,
            
            # TTFT metrics
            ttft_mean=statistics.mean(ttft_values),
            ttft_median=statistics.median(ttft_values),
            ttft_p95=self._calculate_percentile(ttft_values, 95),
            ttft_p99=self._calculate_percentile(ttft_values, 99),
            ttft_min=min(ttft_values),
            ttft_max=max(ttft_values),
            
            # TPOT metrics
            tpot_mean=statistics.mean(tpot_values),
            tpot_median=statistics.median(tpot_values),
            tpot_p95=self._calculate_percentile(tpot_values, 95),
            tpot_p99=self._calculate_percentile(tpot_values, 99),
            tpot_min=min(tpot_values),
            tpot_max=max(tpot_values),
            
            # E2E Latency metrics
            e2e_mean=statistics.mean(e2e_values),
            e2e_median=statistics.median(e2e_values),
            e2e_p95=self._calculate_percentile(e2e_values, 95),
            e2e_p99=self._calculate_percentile(e2e_values, 99),
            e2e_min=min(e2e_values),
            e2e_max=max(e2e_values),
            
            # Throughput metrics
            throughput_mean=statistics.mean(throughput_values),
            throughput_median=statistics.median(throughput_values),
            throughput_p95=self._calculate_percentile(throughput_values, 95),
            throughput_p99=self._calculate_percentile(throughput_values, 99),
            throughput_min=min(throughput_values),
            throughput_max=max(throughput_values),
            
            # Overall metrics
            overall_throughput=overall_throughput,
            test_duration=test_duration
        )
    
    async def _send_request(self, session: aiohttp.ClientSession, 
                          request_data: Dict, request_id: str) -> PerformanceMetrics:
        """Send a single request and measure performance."""
        start_time = time.time()
        first_token_time = None
        completion_tokens = 0
        
        try:
            async with session.post(
                f"{self.server_url}/v1/chat/completions",
                json=request_data
            ) as response:
                if response.status != 200:
                    raise Exception(f"HTTP {response.status}: {await response.text()}")
                
                if request_data.get("stream", False):
                    # Handle streaming response
                    async for line in response.content:
                        line = line.decode('utf-8').strip()
                        if line.startswith('data: '):
                            data_str = line[6:]
                            if data_str == '[DONE]':
                                break
                            
                            try:
                                data = json.loads(data_str)
                                if 'choices' in data and data['choices']:
                                    choice = data['choices'][0]
                                    if 'delta' in choice and 'content' in choice['delta']:
                                        if first_token_time is None:
                                            first_token_time = time.time()
                                        completion_tokens += 1
                            except json.JSONDecodeError:
                                continue
                else:
                    # Handle non-streaming response
                    result = await response.json()
                    first_token_time = time.time()
                    if 'usage' in result:
                        completion_tokens = result['usage'].get('completion_tokens', 0)
                    else:
                        # Estimate tokens from content
                        content = result.get('choices', [{}])[0].get('message', {}).get('content', '')
                        completion_tokens = len(content.split())
        
        except Exception as e:
            logger.error(f"Request {request_id} failed: {e}")
            raise
        
        end_time = time.time()
        
        # Calculate metrics
        e2e_latency = end_time - start_time
        ttft = (first_token_time - start_time) if first_token_time else e2e_latency
        
        prompt_tokens = len(request_data.get('messages', [{}])[-1].get('content', '').split())
        total_tokens = prompt_tokens + completion_tokens
        
        tpot = (end_time - first_token_time) / completion_tokens if first_token_time and completion_tokens > 0 else 0
        throughput = total_tokens / e2e_latency if e2e_latency > 0 else 0
        
        return PerformanceMetrics(
            request_id=request_id,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            ttft=ttft,
            tpot=tpot,
            e2e_latency=e2e_latency,
            throughput=throughput,
            timestamp=start_time
        )
    
    async def run_performance_test(self, 
                                 model_name: str,
                                 prompts: List[str],
                                 max_tokens: int = 100,
                                 temperature: float = 0.7,
                                 concurrent_requests: int = 1,
                                 stream: bool = True) -> AggregatedMetrics:
        """Run performance test with given parameters."""
        logger.info(f"Starting performance test with {len(prompts)} prompts, "
                   f"{concurrent_requests} concurrent requests")
        
        self.test_start_time = time.time()
        
        # Prepare requests
        requests_data = []
        for i, prompt in enumerate(prompts):
            request_data = {
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": stream
            }
            requests_data.append((request_data, f"req_{i}"))
        
        # Execute requests with concurrency control
        semaphore = asyncio.Semaphore(concurrent_requests)
        
        async def bounded_request(session, request_data, request_id):
            async with semaphore:
                return await self._send_request(session, request_data, request_id)
        
        async with aiohttp.ClientSession() as session:
            tasks = [
                bounded_request(session, req_data, req_id)
                for req_data, req_id in requests_data
            ]
            
            self.metrics = await asyncio.gather(*tasks)
        
        self.test_end_time = time.time()
        
        return self._aggregate_metrics()
    
    def print_results(self, metrics: AggregatedMetrics):
        """Print performance results to console."""
        print("\n" + "="*80)
        print("PERFORMANCE TEST RESULTS")
        print("="*80)
        
        print(f"\nTest Summary:")
        print(f"  Total Requests: {metrics.total_requests}")
        print(f"  Total Tokens: {metrics.total_tokens:,}")
        print(f"  Prompt Tokens: {metrics.total_prompt_tokens:,}")
        print(f"  Completion Tokens: {metrics.total_completion_tokens:,}")
        print(f"  Test Duration: {metrics.test_duration:.2f}s")
        print(f"  Overall Throughput: {metrics.overall_throughput:.2f} tokens/s")
        
        print(f"\nTime to First Token (TTFT):")
        print(f"  Mean: {metrics.ttft_mean:.3f}s")
        print(f"  Median: {metrics.ttft_median:.3f}s")
        print(f"  P95: {metrics.ttft_p95:.3f}s")
        print(f"  P99: {metrics.ttft_p99:.3f}s")
        print(f"  Min: {metrics.ttft_min:.3f}s")
        print(f"  Max: {metrics.ttft_max:.3f}s")
        
        print(f"\nTime Per Output Token (TPOT):")
        print(f"  Mean: {metrics.tpot_mean:.3f}s")
        print(f"  Median: {metrics.tpot_median:.3f}s")
        print(f"  P95: {metrics.tpot_p95:.3f}s")
        print(f"  P99: {metrics.tpot_p99:.3f}s")
        print(f"  Min: {metrics.tpot_min:.3f}s")
        print(f"  Max: {metrics.tpot_max:.3f}s")
        
        print(f"\nEnd-to-End Latency:")
        print(f"  Mean: {metrics.e2e_mean:.3f}s")
        print(f"  Median: {metrics.e2e_median:.3f}s")
        print(f"  P95: {metrics.e2e_p95:.3f}s")
        print(f"  P99: {metrics.e2e_p99:.3f}s")
        print(f"  Min: {metrics.e2e_min:.3f}s")
        print(f"  Max: {metrics.e2e_max:.3f}s")
        
        print(f"\nThroughput (tokens/s):")
        print(f"  Mean: {metrics.throughput_mean:.2f}")
        print(f"  Median: {metrics.throughput_median:.2f}")
        print(f"  P95: {metrics.throughput_p95:.2f}")
        print(f"  P99: {metrics.throughput_p99:.2f}")
        print(f"  Min: {metrics.throughput_min:.2f}")
        print(f"  Max: {metrics.throughput_max:.2f}")
        
        print("="*80)
    
    def save_results(self, metrics: AggregatedMetrics, individual_metrics: bool = True):
        """Save performance results to log file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"mim_profile_{timestamp}.log"
        log_path = os.path.join(self.log_dir, log_filename)
        
        try:
            with open(log_path, 'w') as f:
                f.write(f"Performance Test Results - {datetime.now().isoformat()}\n")
                f.write("="*80 + "\n\n")
                
                # Write aggregated metrics
                f.write("AGGREGATED METRICS:\n")
                f.write(json.dumps(asdict(metrics), indent=2))
                f.write("\n\n")
                
                # Write individual metrics if requested
                if individual_metrics:
                    f.write("INDIVIDUAL REQUEST METRICS:\n")
                    for metric in self.metrics:
                        f.write(json.dumps(asdict(metric), indent=2))
                        f.write("\n")
            
            logger.info(f"Performance results saved to: {log_path}")
            print(f"\nResults saved to: {log_path}")
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            print(f"Error saving results: {e}")


def get_default_prompts() -> List[str]:
    """Get default test prompts."""
    return [
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
        "Write a short story about a robot learning to paint.",
        "How do neural networks work?",
        "What are the benefits of renewable energy?",
        "Describe the process of photosynthesis.",
        "What is the difference between AI and machine learning?",
        "Write a poem about the ocean.",
        "Explain the theory of relativity.",
        "What are the main causes of climate change?"
    ]


async def main():
    parser = argparse.ArgumentParser(description="Performance testing tool for vLLM MxExt")
    parser.add_argument("--server-url", default="http://localhost:8000",
                       help="Server URL (default: http://localhost:8000)")
    parser.add_argument("--model", required=True,
                       help="Model name to test")
    parser.add_argument("--prompts-file", 
                       help="File containing prompts (one per line)")
    parser.add_argument("--num-prompts", type=int, default=10,
                       help="Number of default prompts to use (default: 10)")
    parser.add_argument("--max-tokens", type=int, default=100,
                       help="Maximum tokens per response (default: 100)")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Temperature for generation (default: 0.7)")
    parser.add_argument("--concurrent-requests", type=int, default=1,
                       help="Number of concurrent requests (default: 1)")
    parser.add_argument("--stream", action="store_true", default=True,
                       help="Use streaming responses (default: True)")
    parser.add_argument("--no-stream", action="store_false", dest="stream",
                       help="Disable streaming responses")
    parser.add_argument("--log-dir", default="/opt/mim/log",
                       help="Directory to save log files (default: /opt/mim/log)")
    parser.add_argument("--no-save", action="store_true",
                       help="Don't save results to file")
    parser.add_argument("--save-individual", action="store_true",
                       help="Save individual request metrics")
    
    args = parser.parse_args()
    
    # Load prompts
    if args.prompts_file:
        try:
            with open(args.prompts_file, 'r') as f:
                prompts = [line.strip() for line in f if line.strip()]
        except Exception as e:
            logger.error(f"Failed to load prompts file: {e}")
            return 1
    else:
        default_prompts = get_default_prompts()
        prompts = default_prompts[:args.num_prompts]
    
    if not prompts:
        logger.error("No prompts to test")
        return 1
    
    # Initialize tester
    tester = PerformanceTester(args.server_url, args.log_dir)
    
    try:
        # Run performance test
        metrics = await tester.run_performance_test(
            model_name=args.model,
            prompts=prompts,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            concurrent_requests=args.concurrent_requests,
            stream=args.stream
        )
        
        # Print results
        tester.print_results(metrics)
        
        # Save results
        if not args.no_save:
            tester.save_results(metrics, args.save_individual)
        
        return 0
        
    except Exception as e:
        logger.error(f"Performance test failed: {e}")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))