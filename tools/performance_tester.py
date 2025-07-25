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
import logging
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
import requests
import aiohttp

# 设置环境变量确保日志配置生效
os.environ.setdefault('VLLM_MXEXT_CONFIGURE_LOGGING', '1')
os.environ.setdefault('VLLM_MXEXT_JSONL_LOGGING', '0')

try:
    from vllm_mxext.logger import init_logger, configure_logger
    configure_logger("")
    logger = init_logger(__name__)
except ImportError as e:
    # 备选方案：使用标准logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    logger.warning(f"Failed to import vllm_mxext.logger: {e}, using standard logging")

# 确保日志级别正确
logger.setLevel(logging.INFO)

# 测试日志输出
logger.info("Performance tester initialized successfully")


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
    
    def __init__(self, server_url: str = "http://0.0.0.0:8000", 
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
    """Get default test prompts with varying token lengths (10-1024 tokens)."""
    return [
        # Short prompts (10-20 tokens)
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
        "How do neural networks work?",
        "What are the benefits of renewable energy?",
        "What is the difference between AI and machine learning?",
        "Write a poem about the ocean.",
        "Explain the theory of relativity.",
        "What are the main causes of climate change?",
        "How does photosynthesis work?",
        "What is blockchain technology?",
        
        # Medium prompts (50-100 tokens)
        "Write a short story about a robot learning to paint and discovering the beauty of art for the first time.",
        "Explain the process of machine learning model training, including data preprocessing, feature selection, and validation techniques.",
        "Describe the impact of artificial intelligence on modern healthcare, including diagnostic tools, treatment recommendations, and patient care improvements.",
        "Compare and contrast different renewable energy sources such as solar, wind, hydroelectric, and geothermal power in terms of efficiency and environmental impact.",
        "Analyze the role of cryptocurrency in the global economy and discuss its potential benefits and risks for traditional financial systems.",
        "Explain the concept of quantum entanglement and its applications in quantum computing and quantum communication technologies.",
        "Describe the evolution of programming languages from assembly to modern high-level languages and their impact on software development.",
        "Discuss the ethical implications of autonomous vehicles, including decision-making algorithms, liability issues, and societal acceptance.",
        "Explain the principles of sustainable agriculture and how technology can help address global food security challenges.",
        "Analyze the impact of social media on human communication patterns and its effects on mental health and social relationships.",
        
        # Long prompts (200-400 tokens)
        "Write a comprehensive analysis of the current state of artificial intelligence research, including recent breakthroughs in natural language processing, computer vision, and reinforcement learning. Discuss the challenges that researchers face in developing more advanced AI systems, such as the need for better data quality, computational resources, and algorithmic improvements. Also, explore the potential applications of AI in various industries including healthcare, finance, transportation, and education, while addressing concerns about job displacement, privacy, and the need for responsible AI development practices.",
        "Explain the complex relationship between climate change and global economic systems. Discuss how rising temperatures, extreme weather events, and sea-level rise affect agricultural productivity, supply chains, and infrastructure development. Analyze the economic costs of climate adaptation and mitigation strategies, including the transition to renewable energy sources, carbon pricing mechanisms, and international cooperation frameworks. Consider the role of technology innovation in addressing climate challenges and the potential for green economic growth models that balance environmental sustainability with economic prosperity.",
        "Describe the evolution of cybersecurity threats in the digital age and the corresponding development of defense mechanisms. Analyze different types of cyber attacks including malware, phishing, ransomware, and advanced persistent threats, explaining how they exploit vulnerabilities in computer systems and networks. Discuss the importance of cybersecurity frameworks, encryption technologies, and user education in protecting sensitive information. Explore the challenges faced by organizations in maintaining security while enabling digital transformation and remote work capabilities.",
        "Examine the role of biotechnology in modern medicine, focusing on gene therapy, personalized medicine, and the development of new pharmaceutical treatments. Discuss how advances in genomics, proteomics, and bioinformatics are revolutionizing our understanding of human diseases and enabling more targeted therapeutic approaches. Analyze the ethical considerations surrounding genetic engineering, clinical trials, and access to advanced medical treatments. Consider the potential for biotechnology to address global health challenges and improve quality of life for patients worldwide.",
        "Analyze the impact of globalization on cultural diversity and local communities. Discuss how increased connectivity, trade, and migration have facilitated cultural exchange while also raising concerns about cultural homogenization and the preservation of traditional practices. Examine the role of technology and social media in shaping global cultural trends and their effects on local identities. Consider strategies for maintaining cultural diversity while embracing the benefits of global interconnectedness and cooperation.",
        
        # Very long prompts (500-800 tokens)
        "Provide a detailed explanation of the principles and applications of machine learning in modern data science, covering supervised learning algorithms such as linear regression, decision trees, random forests, and support vector machines, as well as unsupervised learning techniques including clustering algorithms like k-means and hierarchical clustering, and dimensionality reduction methods such as principal component analysis and t-SNE. Discuss the importance of data preprocessing, feature engineering, and model validation techniques including cross-validation and holdout methods. Explain how deep learning has revolutionized the field with neural networks, convolutional neural networks for image processing, recurrent neural networks for sequence data, and transformer architectures for natural language processing. Address the challenges of overfitting, underfitting, and bias in machine learning models, and describe strategies for model interpretability and explainable AI. Consider the ethical implications of automated decision-making systems and the importance of fairness, accountability, and transparency in AI applications across different domains such as healthcare, finance, criminal justice, and hiring processes.",
        "Examine the complex interplay between technological innovation and social change throughout human history, from the agricultural revolution to the industrial revolution and the current digital transformation. Analyze how major technological breakthroughs such as the printing press, steam engine, electricity, telecommunications, and the internet have reshaped social structures, economic systems, and cultural practices. Discuss the concept of technological determinism versus social shaping of technology, exploring how society influences technological development while technology simultaneously transforms social relationships and institutions. Consider the role of innovation ecosystems, including research institutions, venture capital, government policies, and entrepreneurial networks in fostering technological advancement. Examine contemporary challenges such as the digital divide, technological unemployment, privacy concerns, and the need for digital literacy in an increasingly connected world. Analyze the potential future implications of emerging technologies such as artificial intelligence, quantum computing, biotechnology, and nanotechnology on society, economy, and human relationships.",
        "Discuss the evolution of sustainable development practices and their implementation across different sectors of the global economy. Examine the concept of the triple bottom line approach that balances economic prosperity, environmental protection, and social equity in business operations and policy-making. Analyze various sustainability frameworks and standards such as the UN Sustainable Development Goals, ESG (Environmental, Social, and Governance) criteria, and circular economy principles. Explore how companies are integrating sustainability into their core business strategies through initiatives such as renewable energy adoption, waste reduction, sustainable supply chain management, and stakeholder engagement. Consider the role of government regulations, international agreements, and consumer behavior in driving sustainable practices. Discuss the challenges and opportunities associated with measuring and reporting sustainability performance, including the development of standardized metrics and the use of technology for environmental monitoring and impact assessment. Examine case studies of successful sustainability initiatives and their broader implications for achieving long-term environmental and social goals.",
        
        # Extra long prompts (800-1024 tokens)
        "Conduct a comprehensive analysis of the current state and future prospects of renewable energy technologies, examining the technical, economic, and policy factors that influence their adoption and deployment worldwide. Begin by discussing the fundamental principles behind different renewable energy sources including solar photovoltaic and thermal systems, wind turbines both onshore and offshore, hydroelectric power generation including pumped storage systems, geothermal energy extraction, and emerging technologies such as tidal and wave energy converters. Analyze the technological improvements that have led to significant cost reductions in renewable energy systems over the past decade, including advances in materials science, manufacturing processes, and system efficiency optimization. Examine the role of energy storage technologies, particularly battery systems, in addressing the intermittency challenges associated with renewable energy sources and enabling greater grid integration. Discuss the importance of smart grid technologies, demand response systems, and grid modernization efforts in accommodating higher penetrations of renewable energy. Consider the economic aspects of renewable energy deployment, including levelized cost of electricity calculations, financing mechanisms, government incentives and subsidies, and the impact on electricity markets and pricing structures. Analyze the policy frameworks and regulatory environments that support or hinder renewable energy development, including renewable portfolio standards, feed-in tariffs, carbon pricing mechanisms, and international climate agreements. Examine the social and environmental benefits of renewable energy adoption, including job creation, air quality improvements, and greenhouse gas emission reductions, while also addressing potential challenges such as land use requirements, visual impacts, and effects on local communities. Finally, explore future trends and emerging technologies in the renewable energy sector, including floating solar installations, advanced wind turbine designs, green hydrogen production, and the integration of artificial intelligence and machine learning in energy system optimization.",
        "Explore the multifaceted relationship between artificial intelligence and human creativity, examining how AI technologies are transforming creative industries while raising fundamental questions about the nature of creativity, authorship, and artistic expression. Begin by analyzing the current applications of AI in various creative domains including music composition and production, visual arts and digital design, creative writing and content generation, film and video production, and interactive entertainment such as video games. Discuss the technical foundations of creative AI systems, including generative adversarial networks (GANs), variational autoencoders (VAEs), transformer-based language models, and reinforcement learning approaches that enable machines to produce novel and aesthetically pleasing content. Examine the collaborative potential between human creators and AI systems, exploring how artists, writers, musicians, and designers are incorporating AI tools into their creative workflows to enhance productivity, explore new artistic possibilities, and push the boundaries of traditional creative practices. Consider the philosophical and ethical implications of AI-generated content, including questions about intellectual property rights, the value and authenticity of machine-created art, and the potential impact on human creative professionals and cultural industries. Analyze the economic dimensions of AI in creativity, including new business models, market disruptions, and the democratization of creative tools that enable broader participation in creative activities. Discuss the challenges and limitations of current AI creative systems, such as the need for large training datasets, potential biases in generated content, and the difficulty of achieving true understanding and intentionality in creative expression. Examine case studies of successful AI-human creative collaborations and their reception by audiences and critics. Finally, speculate on the future evolution of AI creativity, considering emerging technologies, potential breakthroughs in artificial general intelligence, and the long-term implications for human culture and artistic expression."
    ]


async def main():
    parser = argparse.ArgumentParser(description="Performance testing tool for vLLM MxExt")
    parser.add_argument("--server-url", default="http://0.0.0.0:8000",
                       help="Server URL (default: http://0.0.0.0:8000)")
    parser.add_argument("--model", required=True,
                       help="Model name to test")
    parser.add_argument("--prompts-file", 
                       help="File containing prompts (one per line)")
    parser.add_argument("--num-prompts", type=int, default=10,
                       help="Number of default prompts to use when no prompts-file is specified (default: 10)")
    parser.add_argument("--num-requests", type=int, 
                       help="Total number of requests to send (overrides num-prompts)")
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
                file_prompts = [line.strip() for line in f if line.strip()]
            
            if not file_prompts:
                logger.error(f"No valid prompts found in file: {args.prompts_file}")
                return 1
            
            # Determine number of requests
            if args.num_requests:
                num_requests = args.num_requests
            else:
                num_requests = len(file_prompts)
            
            # Generate prompts based on num_requests
            if num_requests <= len(file_prompts):
                # Take first num_requests prompts
                prompts = file_prompts[:num_requests]
            else:
                # Cycle through prompts to reach num_requests
                prompts = []
                for i in range(num_requests):
                    prompts.append(file_prompts[i % len(file_prompts)])
            
            logger.info(f"Loaded {len(file_prompts)} prompts from file, using {len(prompts)} requests")
            
        except Exception as e:
            logger.error(f"Failed to load prompts file: {e}")
            return 1
    else:
        # Use default prompts
        default_prompts = get_default_prompts()
        
        if args.num_requests:
            num_requests = args.num_requests
            # Cycle through default prompts to reach num_requests
            prompts = []
            for i in range(num_requests):
                prompts.append(default_prompts[i % len(default_prompts)])
        else:
            # Use num_prompts (legacy behavior)
            prompts = default_prompts[:args.num_prompts]
        
        logger.info(f"Using {len(prompts)} default prompts")
    
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
