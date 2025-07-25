"""Web-based monitoring dashboard for vLLM MxExt."""

import asyncio
import json
import time
import os
import subprocess
import psutil
from typing import Dict, List, Optional
from datetime import datetime, timedelta

from fastapi import FastAPI, WebSocket, Request, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from vllm_mxext.logger import init_logger
from vllm_mxext.monitoring.advanced_metrics import AdvancedMetricsCollector

logger = init_logger(__name__)

def collect_metax_gpu_metrics() -> List[Dict]:
    """Collect MetaX GPU metrics using mx-smi command."""
    print("=" * 60)
    print("Starting MetaX GPU metrics collection test...")
    print("=" * 60)
    
    gpu_metrics = []
    
    try:
        print("Step 1: Executing mx-smi command...")
        result = subprocess.run(['mx-smi'], capture_output=True, text=True, timeout=10)
        
        if result.returncode != 0:
            print(f"❌ ERROR: mx-smi command failed with return code {result.returncode}")
            return gpu_metrics
        
        print("✅ mx-smi command executed successfully")
        
        output = result.stdout
        lines = output.split('\n')
        
        print(f"\nStep 2: Parsing GPU information...")
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Look for GPU info lines that start with "| <gpu_id>"
            if line.startswith('|') and len(line.split('|')) >= 4:
                parts = [part.strip() for part in line.split('|') if part.strip()]
                
                # Check if this is a GPU info line by looking for GPU ID at the start
                if len(parts) >= 3:
                    # Extract GPU ID from the first part (e.g., "0           MetaX C280")
                    first_part = parts[0].strip()
                    print(f"\n  Processing potential GPU line {i}: '{line}'")
                    print(f"    First part: '{first_part}'")
                    
                    if first_part:  # Make sure it's not empty
                        gpu_id_match = first_part.split()[0]  # Get the first word
                        print(f"    First word: '{gpu_id_match}'")
                        
                        # Check if the first word is a digit (GPU ID)
                        if gpu_id_match.isdigit():
                            print(f"    ✅ Found GPU ID: {gpu_id_match}")
                            try:
                                gpu_id = int(gpu_id_match)
                                
                                # Extract GPU name (everything after the GPU ID)
                                gpu_name_parts = first_part.split()[1:]  # Skip the first word (GPU ID)
                                gpu_name = ' '.join(gpu_name_parts)  # Join the rest as GPU name
                                
                                bus_id = parts[1].strip()    # e.g., "0000:17:00.0"
                                gpu_util_str = parts[2].strip()  # e.g., "0%"
                                gpu_util = int(gpu_util_str.replace('%', ''))
                                
                                print(f"    - GPU ID: {gpu_id}")
                                print(f"    - GPU Name: '{gpu_name}'")
                                print(f"    - Bus ID: {bus_id}")
                                print(f"    - GPU Utilization: {gpu_util}%")
                                
                                # Get the next line for temperature, power, and memory info
                                if i + 1 < len(lines):
                                    next_line = lines[i + 1].strip()
                                    print(f"    Next line {i+1}: '{next_line}'")
                                    next_parts = [part.strip() for part in next_line.split('|') if part.strip()]
                                    print(f"    Next line parts: {next_parts}")
                                    
                                    if len(next_parts) >= 2:
                                        # Parse temperature and power from first part (e.g., "40C         73W / 280W")
                                        temp_power_str = next_parts[0].strip()
                                        print(f"    Temp/Power string: '{temp_power_str}'")
                                        temp_power_parts = temp_power_str.split()
                                        print(f"    Temp/Power parts: {temp_power_parts}")
                                        
                                        # Extract temperature (e.g., "40C")
                                        temperature = int(temp_power_parts[0].replace('C', ''))
                                        
                                        # Extract power (e.g., "73W / 280W")
                                        power_str = ' '.join(temp_power_parts[1:])  # "73W / 280W"
                                        print(f"    Power string: '{power_str}'")
                                        power_parts = power_str.split('/')
                                        power_usage = int(power_parts[0].strip().replace('W', ''))
                                        power_limit = int(power_parts[1].strip().replace('W', ''))
                                        
                                        # Parse memory (e.g., "54995/65536 MiB")
                                        memory_str = next_parts[1].strip()
                                        print(f"    Memory string: '{memory_str}'")
                                        memory_parts = memory_str.replace('MiB', '').split('/')
                                        memory_used_mib = int(memory_parts[0].strip())
                                        memory_total_mib = int(memory_parts[1].strip())
                                        
                                        print(f"    - Temperature: {temperature}°C")
                                        print(f"    - Power: {power_usage}W / {power_limit}W")
                                        print(f"    - Memory: {memory_used_mib}/{memory_total_mib} MiB")
                                        
                                        # Calculate derived values
                                        memory_percent = (memory_used_mib / memory_total_mib) * 100 if memory_total_mib > 0 else 0
                                        power_percent = (power_usage / power_limit) * 100 if power_limit > 0 else 0
                                        
                                        gpu_metric = {
                                            'id': gpu_id,
                                            'name': gpu_name,
                                            'load': gpu_util,
                                            'memory_used': memory_used_mib,
                                            'memory_total': memory_total_mib,
                                            'memory_percent': memory_percent,
                                            'temperature': temperature,
                                            'power_usage': power_usage,
                                            'power_limit': power_limit,
                                            'power_percent': power_percent,
                                            'bus_id': bus_id
                                        }
                                        
                                        gpu_metrics.append(gpu_metric)
                                        print(f"    ✅ Successfully processed GPU {gpu_id}")
                                        
                            except (ValueError, IndexError) as e:
                                print(f"    ❌ Error parsing GPU line: {line}, error: {e}")
                        else:
                            print(f"    ❌ First word '{gpu_id_match}' is not a digit")
                            
            i += 1
            
        print(f"\nStep 3: Final results")
        print(f"  - Total GPUs processed: {len(gpu_metrics)}")
        
        return gpu_metrics
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return gpu_metrics


class MetricsCollector:
    """Collects and aggregates metrics for the dashboard."""
    
    def __init__(self, stat_logger=None):
        self.stat_logger = stat_logger
        self.advanced_collector = AdvancedMetricsCollector()
        self.last_network_stats = None
        
    async def collect_system_metrics(self) -> Dict:
        """Collect system-level metrics."""
        try:
            # 获取CPU使用率 - 使用更长的间隔获取准确数据
            cpu_percent_total = psutil.cpu_percent(interval=1.0)  # 增加间隔时间
            cpu_percent = psutil.cpu_percent(interval=None, percpu=True)
            
            # 获取内存信息
            memory = psutil.virtual_memory()
            
            # 获取磁盘信息
            disk = psutil.disk_usage('/')
            
            # 获取网络信息
            network = psutil.net_io_counters()
            
            # 计算网络吞吐量
            network_throughput = 0
            if self.last_network_stats:
                time_diff = time.time() - self.last_network_stats['timestamp']
                if time_diff > 0:
                    bytes_sent_diff = network.bytes_sent - self.last_network_stats['bytes_sent']
                    bytes_recv_diff = network.bytes_recv - self.last_network_stats['bytes_recv']
                    network_throughput = (bytes_sent_diff + bytes_recv_diff) / time_diff / (1024 * 1024)  # MB/s
            
            self.last_network_stats = {
                'timestamp': time.time(),
                'bytes_sent': network.bytes_sent,
                'bytes_recv': network.bytes_recv
            }
            
            # GPU指标收集
            gpu_metrics = []
            try:
                gpu_metrics = collect_metax_gpu_metrics()
                if gpu_metrics:
                    logger.info(f"Successfully collected {len(gpu_metrics)} GPU metrics")
                else:
                    logger.warning("No GPU metrics collected")
            except Exception as e:
                logger.error(f"Failed to collect GPU metrics: {e}")

            # 构建系统指标字典
            system_metrics = {
                'timestamp': time.time(),
                'cpu': {
                    'usage_per_core': cpu_percent,
                    'usage_total': float(cpu_percent_total)  # 确保是float类型
                },
                'memory': {
                    'used': int(memory.used),
                    'total': int(memory.total),
                    'percent': float(memory.percent)  # 确保是float类型
                },
                'disk': {
                    'used': int(disk.used),
                    'total': int(disk.total),
                    'percent': float((disk.used / disk.total) * 100)  # 确保是float类型
                },
                'network': {
                    'bytes_sent': int(network.bytes_sent),
                    'bytes_recv': int(network.bytes_recv),
                    'throughput_mbps': float(network_throughput)  # 确保是float类型
                },
                'gpu': gpu_metrics
            }
            
            # 记录指标用于时间序列
            self.advanced_collector.record_metric('cpu_usage', system_metrics['cpu']['usage_total'])
            self.advanced_collector.record_metric('memory_usage', system_metrics['memory']['percent'])
            self.advanced_collector.record_metric('disk_usage', system_metrics['disk']['percent'])
            self.advanced_collector.record_metric('network_throughput', network_throughput)
            
            if gpu_metrics:
                avg_gpu_load = sum(gpu['load'] for gpu in gpu_metrics) / len(gpu_metrics)
                avg_gpu_memory = sum(gpu['memory_percent'] for gpu in gpu_metrics) / len(gpu_metrics)
                avg_gpu_temp = sum(gpu.get('temperature', 0) for gpu in gpu_metrics) / len(gpu_metrics)
                avg_gpu_power = sum(gpu.get('power_percent', 0) for gpu in gpu_metrics) / len(gpu_metrics)
                
                self.advanced_collector.record_metric('gpu_utilization', avg_gpu_load)
                self.advanced_collector.record_metric('gpu_memory_usage', avg_gpu_memory)
                self.advanced_collector.record_metric('gpu_temperature', avg_gpu_temp)
                self.advanced_collector.record_metric('gpu_power_usage', avg_gpu_power)
            
            logger.debug(f"Collected system metrics: CPU={cpu_percent_total}%, Memory={memory.percent}%, Disk={(disk.used/disk.total)*100}%, Network={network_throughput}MB/s")
            return system_metrics
        
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            return {
                'timestamp': time.time(), 
                'error': str(e),
                'cpu': {'usage_total': 0, 'usage_per_core': []},
                'memory': {'used': 0, 'total': 0, 'percent': 0},
                'disk': {'used': 0, 'total': 0, 'percent': 0},
                'network': {'bytes_sent': 0, 'bytes_recv': 0, 'throughput_mbps': 0},
                'gpu': []
            }
    
    async def collect_llm_metrics(self) -> Dict:
        """Collect LLM-specific metrics."""
        try:
            # 首先检查stat_logger是否存在且有正确的方法
            if self.stat_logger and hasattr(self.stat_logger, 'get_dashboard_metrics'):
                try:
                    metrics = self.stat_logger.get_dashboard_metrics()
                    logger.info(f"Retrieved LLM metrics from stat_logger: {metrics}")
                    
                    # 验证数据结构
                    if isinstance(metrics, dict) and 'requests' in metrics and 'performance' in metrics:
                        # 记录时间序列数据
                        if 'performance' in metrics:
                            perf = metrics['performance']
                            if perf.get('ttft_p50'):
                                self.advanced_collector.record_metric('ttft', perf['ttft_p50'])
                            if perf.get('throughput'):
                                self.advanced_collector.record_metric('throughput', perf['throughput'])
                                
                        if 'requests' in metrics:
                            self.advanced_collector.record_metric('active_requests', metrics['requests'].get('active', 0))
                        
                        return {
                            'timestamp': time.time(),
                            **metrics
                        }
                    else:
                        logger.warning("stat_logger returned invalid data structure")
                    
                except Exception as e:
                    logger.error(f"Error calling stat_logger.get_dashboard_metrics(): {e}")
            
            # 如果没有stat_logger或调用失败，提供模拟数据
            logger.info("Using simulated LLM metrics for testing")
            import random
            
            # 生成更真实的模拟数据
            current_time = time.time()
            simulated_metrics = {
                'timestamp': current_time,
                'requests': {
                    'active': random.randint(1, 8),
                    'queued': random.randint(0, 3),
                    'completed': random.randint(500, 2000)
                },
                'performance': {
                    'ttft_p50': random.uniform(80, 150),
                    'ttft_p95': random.uniform(150, 250),
                    'ttft_p99': random.uniform(200, 350),
                    'ttot_avg': random.uniform(15, 35),
                    'throughput': random.uniform(25, 85)
                }
            }
            
            # 记录模拟数据到时间序列
            self.advanced_collector.record_metric('ttft', simulated_metrics['performance']['ttft_p50'])
            self.advanced_collector.record_metric('throughput', simulated_metrics['performance']['throughput'])
            self.advanced_collector.record_metric('active_requests', simulated_metrics['requests']['active'])
            
            logger.debug(f"Generated simulated LLM metrics: {simulated_metrics}")
            return simulated_metrics
            
        except Exception as e:
            logger.error(f"Error collecting LLM metrics: {e}")
            return {
                'timestamp': time.time(), 
                'error': str(e),
                'requests': {'active': 0, 'queued': 0, 'completed': 0},
                'performance': {'ttft_p50': 0, 'ttft_p95': 0, 'ttft_p99': 0, 'ttot_avg': 0, 'throughput': 0}
            }


class DashboardManager:
    """Manages the monitoring dashboard."""
    
    def __init__(self, stat_logger=None):
        self.metrics_collector = MetricsCollector(stat_logger)
        self.connected_clients = set()
        
    def setup_routes(self, app: FastAPI):
        """Setup dashboard routes."""
        
        @app.get("/dashboard", response_class=HTMLResponse)
        async def dashboard_home():
            return self.get_dashboard_html()
            
        @app.get("/api/metrics/current")
        async def get_current_metrics():
            system_metrics = await self.metrics_collector.collect_system_metrics()
            llm_metrics = await self.metrics_collector.collect_llm_metrics()
            return JSONResponse({
                'system': system_metrics,
                'llm': llm_metrics
            })
            
        @app.get("/api/metrics/history")
        async def get_metrics_history(timerange: str = "1h"):
            historical_data = self.metrics_collector.advanced_collector.get_aggregated_data(timerange)
            return JSONResponse(historical_data)
            
        @app.websocket("/ws/metrics")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            self.connected_clients.add(websocket)
            try:
                while True:
                    await asyncio.sleep(1)  # Send updates every second
                    metrics = {
                        'system': await self.metrics_collector.collect_system_metrics(),
                        'llm': await self.metrics_collector.collect_llm_metrics()
                    }
                    await websocket.send_text(json.dumps(metrics))
            except WebSocketDisconnect:
                logger.info("WebSocket client disconnected")
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
            finally:
                self.connected_clients.discard(websocket)
    
    def get_dashboard_html(self) -> str:
        """Return the dashboard HTML."""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>vLLM MxExt Monitoring Dashboard</title>
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <style>
                body { 
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                    margin: 0; 
                    padding: 20px; 
                    background-color: #f5f5f5;
                }
                .header {
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 20px;
                    border-radius: 10px;
                    margin-bottom: 20px;
                    text-align: center;
                }
                .time-range-controls {
                    text-align: center;
                    margin-bottom: 20px;
                }
                .time-range-btn {
                    background: #667eea;
                    color: white;
                    border: none;
                    padding: 8px 16px;
                    margin: 0 5px;
                    border-radius: 5px;
                    cursor: pointer;
                    transition: background 0.3s;
                }
                .time-range-btn:hover {
                    background: #5a6fd8;
                }
                .time-range-btn.active {
                    background: #4CAF50;
                }
                .dashboard-grid { 
                    display: grid; 
                    grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); 
                    gap: 20px; 
                }
                .metric-card { 
                    background: white;
                    border: none;
                    border-radius: 10px; 
                    padding: 20px; 
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                    transition: transform 0.2s;
                }
                .metric-card:hover {
                    transform: translateY(-2px);
                    box-shadow: 0 8px 15px rgba(0, 0, 0, 0.2);
                }
                .metric-card h3 {
                    margin-top: 0;
                    color: #333;
                    border-bottom: 2px solid #667eea;
                    padding-bottom: 10px;
                }
                .metric-value { 
                    font-size: 2em; 
                    font-weight: bold; 
                    color: #667eea; 
                    margin: 10px 0;
                }
                .gpu-details {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 15px;
                    margin-top: 15px;
                }
                .gpu-card {
                    background: #f8f9fa;
                    border-radius: 8px;
                    padding: 15px;
                    border-left: 4px solid #667eea;
                }
                .gpu-card h4 {
                    margin: 0 0 10px 0;
                    color: #333;
                    font-size: 1.1em;
                }
                .gpu-metric {
                    display: flex;
                    justify-content: space-between;
                    margin: 5px 0;
                    font-size: 0.9em;
                }
                .gpu-metric .label {
                    color: #666;
                }
                .gpu-metric .value {
                    font-weight: bold;
                    color: #333;
                }
                .chart-container { 
                    position: relative; 
                    height: 300px; 
                }
                .status-indicator {
                    display: inline-block;
                    width: 12px;
                    height: 12px;
                    border-radius: 50%;
                    margin-right: 8px;
                }
                .status-online { background-color: #4CAF50; }
                .status-warning { background-color: #FF9800; }
                .status-error { background-color: #F44336; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🚀 Metax MIM Monitoring Dashboard</h1>
                <p><span id="connectionStatus" class="status-indicator status-online"></span>Real-time System Monitoring</p>
            </div>
            
            <div class="time-range-controls">
                <button class="time-range-btn active" onclick="changeTimeRange('1m')">1Min</button>
                <button class="time-range-btn" onclick="changeTimeRange('5m')">5Min</button>
                <button class="time-range-btn" onclick="changeTimeRange('1h')">1Hour</button>
                <button class="time-range-btn" onclick="changeTimeRange('6h')">6Hours</button>
                <button class="time-range-btn" onclick="changeTimeRange('24h')">24Hours</button>
            </div>
            
            <div class="dashboard-grid">
                <div class="metric-card">
                    <h3>💻 System CPU Usage</h3>
                    <div class="metric-value" id="cpuValue">0%</div>
                    <div class="chart-container">
                        <canvas id="cpuChart"></canvas>
                    </div>
                </div>
                <div class="metric-card">
                    <h3>🧠 Memory Usage</h3>
                    <div class="metric-value" id="memoryValue">0%</div>
                    <div class="chart-container">
                        <canvas id="memoryChart"></canvas>
                    </div>
                </div>
                <div class="metric-card">
                    <h3>💾 Disk Usage</h3>
                    <div class="metric-value" id="diskValue">0%</div>
                    <div class="chart-container">
                        <canvas id="diskChart"></canvas>
                    </div>
                </div>
                <div class="metric-card">
                    <h3>🌐 Network I/O</h3>
                    <div class="metric-value" id="networkValue">0 MB/s</div>
                    <div class="chart-container">
                        <canvas id="networkChart"></canvas>
                    </div>
                </div>
                <div class="metric-card">
                    <h3>🎮 GPU Overview</h3>
                    <div class="metric-value" id="gpuValue">0%</div>
                    <div id="gpuDetails" class="gpu-details">
                        <!-- GPU details will be populated here -->
                    </div>
                </div>
                <div class="metric-card">
                    <h3>⚡ LLM Performance</h3>
                    <div class="metric-value" id="llmValue">0 req/s</div>
                    <div class="chart-container">
                        <canvas id="llmChart"></canvas>
                    </div>
                </div>
                <div class="metric-card">
                    <h3>📊 Request Statistics</h3>
                    <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; text-align: center;">
                        <div>
                            <div class="metric-value" id="activeRequests">0</div>
                            <small>Active</small>
                        </div>
                        <div>
                            <div class="metric-value" id="queuedRequests">0</div>
                            <small>Queued</small>
                        </div>
                        <div>
                            <div class="metric-value" id="completedRequests">0</div>
                            <small>Completed</small>
                        </div>
                    </div>
                </div>
                <div class="metric-card">
                    <h3>⏱️ Performance Metrics</h3>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
                        <div>
                            <div class="metric-value" id="ttftValue">0ms</div>
                            <small>TTFT (P95)</small>
                        </div>
                        <div>
                            <div class="metric-value" id="throughputValue">0</div>
                            <small>Tokens/sec</small>
                        </div>
                    </div>
                </div>
            </div>
            
            <script>
                // Dashboard JavaScript for real-time metrics visualization
                class DashboardManager {
                    constructor() {
                        this.charts = {};
                        this.websocket = null;
                        this.currentTimeRange = '1m';
                        this.initializeCharts();
                        this.connectWebSocket();
                    }

                    initializeCharts() {
                        const commonOptions = {
                            responsive: true,
                            maintainAspectRatio: false,
                            animation: { duration: 0 },
                            plugins: {
                                legend: { display: true, position: 'top' }
                            }
                        };

                        // CPU Chart
                        this.charts.cpu = new Chart(document.getElementById('cpuChart'), {
                            type: 'line',
                            data: {
                                labels: [],
                                datasets: [{
                                    label: 'CPU Usage %',
                                    data: [],
                                    borderColor: 'rgb(102, 126, 234)',
                                    backgroundColor: 'rgba(102, 126, 234, 0.1)',
                                    tension: 0.4,
                                    fill: true
                                }]
                            },
                            options: {
                                ...commonOptions,
                                scales: {
                                    y: { beginAtZero: true, max: 100 }
                                }
                            }
                        });

                        // Memory Chart (Doughnut)
                        this.charts.memory = new Chart(document.getElementById('memoryChart'), {
                            type: 'doughnut',
                            data: {
                                labels: ['Used', 'Free'],
                                datasets: [{
                                    data: [0, 100],
                                    backgroundColor: ['#FF6384', '#36A2EB'],
                                    borderWidth: 2
                                }]
                            },
                            options: commonOptions
                        });

                        // Disk Chart (Doughnut)
                        this.charts.disk = new Chart(document.getElementById('diskChart'), {
                            type: 'doughnut',
                            data: {
                                labels: ['Used', 'Free'],
                                datasets: [{
                                    data: [0, 100],
                                    backgroundColor: ['#FFCE56', '#4BC0C0'],
                                    borderWidth: 2
                                }]
                            },
                            options: commonOptions
                        });

                        // Network Chart
                        this.charts.network = new Chart(document.getElementById('networkChart'), {
                            type: 'line',
                            data: {
                                labels: [],
                                datasets: [{
                                    label: 'Network I/O (MB/s)',
                                    data: [],
                                    borderColor: 'rgb(75, 192, 192)',
                                    backgroundColor: 'rgba(75, 192, 192, 0.1)',
                                    tension: 0.4,
                                    fill: true
                                }]
                            },
                            options: {
                                ...commonOptions,
                                scales: {
                                    y: { beginAtZero: true }
                                }
                            }
                        });

                        // GPU Chart
                        this.charts.gpu = new Chart(document.getElementById('gpuChart'), {
                            type: 'bar',
                            data: {
                                labels: [],
                                datasets: [{
                                    label: 'GPU Load %',
                                    data: [],
                                    backgroundColor: 'rgba(153, 102, 255, 0.8)',
                                    borderColor: 'rgba(153, 102, 255, 1)',
                                    borderWidth: 1
                                }]
                            },
                            options: {
                                ...commonOptions,
                                scales: {
                                    y: { beginAtZero: true, max: 100 }
                                }
                            }
                        });

                        // LLM Performance Chart
                        this.charts.llm = new Chart(document.getElementById('llmChart'), {
                            type: 'line',
                            data: {
                                labels: [],
                                datasets: [
                                    {
                                        label: 'TTFT (ms)',
                                        data: [],
                                        borderColor: 'rgb(255, 99, 132)',
                                        backgroundColor: 'rgba(255, 99, 132, 0.1)',
                                        yAxisID: 'y',
                                        tension: 0.4
                                    },
                                    {
                                        label: 'Throughput (tokens/s)',
                                        data: [],
                                        borderColor: 'rgb(54, 162, 235)',
                                        backgroundColor: 'rgba(54, 162, 235, 0.1)',
                                        yAxisID: 'y1',
                                        tension: 0.4
                                    }
                                ]
                            },
                            options: {
                                ...commonOptions,
                                scales: {
                                    y: { 
                                        type: 'linear', 
                                        display: true, 
                                        position: 'left',
                                        title: { display: true, text: 'TTFT (ms)' }
                                    },
                                    y1: { 
                                        type: 'linear', 
                                        display: true, 
                                        position: 'right',
                                        title: { display: true, text: 'Tokens/sec' },
                                        grid: { drawOnChartArea: false }
                                    }
                                }
                            }
                        });
                    }

                    connectWebSocket() {
                        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                        const wsUrl = `${protocol}//${window.location.host}/ws/metrics`;
                        
                        console.log('Connecting to WebSocket:', wsUrl);
                        this.websocket = new WebSocket(wsUrl);
                        
                        this.websocket.onopen = () => {
                            console.log('WebSocket connected successfully');
                            this.updateConnectionStatus(true);
                        };
                        
                        this.websocket.onmessage = (event) => {
                            try {
                                const data = JSON.parse(event.data);
                                console.log('=== WebSocket Data Received ===');
                                console.log('Raw event data:', event.data);
                                console.log('Parsed data:', data);
                                
                                // 确保DOM已加载完成
                                if (document.readyState === 'loading') {
                                    console.warn('DOM not ready, deferring update');
                                    document.addEventListener('DOMContentLoaded', () => {
                                        this.updateMetricValues(data);
                                        this.updateCharts(data);
                                    });
                                } else {
                                    // 立即更新
                                    console.log('Updating metrics and charts...');
                                    this.updateMetricValues(data);
                                    this.updateCharts(data);
                                }
                                
                            } catch (error) {
                                console.error('Error parsing WebSocket data:', error);
                                console.error('Raw data that failed to parse:', event.data);
                            }
                        };
                        
                        this.websocket.onclose = (event) => {
                            console.log('WebSocket disconnected:', event.code, event.reason);
                            this.updateConnectionStatus(false);
                            setTimeout(() => this.connectWebSocket(), 5000);
                        };

                        this.websocket.onerror = (error) => {
                            console.error('WebSocket error:', error);
                            this.updateConnectionStatus(false);
                        };
                    }

                    updateConnectionStatus(connected) {
                        const statusElement = document.getElementById('connectionStatus');
                        if (connected) {
                            statusElement.className = 'status-indicator status-online';
                        } else {
                            statusElement.className = 'status-indicator status-error';
                        }
                    }

                    updateMetricValues(data) {
                        console.log('=== updateMetricValues called ===');
                        console.log('Full data:', JSON.stringify(data, null, 2));
                        
                        // 强制更新系统指标
                        if (data && data.system) {
                            console.log('Processing system data...');
                            
                            // CPU更新
                            if (data.system.cpu && typeof data.system.cpu.usage_total === 'number') {
                                const cpuValue = data.system.cpu.usage_total;
                                const cpuElement = document.getElementById('cpuValue');
                                if (cpuElement) {
                                    cpuElement.textContent = `${cpuValue.toFixed(1)}%`;
                                    console.log(`✅ CPU updated: ${cpuValue.toFixed(1)}%`);
                                } else {
                                    console.error('❌ cpuValue element not found');
                                }
                            } else {
                                console.warn('❌ CPU data invalid:', data.system.cpu);
                            }
                            
                            // Memory更新
                            if (data.system.memory && typeof data.system.memory.percent === 'number') {
                                const memoryValue = data.system.memory.percent;
                                const memoryElement = document.getElementById('memoryValue');
                                if (memoryElement) {
                                    memoryElement.textContent = `${memoryValue.toFixed(1)}%`;
                                    console.log(`✅ Memory updated: ${memoryValue.toFixed(1)}%`);
                                } else {
                                    console.error('❌ memoryValue element not found');
                                }
                            } else {
                                console.warn('❌ Memory data invalid:', data.system.memory);
                            }
                            
                            // Disk更新
                            if (data.system.disk && typeof data.system.disk.percent === 'number') {
                                const diskValue = data.system.disk.percent;
                                const diskElement = document.getElementById('diskValue');
                                if (diskElement) {
                                    diskElement.textContent = `${diskValue.toFixed(1)}%`;
                                    console.log(`✅ Disk updated: ${diskValue.toFixed(1)}%`);
                                } else {
                                    console.error('❌ diskValue element not found');
                                }
                            } else {
                                console.warn('❌ Disk data invalid:', data.system.disk);
                            }
                            
                            // Network更新
                            if (data.system.network && typeof data.system.network.throughput_mbps === 'number') {
                                const networkValue = data.system.network.throughput_mbps;
                                const networkElement = document.getElementById('networkValue');
                                if (networkElement) {
                                    networkElement.textContent = `${networkValue.toFixed(2)} MB/s`;
                                    console.log(`✅ Network updated: ${networkValue.toFixed(2)} MB/s`);
                                } else {
                                    console.error('❌ networkValue element not found');
                                }
                            } else {
                                console.warn('❌ Network data invalid:', data.system.network);
                            }
                            
                            // GPU更新
                            if (data.system.gpu && Array.isArray(data.system.gpu) && data.system.gpu.length > 0) {
                                const avgGpuLoad = data.system.gpu.reduce((sum, gpu) => sum + (gpu.load || 0), 0) / data.system.gpu.length;
                                const gpuElement = document.getElementById('gpuValue');
                                if (gpuElement) {
                                    gpuElement.textContent = `${avgGpuLoad.toFixed(1)}%`;
                                    console.log(`✅ GPU updated: ${avgGpuLoad.toFixed(1)}%`);
                                } else {
                                    console.error('❌ gpuValue element not found');
                                }
                                this.updateGpuDetails(data.system.gpu);
                            } else {
                                console.warn('❌ GPU data invalid or empty:', data.system.gpu);
                                const gpuElement = document.getElementById('gpuValue');
                                if (gpuElement) {
                                    gpuElement.textContent = 'N/A';
                                }
                            }
                        } else {
                            console.error('❌ No system data received');
                        }

                        // 更新LLM指标
                        if (data && data.llm) {
                            console.log('Processing LLM data...');
                            
                            if (data.llm.requests) {
                                const activeElement = document.getElementById('activeRequests');
                                const queuedElement = document.getElementById('queuedRequests');
                                const completedElement = document.getElementById('completedRequests');
                                
                                if (activeElement) activeElement.textContent = data.llm.requests.active || 0;
                                if (queuedElement) queuedElement.textContent = data.llm.requests.queued || 0;
                                if (completedElement) completedElement.textContent = data.llm.requests.completed || 0;
                                
                                console.log('✅ LLM requests updated');
                            }
                            
                            if (data.llm.performance) {
                                const ttftValue = data.llm.performance.ttft_p95 || 0;
                                const throughputValue = data.llm.performance.throughput || 0;
                                
                                const ttftElement = document.getElementById('ttftValue');
                                const throughputElement = document.getElementById('throughputValue');
                                const llmElement = document.getElementById('llmValue');
                                
                                if (ttftElement) ttftElement.textContent = `${ttftValue.toFixed(1)}ms`;
                                if (throughputElement) throughputElement.textContent = `${throughputValue.toFixed(1)}`;
                                if (llmElement) llmElement.textContent = `${throughputValue.toFixed(1)} tok/s`;
                                
                                console.log('✅ LLM performance updated');
                            }
                        } else {
                            console.error('❌ No LLM data received');
                        }
                    }

                    updateGpuDetails(gpus) {
                        const gpuDetailsContainer = document.getElementById('gpuDetails');
                        if (!gpuDetailsContainer) return;
                        
                        gpuDetailsContainer.innerHTML = '';
                        
                        gpus.forEach(gpu => {
                            const gpuCard = document.createElement('div');
                            gpuCard.className = 'gpu-card';
                            
                            gpuCard.innerHTML = `
                                <h4>GPU ${gpu.id}: ${gpu.name}</h4>
                                <div class="gpu-metric">
                                    <span class="label">Utilization:</span>
                                    <span class="value">${gpu.load.toFixed(1)}%</span>
                                </div>
                                <div class="gpu-metric">
                                    <span class="label">Memory:</span>
                                    <span class="value">${gpu.memory_used}/${gpu.memory_total} MiB (${gpu.memory_percent?.toFixed(1) || 0}%)</span>
                                </div>
                                <div class="gpu-metric">
                                    <span class="label">Temperature:</span>
                                    <span class="value">${gpu.temperature || 'N/A'}°C</span>
                                </div>
                                ${gpu.power_usage !== undefined ? `
                                <div class="gpu-metric">
                                    <span class="label">Power:</span>
                                    <span class="value">${gpu.power_usage}/${gpu.power_limit}W (${gpu.power_percent?.toFixed(1) || 0}%)</span>
                                </div>
                                ` : ''}
                            `;
                            
                            gpuDetailsContainer.appendChild(gpuCard);
                        });
                    }

                    updateCharts(data) {
                        const timestamp = new Date().toLocaleTimeString();
                        
                        // Update CPU chart
                        if (data.system?.cpu?.usage_total !== undefined) {
                            this.updateLineChart(this.charts.cpu, timestamp, data.system.cpu.usage_total);
                        }
                        
                        // Update Memory chart
                        if (data.system?.memory?.percent !== undefined) {
                            const memoryPercent = data.system.memory.percent;
                            this.charts.memory.data.datasets[0].data = [memoryPercent, 100 - memoryPercent];
                            this.charts.memory.update('none');
                        }
                        
                        // Update Disk chart
                        if (data.system?.disk?.percent !== undefined) {
                            const diskPercent = data.system.disk.percent;
                            this.charts.disk.data.datasets[0].data = [diskPercent, 100 - diskPercent];
                            this.charts.disk.update('none');
                        }
                        
                        // Update Network chart
                        if (data.system?.network?.throughput_mbps !== undefined) {
                            this.updateLineChart(this.charts.network, timestamp, data.system.network.throughput_mbps);
                        }
                        
                        // Update GPU chart
                        if (data.system?.gpu && data.system.gpu.length > 0) {
                            this.charts.gpu.data.labels = data.system.gpu.map(gpu => `GPU ${gpu.id}`);
                            this.charts.gpu.data.datasets[0].data = data.system.gpu.map(gpu => gpu.load);
                            this.charts.gpu.update('none');
                        }
                        
                        // Update LLM chart
                        if (data.llm?.performance) {
                            this.updateLineChart(this.charts.llm, timestamp, 
                                data.llm.performance.ttft_p95 || 0, 0);
                            this.updateLineChart(this.charts.llm, timestamp, 
                                data.llm.performance.throughput || 0, 1);
                        }
                    }

                    updateLineChart(chart, label, value, datasetIndex = 0) {
                        if (datasetIndex === 0) {
                            chart.data.labels.push(label);
                        }
                        chart.data.datasets[datasetIndex].data.push(value);
                        
                        // Keep only last 50 data points
                        if (chart.data.labels.length > 50) {
                            chart.data.labels.shift();
                            chart.data.datasets.forEach(dataset => dataset.data.shift());
                        }
                        
                        chart.update('none');
                    }

                    async loadHistoricalData(timeRange) {
                        try {
                            const response = await fetch(`/api/metrics/history?timerange=${timeRange}`);
                            const data = await response.json();
                            
                            // Update charts with historical data
                            this.updateChartsWithHistoricalData(data);
                        } catch (error) {
                            console.error('Error loading historical data:', error);
                        }
                    }

                    updateChartsWithHistoricalData(data) {
                        // Clear existing data
                        Object.values(this.charts).forEach(chart => {
                            if (chart.data.labels) {
                                chart.data.labels = [];
                                chart.data.datasets.forEach(dataset => dataset.data = []);
                            }
                        });

                        // Populate with historical data
                        if (data.cpu_usage?.data_points) {
                            data.cpu_usage.data_points.forEach(([timestamp, value]) => {
                                const time = new Date(timestamp * 1000).toLocaleTimeString();
                                this.charts.cpu.data.labels.push(time);
                                this.charts.cpu.data.datasets[0].data.push(value);
                            });
                        }

                        if (data.network_throughput?.data_points) {
                            data.network_throughput.data_points.forEach(([timestamp, value]) => {
                                const time = new Date(timestamp * 1000).toLocaleTimeString();
                                this.charts.network.data.labels.push(time);
                                this.charts.network.data.datasets[0].data.push(value);
                            });
                        }

                        if (data.ttft?.data_points && data.throughput?.data_points) {
                            data.ttft.data_points.forEach(([timestamp, value], index) => {
                                const time = new Date(timestamp * 1000).toLocaleTimeString();
                                if (index === 0) this.charts.llm.data.labels.push(time);
                                this.charts.llm.data.datasets[0].data.push(value);
                            });
                            
                            data.throughput.data_points.forEach(([timestamp, value]) => {
                                this.charts.llm.data.datasets[1].data.push(value);
                            });
                        }

                        // Update all charts
                        Object.values(this.charts).forEach(chart => chart.update());
                    }
                }

                // Global functions
                function changeTimeRange(timeRange) {
                    // Update active button
                    document.querySelectorAll('.time-range-btn').forEach(btn => {
                        btn.classList.remove('active');
                    });
                    event.target.classList.add('active');
                    
                    // Load historical data
                    if (window.dashboardManager) {
                        window.dashboardManager.currentTimeRange = timeRange;
                        window.dashboardManager.loadHistoricalData(timeRange);
                    }
                }

                // Initialize dashboard when page loads
                document.addEventListener('DOMContentLoaded', function() {
                    window.dashboardManager = new DashboardManager();
                });
            </script>
        </body>
        </html>
        """
