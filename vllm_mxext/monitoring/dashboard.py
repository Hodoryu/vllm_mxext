"""Web-based monitoring dashboard for vLLM MxExt."""

import asyncio
import json
import time
import os
from typing import Dict, List, Optional
from datetime import datetime, timedelta

from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import psutil

from vllm_mxext.logger import init_logger
from vllm_mxext.monitoring.advanced_metrics import AdvancedMetricsCollector

logger = init_logger(__name__)

class MetricsCollector:
    """Collects and aggregates metrics for the dashboard."""
    
    def __init__(self, stat_logger=None):
        self.stat_logger = stat_logger
        self.advanced_collector = AdvancedMetricsCollector()
        
    async def collect_system_metrics(self) -> Dict:
        """Collect system-level metrics."""
        try:
            cpu_percent = psutil.cpu_percent(interval=None, percpu=True)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            network = psutil.net_io_counters()
            
            # GPU metrics - handle import gracefully
            gpu_metrics = []
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                for gpu in gpus:
                    gpu_metrics.append({
                        'id': gpu.id,
                        'name': gpu.name,
                        'load': gpu.load * 100,
                        'memory_used': gpu.memoryUsed,
                        'memory_total': gpu.memoryTotal,
                        'temperature': gpu.temperature
                    })
            except ImportError:
                logger.warning("GPUtil not available, GPU metrics disabled")
            except Exception as e:
                logger.warning(f"Error collecting GPU metrics: {e}")
                
            system_metrics = {
                'timestamp': time.time(),
                'cpu': {
                    'usage_per_core': cpu_percent,
                    'usage_total': psutil.cpu_percent()
                },
                'memory': {
                    'used': memory.used,
                    'total': memory.total,
                    'percent': memory.percent
                },
                'disk': {
                    'used': disk.used,
                    'total': disk.total,
                    'percent': (disk.used / disk.total) * 100
                },
                'network': {
                    'bytes_sent': network.bytes_sent,
                    'bytes_recv': network.bytes_recv
                },
                'gpu': gpu_metrics
            }
            
            # Record metrics for time-series
            self.advanced_collector.record_metric('cpu_usage', system_metrics['cpu']['usage_total'])
            self.advanced_collector.record_metric('memory_usage', system_metrics['memory']['percent'])
            if gpu_metrics:
                avg_gpu_load = sum(gpu['load'] for gpu in gpu_metrics) / len(gpu_metrics)
                self.advanced_collector.record_metric('gpu_utilization', avg_gpu_load)
                
            return system_metrics
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            return {'timestamp': time.time(), 'error': str(e)}
    
    async def collect_llm_metrics(self) -> Dict:
        """Collect LLM-specific metrics."""
        try:
            if hasattr(self.stat_logger, 'get_dashboard_metrics'):
                metrics = self.stat_logger.get_dashboard_metrics()
                
                # Record for time-series
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
                return {
                    'timestamp': time.time(),
                    'requests': {'active': 0, 'queued': 0, 'completed': 0},
                    'performance': {'ttft_p50': 0, 'ttft_p95': 0, 'ttft_p99': 0, 'ttot_avg': 0, 'throughput': 0}
                }
        except Exception as e:
            logger.error(f"Error collecting LLM metrics: {e}")
            return {'timestamp': time.time(), 'error': str(e)}


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
                .controls {
                    margin-bottom: 20px;
                    text-align: center;
                }
                .time-range-btn {
                    background: #667eea;
                    color: white;
                    border: none;
                    padding: 8px 16px;
                    margin: 0 5px;
                    border-radius: 5px;
                    cursor: pointer;
                    transition: background 0.2s;
                }
                .time-range-btn:hover {
                    background: #5a6fd8;
                }
                .time-range-btn.active {
                    background: #764ba2;
                }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🚀 vLLM MxExt Monitoring Dashboard</h1>
                <p><span id="connectionStatus" class="status-indicator status-online"></span>Real-time System Monitoring</p>
            </div>
            
            <div class="controls">
                <button class="time-range-btn active" onclick="changeTimeRange('1m')">1 Min</button>
                <button class="time-range-btn" onclick="changeTimeRange('5m')">5 Min</button>
                <button class="time-range-btn" onclick="changeTimeRange('1h')">1 Hour</button>
                <button class="time-range-btn" onclick="changeTimeRange('6h')">6 Hours</button>
                <button class="time-range-btn" onclick="changeTimeRange('24h')">24 Hours</button>
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
                    <h3>🎮 GPU Utilization</h3>
                    <div class="metric-value" id="gpuValue">0%</div>
                    <div class="chart-container">
                        <canvas id="gpuChart"></canvas>
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
            <script src="/static/dashboard.js"></script>
        </body>
        </html>
        """