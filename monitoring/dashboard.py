"""Web-based monitoring dashboard for vLLM MxExt."""

import asyncio
import json
import time
from typing import Dict, List, Optional
from datetime import datetime, timedelta

from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import psutil
import GPUtil

from vllm_mxext.engine.metrics import StatLogger, NimMetrics


class MetricsCollector:
    """Collects and aggregates metrics for the dashboard."""
    
    def __init__(self, stat_logger: Optional[StatLogger] = None):
        self.stat_logger = stat_logger
        self.metrics_history = []
        self.max_history_size = 10000
        
    async def collect_system_metrics(self) -> Dict:
        """Collect system-level metrics."""
        cpu_percent = psutil.cpu_percent(interval=None, percpu=True)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        network = psutil.net_io_counters()
        
        # GPU metrics
        gpu_metrics = []
        try:
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
        except Exception:
            pass
            
        return {
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
    
    async def collect_llm_metrics(self) -> Dict:
        """Collect LLM-specific metrics."""
        if not self.stat_logger:
            return {}
            
        # Extract metrics from StatLogger/NimMetrics
        return {
            'timestamp': time.time(),
            'requests': {
                'active': 0,  # Will be populated from actual metrics
                'queued': 0,
                'completed': 0
            },
            'performance': {
                'ttft_p50': 0,
                'ttft_p95': 0,
                'ttft_p99': 0,
                'ttot_avg': 0,
                'throughput': 0
            }
        }


class DashboardManager:
    """Manages the monitoring dashboard."""
    
    def __init__(self):
        self.metrics_collector = MetricsCollector()
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
            # Return historical metrics based on timerange
            return JSONResponse(self.get_historical_metrics(timerange))
            
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
            except Exception:
                pass
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
            <style>
                body { font-family: Arial, sans-serif; margin: 0; padding: 20px; }
                .dashboard-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; }
                .metric-card { border: 1px solid #ddd; border-radius: 8px; padding: 20px; }
                .metric-value { font-size: 2em; font-weight: bold; color: #2196F3; }
                .chart-container { position: relative; height: 300px; }
            </style>
        </head>
        <body>
            <h1>vLLM MxExt Monitoring Dashboard</h1>
            <div class="dashboard-grid">
                <div class="metric-card">
                    <h3>System CPU Usage</h3>
                    <div class="chart-container">
                        <canvas id="cpuChart"></canvas>
                    </div>
                </div>
                <div class="metric-card">
                    <h3>Memory Usage</h3>
                    <div class="chart-container">
                        <canvas id="memoryChart"></canvas>
                    </div>
                </div>
                <div class="metric-card">
                    <h3>GPU Utilization</h3>
                    <div class="chart-container">
                        <canvas id="gpuChart"></canvas>
                    </div>
                </div>
                <div class="metric-card">
                    <h3>LLM Performance</h3>
                    <div class="chart-container">
                        <canvas id="llmChart"></canvas>
                    </div>
                </div>
            </div>
            <script src="/static/dashboard.js"></script>
        </body>
        </html>
        """
    
    def get_historical_metrics(self, timerange: str) -> List[Dict]:
        """Get historical metrics for specified timerange."""
        # Implementation for historical data retrieval
        return []