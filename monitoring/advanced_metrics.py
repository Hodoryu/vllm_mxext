"""Advanced metrics collection and analysis for the dashboard."""

import asyncio
import time
from collections import deque
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class TimeSeriesPoint:
    timestamp: float
    value: float

class TimeSeriesBuffer:
    """Efficient time-series data storage with automatic cleanup."""
    
    def __init__(self, max_age_seconds: int = 86400):  # 24 hours default
        self.data = deque()
        self.max_age = max_age_seconds
    
    def add_point(self, value: float, timestamp: Optional[float] = None):
        if timestamp is None:
            timestamp = time.time()
        
        self.data.append(TimeSeriesPoint(timestamp, value))
        self._cleanup_old_data()
    
    def get_data_for_range(self, start_time: float, end_time: float) -> List[TimeSeriesPoint]:
        return [point for point in self.data 
                if start_time <= point.timestamp <= end_time]
    
    def _cleanup_old_data(self):
        cutoff_time = time.time() - self.max_age
        while self.data and self.data[0].timestamp < cutoff_time:
            self.data.popleft()

class AdvancedMetricsCollector:
    """Advanced metrics collection with time-series storage."""
    
    def __init__(self):
        self.buffers = {
            'cpu_usage': TimeSeriesBuffer(),
            'memory_usage': TimeSeriesBuffer(),
            'gpu_utilization': TimeSeriesBuffer(),
            'ttft': TimeSeriesBuffer(),
            'throughput': TimeSeriesBuffer(),
            'active_requests': TimeSeriesBuffer()
        }
    
    def record_metric(self, metric_name: str, value: float):
        if metric_name in self.buffers:
            self.buffers[metric_name].add_point(value)
    
    def get_aggregated_data(self, timerange: str) -> Dict:
        """Get aggregated metrics for specified timerange."""
        time_ranges = {
            '1m': 60,
            '5m': 300,
            '1h': 3600,
            '6h': 21600,
            '24h': 86400,
            '7d': 604800
        }
        
        seconds = time_ranges.get(timerange, 3600)
        end_time = time.time()
        start_time = end_time - seconds
        
        result = {}
        for metric_name, buffer in self.buffers.items():
            data_points = buffer.get_data_for_range(start_time, end_time)
            if data_points:
                values = [point.value for point in data_points]
                result[metric_name] = {
                    'min': min(values),
                    'max': max(values),
                    'avg': sum(values) / len(values),
                    'current': values[-1] if values else 0,
                    'data_points': [(point.timestamp, point.value) for point in data_points]
                }
            else:
                result[metric_name] = {
                    'min': 0, 'max': 0, 'avg': 0, 'current': 0, 'data_points': []
                }
        
        return result