# Add dashboard integration methods to StatLogger class

    def get_dashboard_metrics(self) -> Dict:
        """Get metrics formatted for dashboard consumption."""
        # This method should be added to the existing StatLogger class
        dashboard_metrics = getattr(self, 'dashboard_metrics', {
            'active_requests': 0,
            'queued_requests': 0,
            'completed_requests': 0,
            'ttft_samples': [],
            'ttot_samples': [],
            'throughput_samples': []
        })
        
        return {
            'requests': {
                'active': dashboard_metrics.get('active_requests', 0),
                'queued': dashboard_metrics.get('queued_requests', 0),
                'completed': dashboard_metrics.get('completed_requests', 0)
            },
            'performance': {
                'ttft_p50': self._calculate_percentile(dashboard_metrics.get('ttft_samples', []), 50),
                'ttft_p95': self._calculate_percentile(dashboard_metrics.get('ttft_samples', []), 95),
                'ttft_p99': self._calculate_percentile(dashboard_metrics.get('ttft_samples', []), 99),
                'ttot_avg': sum(dashboard_metrics.get('ttot_samples', [])) / len(dashboard_metrics.get('ttot_samples', [])) if dashboard_metrics.get('ttot_samples') else 0,
                'throughput': sum(dashboard_metrics.get('throughput_samples', [])) / len(dashboard_metrics.get('throughput_samples', [])) if dashboard_metrics.get('throughput_samples') else 0
            }
        }
    
    def _calculate_percentile(self, samples: List[float], percentile: int) -> float:
        """Calculate percentile from samples."""
        if not samples:
            return 0
        sorted_samples = sorted(samples)
        index = int((percentile / 100) * len(sorted_samples))
        return sorted_samples[min(index, len(sorted_samples) - 1)]
    
    def update_dashboard_metrics(self, metric_type: str, value: float):
        """Update dashboard-specific metrics."""
        if not hasattr(self, 'dashboard_metrics'):
            self.dashboard_metrics = {
                'active_requests': 0,
                'queued_requests': 0,
                'completed_requests': 0,
                'ttft_samples': [],
                'ttot_samples': [],
                'throughput_samples': []
            }
        
        if metric_type in ['ttft_samples', 'ttot_samples', 'throughput_samples']:
            samples = self.dashboard_metrics[metric_type]
            samples.append(value)
            # Keep only last 100 samples
            if len(samples) > 100:
                samples.pop(0)
        elif metric_type in self.dashboard_metrics:
            self.dashboard_metrics[metric_type] = value