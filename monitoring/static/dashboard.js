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

        // Memory Chart
        this.charts.memory = new Chart(document.getElementById('memoryChart'), {
            type: 'doughnut',
            data: {
                labels: ['Used', 'Free'],
                datasets: [{
                    data: [0, 100],
                    backgroundColor: ['#FF6384', '#36A2EB'],
                    borderWidth: 0
                }]
            },
            options: {
                ...commonOptions,
                cutout: '60%'
            }
        });

        // Disk Chart
        this.charts.disk = new Chart(document.getElementById('diskChart'), {
            type: 'doughnut',
            data: {
                labels: ['Used', 'Free'],
                datasets: [{
                    data: [0, 100],
                    backgroundColor: ['#FF9F40', '#4BC0C0'],
                    borderWidth: 0
                }]
            },
            options: {
                ...commonOptions,
                cutout: '60%'
            }
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
        
        this.websocket = new WebSocket(wsUrl);
        
        this.websocket.onopen = () => {
            console.log('WebSocket connected');
            this.updateConnectionStatus(true);
        };
        
        this.websocket.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                this.updateCharts(data);
                this.updateMetricValues(data);
            } catch (error) {
                console.error('Error parsing WebSocket data:', error);
            }
        };
        
        this.websocket.onclose = () => {
            console.log('WebSocket disconnected, attempting to reconnect...');
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
        // Update metric value displays
        if (data.system) {
            document.getElementById('cpuValue').textContent = 
                `${data.system.cpu?.usage_total?.toFixed(1) || 0}%`;
            document.getElementById('memoryValue').textContent = 
                `${data.system.memory?.percent?.toFixed(1) || 0}%`;
            document.getElementById('diskValue').textContent = 
                `${data.system.disk?.percent?.toFixed(1) || 0}%`;
            document.getElementById('networkValue').textContent = 
                `${(data.system.network?.throughput_mbps || 0).toFixed(2)} MB/s`;
            
            if (data.system.gpu && data.system.gpu.length > 0) {
                const avgGpuLoad = data.system.gpu.reduce((sum, gpu) => sum + gpu.load, 0) / data.system.gpu.length;
                document.getElementById('gpuValue').textContent = `${avgGpuLoad.toFixed(1)}%`;
                
                // Update GPU details
                this.updateGpuDetails(data.system.gpu);
            }
        }

        if (data.llm) {
            if (data.llm.requests) {
                document.getElementById('activeRequests').textContent = data.llm.requests.active || 0;
                document.getElementById('queuedRequests').textContent = data.llm.requests.queued || 0;
                document.getElementById('completedRequests').textContent = data.llm.requests.completed || 0;
            }
            
            if (data.llm.performance) {
                document.getElementById('ttftValue').textContent = 
                    `${(data.llm.performance.ttft_p95 || 0).toFixed(1)}ms`;
                document.getElementById('throughputValue').textContent = 
                    `${(data.llm.performance.throughput || 0).toFixed(1)}`;
                document.getElementById('llmValue').textContent = 
                    `${(data.llm.performance.throughput || 0).toFixed(1)} tok/s`;
            }
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
function changeTimeRange(range) {
    // Update button states
    document.querySelectorAll('.time-range-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    event.target.classList.add('active');
    
    // Load historical data for the selected range
    if (window.dashboardManager) {
        window.dashboardManager.currentTimeRange = range;
        window.dashboardManager.loadHistoricalData(range);
    }
}

// Initialize dashboard when page loads
document.addEventListener('DOMContentLoaded', () => {
    window.dashboardManager = new DashboardManager();
});
