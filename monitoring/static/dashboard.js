// Dashboard JavaScript for real-time metrics visualization

class DashboardManager {
    constructor() {
        this.charts = {};
        this.websocket = null;
        this.initializeCharts();
        this.connectWebSocket();
    }

    initializeCharts() {
        // CPU Chart
        this.charts.cpu = new Chart(document.getElementById('cpuChart'), {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'CPU Usage %',
                    data: [],
                    borderColor: 'rgb(75, 192, 192)',
                    tension: 0.1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
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
                    backgroundColor: ['#FF6384', '#36A2EB']
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false
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
                    backgroundColor: 'rgba(153, 102, 255, 0.6)'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
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
                        yAxisID: 'y'
                    },
                    {
                        label: 'Throughput (tokens/s)',
                        data: [],
                        borderColor: 'rgb(54, 162, 235)',
                        yAxisID: 'y1'
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: { type: 'linear', display: true, position: 'left' },
                    y1: { type: 'linear', display: true, position: 'right' }
                }
            }
        });
    }

    connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/metrics`;
        
        this.websocket = new WebSocket(wsUrl);
        
        this.websocket.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.updateCharts(data);
        };
        
        this.websocket.onclose = () => {
            setTimeout(() => this.connectWebSocket(), 5000);
        };
    }

    updateCharts(data) {
        const timestamp = new Date().toLocaleTimeString();
        
        // Update CPU chart
        this.updateLineChart(this.charts.cpu, timestamp, data.system.cpu.usage_total);
        
        // Update Memory chart
        const memoryPercent = data.system.memory.percent;
        this.charts.memory.data.datasets[0].data = [memoryPercent, 100 - memoryPercent];
        this.charts.memory.update();
        
        // Update GPU chart
        if (data.system.gpu && data.system.gpu.length > 0) {
            this.charts.gpu.data.labels = data.system.gpu.map(gpu => gpu.name);
            this.charts.gpu.data.datasets[0].data = data.system.gpu.map(gpu => gpu.load);
            this.charts.gpu.update();
        }
        
        // Update LLM chart
        if (data.llm.performance) {
            this.updateLineChart(this.charts.llm, timestamp, 
                data.llm.performance.ttft_p50, 0);
            this.updateLineChart(this.charts.llm, timestamp, 
                data.llm.performance.throughput, 1);
        }
    }

    updateLineChart(chart, label, value, datasetIndex = 0) {
        chart.data.labels.push(label);
        chart.data.datasets[datasetIndex].data.push(value);
        
        // Keep only last 50 data points
        if (chart.data.labels.length > 50) {
            chart.data.labels.shift();
            chart.data.datasets[datasetIndex].data.shift();
        }
        
        chart.update('none');
    }
}

// Initialize dashboard when page loads
document.addEventListener('DOMContentLoaded', () => {
    new DashboardManager();
});