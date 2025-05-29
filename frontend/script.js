console.log("script.js loaded");

// Basic structure for fetching and updating metrics
// This will be expanded in subsequent tasks

// Configuration
const API_BASE_URL = '/api/v1'; // Assuming the FastAPI server is serving on the same host/port
let refreshInterval = 10000; // Default 10 seconds, will be updated from UI
let historicalTimeWindow = 'hourly'; // Default, will be updated from UI

// DOM Elements
const refreshIntervalInput = document.getElementById('refreshInterval');
const timeWindowSelect = document.getElementById('timeWindow');

// Real-time metric display elements
const dialRequestsRunning = document.getElementById('dial-requests-running');
const dialGpuCacheUsage = document.getElementById('dial-gpu-cache-usage');

// Historical metric display elements
const histDialE2eLatency = document.getElementById('hist-dial-e2e-latency');

// --- Function to parse Prometheus text data for specific real-time metrics ---
function parsePrometheusTextToRealtimeMetrics(textData) {
    const parsedMetrics = {
        num_requests_running: null,
        gpu_cache_usage_perc: null
    };
    const lines = textData.split('\n');

    for (const line of lines) {
        if (line.startsWith('#')) {
            continue; // Skip comments
        }

        // For num_requests_running
        // Example line: num_requests_running 5
        // Example line with labels: num_requests_running{worker_id="0"} 2
        // We will take the first one found or sum if multiple (currently first)
        if (line.startsWith('num_requests_running')) {
            if (parsedMetrics.num_requests_running === null) { // Take the first one
                const parts = line.trim().split(' ');
                const value = parseFloat(parts[parts.length - 1]);
                if (!isNaN(value)) {
                    parsedMetrics.num_requests_running = value;
                }
            }
            // To sum:
            // const parts = line.trim().split(' ');
            // const value = parseFloat(parts[parts.length - 1]);
            // if (!isNaN(value)) {
            //     parsedMetrics.num_requests_running = (parsedMetrics.num_requests_running || 0) + value;
            // }
        }

        // For gpu_cache_usage_perc
        // Example line: gpu_cache_usage_perc 0.75
        if (line.startsWith('gpu_cache_usage_perc')) {
             if (parsedMetrics.gpu_cache_usage_perc === null) { // Take the first one
                const parts = line.trim().split(' ');
                const value = parseFloat(parts[parts.length - 1]);
                if (!isNaN(value)) {
                    parsedMetrics.gpu_cache_usage_perc = value;
                }
            }
        }
    }
    return parsedMetrics;
}


// --- Function to fetch real-time metrics ---
async function fetchRealtimeMetrics() {
    try {
        const response = await fetch('/metrics'); // Fetch from the Prometheus endpoint
        if (!response.ok) {
            throw new Error(`Failed to fetch /metrics: ${response.status} ${response.statusText}`);
        }
        const textData = await response.text();
        const metrics = parsePrometheusTextToRealtimeMetrics(textData);

        if (dialRequestsRunning) {
            dialRequestsRunning.textContent = metrics.num_requests_running !== null ? metrics.num_requests_running : 'N/A';
        }
        if (dialGpuCacheUsage) {
            dialGpuCacheUsage.textContent = metrics.gpu_cache_usage_perc !== null ? (metrics.gpu_cache_usage_perc * 100).toFixed(0) + '%' : 'N/A';
        }
        console.log('Real-time metrics updated from /metrics endpoint.');

    } catch (error) {
        console.error('Error fetching real-time metrics:', error);
        if(dialRequestsRunning) dialRequestsRunning.textContent = 'Error';
        if(dialGpuCacheUsage) dialGpuCacheUsage.textContent = 'Error';
    }
}

// --- Function to fetch historical metrics ---
async function fetchHistoricalMetrics() {
    const metricName = 'vllm_e2e_request_latency_seconds'; // Example metric
    const aggFunction = 'AVG'; // Example aggregation

    // Determine start and end times based on selected window
    const now = new Date();
    let startTime;

    switch (historicalTimeWindow) {
        case 'hourly':
            startTime = new Date(now.getTime() - 60 * 60 * 1000);
            break;
        case 'daily':
            startTime = new Date(now.getTime() - 24 * 60 * 60 * 1000);
            break;
        case 'weekly':
            startTime = new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000);
            break;
        default:
            startTime = new Date(now.getTime() - 60 * 60 * 1000); // Default to hourly
    }

    const startTimeISO = startTime.toISOString();
    const endTimeISO = now.toISOString();

    const url = `${API_BASE_URL}/historical_metrics/${metricName}?start_time_iso=${startTimeISO}&end_time_iso=${endTimeISO}&period=${historicalTimeWindow === 'weekly' ? 'daily' : 'hourly'}&agg=${aggFunction}`;
    // Note: For 'weekly', we might want 'daily' period aggregation.
    // The historical API supports 'hourly' and 'daily' periods. So for 'weekly' window, we might show daily averages.

    try {
        console.log(`Fetching historical data from: ${url}`);
        const response = await fetch(url);
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(`API Error (${response.status}): ${errorData.detail || 'Failed to fetch historical data'}`);
        }
        const data = await response.json();
        console.log('Historical data received:', data);

        if (data.length > 0) {
            // Assuming we want the latest aggregated value for the dial, or an average of averages.
            // For simplicity, let's take the last value if available, or average them.
            let valueToShow = "N/A";
            if (metricName === 'vllm_e2e_request_latency_seconds' && aggFunction === 'AVG') {
                // Average of the aggregated averages over the period
                const totalSum = data.reduce((sum, item) => sum + item.aggregated_value, 0);
                valueToShow = (totalSum / data.length).toFixed(4) + 's';
            }
            // Add more specific logic for other metrics/aggregations if needed
            
            if(histDialE2eLatency) histDialE2eLatency.textContent = valueToShow;

        } else {
            if(histDialE2eLatency) histDialE2eLatency.textContent = 'No Data';
        }
         console.log('Historical metrics updated.');

    } catch (error) {
        console.error('Error fetching historical metrics:', error);
        if(histDialE2eLatency) histDialE2eLatency.textContent = 'Error';
    }
}


// --- Interval Timers ---
let realtimeTimerId;
let historicalTimerId;

function setupTimers() {
    // Clear existing timers
    if (realtimeTimerId) clearInterval(realtimeTimerId);
    if (historicalTimerId) clearInterval(historicalTimerId);

    // Setup new timers with the current refreshInterval value
    if (refreshInterval > 0) {
        fetchRealtimeMetrics(); // Initial fetch
        realtimeTimerId = setInterval(fetchRealtimeMetrics, refreshInterval);

        fetchHistoricalMetrics(); // Initial fetch
        historicalTimerId = setInterval(fetchHistoricalMetrics, refreshInterval); // Historical also refreshes
    }
}

// --- Event Listeners ---
if (refreshIntervalInput) {
    refreshIntervalInput.addEventListener('change', (event) => {
        const newIntervalSeconds = parseInt(event.target.value, 10);
        if (newIntervalSeconds > 0) {
            refreshInterval = newIntervalSeconds * 1000;
            console.log(`Refresh interval set to ${newIntervalSeconds} seconds.`);
            setupTimers(); // Reset timers with new interval
        }
    });
}

if (timeWindowSelect) {
    timeWindowSelect.addEventListener('change', (event) => {
        historicalTimeWindow = event.target.value;
        console.log(`Historical time window changed to: ${historicalTimeWindow}`);
        fetchHistoricalMetrics(); // Fetch immediately on change
        // Timers will pick up the new window on their next scheduled run, or we can reset them:
        setupTimers();
    });
}

// --- Initial Setup ---
document.addEventListener('DOMContentLoaded', () => {
    // Set initial values from UI if needed (e.g., refresh interval)
    if (refreshIntervalInput) {
        refreshInterval = parseInt(refreshIntervalInput.value, 10) * 1000;
    }
    if (timeWindowSelect) {
        historicalTimeWindow = timeWindowSelect.value;
    }
    
    console.log("DOM fully loaded and parsed. Initializing metrics fetch.");
    setupTimers();
});
