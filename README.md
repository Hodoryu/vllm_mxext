# Vllm Nvext (`vllm_mxext`)

## Introduction

Vllm Nvext (packaged as `vllm_mxext`) is an advanced Python-based inference server designed for high-performance serving of Large Language Models (LLMs). It leverages NVIDIA TensorRT-LLM for optimized execution on NVIDIA GPUs and extends concepts from the vLLM project. Key features include an OpenAI-compatible API, support for LoRA (Low-Rank Adaptation), and a manifest-driven "Model Experience" (MX) Hub for streamlined model management and deployment.

## Features

*   **High-Performance Inference:** Utilizes NVIDIA TensorRT-LLM for optimized LLM execution.
*   **Asynchronous Request Handling:** Built with FastAPI and Uvicorn for concurrent processing.
*   **OpenAI-Compatible API:** Allows easy integration with existing OpenAI API clients and workflows.
*   **LoRA Support:** Enables dynamic loading and serving of models adapted with LoRA.
*   **Manifest-Driven Model Management:** The MX Hub simplifies model configuration and deployment using manifest files.
*   **Hardware-Aware Configuration:** Adapts settings based on detected NVIDIA GPU hardware.
*   **Metrics and Monitoring:** Includes Prometheus-compatible metrics for observing server performance.
*   **Configurable Logging:** Flexible logging system with support for JSONL format and environment variable controls.

## Dependencies

### Python
*   Python >= 3.8

### External Python Packages
The project relies on the following external Python packages. It is recommended to add these to the `project.dependencies` section of `pyproject.toml` for automatic installation.
*   `vllm`
*   `numpy`
*   `packaging`
*   `prometheus_client`
*   `tensorrt_llm`
*   `transformers`
*   `pydantic`
*   `uvloop` (recommended for FastAPI performance)
*   `fastapi`
*   `starlette`
*   `typing_extensions`
*   `torch`
*   `pyyaml`
*   `nim_hub`
*   `nvidia-lora-conversions`
*   `minio`
*   `progress` (or a similar progress bar library like `progressbar2` or `tqdm`)
*   `requests`
*   `tqdm`
*   `modelscope`

## Installation

1.  **Prerequisites:**
    *   Ensure Python (>=3.8) and `pip` are installed.
    *   Install the external Python packages listed above. (e.g., `pip install fastapi uvicorn pydantic "tensorrt_llm>=0.9.0" ...`)
    *   NVIDIA GPU drivers, CUDA toolkit compatible with TensorRT-LLM.

2.  **Clone the Repository:**
    ```bash
    git clone <your_repository_url>
    cd vllm_mxext # Or your repository name
    ```

3.  **Install the Package:**
    *   For editable mode (recommended for development):
        ```bash
        pip install -e .
        ```
    *   For standard installation:
        ```bash
        pip install .
        ```

## Configuration

### Environment Variables
The project's behavior, particularly logging, can be configured via environment variables defined in `envs.py`:
*   `VLLM_MXEXT_CONFIGURE_LOGGING`: (Default: `1`) Set to `0` to disable automatic logging configuration.
*   `VLLM_MXEXT_LOGGING_CONFIG_PATH`: Path to a custom logging configuration file.
*   `VLLM_MXEXT_LOG_LEVEL`: (Default: `INFO`) Logging level (e.g., `DEBUG`, `INFO`, `WARNING`).
*   `VLLM_MXEXT_JSONL_LOGGING`: (Default: `0`) Set to `1` for JSONL formatted logs.

### Model Configuration (MX Hub)
The "Model Experience" (MX) Hub (`hub/`) uses manifest files (e.g., `model_manifest.yaml`) to manage model selection, profiles, and hardware-specific configurations. Key environment variables for the hub include:
*   `MIM_MODEL_NAME`: Specifies the model to be loaded (maps to entries in the manifest).
*   Refer to `hub/mx_download.py` and `nim_hub` documentation for more details on manifest structure and available configurations.

### LoRA Configuration
LoRA (Low-Rank Adaptation) support can be configured via environment variables:
*   `VLLM_ALLOW_RUNTIME_LORA_UPDATING`: (Default: `false`) Set to `true` to enable dynamic LoRA loading/unloading.
*   `VLLM_MAX_LORA_RANK`: (Default: `32`) Maximum LoRA rank supported.
*   `VLLM_MAX_LORAS`: (Default: `8`) Maximum number of LoRA adapters that can be loaded simultaneously.
*   `MIM_MAX_GPU_LORAS`: (Default: `8`) Maximum LoRA adapters on GPU.
*   `MIM_MAX_CPU_LORAS`: (Default: `16`) Maximum LoRA adapters on CPU.

## Usage / Quick Start

### Default Configuration Startup

#### Basic Server Startup
```bash
# Start with default configuration
python -m vllm_mxext.entrypoints.openai.api_server --host 0.0.0.0 --port 8000 --model /path/to/your/model

# Start with environment variables
export MIM_MODEL_NAME="your_model_name"
python -m vllm_mxext.entrypoints.openai.api_server
```

#### Start with LoRA Support
```bash
# Start server with LoRA support enabled
python scripts/start_with_lora.py --model /path/to/base/model --host 0.0.0.0 --port 8000

# Start with custom LoRA configuration
python scripts/start_with_lora.py --model /path/to/base/model --lora-config examples/lora_config_example.yaml
```

### Testing the Server

#### Basic Health Check
```bash
# Check server status
curl http://0.0.0.0:8000/health

# List available models
curl http://0.0.0.0:8000/v1/models
```

#### Basic Inference Test
```bash
# Test chat completions
curl -X POST http://0.0.0.0:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "your_model_name",
    "messages": [{"role": "user", "content": "Hello, how are you?"}],
    "max_tokens": 100,
    "temperature": 0.7
  }'

# Test completions
curl -X POST http://0.0.0.0:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "your_model_name",
    "prompt": "The capital of France is",
    "max_tokens": 50
  }'
```

## LoRA Adapter Usage

### LoRA Configuration File

Create a LoRA configuration file (`lora_config.yaml`):

```yaml
# LoRA Configuration
base_model:
  model_path: "/path/to/base/model"
  model_name: "llama-2-7b-chat"

lora_adapters:
  - name: "math_tuned"
    path: "/path/to/lora/math_adapter"
    format: "huggingface"
    description: "LoRA adapter for mathematical reasoning"
    
  - name: "code_tuned"
    path: "/path/to/lora/code_adapter"
    format: "huggingface"
    description: "LoRA adapter for code generation"

lora_settings:
  max_lora_rank: 64
  max_loras: 8
  enable_dynamic_loading: true
  cache_size: 4

auto_load:
  - "math_tuned"
```

### Dynamic LoRA Management via API

#### Load LoRA Adapter
```bash
curl -X POST http://0.0.0.0:8000/v1/load_lora_adapter \
  -H "Content-Type: application/json" \
  -d '{
    "lora_name": "math_tuned",
    "lora_path": "/path/to/math_lora_adapter"
  }'
```

#### List Loaded LoRA Adapters
```bash
curl http://0.0.0.0:8000/v1/lora_adapters
```

#### Unload LoRA Adapter
```bash
curl -X POST http://0.0.0.0:8000/v1/unload_lora_adapter \
  -H "Content-Type: application/json" \
  -d '{
    "lora_name": "math_tuned"
  }'
```

### LoRA Inference

#### Using LoRA with Chat Completions
```bash
curl -X POST http://0.0.0.0:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "base_model:math_tuned",
    "messages": [
      {"role": "user", "content": "Solve this equation: 2x + 5 = 13"}
    ],
    "max_tokens": 150,
    "temperature": 0.7
  }'
```

#### Using LoRA with Completions
```bash
curl -X POST http://0.0.0.0:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "base_model:code_tuned",
    "prompt": "def fibonacci(n):",
    "max_tokens": 200
  }'
```

### CLI Tool for LoRA Management

#### Install and Use LoRA Manager CLI
```bash
# Load a LoRA adapter
python -m vllm_mxext.tools.lora_manager load math_tuned /path/to/math_lora_adapter

# List loaded LoRA adapters
python -m vllm_mxext.tools.lora_manager list

# Test LoRA inference
python -m vllm_mxext.tools.lora_manager test math_tuned "What is the derivative of x^2?"

# Unload a LoRA adapter
python -m vllm_mxext.tools.lora_manager unload math_tuned

# Use custom server URL
python -m vllm_mxext.tools.lora_manager --server-url http://your-server:8000 list
```

### Custom LoRA Configuration Startup

#### Method 1: Using Configuration File
```bash
# Create custom LoRA config
cat > my_lora_config.yaml << EOF
base_model:
  model_path: "/models/llama-2-7b"
  model_name: "llama-2-7b"

lora_adapters:
  - name: "custom_adapter"
    path: "/lora_adapters/my_custom_adapter"
    format: "huggingface"

lora_settings:
  max_lora_rank: 128
  max_loras: 16
  enable_dynamic_loading: true
EOF

# Start server with custom config
python scripts/start_with_lora.py --model /models/llama-2-7b --lora-config my_lora_config.yaml
```

#### Method 2: Using Environment Variables
```bash
# Set LoRA environment variables
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=true
export VLLM_MAX_LORA_RANK=128
export VLLM_MAX_LORAS=16
export MIM_MAX_GPU_LORAS=16

# Start server
python -m vllm_mxext.entrypoints.openai.api_server --model /models/llama-2-7b
```

### Programmatic Usage (Offline Inference)
The `TRTLLM` class in `entrypoints/nvllm.py` can be used for direct programmatic inference with TensorRT-LLM models:
```python
# Example (conceptual - refer to nvllm.py for actual usage)
# from entrypoints.nvllm import TRTLLM
# engine_args = Ellipsis # Configure engine arguments
# engine = TRTLLM(**engine_args)
# results = engine.generate(prompts=["My prompt"])
# print(results)
```

### Python SDK Usage Example

```python
#!/usr/bin/env python3
import requests
import json

class VllmMxextClient:
    def __init__(self, base_url="http://0.0.0.0:8000"):
        self.base_url = base_url.rstrip('/')
    
    def load_lora(self, lora_name: str, lora_path: str):
        """Load a LoRA adapter"""
        response = requests.post(
            f"{self.base_url}/v1/load_lora_adapter",
            json={"lora_name": lora_name, "lora_path": lora_path}
        )
        return response.json()
    
    def chat_with_lora(self, lora_name: str, messages: list):
        """Chat using LoRA adapter"""
        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": f"base_model:{lora_name}",
                "messages": messages,
                "max_tokens": 150
            }
        )
        return response.json()

# Usage
client = VllmMxextClient()
client.load_lora("math_tuned", "/path/to/math_adapter")
result = client.chat_with_lora("math_tuned", [
    {"role": "user", "content": "Solve: 2x + 5 = 13"}
])
print(json.dumps(result, indent=2))
```

## Architecture

### High-Level Overview
`vllm_mxext` is an inference server optimized for NVIDIA TensorRT-LLM. It integrates a "Model Experience" (MX) Hub for managing model configurations, supports LoRA for model adaptation, and exposes LLMs through an OpenAI-compatible API and a direct Python interface.

### Core Components

*   **Model Experience Hub (`hub/`)**:
    *   Manages model selection (via `mx_download.py`, `nim_hub`), profiling, and configuration injection based on manifests and hardware inspection (`hardware_inspect.py`).
*   **TensorRT-LLM Engine (`engine/`)**:
    *   The core inference unit, centered around `AsyncTRTLLMEngine`. It handles asynchronous requests, model execution via `TrtllmModelRunner`, and integrates LoRA adapters. An `AsyncLLMEngineFactory` selects the appropriate backend (vLLM or TRT-LLM).
*   **LoRA Subsystem (`lora/`)**:
    *   Responsible for loading, converting (`lora/source.py`), and applying LoRA adapters during inference.
*   **Entrypoints (`entrypoints/`)**:
    *   `openai/api_server.py`: Provides the OpenAI-compatible REST API.
    *   `nvllm.py`: Offers the `TRTLLM` class for programmatic (offline) inference.
*   **Supporting Utilities**:
    *   `trtllm/`: TensorRT-LLM specific utilities.
    *   `utils/`: General project utilities (caching, MinIO access).
    *   `logging/`, `logger.py`, `envs.py`: Logging and environment-based configuration.

## Workflow: API Request Lifecycle (Text-Based Flowchart)

1.  **Request Ingestion**: API request hits `entrypoints.openai.api_server`.
2.  **Engine Interaction**: Server forwards request to an `EngineClient`.
3.  **Engine Initialization (if needed)**:
    *   **MX Hub**: Model manifests are parsed (`hub.mx_download`), hardware checked (`hub.hardware_inspect`), and configuration injected into engine arguments.
    *   **Engine Selection**: `AsyncLLMEngineFactory` chooses TRT-LLM or vLLM engine.
    *   **Core Engine Init**: `AsyncTRTLLMEngine` initializes `TrtllmModelRunner` and `TrtllmLoraSource` (if LoRA is used). Model loads on GPUs.
4.  **Request Processing**: `AsyncTRTLLMEngine` queues and schedules the request. `TrtllmModelRunner` executes inference using TensorRT-LLM, applying LoRA if specified.
5.  **Response Generation**: Results are streamed back to the API server and formatted.
6.  **Response Dispatch**: HTTP response sent to the user.
7.  **Logging/Metrics**: Activities are logged and metrics updated throughout.

## API Endpoints

### Standard OpenAI-Compatible Endpoints
- `GET /v1/models` - List available models
- `POST /v1/chat/completions` - Chat completions
- `POST /v1/completions` - Text completions
- `GET /health` - Health check

### LoRA Management Endpoints
- `POST /v1/load_lora_adapter` - Load a LoRA adapter
- `POST /v1/unload_lora_adapter` - Unload a LoRA adapter
- `GET /v1/lora_adapters` - List loaded LoRA adapters

## Advantages

*   **High Performance:** Optimized for NVIDIA GPUs via TensorRT-LLM.
*   **Async Architecture:** Efficient handling of concurrent requests.
*   **OpenAI Compatibility:** Easy integration with existing ecosystems.
*   **Flexible LoRA Support:** Serve adapted models efficiently.
*   **Streamlined Model Management:** MX Hub simplifies deployment with manifests.
*   **Hardware-Aware:** Tailors configuration to detected hardware.
*   **Monitoring Capabilities:** Built-in Prometheus metrics.

## Disadvantages & Considerations

*   **NVIDIA GPU Dependency:** Primarily targets NVIDIA hardware.
*   **System Complexity:** Multi-component architecture can be complex to grasp initially.
*   **Configuration Curve:** Requires understanding of manifests and environment variables.
*   **Build/Installation:** Dependencies like TensorRT-LLM can have specific system requirements.
*   **Documentation:** (This README aims to improve it!) Detailed internal documentation might be needed for contributors.
*   **Resource Intensive:** LLMs demand significant GPU memory and compute.

## Performance Testing

### Performance Testing Tool

The project includes a comprehensive performance testing tool to measure LLM inference performance metrics:

- **Tokens Throughput** (tokens/s) - Overall and per-request throughput
- **TTFT** (Time to First Token) - Time until first token is generated
- **TPOT** (Time Per Output Token) - Average time per output token
- **End-to-End Latency** - Complete request processing time

#### Key Features

🚀 **Core Capabilities**
- Comprehensive performance metrics collection
- Statistical analysis with percentiles (P95, P99)
- Concurrent request testing
- Streaming and non-streaming response support
- Automatic result logging with timestamps

📊 **Statistical Analysis**
- Mean, Median, P95, P99, Min, Max values
- Individual and aggregated metrics
- JSON-formatted structured output
- Time-series performance tracking

🔧 **Technical Features**
- Asynchronous concurrent request handling
- Configurable test parameters
- YAML configuration file support
- Flexible prompt management
- Automatic log directory creation

#### Basic Usage

```bash
# Test with default prompts
python -m vllm_mxext.tools.performance_tester --model your_model_name

# Test with custom prompts file
python -m vllm_mxext.tools.performance_tester --model your_model_name --prompts-file prompts.txt

# Test with concurrent requests
python -m vllm_mxext.tools.performance_tester --model your_model_name --concurrent-requests 4

# Test with custom parameters
python -m vllm_mxext.tools.performance_tester \
  --model your_model_name \
  --max-tokens 200 \
  --temperature 0.8 \
  --concurrent-requests 8 \
  --stream
```

#### Configuration-Based Testing

Create a YAML configuration file:

```yaml
# performance_config.yaml
server:
  url: "http://0.0.0.0:8000"
  model: "llama-2-7b-chat"

test_params:
  max_tokens: 150
  temperature: 0.7
  concurrent_requests: 4
  stream: true

prompts:
  - "What is artificial intelligence?"
  - "Explain machine learning concepts."
  - "Write a story about robots."

output:
  log_dir: "/opt/mim/log"
  save_individual_metrics: true
  console_output: true
```

Run with configuration:
```bash
python -m vllm_mxext.tools.performance_config_runner performance_config.yaml
```

#### Performance Metrics Output

The tool provides detailed performance metrics with comprehensive statistics:

```
================================================================================
PERFORMANCE TEST RESULTS
================================================================================

Test Summary:
  Total Requests: 10
  Total Tokens: 1,250
  Prompt Tokens: 150
  Completion Tokens: 1,100
  Test Duration: 15.45s
  Overall Throughput: 80.91 tokens/s

Time to First Token (TTFT):
  Mean: 0.245s
  Median: 0.230s
  P95: 0.380s
  P99: 0.420s
  Min: 0.180s
  Max: 0.450s

Time Per Output Token (TPOT):
  Mean: 0.012s
  Median: 0.011s
  P95: 0.018s
  P99: 0.020s
  Min: 0.008s
  Max: 0.025s

End-to-End Latency:
  Mean: 1.545s
  Median: 1.520s
  P95: 2.100s
  P99: 2.250s
  Min: 1.200s
  Max: 2.300s

Throughput (tokens/s):
  Mean: 80.91
  Median: 82.15
  P95: 95.20
  P99: 98.50
  Min: 65.30
  Max: 105.80
================================================================================

Results saved to: /opt/mim/log/mim_profile_20241201_143022.log
```

#### Command Line Options

```bash
python -m vllm_mxext.tools.performance_tester --help

Required Arguments:
  --model MODEL            Model name to test

Optional Arguments:
  --server-url URL         Server URL (default: http://0.0.0.0:8000)
  --prompts-file FILE      File containing prompts (one per line)
  --num-prompts N          Number of default prompts to use (default: 10)
  --max-tokens N           Maximum tokens per response (default: 100)
  --temperature FLOAT      Temperature for generation (default: 0.7)
  --concurrent-requests N  Number of concurrent requests (default: 1)
  --stream                 Use streaming responses (default: True)
  --no-stream             Disable streaming responses
  --log-dir DIR           Directory to save log files (default: /opt/mim/log)
  --no-save               Don't save results to file
  --save-individual       Save individual request metrics
```

#### Advanced Usage Examples

**Load Testing with High Concurrency:**
```bash
python -m vllm_mxext.tools.performance_tester \
  --model llama-2-7b-chat \
  --concurrent-requests 16 \
  --num-prompts 50 \
  --max-tokens 200 \
  --save-individual
```

**Custom Prompts Testing:**
```bash
# Create prompts file
echo "Explain quantum computing" > test_prompts.txt
echo "Write a Python function" >> test_prompts.txt
echo "Describe machine learning" >> test_prompts.txt

# Run test
python -m vllm_mxext.tools.performance_tester \
  --model llama-2-7b-chat \
  --prompts-file test_prompts.txt \
  --concurrent-requests 4
```

**Non-streaming Performance Test:**
```bash
python -m vllm_mxext.tools.performance_tester \
  --model llama-2-7b-chat \
  --no-stream \
  --concurrent-requests 8 \
  --max-tokens 150
```

#### Log Files and Output

**Automatic Logging:**
- Performance results are automatically saved to `/opt/mim/log/`
- Filename format: `mim_profile_YYYYMMDD_HHMMSS.log`
- Contains both aggregated and individual request metrics

**Log File Contents:**
- Test configuration and parameters
- Aggregated performance statistics (JSON format)
- Individual request metrics (optional)
- Timestamps and test metadata

**Example Log Structure:**
```
Performance Test Results - 2024-12-01T14:30:22

AGGREGATED METRICS:
{
  "total_requests": 10,
  "total_tokens": 1250,
  "ttft_mean": 0.245,
  "tpot_mean": 0.012,
  "e2e_mean": 1.545,
  "overall_throughput": 80.91,
  ...
}

INDIVIDUAL REQUEST METRICS:
{
  "request_id": "req_0",
  "ttft": 0.230,
  "tpot": 0.011,
  "e2e_latency": 1.520,
  "throughput": 82.15,
  ...
}
```

#### Integration with Monitoring

The performance testing tool integrates with the project's monitoring system:

- Metrics are compatible with Prometheus format
- Results can be used for dashboard visualization
- Historical performance tracking capabilities
- Integration with existing logging infrastructure

#### Convenience Scripts

Use the convenience script for quick testing:

```bash
# Direct script execution
python scripts/performance_test.py --model your_model_name --concurrent-requests 4
```

This comprehensive performance testing tool enables thorough evaluation of LLM inference performance under various conditions and loads.

## Troubleshooting

### Common Issues

#### LoRA Loading Issues
```bash
# Check if LoRA path exists and has correct format
ls -la /path/to/lora/adapter/
# Should contain: adapter_config.json, adapter_model.bin (for HuggingFace format)

# Check server logs for LoRA loading errors
tail -f /var/log/vllm_mxext.log
```

#### Memory Issues
```bash
# Monitor GPU memory usage
nvidia-smi

# Reduce max_loras if running out of memory
export VLLM_MAX_LORAS=4
```

#### API Connection Issues
```bash
# Test server connectivity
curl http://0.0.0.0:8000/health

# Check if LoRA endpoints are available
curl http://0.0.0.0:8000/v1/lora_adapters
```

## Contributing
Contributions are welcome! Please open an issue or submit a pull request. (User may want to expand this section).

## License
The licensing for this project is mixed. Parts derived from vLLM are under the Apache 2.0 license. Other parts may be under NVIDIA Proprietary licenses. Please check the source files for specific license headers. (User should verify and update this section accurately).


