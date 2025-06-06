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

## Usage / Quick Start

### Running the OpenAI-Compatible API Server
The primary way to use `vllm_mxext` is by running the OpenAI-compatible API server:

```bash
python -m entrypoints.openai.api_server --host 0.0.0.0 --port 8000
```
Or, if `MIM_MODEL_NAME` and other configurations are set via environment variables:
```bash
python -m entrypoints.openai.api_server
```

Key command-line arguments for `api_server.py` (can also be set/overridden by environment variables):
*   `--host`: Host to bind the server to.
*   `--port`: Port for the server.
*   `--model` or `MIM_MODEL_NAME` (env var): Name of the model to serve (corresponds to manifest entries).
*   `--tensor-parallel-size` or `TP_SIZE` (env var): Tensor parallel degree.
*   ... (refer to `entrypoints.openai.api_server.py` for more arguments)

Once started, the server exposes endpoints like `/v1/chat/completions`, `/v1/completions`, and `/v1/models`.

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

## Contributing
Contributions are welcome! Please open an issue or submit a pull request. (User may want to expand this section).

## License
The licensing for this project is mixed. Parts derived from vLLM are under the Apache 2.0 license. Other parts may be under NVIDIA Proprietary licenses. Please check the source files for specific license headers. (User should verify and update this section accurately).
