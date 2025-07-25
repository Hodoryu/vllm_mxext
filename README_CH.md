# Vllm Nvext (`vllm_mxext`) - 中文文档

## 简介

Vllm Nvext（打包为 `vllm_mxext`）是一个先进的基于Python的推理服务器，专为大型语言模型（LLM）的高性能服务而设计。它利用NVIDIA TensorRT-LLM在NVIDIA GPU上进行优化执行，并扩展了vLLM项目的概念。主要特性包括OpenAI兼容的API、LoRA（低秩适应）支持，以及用于简化模型管理和部署的清单驱动"模型体验"（MX）Hub。

## 特性

*   **高性能推理：** 利用NVIDIA TensorRT-LLM进行优化的LLM执行。
*   **异步请求处理：** 使用FastAPI和Uvicorn构建，支持并发处理。
*   **OpenAI兼容API：** 允许与现有OpenAI API客户端和工作流程轻松集成。
*   **LoRA支持：** 支持动态加载和服务LoRA适配的模型。
*   **清单驱动的模型管理：** MX Hub使用清单文件简化模型配置和部署。
*   **硬件感知配置：** 根据检测到的NVIDIA GPU硬件调整设置。
*   **指标和监控：** 包含Prometheus兼容的指标，用于观察服务器性能。
*   **可配置日志：** 灵活的日志系统，支持JSONL格式和环境变量控制。

## 依赖项

### Python
*   Python >= 3.8

### 外部Python包
项目依赖以下外部Python包。建议将这些添加到 `pyproject.toml` 的 `project.dependencies` 部分以便自动安装。
*   `vllm`
*   `numpy`
*   `packaging`
*   `prometheus_client`
*   `tensorrt_llm`
*   `transformers`
*   `pydantic`
*   `uvloop`（推荐用于FastAPI性能）
*   `fastapi`
*   `starlette`
*   `typing_extensions`
*   `torch`
*   `pyyaml`
*   `nim_hub`
*   `nvidia-lora-conversions`
*   `minio`
*   `progress`（或类似的进度条库如 `progressbar2` 或 `tqdm`）
*   `requests`
*   `tqdm`
*   `modelscope`

## 安装

1.  **前置条件：**
    *   确保安装了Python（>=3.8）和 `pip`。
    *   安装上述列出的外部Python包。（例如：`pip install fastapi uvicorn pydantic "tensorrt_llm>=0.9.0" ...`）
    *   NVIDIA GPU驱动程序，与TensorRT-LLM兼容的CUDA工具包。

2.  **克隆仓库：**
    ```bash
    git clone <your_repository_url>
    cd vllm_mxext # 或您的仓库名称
    ```

3.  **安装包：**
    *   可编辑模式（推荐用于开发）：
        ```bash
        pip install -e .
        ```
    *   标准安装：
        ```bash
        pip install .
        ```

## 配置

### 环境变量
项目的行为，特别是日志记录，可以通过 `envs.py` 中定义的环境变量进行配置：
*   `VLLM_MXEXT_CONFIGURE_LOGGING`：（默认：`1`）设置为 `0` 以禁用自动日志配置。
*   `VLLM_MXEXT_LOGGING_CONFIG_PATH`：自定义日志配置文件的路径。
*   `VLLM_MXEXT_LOG_LEVEL`：（默认：`INFO`）日志级别（例如：`DEBUG`、`INFO`、`WARNING`）。
*   `VLLM_MXEXT_JSONL_LOGGING`：（默认：`0`）设置为 `1` 以使用JSONL格式的日志。

### 模型配置（MX Hub）
"模型体验"（MX）Hub（`hub/`）使用清单文件（例如：`model_manifest.yaml`）来管理模型选择、配置文件和硬件特定配置。Hub的关键环境变量包括：
*   `MIM_MODEL_NAME`：指定要加载的模型（映射到清单中的条目）。
*   有关清单结构和可用配置的更多详细信息，请参考 `hub/mx_download.py` 和 `nim_hub` 文档。

### LoRA配置
LoRA（低秩适应）支持可以通过环境变量进行配置：
*   `VLLM_ALLOW_RUNTIME_LORA_UPDATING`：（默认：`false`）设置为 `true` 以启用动态LoRA加载/卸载。
*   `VLLM_MAX_LORA_RANK`：（默认：`32`）支持的最大LoRA秩。
*   `VLLM_MAX_LORAS`：（默认：`8`）可以同时加载的最大LoRA适配器数量。
*   `MIM_MAX_GPU_LORAS`：（默认：`8`）GPU上的最大LoRA适配器数量。
*   `MIM_MAX_CPU_LORAS`：（默认：`16`）CPU上的最大LoRA适配器数量。

## 使用方法 / 快速开始

### 默认配置启动

#### 基本服务器启动
```bash
# 使用默认配置启动
python -m vllm_mxext.entrypoints.openai.api_server --host 0.0.0.0 --port 8000 --model /path/to/your/model

# 使用环境变量启动
export MIM_MODEL_NAME="your_model_name"
python -m vllm_mxext.entrypoints.openai.api_server
```

#### 启动LoRA支持
```bash
# 启动支持LoRA的服务器
python scripts/start_with_lora.py --model /path/to/base/model --host 0.0.0.0 --port 8000

# 使用自定义LoRA配置启动
python scripts/start_with_lora.py --model /path/to/base/model --lora-config examples/lora_config_example.yaml
```

### 测试服务器

#### 基本健康检查
```bash
# 检查服务器状态
curl http://0.0.0.0:8000/health

# 列出可用模型
curl http://0.0.0.0:8000/v1/models
```

#### 基本推理测试
```bash
# 测试聊天完成
curl -X POST http://0.0.0.0:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "your_model_name",
    "messages": [{"role": "user", "content": "你好，你好吗？"}],
    "max_tokens": 100,
    "temperature": 0.7
  }'

# 测试文本完成
curl -X POST http://0.0.0.0:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "your_model_name",
    "prompt": "法国的首都是",
    "max_tokens": 50
  }'
```

## LoRA适配器使用

### LoRA配置文件

创建LoRA配置文件（`lora_config.yaml`）：

```yaml
# LoRA配置
base_model:
  model_path: "/path/to/base/model"
  model_name: "llama-2-7b-chat"

lora_adapters:
  - name: "math_tuned"
    path: "/path/to/lora/math_adapter"
    format: "huggingface"
    description: "用于数学推理的LoRA适配器"
    
  - name: "code_tuned"
    path: "/path/to/lora/code_adapter"
    format: "huggingface"
    description: "用于代码生成的LoRA适配器"

lora_settings:
  max_lora_rank: 64
  max_loras: 8
  enable_dynamic_loading: true
  cache_size: 4

auto_load:
  - "math_tuned"
```

### 通过API动态管理LoRA

#### 加载LoRA适配器
```bash
curl -X POST http://0.0.0.0:8000/v1/load_lora_adapter \
  -H "Content-Type: application/json" \
  -d '{
    "lora_name": "math_tuned",
    "lora_path": "/path/to/math_lora_adapter"
  }'
```

#### 列出已加载的LoRA适配器
```bash
curl http://0.0.0.0:8000/v1/lora_adapters
```

#### 卸载LoRA适配器
```bash
curl -X POST http://0.0.0.0:8000/v1/unload_lora_adapter \
  -H "Content-Type: application/json" \
  -d '{
    "lora_name": "math_tuned"
  }'
```

### LoRA推理

#### 使用LoRA进行聊天完成
```bash
curl -X POST http://0.0.0.0:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "base_model:math_tuned",
    "messages": [
      {"role": "user", "content": "解这个方程：2x + 5 = 13"}
    ],
    "max_tokens": 150,
    "temperature": 0.7
  }'
```

#### 使用LoRA进行文本完成
```bash
curl -X POST http://0.0.0.0:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "base_model:code_tuned",
    "prompt": "def fibonacci(n):",
    "max_tokens": 200
  }'
```

### LoRA管理CLI工具

#### 安装和使用LoRA管理器CLI
```bash
# 加载LoRA适配器
python -m vllm_mxext.tools.lora_manager load math_tuned /path/to/math_lora_adapter

# 列出已加载的LoRA适配器
python -m vllm_mxext.tools.lora_manager list

# 测试LoRA推理
python -m vllm_mxext.tools.lora_manager test math_tuned "x^2的导数是什么？"

# 卸载LoRA适配器
python -m vllm_mxext.tools.lora_manager unload math_tuned

# 使用自定义服务器URL
python -m vllm_mxext.tools.lora_manager --server-url http://your-server:8000 list
```

### 自定义LoRA配置启动

#### 方法1：使用配置文件
```bash
# 创建自定义LoRA配置
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

# 使用自定义配置启动服务器
python scripts/start_with_lora.py --model /models/llama-2-7b --lora-config my_lora_config.yaml
```

#### 方法2：使用环境变量
```bash
# 设置LoRA环境变量
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=true
export VLLM_MAX_LORA_RANK=128
export VLLM_MAX_LORAS=16
export MIM_MAX_GPU_LORAS=16

# 启动服务器
python -m vllm_mxext.entrypoints.openai.api_server --model /models/llama-2-7b
```

### 编程使用（离线推理）
`entrypoints/nvllm.py` 中的 `TRTLLM` 类可用于使用TensorRT-LLM模型进行直接编程推理：
```python
# 示例（概念性 - 请参考nvllm.py了解实际使用方法）
# from entrypoints.nvllm import TRTLLM
# engine_args = Ellipsis # 配置引擎参数
# engine = TRTLLM(**engine_args)
# results = engine.generate(prompts=["我的提示"])
# print(results)
```

### Python SDK使用示例

```python
#!/usr/bin/env python3
import requests
import json

class VllmMxextClient:
    def __init__(self, base_url="http://0.0.0.0:8000"):
        self.base_url = base_url.rstrip('/')
    
    def load_lora(self, lora_name: str, lora_path: str):
        """加载LoRA适配器"""
        response = requests.post(
            f"{self.base_url}/v1/load_lora_adapter",
            json={"lora_name": lora_name, "lora_path": lora_path}
        )
        return response.json()
    
    def chat_with_lora(self, lora_name: str, messages: list):
        """使用LoRA适配器进行聊天"""
        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": f"base_model:{lora_name}",
                "messages": messages,
                "max_tokens": 150
            }
        )
        return response.json()

# 使用方法
client = VllmMxextClient()
client.load_lora("math_tuned", "/path/to/math_adapter")
result = client.chat_with_lora("math_tuned", [
    {"role": "user", "content": "解方程：2x + 5 = 13"}
])
print(json.dumps(result, indent=2))
```

## 架构

### 高级概述
`vllm_mxext` 是一个针对NVIDIA TensorRT-LLM优化的推理服务器。它集成了用于管理模型配置的"模型体验"（MX）Hub，支持用于模型适应的LoRA，并通过OpenAI兼容的API和直接Python接口公开LLM。

### 核心组件

*   **模型体验Hub（`hub/`）**：
    *   管理模型选择（通过 `mx_download.py`、`nim_hub`）、配置文件，以及基于清单和硬件检查（`hardware_inspect.py`）的配置注入。
*   **TensorRT-LLM引擎（`engine/`）**：
    *   核心推理单元，以 `AsyncTRTLLMEngine` 为中心。它处理异步请求，通过 `TrtllmModelRunner` 执行模型，并集成LoRA适配器。`AsyncLLMEngineFactory` 选择适当的后端（vLLM或TRT-LLM）。
*   **LoRA子系统（`lora/`）**：
    *   负责在推理期间加载、转换（`lora/source.py`）和应用LoRA适配器。
*   **入口点（`entrypoints/`）**：
    *   `openai/api_server.py`：提供OpenAI兼容的REST API。
    *   `nvllm.py`：为编程（离线）推理提供 `TRTLLM` 类。
*   **支持工具**：
    *   `trtllm/`：TensorRT-LLM特定工具。
    *   `utils/`：通用项目工具（缓存、MinIO访问）。
    *   `logging/`、`logger.py`、`envs.py`：日志记录和基于环境的配置。

## 工作流程：API请求生命周期（基于文本的流程图）

1.  **请求接收**：API请求到达 `entrypoints.openai.api_server`。
2.  **引擎交互**：服务器将请求转发给 `EngineClient`。
3.  **引擎初始化（如果需要）**：
    *   **MX Hub**：解析模型清单（`hub.mx_download`），检查硬件（`hub.hardware_inspect`），并将配置注入引擎参数。
    *   **引擎选择**：`AsyncLLMEngineFactory` 选择TRT-LLM或vLLM引擎。
    *   **核心引擎初始化**：`AsyncTRTLLMEngine` 初始化 `TrtllmModelRunner` 和 `TrtllmLoraSource`（如果使用LoRA）。模型加载到GPU上。
4.  **请求处理**：`AsyncTRTLLMEngine` 排队和调度请求。`TrtllmModelRunner` 使用TensorRT-LLM执行推理，如果指定则应用LoRA。
5.  **响应生成**：结果流式传输回API服务器并格式化。
6.  **响应分发**：HTTP响应发送给用户。
7.  **日志记录/指标**：在整个过程中记录活动并更新指标。

## API端点

### 标准OpenAI兼容端点
- `GET /v1/models` - 列出可用模型
- `POST /v1/chat/completions` - 聊天完成
- `POST /v1/completions` - 文本完成
- `GET /health` - 健康检查

### LoRA管理端点
- `POST /v1/load_lora_adapter` - 加载LoRA适配器
- `POST /v1/unload_lora_adapter` - 卸载LoRA适配器
- `GET /v1/lora_adapters` - 列出已加载的LoRA适配器

## 优势

*   **高性能：** 通过TensorRT-LLM针对NVIDIA GPU进行优化。
*   **异步架构：** 高效处理并发请求。
*   **OpenAI兼容性：** 与现有生态系统轻松集成。
*   **灵活的LoRA支持：** 高效服务适配模型。
*   **简化的模型管理：** MX Hub使用清单简化部署。
*   **硬件感知：** 根据检测到的硬件定制配置。
*   **监控能力：** 内置Prometheus指标。

## 缺点和注意事项

*   **NVIDIA GPU依赖：** 主要针对NVIDIA硬件。
*   **系统复杂性：** 多组件架构最初可能难以理解。
*   **配置学习曲线：** 需要理解清单和环境变量。
*   **构建/安装：** TensorRT-LLM等依赖项可能有特定的系统要求。
*   **文档：** （本README旨在改进它！）贡献者可能需要详细的内部文档。
*   **资源密集：** LLM需要大量的GPU内存和计算资源。

## 故障排除

### 常见问题

#### LoRA加载问题
```bash
# 检查LoRA路径是否存在且格式正确
ls -la /path/to/lora/adapter/
# 应包含：adapter_config.json, adapter_model.bin（对于HuggingFace格式）

# 检查服务器日志中的LoRA加载错误
tail -f /var/log/vllm_mxext.log
```

#### 内存问题
```bash
# 监控GPU内存使用情况
nvidia-smi

# 如果内存不足，减少max_loras
export VLLM_MAX_LORAS=4
```

#### API连接问题
```bash
# 测试服务器连接
curl http://0.0.0.0:8000/health

# 检查LoRA端点是否可用
curl http://0.0.0.0:8000/v1/lora_adapters
```

## 性能测试

### 性能测试工具

该项目包含一个全面的性能测试工具，用于测量LLM推理性能指标：

- **Tokens吞吐量** (tokens/s) - 总体和单请求吞吐量
- **TTFT** (Time to First Token) - 首个token生成时间
- **TPOT** (Time Per Output Token) - 每个输出token的平均时间
- **End-to-End Latency** - 完整请求处理时间

#### 核心特性

🚀 **核心功能**
- 全面的性能指标收集
- 百分位数统计分析（P95、P99）
- 并发请求测试
- 流式和非流式响应支持
- 带时间戳的自动结果记录

📊 **统计分析**
- 均值、中位数、P95、P99、最小值、最大值
- 单个和聚合指标
- JSON格式的结构化输出
- 时间序列性能跟踪

🔧 **技术特性**
- 异步并发请求处理
- 可配置的测试参数
- YAML配置文件支持
- 灵活的提示词管理
- 自动日志目录创建

#### 基本使用方法

```bash
# 使用默认提示词测试
python -m vllm_mxext.tools.performance_tester --model your_model_name

# 使用自定义提示词文件测试
python -m vllm_mxext.tools.performance_tester --model your_model_name --prompts-file prompts.txt

# 并发请求测试
python -m vllm_mxext.tools.performance_tester --model your_model_name --concurrent-requests 4

# 自定义参数测试
python -m vllm_mxext.tools.performance_tester \
  --model your_model_name \
  --max-tokens 200 \
  --temperature 0.8 \
  --concurrent-requests 8 \
  --stream
```

#### 基于配置文件的测试

创建YAML配置文件：

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
  - "什么是人工智能？"
  - "解释机器学习概念。"
  - "写一个关于机器人的故事。"

output:
  log_dir: "/opt/mim/log"
  save_individual_metrics: true
  console_output: true
```

使用配置文件运行：
```bash
python -m vllm_mxext.tools.performance_config_runner performance_config.yaml
```

#### 性能指标输出

工具提供详细的性能指标和全面的统计信息：

```
================================================================================
性能测试结果
================================================================================

测试摘要:
  总请求数: 10
  总Token数: 1,250
  提示Token数: 150
  完成Token数: 1,100
  测试持续时间: 15.45s
  总体吞吐量: 80.91 tokens/s

首个Token时间 (TTFT):
  均值: 0.245s
  中位数: 0.230s
  P95: 0.380s
  P99: 0.420s
  最小值: 0.180s
  最大值: 0.450s

每个输出Token时间 (TPOT):
  均值: 0.012s
  中位数: 0.011s
  P95: 0.018s
  P99: 0.020s
  最小值: 0.008s
  最大值: 0.025s

端到端延迟:
  均值: 1.545s
  中位数: 1.520s
  P95: 2.100s
  P99: 2.250s
  最小值: 1.200s
  最大值: 2.300s

吞吐量 (tokens/s):
  均值: 80.91
  中位数: 82.15
  P95: 95.20
  P99: 98.50
  最小值: 65.30
  最大值: 105.80
================================================================================

结果已保存到: /opt/mim/log/mim_profile_20241201_143022.log
```

#### 命令行选项

```bash
python -m vllm_mxext.tools.performance_tester --help

必需参数:
  --model MODEL            要测试的模型名称

可选参数:
  --server-url URL         服务器URL (默认: http://0.0.0.0:8000)
  --prompts-file FILE      包含提示词的文件 (每行一个)
  --num-prompts N          使用的默认提示词数量 (默认: 10)
  --max-tokens N           每个响应的最大token数 (默认: 100)
  --temperature FLOAT      生成温度 (默认: 0.7)
  --concurrent-requests N  并发请求数 (默认: 1)
  --stream                 使用流式响应 (默认: True)
  --no-stream             禁用流式响应
  --log-dir DIR           保存日志文件的目录 (默认: /opt/mim/log)
  --no-save               不保存结果到文件
  --save-individual       保存单个请求指标
```

#### 高级使用示例

**高并发负载测试：**
```bash
python -m vllm_mxext.tools.performance_tester \
  --model llama-2-7b-chat \
  --concurrent-requests 16 \
  --num-prompts 50 \
  --max-tokens 200 \
  --save-individual
```

**自定义提示词测试：**
```bash
# 创建提示词文件
echo "解释量子计算" > test_prompts.txt
echo "写一个Python函数" >> test_prompts.txt
echo "描述机器学习" >> test_prompts.txt

# 运行测试
python -m vllm_mxext.tools.performance_tester \
  --model llama-2-7b-chat \
  --prompts-file test_prompts.txt \
  --concurrent-requests 4
```

**非流式性能测试：**
```bash
python -m vllm_mxext.tools.performance_tester \
  --model llama-2-7b-chat \
  --no-stream \
  --concurrent-requests 8 \
  --max-tokens 150
```

#### 日志文件和输出

**自动日志记录：**
- 性能结果自动保存到 `/opt/mim/log/`
- 文件名格式：`mim_profile_YYYYMMDD_HHMMSS.log`
- 包含聚合和单个请求指标

**日志文件内容：**
- 测试配置和参数
- 聚合性能统计（JSON格式）
- 单个请求指标（可选）
- 时间戳和测试元数据

**日志结构示例：**
```
性能测试结果 - 2024-12-01T14:30:22

聚合指标:
{
  "total_requests": 10,
  "total_tokens": 1250,
  "ttft_mean": 0.245,
  "tpot_mean": 0.012,
  "e2e_mean": 1.545,
  "overall_throughput": 80.91,
  ...
}

单个请求指标:
{
  "request_id": "req_0",
  "ttft": 0.230,
  "tpot": 0.011,
  "e2e_latency": 1.520,
  "throughput": 82.15,
  ...
}
```

#### 与监控系统集成

性能测试工具与项目的监控系统集成：

- 指标与Prometheus格式兼容
- 结果可用于仪表板可视化
- 历史性能跟踪功能
- 与现有日志基础设施集成

#### 便捷脚本

使用便捷脚本进行快速测试：

```bash
# 直接脚本执行
python scripts/performance_test.py --model your_model_name --concurrent-requests 4
```

#### 性能测试最佳实践

**测试准备：**
1. 确保服务器正常运行并已加载模型
2. 准备多样化的测试提示词
3. 根据硬件配置调整并发数

**测试策略：**
- 从低并发开始，逐步增加负载
- 测试不同长度的提示词和响应
- 比较流式和非流式性能
- 记录不同温度设置下的性能

**结果分析：**
- 关注P95和P99延迟指标
- 监控吞吐量随并发数的变化
- 分析TTFT和TPOT的分布
- 识别性能瓶颈和优化机会

这个全面的性能测试工具能够在各种条件和负载下彻底评估LLM推理性能。

## 贡献
欢迎贡献！请开启issue或提交pull request。（用户可能希望扩展此部分）。

## 许可证
此项目的许可证是混合的。从vLLM派生的部分采用Apache 2.0许可证。其他部分可能采用NVIDIA专有许可证。请检查源文件以获取特定的许可证标头。（用户应验证并准确更新此部分）。
