# Parlor Qwen

基于 [Parlor](https://github.com/fikrikarim/parlor) 的增强版本，在原有 Gemma 4 + Kokoro 方案基础上，引入了 **Qwen3 全家桶**（ASR + LLM + TTS），针对中文场景深度优化。

**新增能力：**
- **Qwen3-ASR** — 语音识别，支持中文/英文，自动检测模型
- **Qwen3.6-A3B** — 语言模型，MLX 运行时，流式输出
- **Qwen3-TTS** — 语音合成，支持说话人选择、声音设计、语速控制
- **流式句子级 TTS** — LLM 生成过程中语音已开始播放，大幅降低首句延迟
- **Runtime 协议** — 统一接口，LiteRT-LM 和 MLX 双运行时无缝切换

> 原项目：[fikrikarim/parlor](https://github.com/fikrikarim/parlor) — 本地实时多模态 AI 语音+视觉助手

https://github.com/user-attachments/assets/cb0ffb2e-f84f-48e7-872c-c5f7b5c6d51f

> **研究预览。** 这是一个早期实验项目，可能存在一些问题。

## 为什么做这个？

原版 Parlor 使用 Gemma 4 + Kokoro，英文体验很好，但中文支持有限。本 fork 的目标：

1. **中文 ASR** — Qwen3-ASR 替代 Gemma 4 原生音频理解，中文识别更准确
2. **中文 TTS** — Qwen3-TTS 替代 Kokoro，中文语音更自然
3. **流式输出** — MLX 运行时支持 token 级流式，配合句子检测实现 TTS 提前介入
4. **保持兼容** — Gemma 4 + Kokoro 方案仍然可用，通过环境变量切换

## 工作原理

```
浏览器（麦克风 + 摄像头）
    │
    │  WebSocket（音频 PCM + JPEG 帧）
    ▼
FastAPI 服务器
    ├── LLM 运行时
    │   ├── LiteRT-LM（GPU）→ Gemma 4 E2B — 理解语音 + 视觉
    │   └── MLX VLM（Apple GPU）→ Qwen3.6-A3B / Gemma 4 MLX — 流式输出
    ├── ASR：Qwen3-ASR（语音转文字）
    └── TTS：Qwen3-TTS / Kokoro / macOS say — 语音合成
    │
    │  WebSocket（流式音频块）
    ▼
浏览器（播放 + 转录文本）
```

### 核心特性

- **浏览器端 VAD** ([Silero VAD](https://github.com/ricky0123/vad)) — 免提操作，无需按键说话
- **打断功能 (Barge-in)** — 说话即可中断 AI 回复
- **流式句子级 TTS** — `StreamingSentenceDetector` 增量处理 token，按句切分，`StreamingTTSHandler` 异步生成语音
- **连续帧采样** — 8 帧环形缓冲区，1.5 秒间隔，关键帧选择
- **响应长度控制** — 短 / 中 / 长
- **对话历史滑动窗口** — `MAX_HISTORY_TURNS` 控制上限

## 系统要求

- Python 3.12+
- macOS (Apple Silicon) 或 Linux (支持的 GPU)
- ~3 GB 可用内存

## 快速开始

```bash
git clone https://github.com/YOUR_USERNAME/parlor-qwen.git
cd parlor-qwen

# 安装 uv（如果没有的话）
curl -LsSf https://astral.sh/uv/install.sh | sh

cd src
uv sync

# 方式一：Qwen 全家桶（推荐中文场景）
./run-parlor-qwen.sh

# 方式二：Gemma 4 + Kokoro（原版方案）
uv run server.py
```

打开 http://localhost:9091，授予摄像头和麦克风权限，开始对话。

## 两套模型栈

### Qwen 全家桶（中文优化）

```bash
./run-parlor-qwen.sh
```

| 组件 | 模型 | 路径 |
|---|---|---|
| ASR | Qwen3-ASR-0.6B | `~/models/asr/Qwen3-ASR-0.6B` |
| LLM | Qwen3.6-A3B-opus-mxfp4 | `~/models/llm/Qwen3.6-A3B-opus-mxfp4-h` |
| TTS | Qwen3-TTS-1.7B | `~/models/tts/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit` |

### Gemma 4 + Kokoro（原版方案）

```bash
uv run server.py
```

| 组件 | 模型 | 说明 |
|---|---|---|
| LLM | Gemma 4 E2B | 自动从 HuggingFace 下载 |
| TTS | Kokoro-82M | MLX (Mac) 或 ONNX (Linux) |

## 配置

所有配置通过 `.env` 文件或环境变量设置。详见 [`.env.example`](.env.example)。

### LLM

| 变量 | 默认值 | 说明 |
|---|---|---|
| `MODEL_PATH` | 自动下载 Gemma 4 E2B | 模型路径（MLX 目录或 `.litertlm` 文件） |
| `SYSTEM_PROMPT` | 内置默认 | 自定义系统提示词 |
| `MAX_TOKENS` | `192` | 最大生成 token 数 |
| `TEMPERATURE` | `0.2` | 生成温度 |
| `LONG_MAX_TOKENS` | `4096` | 响应长度为"长"时的最大 token 数 |
| `MAX_HISTORY_TURNS` | `20` | 对话历史保留轮数上限 |
| `PORT` | `9091` | 服务器端口 |

### ASR（语音识别）

| 变量 | 默认值 | 说明 |
|---|---|---|
| `ASR_BACKEND` | 自动检测 | 后端：`qwen` 或 `none` |
| `ASR_MODEL_PATH` | 自动检测 `~/models/asr/` | Qwen3-ASR 模型路径 |
| `ASR_LANGUAGE` | `chinese` | 识别语言 |
| `ASR_MAX_TOKENS` | `4096` | 最大 token 数 |

### TTS（语音合成）

| 变量 | 默认值 | 说明 |
|---|---|---|
| `TTS_MODEL_PATH` | 自动检测 `~/models/tts/` | Qwen3-TTS 模型路径 |
| `QWEN_TTS_SPEAKER` | `serena` | Qwen TTS 说话人 |
| `QWEN_TTS_INSTRUCT` | 无 | Qwen TTS 指令提示 |
| `QWEN_TTS_VOICE_DESIGN` | 中文女声描述 | 声音设计（voice_design 类型） |
| `QWEN_TTS_SPEED` | `1.0` | 语速 |
| `KOKORO_MODEL_PATH` | `~/models/tts/Kokoro-82M-bf16` | Kokoro MLX 模型路径 |
| `DISABLE_TTS` | 无 | 设为 `1` 禁用 TTS |

## TTS 后端（5 种，自动选择）

| 后端 | 引擎 | 平台 |
|---|---|---|
| Qwen3-TTS | mlx-audio | macOS (Apple Silicon) |
| Kokoro MLX | mlx-audio | macOS (Apple Silicon) |
| Kokoro ONNX | kokoro-onnx | Linux (CPU) |
| macOS `say` | 系统 | macOS |
| Disabled | 无操作 | 调试用 |

## 运行时（2 种）

| 运行时 | 模型格式 | 平台 | 流式输出 |
|---|---|---|---|
| LiteRTRuntime | `.litertlm` | macOS/Linux (GPU) | 否（工具调用模式） |
| MlxRuntime | 目录 (config.json + safetensors) | macOS (Apple Silicon) | 是 |

## 性能（Apple M3 Pro）

| 阶段 | 耗时 |
|---|---|
| 语音 + 视觉理解 | ~1.8-2.2s |
| 响应生成（~25 token） | ~0.3s |
| 语音合成（1-3 句） | ~0.3-0.7s |
| **端到端总耗时** | **~2.5-3.0s** |

解码速度：GPU 约 83 token/s（Apple M3 Pro）。

## 项目结构

```
src/
├── server.py              # FastAPI WebSocket 服务器 + LLM 推理 + Runtime 协议 + TTS Handler
├── tts.py                 # 平台感知 TTS（Qwen3-TTS / Kokoro MLX / Kokoro ONNX / macOS say）
├── asr.py                 # 语音识别（Qwen3-ASR via mlx-audio）
├── index.html             # 前端 UI（VAD、摄像头、音频播放、波形可视化）
├── pyproject.toml         # 依赖
└── benchmarks/
    ├── bench.py           # 端到端 WebSocket 基准测试
    └── benchmark_tts.py   # TTS 后端对比测试
```

## 与原版 Parlor 的区别

| 特性 | 原版 Parlor | Parlor Qwen |
|---|---|---|
| ASR | Gemma 4 原生音频 | Qwen3-ASR（可选） |
| TTS | Kokoro | Kokoro + Qwen3-TTS |
| LLM 流式输出 | 否（LiteRT-LM 工具调用） | 是（MLX 运行时） |
| 句子级流式 TTS | 否 | 是 |
| 中文优化 | 基础 | 深度优化（ASR/TTS/标点） |
| 对话历史管理 | 无限制 | 滑动窗口 |

## 致谢

- [Parlor](https://github.com/fikrikarim/parlor) by [fikrikarim](https://github.com/fikrikarim) — 原始项目架构与设计
- [Gemma 4](https://ai.google.dev/gemma) by Google DeepMind
- [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM) by Google AI Edge
- [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M) TTS by Hexgrad
- [Silero VAD](https://github.com/snakers4/silero-vad) 浏览器端语音活动检测
- [Qwen3](https://qwenlm.github.io/) ASR + TTS by Alibaba

## 许可证

[Apache 2.0](LICENSE)
