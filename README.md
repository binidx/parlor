# Parlor

本地实时多模态 AI 助手。在你的设备上进行自然的语音和视觉对话，完全离线运行，无需云端依赖。

Parlor 支持两套模型栈：
- **Gemma 4 E2B** (视觉理解) + **Kokoro** (语音合成) — 默认方案
- **Qwen3 系列** — ASR (语音识别) + LLM (语言模型) + TTS (语音合成) — 中文优化方案

https://github.com/user-attachments/assets/cb0ffb2e-f84f-48e7-872c-c5f7b5c6d51f

> **研究预览。** 这是一个早期实验项目，可能存在一些问题。

## 为什么做这个？

我在[自托管一个完全免费的语音 AI](https://www.fikrikarim.com/bule-ai-initial-release/)，帮助人们学习英语口语，有数百名月活用户。我一直在思考如何在保持免费的同时让它可持续发展。

答案很明显：完全在设备上运行，消除服务器成本。半年前我还需要一张 RTX 5090 才能实时运行语音模型。

Google 刚刚发布了一个足够小的模型，我可以在 M3 Pro 上实时运行，而且还有视觉能力！当然你不能用它做 AI 编程，但对于学习新语言的人来说这是一个游戏规则改变者。想象几年后人们可以在手机上本地运行它，用摄像头对准物体并讨论它们。这个模型是多语言的，人们随时可以切换回母语。这本质上就是 OpenAI 几年前演示的功能。

## 工作原理

```
浏览器（麦克风 + 摄像头）
    │
    │  WebSocket（音频 PCM + JPEG 帧）
    ▼
FastAPI 服务器
    ├── LLM 运行时
    │   ├── LiteRT-LM（GPU）→ Gemma 4 E2B — 理解语音 + 视觉
    │   └── MLX VLM（Apple GPU）→ Qwen3 / Gemma 4 MLX 模型 — 流式输出
    ├── ASR：Qwen3-ASR（可选，语音转文字）
    └── TTS：Kokoro / Qwen3-TTS / macOS say — 语音合成
    │
    │  WebSocket（流式音频块）
    ▼
浏览器（播放 + 转录文本）
```

### 核心特性

- **浏览器端 VAD** ([Silero VAD](https://github.com/ricky0123/vad)) — 免提操作，无需按键说话
- **打断功能 (Barge-in)** — 说话即可中断 AI 回复
- **流式句子级 TTS** — LLM 还在生成时，第一句的语音已开始播放
- **`StreamingSentenceDetector`** — 增量处理 token，剥离 `<think>` 块，定位 `RESPONSE:` 标记，按句边界切分
- **`Runtime` 协议** — 统一的运行时接口，`LiteRTRuntime` 和 `MlxRuntime` 均满足此协议
- **`BatchTTSHandler` / `StreamingTTSHandler`** — 封装 TTS 生命周期，支持批量和流式两种模式
- **连续帧采样** — 8 帧环形缓冲区，1.5 秒间隔，关键帧选择（灰度像素差分）
- **响应长度控制** — 短 / 中 / 长
- **对话历史滑动窗口** — `MAX_HISTORY_TURNS` 控制上限，防止延迟退化

## 系统要求

- Python 3.12+
- macOS (Apple Silicon) 或 Linux (支持的 GPU)
- ~3 GB 可用内存

## 快速开始

```bash
git clone https://github.com/fikrikarim/parlor.git
cd parlor

# 安装 uv（如果没有的话）
curl -LsSf https://astral.sh/uv/install.sh | sh

cd src
uv sync
uv run server.py
```

打开 http://localhost:9091，授予摄像头和麦克风权限，开始对话。

首次运行时模型会自动下载（Gemma 4 E2B 约 2.6 GB，加上 TTS 模型）。

在 macOS 上，你也可以将 `MODEL_PATH` 指向本地 MLX Gemma 4 模型目录（`config.json` + `*.safetensors`），Parlor 会使用 `mlx-vlm` 代替 LiteRT-LM。

## Qwen 模型栈

使用中文优化的 Qwen 全家桶（ASR + LLM + TTS）：

```bash
./run-parlor-qwen.sh
```

默认模型路径：

- LLM：`~/models/llm/Qwen3.6-A3B-opus-mxfp4-h`
- ASR：`~/models/asr/Qwen3-ASR-0.6B` 或自动从 HuggingFace 下载 `Qwen/Qwen3-ASR-0.6B`
- TTS：`~/models/tts/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit`

关键环境变量：`ASR_MODEL_PATH`、`ASR_LANGUAGE`（默认 `chinese`）、`TTS_MODEL_PATH`、`QWEN_TTS_SPEAKER`（默认 `serena`）、`QWEN_TTS_INSTRUCT`

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
| `ASR_SYSTEM_PROMPT` | 无 | 自定义 ASR 提示词 |

### TTS（语音合成）

| 变量 | 默认值 | 说明 |
|---|---|---|
| `DISABLE_TTS` | 无 | 设为 `1` 禁用 TTS |
| `TTS_BACKEND` | 自动检测 | 强制指定后端：`say` / `system` / `macos` |
| `TTS_MODEL_PATH` | 自动检测 `~/models/tts/` | Qwen3-TTS 模型路径 |
| `QWEN_TTS_LANGUAGE` | `chinese` | Qwen TTS 语言 |
| `QWEN_TTS_SPEAKER` | `serena` | Qwen TTS 说话人 |
| `QWEN_TTS_INSTRUCT` | 无 | Qwen TTS 指令提示 |
| `QWEN_TTS_VOICE_DESIGN` | 中文女声描述 | Qwen TTS 声音设计（voice_design 类型） |
| `QWEN_TTS_SPEED` | `1.0` | Qwen TTS 语速 |
| `KOKORO_MODEL_PATH` | `~/models/tts/Kokoro-82M-bf16` | Kokoro MLX 模型路径 |
| `KOKORO_ONNX_MODEL_PATH` | 自动下载 | Kokoro ONNX 模型路径 |
| `KOKORO_ONNX_VOICES_PATH` | 自动下载 | Kokoro ONNX 声音文件路径 |
| `SYSTEM_TTS_VOICE` | `Tingting` | macOS 系统 TTS 声音 |
| `SYSTEM_TTS_RATE` | `180` | macOS 系统 TTS 语速 |

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
├── tts.py                 # 平台感知 TTS（MLX on Mac, ONNX on Linux, Qwen3-TTS）
├── asr.py                 # 语音识别（Qwen3-ASR via mlx-audio）
├── index.html             # 前端 UI（VAD、摄像头、音频播放、波形可视化）
├── pyproject.toml         # 依赖
└── benchmarks/
    ├── bench.py           # 端到端 WebSocket 基准测试
    └── benchmark_tts.py   # TTS 后端对比测试
```

## API 端点

| 端点 | 说明 |
|---|---|
| `GET /` | 前端页面 |
| `GET /config` | 模型标签、后端信息、系统提示词默认值 |
| `WebSocket /ws` | 双向音频/文本/图像流 |

## 致谢

- [Gemma 4](https://ai.google.dev/gemma) by Google DeepMind
- [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM) by Google AI Edge
- [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M) TTS by Hexgrad
- [Silero VAD](https://github.com/snakers4/silero-vad) 浏览器端语音活动检测
- [Qwen3](https://qwenlm.github.io/) ASR + TTS by Alibaba

## 许可证

[Apache 2.0](LICENSE)
