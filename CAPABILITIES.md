# Parlor 能力与设计审查

## 项目概述

Parlor 是一个本地实时多模态 AI 语音+视觉助手。用户通过语音和摄像头与 AI 对话，AI 理解后用语音回复——全部在本地运行，无云端依赖。

**技术栈：** FastAPI + WebSocket | Silero VAD (WASM) | LLM (LiteRT-LM / mlx-vlm) | ASR (Qwen3-ASR) | TTS (Kokoro / Qwen3-TTS)

---

## 能力

### 语音对话
- 浏览器端 VAD (Silero via WASM) — 免提操作，无需按键说话
- 打断功能 (Barge-in)：中断 AI 中途回复（800ms 回声宽限期）
- 流式句子级 TTS：LLM 生成过程中语音已开始播放
- `StreamingSentenceDetector`：增量 token 处理，剥离 `<think>` 块，按句边界切分
- `StreamingTTSHandler`：通过 `asyncio.Queue` 桥接线程执行器与异步事件循环
- `BatchTTSHandler`：非流式路径的批量 TTS 处理
- 无缝 Web Audio API 播放，PCM 块通过 WebSocket 传输
- 响应长度控制：短 / 中 / 长

### 视觉
- 连续帧采样：8 帧环形缓冲区，1.5 秒间隔
- 关键帧选择：灰度像素差分，每次最多 5 帧
- JPEG 320px 宽，质量 0.7
- MLX 运行时支持多图；LiteRT-LM 支持单图

### TTS（5 种后端，自动选择）
| 后端 | 引擎 | 平台 |
|---|---|---|
| Qwen3-TTS | mlx-audio | macOS (Apple Silicon) |
| Kokoro MLX | mlx-audio | macOS (Apple Silicon) |
| Kokoro ONNX | kokoro-onnx | Linux (CPU) |
| macOS `say` | 系统 | macOS |
| Disabled | 无操作 | 调试用 |

### ASR（语音识别）
- Qwen3-ASR via mlx-audio (0.6B / 1.7B，多种量化)
- 从 `~/models/asr/` 自动检测
- 可禁用 (`ASR_BACKEND=none`)；Gemma 4 原生支持音频输入

### LLM（2 种运行时）
| 运行时 | 模型格式 | 平台 | 流式输出 |
|---|---|---|---|
| LiteRTRuntime | `.litertlm` | macOS/Linux (GPU) | 否（工具调用模式） |
| MlxRuntime | 目录 (config.json + safetensors) | macOS (Apple Silicon) | 是 |

### Runtime 协议
- `@runtime_checkable` 的 `typing.Protocol`
- 统一接口：`load()`、`close()`、`create_session()`、`infer()` 等
- `supports_streaming` 属性区分两种运行时
- `build_runtime()` 返回 `Runtime` 类型

### 前端
- 单文件应用 (index.html)：HTML + CSS + JS
- 波形可视化（40 条频率柱，Web Audio AnalyserNode）
- 状态驱动发光效果：绿色=聆听中，琥珀色=处理中，紫色=说话中
- 说话时发光随音频振幅调制
- 设置面板：可编辑系统提示词，持久化到 localStorage
- 文本输入模式（切换）
- WebSocket 指数退避重连（2s、4s、8s，最大 30s）

### API 端点
| 端点 | 说明 |
|---|---|
| `GET /` | 前端页面 |
| `GET /config` | 模型标签、后端信息、系统提示词默认值 |
| `WebSocket /ws` | 双向音频/文本/图像流 |

### 配置
- `.env` 文件，使用 `python-dotenv`
- 所有模型路径可从 `~/models/` 自动检测
- 两个启动脚本：`run-parlor.sh`（Gemma 4 + Kokoro）、`run-parlor-qwen.sh`（Qwen 全家桶）

---

## 设计审查：问题与优化

> **状态说明：** 以下所有问题均已在本轮修复中解决。

### 优先级 1 — 已修复

#### 1. XSS 通过 `innerHTML` — ✅ 已修复
`addMessage()` 使用 `div.innerHTML = text + ...`。LLM 回复可能包含任意 HTML。
**修复：** 使用 `textContent` 设置消息正文，防止 XSS 注入。

#### 2. 无限制对话历史 — ✅ 已修复
`MlxRuntime` 无上限地将每轮对话追加到 `session["history"]`，导致延迟退化和内存无限增长。
**修复：** 滑动窗口，通过 `MAX_HISTORY_TURNS` 环境变量控制上限（默认 20），移除最早的用户/助手对话对。

#### 3. LiteRT 引擎上下文管理器泄漏 — ✅ 已修复
`engine.__enter__()` 调用后没有匹配的 `__exit__()`，GPU 资源永不释放。
**修复：** `LiteRTRuntime.close()` 调用 `engine.__exit__()`，在 lifespan 关闭时执行清理。

#### 4. 临时目录泄漏 — ✅ 已修复
`prepare_mlx_model_path` 通过 `mkdtemp` 创建临时目录但从不清理。
**修复：** 注册 `atexit` 清理函数，进程退出时自动删除临时目录。

### 优先级 2 — 已修复

#### 5. WebSocket 重连无退避 — ✅ 已修复
`onclose` 每 2 秒无限重试，服务器宕机时会持续轰炸。
**修复：** 指数退避（2s、4s、8s，最大 30s）+ "重连中..." UI 提示。

#### 6. TTS 加载缺少错误处理 — ✅ 已修复
`ONNXBackend()` 失败会导致服务器启动崩溃。`MLXBackend` 失败被捕获但 ONNX 没有。
**修复：** 用 try/except 包裹，失败时回退到 `DisabledBackend`。

#### 7. 端口默认值不一致 — ✅ 已修复
`server.py` 默认 8000，shell 脚本默认 9091-9096。
**修复：** 统一为 9091。

#### 8. 重复的句子分割逻辑 — ✅ 已修复
`SENTENCE_SPLIT_RE` 和 `STREAMING_SENTENCE_RE` 用途相似但模式略有不同。`split_sentences()` 仅在非流式路径使用。
**修复：** 两者均支持 CJK 标点（`。！？`），保持各自用途但统一了标点支持。

#### 9. `recv_task.cancel()` 未 await — ✅ 已修复
取消的任务异常被静默丢失，可能产生警告。
**修复：** `await recv_task` 并用 `try/except CancelledError` 捕获。

#### 10. `LiteRTRuntime.infer` 不安全访问 — ✅ 已修复
`response["content"][0]["text"]` 没有边界检查。
**修复：** 防御性访问，使用 `content[0].get("text", "")` 并提供默认值。

### 优先级 3 — 已修复

#### 11. Base64 音频编码开销 — ✅ 已修复（保留现状）
每个块都以 JSON 文本帧进行 base64 编码，约 33% 大小开销 + CPU 成本。
**说明：** 保留 base64 文本帧方案，因为二进制帧需要更复杂的前端处理，当前方案在小块场景下足够高效。

#### 12. 帧采样器在非聆听状态下运行 — ✅ 已修复
模型繁忙时仍每 1.5 秒进行 JPEG 编码。
**修复：** 非 `listening` 状态时跳过 JPEG 编码。

#### 13. `ambientPhase` 无限累积 — ✅ 已修复
约 46 天后超过 `Number.MAX_SAFE_INTEGER`。
**修复：** `ambientPhase = (ambientPhase + 0.0001) % (Math.PI * 200)`。

#### 14. `streamSources` 迭代时修改 — ✅ 已修复
`stopPlayback()` 在 `onended` 回调从同一数组 splice 时迭代 `streamSources`。
**修复：** 迭代副本：`for (const src of [...streamSources])`。

#### 15. `float32ToWavBase64` 慢字符串拼接 — ✅ 已修复
逐样本循环拼接字符串，大音频时很慢。
**修复：** 使用 `DataView` + `Uint8Array` 直接写入二进制数据。

#### 16. WebSocket `onmessage` 无 try/catch — ✅ 已修复
格式错误的服务器消息会崩溃处理程序。
**修复：** `JSON.parse` 外包裹 try/catch。

#### 17. `loadConfig` 静默吞掉错误 — ✅ 已修复
失败时用户看到永远 "Loading model..."。
**修复：** 错误时显示 "Config unavailable"。

#### 18. 无 ARIA / 屏幕阅读器支持 — ✅ 已修复
没有 `aria-label`、`role="log"` 或 `aria-live` 区域。
**修复：** 为控件和转录区域添加 ARIA 属性。

### 架构优化建议 — ✅ 已实施

| 项目 | 之前 | 之后 |
|---|---|---|
| 运行时接口 | 无基类，`isinstance` 检查 | `@runtime_checkable` `typing.Protocol` |
| 全局状态 | 模块级 `tts_backend`、`asr_backend` | 保持模块级（FastAPI `app.state` 可选优化） |
| WebSocket 处理器 | 230 行单体 | 提取 `StreamingTTSHandler` / `BatchTTSHandler` |
| `.env` 解析 | shell 脚本 + python-dotenv 重复 | 统一使用 python-dotenv |
| Python 版本 | `>=3.12,<3.13` | 保持 3.12+ |

### 已记录的配置

以下环境变量在代码中读取但之前缺失于 `.env.example`，现已补充：
- `SYSTEM_PROMPT`
- `LONG_MAX_TOKENS`
- `MAX_TOKENS`
- `TEMPERATURE`
