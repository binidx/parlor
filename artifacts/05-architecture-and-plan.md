# 架构与实施计划

## 目标硬件
- **Apple M3 Pro，18 GB 统一内存**
- macOS

## 技术栈选择

| 组件 | 选择 | 原因 |
|-----------|--------|-----|
| **LLM 推理** | LiteRT-LM C++ (Metal GPU) | 最快、最小的模型（3.65 GB），原生支持音频+视觉 |
| **模型** | Gemma 4 E4B-it (.litertlm) | 原生多模态：文本 + 图像 + 音频 |
| **TTS** | Kokoro-82M | <300ms，82M 参数，质量出色 |
| **Web 前端** | HTML/JS + WebRTC 或 WebSocket | 通过 getUserMedia 捕获摄像头和麦克风 |
| **HTTP 服务器** | C++ + cpp-httplib（或类似库） | 封装 LiteRT-LM 引擎，提供 SSE/WebSocket 服务 |

## 架构图

```
┌─────────────────────────────────────────────────────┐
│                    浏览器                             │
│                                                      │
│  getUserMedia() → 摄像头 (1fps JPEG) + 麦克风 (PCM)  │
│       │                                    │         │
│       └──── WebSocket ─────────────────────┘         │
│                    │                                  │
└────────────────────┼──────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│              C++ 服务器 (Metal GPU)                   │
│                                                      │
│  ┌─────────────────────────────────────────────┐    │
│  │        LiteRT-LM 引擎                        │    │
│  │        Gemma 4 E4B (3.65 GB, Metal GPU)     │    │
│  │                                              │    │
│  │  输入：                                      │    │
│  │   • 音频 blob (WAV) — 原生支持，无需 STT     │    │
│  │   • 图像 blob (JPEG) — 摄像头帧              │    │
│  │   • 文本 — 对话上下文                        │    │
│  │                                              │    │
│  │  输出：                                      │    │
│  │   • 流式文本 token                           │    │
│  └──────────────┬──────────────────────────────┘    │
│                  │ 流式文本                            │
│                  ▼                                    │
│  ┌─────────────────────────────────────────────┐    │
│  │        Kokoro TTS (82M, ONNX)               │    │
│  │        句子级流式处理                         │    │
│  │        输出：PCM 音频分块                     │    │
│  └──────────────┬──────────────────────────────┘    │
│                  │                                    │
└──────────────────┼────────────────────────────────────┘
                   │ 音频（WebSocket 二进制帧）
                   ▼
              浏览器扬声器（Web Audio API）
```

## 内存预算（M3 Pro 18 GB）

| 组件 | 预估内存 |
|-----------|-----------------|
| Gemma 4 E4B (.litertlm, Metal) | ~3.65 GB 模型 + ~1.5 GB GPU 工作内存 | 
| Kokoro TTS (ONNX) | ~200 MB |
| C++ 服务器 + 缓冲区 | ~200 MB |
| 浏览器 + 操作系统 | ~4 GB |
| **总计** | **~9.5 GB** |
| **剩余空间** | **~8.5 GB** |

## 数据流

### 用户说话 + 展示摄像头 → AI 语音回复

1. **浏览器**以 1 fps 捕获摄像头帧（JPEG，~50-100 KB）
2. **浏览器**捕获麦克风音频，检测静音（JS 或服务端 VAD）
3. 用户轮次结束时：通过 WebSocket 发送音频分块（WAV，最长 ~30s）+ 最新帧
4. **C++ 服务器**接收两者，构造多模态消息：
   ```json
   {
     "role": "user",
     "content": [
       {"type": "audio", "blob": "<base64_wav>"},
       {"type": "image", "blob": "<base64_jpeg>"},
       {"type": "text", "text": "The user is speaking to you while showing their camera. Respond conversationally."}
     ]
   }
   ```
5. **LiteRT-LM** 使用 Gemma 4 E4B（Metal GPU）处理，流式输出 token
6. **Kokoro TTS** 在 LLM 继续生成的同时，将第一个完整句子转换为音频
7. **服务器**通过 WebSocket 发送音频分块
8. **浏览器**通过 Web Audio API 播放音频

## 实测基准（M3 Pro 18GB, Metal GPU）

使用 `litert_lm_main` 测试，Gemma 4 E4B（3.4 GB），Metal 原生后端：

| 指标 | 数值 |
|--------|-------|
| **TTFT** | **0.38s** |
| **Prefill** | **61.3 tok/s** |
| **Decode** | **26.5 tok/s** |
| 初始化时间 | ~5.9s（一次性） |
| 140 token 输出 | 总计 5.3s |

**重要提示**：必须创建符号链接 `libLiteRtMetalAccelerator.dylib` → `libLiteRtGpuAccelerator.dylib` 以启用原生 Metal。否则会回退到 WebGPU（19 tok/s，1.18s TTFT）。

## 延迟目标（基于实测数据修订）

| 阶段 | 目标 | 备注 |
|-------|--------|-------|
| 音频采集 + VAD | 语音结束后 ~200ms | Silero VAD 或浏览器端 |
| WebSocket 往返 | <10ms | localhost |
| LiteRT-LM TTFT (GPU) | **~380ms** | M3 Pro Metal 实测 |
| LiteRT-LM decode | **~26.5 tok/s** | M3 Pro Metal 实测 |
| 第一个句子完成 | ~1s | ~26 tokens，26.5 tok/s |
| Kokoro TTS 首段音频 | <300ms | 从第一个完整句子开始 |
| **总感知延迟** | **~1.5-2s** | 从用户停止说话到 AI 语音开始 |

## 实施阶段

### 阶段一：验证 LiteRT-LM 构建与 GPU 推理
1. 克隆 LiteRT-LM 到 ~/workspace
2. 在 macOS 上使用 Bazel 构建
3. 下载 Gemma 4 E4B 模型
4. 验证 GPU 后端的文本生成
5. 测试多模态（图像 + 文本）推理
6. 测试音频输入

### 阶段二：C++ HTTP/WebSocket 服务器
1. 添加 cpp-httplib 或类似轻量级 HTTP 库
2. 实现 WebSocket 端点用于流式传输
3. 接受多模态输入（音频 blob + 图像 blob）
4. 以 SSE 或 WebSocket 消息流式返回 LLM token
5. 集成 Kokoro TTS（通过 ONNX C++ 运行时或子进程）

### 阶段三：Web 前端
1. HTML/JS 页面，使用 getUserMedia 捕获摄像头和麦克风
2. 与 C++ 服务器的 WebSocket 连接
3. 客户端 VAD（Silero WASM 或简单能量检测）
4. 以 1 fps 采样帧
5. Web Audio API 播放 TTS 音频
6. 简单的聊天 UI 显示对话

### 阶段四：完善与优化
1. 调整视觉 token 预算（为速度从 70 开始）
2. 优化音频分块大小以降低延迟
3. 添加对话历史管理
4. 处理中断（用户开始说话时取消生成）
5. 添加系统提示词调优以实现自然对话

## 待解决问题 / 风险

1. **macOS 上的 Bazel 构建** — 可能存在依赖问题，需要验证
2. **LiteRT-LM Gemma 4 E4B 模型可用性** — 确认 E4B 的 .litertlm 文件存在（不仅仅是 E2B）
3. **原生输入的音频质量** — Gemma 4 的音频编码器质量 vs 专用 Whisper STT
4. **Kokoro TTS C++ 集成** — 可能需要 Python 子进程或 ONNX C++ 运行时
5. **M3 Pro GPU 性能** — 没有公开基准，需要实测
6. **C++ 中的 WebSocket 库** — cpp-httplib 不原生支持 WebSocket；可能需要 uWebSockets、Boost.Beast 或类似库

## 备选方案：Python + C++ 混合架构

如果纯 C++ 对服务器层来说过于复杂：

```
浏览器 ←→ Python FastAPI (WebSocket, Kokoro TTS)
                ↕
           C++ LiteRT-LM 引擎 (Metal GPU)
           通过：subprocess CLI | 共享内存 | Unix socket
```

这样可以用 Python 处理"胶水"层（WebSocket、TTS、音频编码），同时将热路径（LLM 推理）保留在 C++ 中使用 Metal GPU。`litert_lm_main` CLI 已支持流式输出。

## 需要创建的文件

```
gemma-4/
├── 01-gemma4-model-overview.md      # 模型文档
├── 02-litert-lm-guide.md            # LiteRT-LM 参考
├── 03-open-source-references.md     # 相关项目
├── 04-tts-options.md                # TTS 对比
├── 05-architecture-and-plan.md      # 本文件
├── 06-unsloth-and-alternatives.md   # Unsloth/Ollama/MLX 信息
└── src/                             # 演示源代码（待定）
    ├── server/                      # C++ 服务器封装 LiteRT-LM
    ├── frontend/                    # HTML/JS Web UI
    └── tts/                         # Kokoro TTS 集成
```
