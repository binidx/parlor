# 开源参考项目 — 实时多模态 AI 演示

## 最相关的项目

### 完整管线（摄像头 + 麦克风 + LLM + TTS）

#### 1. Vocalis — 带视觉的语音对话 AI
- **GitHub**：https://github.com/Lex-au/Vocalis
- **Star 数**：约 294
- **技术栈**：React/TypeScript 前端、FastAPI/Python 后端、WebSocket
- **模型**：LLaMA 3 8B（LM Studio）、SmolVLM-256M（视觉）、Faster-Whisper（STT）、Orpheus TTS
- **功能**：对话中打断、AI 主动追问、图像分析、低延迟音频流
- **相关性**：少有的同时包含全部四个元素的项目。React + FastAPI + WebSocket 架构的良好参考。

#### 2. MiniCPM-o — 端到端多模态全双工流式交互
- **GitHub**：https://github.com/OpenBMB/MiniCPM-o
- **Star 数**：约 24,300
- **技术栈**：PyTorch、llama.cpp、Ollama、vLLM、SGLang、WebRTC
- **模型**：MiniCPM-o 4.5（9B）、SigLip2 视觉、Whisper-medium 音频、CosyVoice2 TTS
- **功能**：全双工实时流式传输、视频/音频输入与语音输出同步进行、主动场景交互
- **相关性**：端到端多模态流式传输的最佳架构参考。

#### 3. Open-LLM-VTuber — 语音 + 视觉 AI 伴侣
- **GitHub**：https://github.com/Open-LLM-VTuber/Open-LLM-VTuber
- **Star 数**：约 6,400
- **技术栈**：Python、JavaScript、Live2D、Docker
- **模型**：Ollama/OpenAI/Gemini（LLM）、Whisper/FunASR（STT）、10+ 种 TTS 引擎
- **功能**：完全离线运行、摄像头/屏幕感知、Live2D 虚拟形象
- **相关性**：展示了如何通过 Ollama 使用本地模型将摄像头感知与语音结合。

### Gemma 专用语音演示（仅音频，无摄像头）

#### 4. macos-local-voice-agents — 最接近的参考项目
- **GitHub**：https://github.com/kwindla/macos-local-voice-agents
- **技术栈**：**Pipecat** + Gemma 3n E4B + MLX Whisper + Kokoro TTS，运行在 **macOS** 上
- **功能**：Silero VAD、Smart Turn v2、SmallWebRTCTransport、子进程隔离的 TTS
- **相关性**：**直接可用**。相同的平台（macOS）、相同的模型家族（Gemma）、相同的框架（Pipecat）。只需添加摄像头支持并升级到 Gemma 4。

#### 5. Local-Voice-AI-Agent — FastRTC + Gemma
- **GitHub**：https://github.com/Utkarsh4412/Local-Voice-AI-Agent
- **Star 数**：约 1
- **技术栈**：FastRTC + Gemma 3（Ollama）+ Moonshine STT + Kokoro TTS
- **相关性**：代码库最小，证明了 FastRTC + Gemma 方案可行。

#### 6. mOrpheus — Gemma + Orpheus TTS
- **GitHub**：https://github.com/Nighthawk42/mOrpheus
- **Star 数**：约 85
- **技术栈**：Gemma 3 12B（LM Studio）+ Whisper STT + Orpheus 3B TTS
- **相关性**：简洁的 Gemma + TTS 集成参考。

### 框架级参考

#### 7. Pipecat — 生产级多模态 AI 框架
- **GitHub**：https://github.com/pipecat-ai/pipecat
- **Star 数**：约 11,000
- **功能**：10+ 种 LLM 提供商、10+ 种 STT 引擎、20+ 种 TTS 引擎、WebRTC 传输
- **内置服务**：`OLLamaLLMService`、`KokoroTTSService`、`PiperTTSService`、`WhisperSTTServiceMLX`
- **关键示例**：
  - `examples/function-calling/function-calling-moondream-video.py` — 摄像头 + 音频 + 视觉 LLM + TTS
  - `examples/function-calling/function-calling-google-video.py` — 使用 Gemini 的相同示例
  - `examples/function-calling/function-calling-ollama.py` — Ollama 集成

#### 8. FastRTC — Gradio 实时通信
- **GitHub**：https://github.com/gradio-app/fastrtc
- **Star 数**：约 4,600
- **关键示例**：`demo/gemini_audio_video/app.py` — 摄像头 + 麦克风 + LLM 的标准模式
- **模式**：继承 `AsyncAudioVideoStreamHandler`，实现 `receive()`、`emit()`、`video_receive()`、`video_emit()`

#### 9. LiveKit Agents
- **GitHub**：https://github.com/livekit/agents
- **Star 数**：约 9,900
- **功能**：生产级 WebRTC 语音 AI 代理、STT/LLM/TTS 管线

### 仅视觉参考

#### 10. SmolVLM 实时摄像头
- **GitHub**：https://github.com/ngxson/smolvlm-realtime-webcam
- **Star 数**：约 5,500
- **简介**：最小化的摄像头 → VLM 循环（单个 HTML 文件 + llama.cpp 服务器）
- **相关性**：持续摄像头画面输入 VLM 推理的最简参考。

#### 11. NVIDIA Live VLM WebUI
- **GitHub**：https://github.com/NVIDIA-AI-IOT/live-vlm-webui
- **Star 数**：约 300
- **简介**：通过摄像头进行实时 VLM 交互的通用 Web UI

### 仅语音参考

#### 12. RealtimeVoiceChat
- **GitHub**：https://github.com/KoljaB/RealtimeVoiceChat
- **Star 数**：约 3,600
- **技术栈**：FastAPI + WebSocket + Ollama + Kokoro/Coqui TTS
- **相关性**：简洁的语音交互循环参考。

#### 13. HuggingFace Speech-to-Speech
- **GitHub**：https://github.com/huggingface/speech-to-speech
- **Star 数**：约 4,600
- **技术栈**：Silero VAD → Whisper → 任意 HF 模型 → MeloTTS/Kokoro
- **功能**：支持 Apple Silicon 的 MLX

## FastRTC Gemini Audio Video — 完整源代码

以下是我们将改编用于演示的精确模式：

```python
from fastrtc import AsyncAudioVideoStreamHandler, Stream, wait_for_item

class GeminiHandler(AsyncAudioVideoStreamHandler):
    def __init__(self):
        super().__init__("mono", output_sample_rate=24000, input_sample_rate=16000)
        self.audio_queue = asyncio.Queue()
        self.video_queue = asyncio.Queue()

    async def start_up(self):
        # 初始化 LLM 连接，启动接收循环
        pass

    async def receive(self, frame: tuple[int, np.ndarray]):
        # 处理来自麦克风的音频输入
        _, array = frame
        await self.session.send(input=encode_audio(array.squeeze()))

    async def emit(self):
        # 返回要播放的音频（TTS 输出）
        array = await wait_for_item(self.audio_queue, 0.01)
        return (self.output_sample_rate, array) if array is not None else None

    async def video_receive(self, frame: np.ndarray):
        # 处理视频帧输入（限制为 1fps）
        if time.time() - self.last_frame_time > 1:
            self.last_frame_time = time.time()
            await self.session.send(input=encode_image(frame))

    async def video_emit(self):
        # 将摄像头画面镜像回 UI
        frame = await wait_for_item(self.video_queue, 0.01)
        return frame if frame is not None else np.zeros((100, 100, 3), dtype=np.uint8)

stream = Stream(handler=GeminiHandler(), modality="audio-video", mode="send-receive")
stream.ui.launch()
```

## Pipecat 摄像头 + 音频管线模式

来自 `function-calling-moondream-video.py`：

```python
transport = SmallWebRTCTransport(
    params=TransportParams(
        video_in_enabled=True,   # 启用摄像头
        audio_in_enabled=True,   # 启用麦克风
        audio_out_enabled=True,  # 启用扬声器
    )
)

# 管线：transport → STT → LLM → TTS → transport
pipeline = Pipeline([
    transport.input(),
    stt,                          # Whisper/Deepgram
    context_aggregator.user(),
    llm,                          # 通过 Ollama 使用 Gemma 4
    tts,                          # Kokoro
    transport.output(),
    context_aggregator.assistant(),
])

# 按需捕获摄像头画面
await maybe_capture_participant_camera(transport, client)
```
