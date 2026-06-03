# TTS（文本转语音）选项

## 推荐方案：Kokoro-82M

- **参数量**：82M（极小）
- **速度**：任意文本长度 <300ms，GPU 上达到 96 倍实时速度
- **质量**：ELO 1,059（开源权重模型中排名最高）
- **语言**：EN、ES、FR、HI、IT、JA、PT、ZH
- **运行时**：ONNX
- **内存**：~200 MB
- **Pipecat 集成**：内置 `KokoroTTSService`

```python
from pipecat.services.kokoro import KokoroTTSService
tts = KokoroTTSService(settings=KokoroTTSService.Settings(voice="af_heart"))
```

## 备选方案

### Piper（~20M 参数）
- 最快的 TTFB（~30ms）。可在 Raspberry Pi 上运行。
- 质量低于 Kokoro。
- Pipecat：`PiperTTSService`

### Orpheus 3B
- 更高质量，支持情感语音和非语言线索。
- 体积大得多（3B 参数），速度更慢。
- 被 mOrpheus 项目与 Gemma 配合使用。

### Dia-1.6B
- 非语言线索（笑声、呼吸声）。支持流式输出。
- 比 Kokoro 更大，质量更高。

### edge-tts（免费，云端）
- 使用 Microsoft Edge 的 TTS API。无需 GPU。
- 质量良好，但需要网络连接。
- `pip install edge-tts`

### ElevenLabs（商业 API）
- 整体质量最佳。
- 存在网络延迟和费用。

## 我们的演示方案

**Kokoro-82M** 是明确的最佳选择：
- 在 M3 Pro 上本地运行，内存占用极小
- 延迟低于 300ms
- Pipecat 集成已就绪
- 质量对于演示来说非常出色

## 流式 TTS 架构

感知延迟的关键优化：**当 LLM 返回第一个完整句子时就立即开始 TTS**，而不是等到完整响应结束后。

```
LLM 输出："The sky is blue. | It reflects..."
                              ↑
                    在第一个句子处开始 TTS
                    同时 LLM 继续生成
```

支持此功能的库：
- **RealtimeTTS**（`pip install realtimetts[all]`）— 封装了 Kokoro、Piper、Edge 等，支持自动句子分块
- **Pipecat** — 在其流水线架构中原生支持此功能
