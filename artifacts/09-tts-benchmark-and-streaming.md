# TTS 基准测试与流式架构

## 基准测试结果（M3 Pro，2026-04-04）

使用 Kokoro-82M 对比 **kokoro-onnx**（ONNX Runtime，CPU）与 **mlx-audio**（MLX，Apple GPU）。
语音：`af_heart`，速度：`1.1x`，预热：2 次运行，测量：5 次运行。

### kokoro-onnx（ONNX Runtime，CPU）

| 输入 | 字符数 | 平均 | 最小 | 音频时长 | RTF |
|------|--------|------|------|----------|-----|
| 短 | 31 | 897 ms | 859 ms | 1.62s | 0.554x |
| 中 | 119 | 2,322 ms | 2,286 ms | 5.48s | 0.423x |
| 长 | 361 | 7,670 ms | 6,577 ms | 21.10s | 0.364x |

### mlx-audio（MLX，Apple GPU）

| 输入 | 字符数 | 平均 | 最小 | 音频时长 | RTF |
|------|--------|------|------|----------|-----|
| 短 | 31 | 173 ms | 165 ms | 2.10s | 0.083x |
| 中 | 119 | 443 ms | 436 ms | 6.10s | 0.073x |
| 长 | 361 | 1,289 ms | 1,163 ms | 21.40s | 0.060x |

### 提速：mlx-audio 相对 kokoro-onnx

| 输入 | kokoro-onnx | mlx-audio | 提速 |
|------|-------------|-----------|------|
| 短 | 897 ms | 173 ms | **5.18x** |
| 中 | 2,322 ms | 443 ms | **5.25x** |
| 长 | 7,670 ms | 1,289 ms | **5.95x** |

### 流式传输（mlx-audio）

注意：Kokoro 按句子生成音频，因此短/中输入只产生 1 个分块。
流式优势在模型将长文本拆分为多个句子时显现。

| 输入 | TTFC 平均 | TTFC 最小 | 总计平均 |
|------|-----------|-----------|----------|
| 短 | 151 ms | 143 ms | 157 ms |
| 中 | 377 ms | 315 ms | 389 ms |
| 长 | 1,300 ms | 1,265 ms | 1,323 ms |

## 关键发现

1. **mlx-audio 快 5-6 倍** — 在 M3 Pro 上实现 RTF 0.06-0.08x（生成音频速度是实时的 12-17 倍）。

2. **kokoro-onnx 硬件利用率不足** — 仅在 CPU 上运行，RTF 为 0.35-0.55x。macOS ARM64 上的标准 onnxruntime pip 包不提供 GPU/ANE 加速。

3. **Kokoro 在内部按句子边界拆分文本。** 模型的 `generate()` 每个句子产出一个结果。这意味着句子级别的流式传输是天然的 — 我们不需要自己拆分文本。

4. **mlx-audio 首次加载较慢（~7s）**，原因是管线初始化（phonemizer、spacy 模型）。之后推理速度很快。模型必须在服务器启动时加载一次。

## 架构变更

### 之前（顺序式，完整响应）

```
用户说话 → VAD 检测结束 → 发送音频+图像到服务器
→ LLM 生成完整响应（2-5s）
→ TTS 从完整文本生成完整音频（kokoro-onnx 需 0.9-7.7s）
→ 通过 WebSocket 发送完整 WAV
→ 客户端播放音频
```

**总首次音频时间：3-13s**

### 之后（流式，句子级）

```
用户说话 → VAD 检测结束 → 发送音频+图像到服务器
→ LLM 生成完整响应（2-5s，因工具调用模式无法流式）
→ 服务器将响应拆分为句子
→ 对每个句子：
    → TTS 生成音频分块（mlx-audio 约 150-400ms）
    → 立即通过 WebSocket 发送 PCM 分块
    → 客户端播放分块（下一个分块生成时已开始播放）
```

**总首次音频时间：2-5s（LLM）+ ~170ms（首句 TTS）**

现在 LLM 是瓶颈，而非 TTS。使用 mlx-audio，即使 4 句的响应也能在 ~600ms 内完成 TTS。

### 渐进式音频传输

不再发送一个大的 base64 编码 WAV：

1. 服务器发送 `{"type": "audio_start", "sample_rate": 24000}` 标记流式传输开始
2. 服务器为每个句子发送 `{"type": "audio_chunk", "audio": "<base64 PCM>"}`
3. 所有分块发送完毕后，服务器发送 `{"type": "audio_end", "tts_time": 0.45}`
4. 客户端使用 AudioWorklet 或缓冲播放，在分块到达时立即播放

### 依赖变更

- **移除**：`kokoro-onnx`（ONNX Runtime，CPU）
- **新增**：`mlx-audio`（MLX，Apple GPU）、`misaki[en]`（phonemizer）、`num2words`

mlx-audio 包引入 `mlx`、`mlx-metal`、`transformers`、`torch`（供 misaki/spacy 使用）。虚拟环境总大小增加，但运行时性能大幅提升。
