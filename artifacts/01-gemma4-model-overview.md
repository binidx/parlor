# Gemma 4 模型概述

## 模型家族

Gemma 4 由 Google DeepMind 于 2026 年 4 月 2 日发布，采用 **Apache 2.0** 许可证。

| 变体 | 总参数量 | 有效参数量 | 上下文长度 | 模态支持 | 显存（4-bit） | 显存（bf16） |
|---------|-------------|-----------------|---------|------------|-------------|-------------|
| **E2B** | 5.1B | 2.3B | 128K | 文本、图像、视频、**音频** | 4-5 GB | 10-15 GB |
| **E4B** | 8B | 4.5B | 128K | 文本、图像、视频、**音频** | 5.5-6 GB | ~16 GB |
| **26B-A4B (MoE)** | 26B | 3.8B 激活参数 | 256K | 文本、图像、视频 | 16-18 GB | ~52 GB |
| **31B (Dense)** | 31B | 31B | 256K | 文本、图像、视频 | 17-20 GB | ~62 GB |

**关键点**：仅 E2B 和 E4B 支持原生音频输入。"E" 代表 "effective"（有效）——逐层嵌入（Per-Layer Embeddings, PLE）使计算成本远低于总参数量。

## E4B 架构

### 文本解码器
- **层数**：42
- **隐藏层大小**：2560
- **中间层（FFN）大小**：10,240
- **每层输入隐藏大小**：256（PLE 特性）
- **注意力头数**：8，KV 头数：2（分组查询注意力）
- **头维度**：256（滑动窗口），512（全局）
- **词汇表大小**：262,144
- **上下文长度**：128K tokens
- **混合注意力**：5 层滑动窗口（512 tokens）+ 1 层全局，重复 7 次
- **KV 共享**：跨 18 层

### 视觉编码器（约 150M 参数）
- **层数**：16
- **隐藏层大小**：768
- **Patch 大小**：16
- **默认输出**：每张图像 280 个 tokens
- **可配置 token 预算**：70、140、280、560、1120
  - 70：分类、快速视频处理
  - 140：快速理解
  - 280：通用多模态对话
  - 560：复杂 UI 推理
  - 1120：OCR、文档、手写识别

### 音频编码器（约 300M 参数）
- **层数**：12
- **隐藏层大小**：1,024
- **特征提取器**：128 个 Mel 频率 bin，16kHz 采样率
- **处理速度**：每个 token 40ms，最多 750 个 tokens = **最长 30 秒**
- **激活函数**：SiLU

### 视频处理
- 帧采样数：32
- 每帧最多 70 个软 tokens
- 最大时长：1 fps 下 60 秒

## 特殊 Tokens

| Token | 字符串 | 用途 |
|-------|--------|---------|
| 图像 | `<\|image\|>` | 图像数据占位符 |
| 音频 | `<\|audio\|>` | 音频数据占位符 |
| 视频 | `<\|video\|>` | 视频数据占位符 |
| 思考 | `<\|think\|>` | 启用推理模式 |
| BOI/EOI | `<\|image>` / `<image\|>` | 图像区域标记 |
| BOA/EOA | `<\|audio>` / `<audio\|>` | 音频区域标记 |
| 工具调用 | `<\|tool_call>` / `<tool_call\|>` | 函数调用 |

## 思考/推理模式

- 在系统提示词开头添加 `<|think|>` 以启用
- 输出格式：`<|channel>thought\n[reasoning]<channel|>[answer]`
- **请勿在实时演示中使用** —— 对话延迟太高

## 推理代码（Transformers）

```python
from transformers import AutoProcessor, AutoModelForMultimodalLM

processor = AutoProcessor.from_pretrained("google/gemma-4-E4B-it")
model = AutoModelForMultimodalLM.from_pretrained("google/gemma-4-E4B-it", dtype="auto", device_map="auto")

# 多模态消息格式
messages = [
    {"role": "user", "content": [
        {"type": "image", "url": "https://example.com/photo.jpg"},
        {"type": "audio", "audio": "https://example.com/audio.wav"},
        {"type": "text", "text": "Describe what you see and hear."}
    ]}
]

inputs = processor.apply_chat_template(messages, tokenize=True, return_dict=True, return_tensors="pt", add_generation_prompt=True).to(model.device)
outputs = model.generate(**inputs, max_new_tokens=512, temperature=1.0, top_p=0.95, top_k=64)
```

## 推荐采样参数
- Temperature：1.0
- Top-p：0.95
- Top-k：64

## 最佳实践
1. 在提示词中将图像/音频内容放在文本之前
2. 音频最大时长：每段 30 秒
3. 视频最大时长：1 fps 下 60 秒
4. 为追求速度使用较低的视觉 token 预算（70-140），为追求细节使用较高预算（560+）
5. 实时应用中请勿启用思考模式

## 基准测试（E4B 指令微调版）

| 基准测试 | 得分 |
|-----------|-------|
| MMLU Pro | 69.4% |
| AIME 2026 | 42.5% |
| LiveCodeBench v6 | 52.0% |
| GPQA Diamond | 58.6% |
| MMMU Pro（视觉） | 52.6% |
| MATH-Vision | 59.5% |
| MMMLU（多语言） | 76.6% |

## 参考来源
- https://huggingface.co/google/gemma-4-E4B
- https://huggingface.co/google/gemma-4-E4B-it
