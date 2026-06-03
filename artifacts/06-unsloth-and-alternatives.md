# Unsloth 与替代推理方案

## Unsloth — 量化 Gemma 4 模型

Unsloth 提供了所有 Gemma 4 变体的优化量化版本。

### 可用格式

| 模型 | GGUF (llama.cpp) | MLX (Apple Silicon) |
|-------|-------------------|---------------------|
| E2B | `unsloth/gemma-4-E2B-it-GGUF` (Q8_0) | 可用 |
| E4B | `unsloth/gemma-4-E4B-it-GGUF` (Q8_0) | `unsloth/gemma-4-E4B-it-UD-MLX-4bit` |
| 26B-A4B | `unsloth/gemma-4-26B-A4B-it-GGUF` (UD-Q4_K_XL) | 可用 |
| 31B | `unsloth/gemma-4-31B-it-GGUF` (UD-Q4_K_XL) | 可用 |

"UD" = Unsloth 动态量化（对不同层选择性地使用不同位宽量化）。

### GGUF 推理（llama.cpp）

```bash
# 构建 llama.cpp
git clone https://github.com/ggml-org/llama.cpp
cmake llama.cpp -B llama.cpp/build -DBUILD_SHARED_LIBS=OFF -DGGML_METAL=ON
cmake --build llama.cpp/build --config Release -j

# 运行 E4B
./llama.cpp/build/bin/llama-cli \
    -hf unsloth/gemma-4-E4B-it-GGUF:Q8_0 \
    --temp 1.0 --top-p 0.95 --top-k 64

# 带视觉功能（需要 mmproj 文件）
hf download unsloth/gemma-4-E4B-it-GGUF \
    --include "*mmproj-BF16*" --include "*Q8_0*"
./llama.cpp/build/bin/llama-cli \
    --model gemma-4-E4B-it-Q8_0.gguf \
    --mmproj mmproj-BF16.gguf \
    --temp 1.0 --top-p 0.95 --top-k 64
```

### MLX 推理（Apple Silicon）

```bash
# 安装
curl -fsSL https://raw.githubusercontent.com/unslothai/unsloth/refs/heads/main/install_gemma4_mlx.sh | sh
source ~/.unsloth/unsloth_gemma4_mlx/bin/activate

# 运行
python -m mlx_lm chat --model unsloth/gemma-4-E4B-it-UD-MLX-4bit --max-tokens 4096
```

### Ollama

```bash
ollama pull gemma4:e4b
ollama run gemma4:e4b
```

在 `http://localhost:11434/v1` 提供 OpenAI 兼容 API。

## M3 Pro 18GB 对比表

| 后端 | 模型大小 | 视觉 | 音频 | GPU | 安装 | 生态 |
|---------|-----------|--------|-------|-----|-------|-----------|
| **LiteRT-LM** | **3.65 GB** | **是** | **是** | **Metal (C++)** | Bazel 构建 | 较小（10 个模型） |
| Ollama | ~6 GB | 是 | 否 | Metal | 一条命令 | 庞大 |
| llama.cpp | ~6 GB (Q8) | 是 (mmproj) | 否 | Metal | cmake | 庞大 (GGUF) |
| MLX | ~5 GB (4-bit) | 是 (mlx-vlm) | 否 | Metal | pip install | 增长中 |
| Transformers | ~16 GB (bf16) | 是 | 是 | 无 Metal | pip install | 完整 |

## 为什么选择 LiteRT-LM 而非其他方案

1. **最小模型**：3.65 GB vs 5-6 GB — 在 18 GB 机器上有更多余量
2. **原生音频**：无需单独的 Whisper STT — 管线更简洁
3. **Google 优化**：他们的引擎、他们的模型、他们的量化
4. **Metal GPU**：C++ API 完全支持 Metal 加速
5. **生产级质量**：为 Chrome 和 Pixel Watch 提供支持

## 备用方案

如果 LiteRT-LM 出现问题（构建失败、缺少 E4B 模型、M3 Pro 性能不佳）：

**备用方案 1：Ollama** — `ollama pull gemma4:e4b`，配合 Pipecat 的 `OLLamaLLMService` 使用。由于 Ollama 不支持音频输入，需添加 faster-whisper 进行 STT。

**备用方案 2：MLX** — Apple Silicon 最佳性能。使用 `mlx-vlm` 处理视觉，`mlx-whisper` 处理 STT。需要更多 Python 代码。

**备用方案 3：llama.cpp server** — `llama-server -hf unsloth/gemma-4-E4B-it-GGUF:Q8_0 --mmproj mmproj-BF16.gguf`。OpenAI 兼容 API。需添加 Whisper 进行 STT。
