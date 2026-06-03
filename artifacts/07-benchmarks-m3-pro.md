# 基准测试 — Apple M3 Pro 18GB

所有基准测试均在 LiteRT-LM v0.10.1、Gemma 4 E2B（2.3B 有效参数，2.58 GB 模型）上运行。

## Python API — GPU (WebGPU) 配置

截至 litert-lm v0.10.1，`Backend.GPU` 在 Python 中可用，尽管文档标注为"即将推出"。
pip 包内置了 WebGPU 加速器，在 macOS 底层使用 Metal。
音频后端受模型限制只能使用 CPU（设置为 GPU 会报 `INVALID_ARGUMENT`）。

```python
engine = litert_lm.Engine(
    MODEL_PATH,
    backend=litert_lm.Backend.GPU,
    vision_backend=litert_lm.Backend.GPU,
    audio_backend=litert_lm.Backend.CPU,  # 模型要求 CPU
)
```

**如何验证 GPU 是否启用：** 检查 stderr 输出 `RegisterAccelerator: name=GPU WebGPU`。

**已知警告（非致命）：** `Could not load symbol LiteRtTopKWebGpuSampler_UpdateConfig` —
回退到 CPU 采样，GPU 计算仍然生效。不影响性能。

## 合成基准测试（1024 prefill, 256 decode）

方法论与 HuggingFace 官方基准测试相同。纯文本，无多模态编码器。
通过 `litert_lm.Benchmark()` Python API 运行。

### GPU (WebGPU/Metal) vs CPU

| 指标 | GPU（缓存后） | GPU（首次运行） | CPU | GPU vs CPU |
|------|-------------|----------------|-----|------------|
| **Prefill** | **2,662 tok/s** | 431 tok/s | 215 tok/s | **12.4x** |
| **Decode** | **83 tok/s** | 70 tok/s | 19 tok/s | **4.4x** |
| **TTFT** | **0.40s** | 2.39s | 4.82s | **12.1x** |
| 初始化时间 | ~0.8s | ~2.1s | ~0.8s | - |

**首次着色器编译惩罚：** WebGPU 在进程启动后首次推理时编译 Metal 着色器（~2.4s TTFT）。后续运行使用缓存的着色器（~0.4s TTFT）。服务器在启动时执行预热 `send_message("Hi")` 以预编译着色器。

### 与 HuggingFace 官方基准测试对比

| 平台 | Prefill (tok/s) | Decode (tok/s) | TTFT |
|------|-----------------|----------------|------|
| **M3 Pro（我们的，WebGPU）** | **2,662** | **83** | **0.40s** |
| M4 Max（官方） | 7,835 | 160 | 0.1s |
| iPhone 17 Pro（官方） | 2,878 | 56.5 | 0.3s |
| S26 Ultra（官方） | 3,808 | 52.1 | 0.3s |
| RTX 4090（官方） | 11,234 | 143 | 0.1s |

我们的 M3 Pro 符合预期 — prefill 与 iPhone 17 Pro 相当，
decode 更快（83 vs 56.5），得益于更高的内存带宽。

## 原生 Metal vs WebGPU（Python）

pip 包仅包含 `libLiteRtWebGpuAccelerator.dylib`。C++ 构建包含
原生 Metal（`libLiteRtMetalAccelerator.dylib`），在 C++ 二进制文件中 prefill 快约 6 倍。然而：

- **原生 Metal 在 Python 中崩溃（SIGSEGV）** — 预编译的 Metal 库与 pip 安装的 Python nanobind 绑定存在 ABI 不兼容问题。
- **DYLD_LIBRARY_PATH 无效** — macOS dyld 在进程启动时读取该变量，而 litert_lm 使用相对于其包目录的 dlopen，而非环境变量。
- **带缓存着色器的 WebGPU 性能与原生 Metal 相当** — 着色器编译完成后，WebGPU prefill（2,662 tok/s）与原生 Metal 相当。差距仅体现在首次运行。

**结论：** 继续使用 pip 包中的 WebGPU。对 Python 用户而言原生 Metal 没有优势。

## 多模态编码器开销（实际场景）

通过隔离每种模态并减去纯文本基线来测量。
编码器在 token prefill 开始之前运行各自的前向传播。

| 组件 | 时间 | 备注 |
|------|------|------|
| **图像编码器（SigLIP，GPU）** | **~0.86s** | 最大的单项开销。320px 输入上采样到 ~912x672，产生 ~268 tokens（默认预算 280）。分辨率无关 — 始终填满预算。 |
| **音频编码器（Conformer，CPU）** | **~0.1-0.4s** | 3.05 亿参数 conformer。~25 tokens/秒音频，上限 750 tokens（30s）。受模型约束强制使用 CPU。 |
| **工具语法开销** | **~0.45s** | 工具调用解析的受限解码设置。每次 `send_message` 都会支付，不仅是首次。LiteRT-LM 内部行为。 |
| **文本 prefill（GPU）** | **~0.08s** | ~300 tokens（系统提示词 + 用户提示词）。很快。 |

### Gemma 4 输入 Token 计数

分词器：SentencePiece，262,144 词表。从 `.litertlm` 包的偏移量 32768 处提取。

**图像 tokens：** 可配置预算（70/140/280/560/1120）。默认为 **280**。
对于 320x240 JPEG 在默认预算下：上采样到 912x672 → **266 软 tokens + 2 特殊 = 总计 268**。
输入分辨率无关 — 图像始终缩放以填满预算。

**音频 tokens：** ~**每秒 25 tokens**（每 token 40ms），上限 750 tokens（30s）。
- 1s → ~25 tokens
- 2s → ~50 tokens
- 5s → ~125 tokens

**文本：** 系统提示词 ~51 tokens，工具定义 ~177 tokens，用户提示词 ~30 tokens。

## 实际场景管线延迟（GPU，E2B）

通过 `bench.py` 对运行中的服务器进行 WebSocket 测量。每轮含图像 + 音频。

### 每轮分解

| 阶段 | 时间 |
|------|------|
| 图像编码器（SigLIP） | ~0.86s |
| 工具受限解码开销 | ~0.45s |
| 音频编码器（2-5s 音频，CPU） | ~0.1-0.4s |
| 文本 + 历史记录 prefill | ~0.1-0.4s |
| **总计 TTFT / prefill** | **~1.8-2.2s** |
| Decode（~25 tokens，80 tok/s） | ~0.3s |
| **总计 LLM** | **~2.1-2.5s** |
| TTS（Kokoro MLX，1-3 句） | ~0.3-0.7s |
| **总计端到端** | **~2.5-3.0s** |

### 多轮对话上下文增长

来自 bench.py（同一 WebSocket 连接，每轮含图像 + 音频）：

| 轮次 | LLM 时间 | 备注 |
|------|----------|------|
| 第 1 轮 | ~1.8s | 新对话 |
| 第 2 轮 | ~2.2s | +前轮上下文 |
| 第 3 轮 | ~2.5s | 上下文增长 |
| 第 4 轮 | ~2.2s | |
| 第 5 轮 | ~2.2s | |

随着对话历史累积，TTFT 每轮增长约 0.1-0.2s。

## 瓶颈总结

| 瓶颈 | 每轮开销 | 可修复？ |
|------|----------|----------|
| **图像编码器（SigLIP）** | 0.86s | 也许 — 将图像预算从 280 降到 70（~0.2s），但 Python API 中不可配置 |
| **工具语法** | 0.45s | 否 — LiteRT-LM 受限解码的内部行为 |
| **音频编码器（CPU）** | 0.1-0.4s | 否 — 模型约束强制使用 CPU |
| **上下文增长** | 0.1-0.2s/轮 | 可以限制对话历史长度 |
| **Decode** | ~0.3s | 已接近硬件极限（80 tok/s） |
| **TTS** | 0.3-0.7s | 已在 GPU 上运行（MLX） |

## 基准测试命令

```bash
# 合成纯文本基准测试（Python API）
python3 -c "
import litert_lm, os
bench = litert_lm.Benchmark(
    os.path.expanduser('~/workspace/LiteRT-LM/run_dir/gemma-4-E2B-it.litertlm'),
    backend=litert_lm.Backend.GPU, prefill_tokens=1024, decode_tokens=256)
r = bench.run()
print(f'Prefill: {r.last_prefill_tokens_per_second:.0f} tok/s')
print(f'Decode: {r.last_decode_tokens_per_second:.0f} tok/s')
print(f'TTFT: {r.time_to_first_token_in_second:.3f}s')
"

# 对运行中服务器的端到端基准测试
# 启动服务器：uv run python server.py
# 然后：uv run python bench.py
```
