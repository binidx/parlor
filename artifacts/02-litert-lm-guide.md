# LiteRT-LM — Gemma 4 推理引擎

## 什么是 LiteRT-LM？

Google 的**生产级、开源推理框架**，用于在边缘设备上部署大语言模型。驱动着 Chrome、Chromebook Plus 和 Pixel Watch 上的端侧 GenAI 功能。

- **许可证**：Apache 2.0
- **GitHub**：https://github.com/google-ai-edge/LiteRT-LM
- **最新版本**：v0.10.1（2026 年 4 月 3 日）
- **Star 数**：约 1,105
- **核心语言**：C++，提供 Python、Kotlin、Swift（开发中）绑定
- **构建系统**：Bazel 7.6.1
- **模型格式**：`.litertlm` 打包文件（预转换，托管在 HuggingFace 的 `litert-community/` 下）

## 为什么在我们的演示中选择 LiteRT-LM

| 特性 | LiteRT-LM | Ollama | llama.cpp | MLX |
|---------|-----------|--------|-----------|-----|
| Gemma 4 E4B 模型大小 | **3.65 GB** | ~6 GB | ~6 GB | ~5 GB |
| 原生音频输入 | **支持** | 不支持 | 不支持 | 不支持 |
| 原生视觉输入 | **支持** | 支持 | 支持 | 支持 |
| Mac 上的 Metal GPU | **支持**（C++ API） | 支持 | 支持 | 支持 |
| Python GPU | **支持**（WebGPU/Metal，v0.10.1） | 支持 | 不适用 | 支持 |
| 生产环境验证 | Chrome、Pixel Watch | 开发工具 | 库 | 研究用途 |

**核心优势**：最小的模型体积（积极的混合精度量化）、原生音频支持（无需单独的 STT 模块）、Google 针对自家模型的深度优化。

## 支持的模型

| 模型 | 类型 | 磁盘大小 |
|-------|------|-------------|
| **Gemma 4 E4B** | 对话 | **3.65 GB** |
| **Gemma 4 E2B** | 对话 | 2.58 GB |
| Gemma 3n E4B | 对话 | 4.24 GB |
| Gemma 3n E2B | 对话 | 2.97 GB |
| Phi-4-mini | 对话 | 3.9 GB |
| Qwen 2.5 1.5B | 对话 | 1.6 GB |

## 性能基准测试（Gemma 4 E2B）

1024 个 prefill tokens + 256 个 decode tokens：

| 平台 | 后端 | Prefill（tok/s） | Decode（tok/s） | TTFT | 内存 |
|----------|---------|-----------------|----------------|------|--------|
| **MacBook Pro M4 Max** | GPU | **7,835** | **160.2** | **0.1s** | 1,623 MB |
| MacBook Pro M4 Max | CPU | 901 | 41.6 | 1.1s | 736 MB |
| Samsung S26 Ultra | GPU | 3,808 | 52.1 | 0.3s | 676 MB |
| iPhone 17 Pro | GPU | 2,878 | 56.5 | 0.3s | 1,450 MB |
| NVIDIA RTX 4090 | GPU | 11,234 | 143.4 | - | 913 MB |
| Raspberry Pi 5 | CPU | 133 | 7.6 | 7.8s | 1,546 MB |

M3 Pro 的基准测试尚未发布，但预计 GPU decode 速度在 60-100+ tok/s 范围内。

## API 语言支持

| 语言 | 状态 | 最佳用途 |
|----------|--------|----------|
| **C++** | **稳定** | 高性能原生开发（原生 Metal GPU） |
| Python | 稳定 | 原型开发与生产部署（CPU + GPU，通过 WebGPU/Metal）。Python 中原生 Metal 因 ABI 不匹配会崩溃。 |
| Kotlin | 稳定 | Android 应用 |
| Swift | **开发中** | iOS/macOS（暂无公开代码） |

## macOS 预构建库

位于 `prebuilt/macos_arm64/` 目录下：

| 库文件 | 用途 |
|---------|---------|
| `libLiteRt.dylib` | 核心运行时 |
| `libLiteRtMetalAccelerator.dylib` | Metal GPU 后端 |
| `libLiteRtWebGpuAccelerator.dylib` | WebGPU 后端 |
| `libGemmaModelConstraintProvider.dylib` | 约束解码 |

这些是运行时加载的插件。引擎本身需要从源码构建。

## macOS 构建指南

```bash
# 前置条件
xcode-select --install
brew install bazelisk

# 克隆仓库
git clone https://github.com/google-ai-edge/LiteRT-LM.git ~/workspace/LiteRT-LM
cd ~/workspace/LiteRT-LM
git checkout v0.10.1
git lfs pull

# 构建 GPU 支持版本
bazel build //runtime/engine:litert_lm_main \
  --define=litert_link_capi_so=true \
  --define=resolve_symbols_in_exec=false

# 设置运行目录
mkdir -p run_dir
cp bazel-bin/runtime/engine/litert_lm_main run_dir/
cp prebuilt/macos_arm64/*.dylib run_dir/

# 下载模型
pip install litert-lm  # 或者：uv tool install litert-lm
litert-lm run --from-huggingface-repo=litert-community/gemma-4-E4B-it-litert-lm \
  gemma-4-E4B-it.litertlm

# 运行
cd run_dir
./litert_lm_main --backend=gpu --model_path=../gemma-4-E4B-it.litertlm
```

## C++ API 参考

### 三层架构

```
Engine（模型加载、硬件后端）
  └── Session（有状态 KV 缓存、prefill/decode）—— 底层接口
  └── Conversation（对话 API、提示词模板、多模态）—— 高层接口（推荐使用）
```

### 创建引擎

```cpp
#include "runtime/engine/engine.h"
#include "runtime/engine/engine_factory.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/executor/executor_settings_base.h"

using namespace litert::lm;

auto model_assets = ModelAssets::Create("/path/to/gemma-4-E4B-it.litertlm");

auto engine_settings = EngineSettings::CreateDefault(
    *std::move(model_assets),
    Backend::GPU,                        // 主 LLM 后端
    /*vision_backend=*/Backend::GPU,     // 用于图像输入
    /*audio_backend=*/Backend::CPU       // 用于音频输入
);

auto engine = EngineFactory::CreateDefault(*std::move(engine_settings));
```

### Conversation API（多模态）

```cpp
#include "runtime/conversation/conversation.h"
#include "runtime/conversation/io_types.h"

// 使用系统提示词创建
auto config = ConversationConfig::Builder()
    .SetPreface(JsonPreface{
        .messages = nlohmann::ordered_json::array({
            {{"role", "system"}, {"content", "You are a helpful assistant."}}
        })
    })
    .Build(*engine);

auto conversation = Conversation::Create(*engine, *config);

// 纯文本
auto response = conversation->SendMessage(
    json{{"role", "user"}, {"content", "Hello!"}});

// 图像 + 文本（摄像头画面）
auto response = conversation->SendMessage(json{
    {"role", "user"},
    {"content", json::array({
        {{"type", "image"}, {"blob", base64_jpeg_data}},
        {{"type", "text"}, {"text", "What do you see?"}}
    })}
});

// 音频 + 文本（麦克风输入）
auto response = conversation->SendMessage(json{
    {"role", "user"},
    {"content", json::array({
        {{"type", "audio"}, {"blob", base64_wav_data}},
        {{"type", "text"}, {"text", "What did the user say?"}}
    })}
});

// 音频 + 图像 + 文本（同时使用！）
auto response = conversation->SendMessage(json{
    {"role", "user"},
    {"content", json::array({
        {{"type", "audio"}, {"blob", base64_wav_data}},
        {{"type", "image"}, {"blob", base64_jpeg_data}},
        {{"type", "text"}, {"text", "The user is speaking to you while showing you their camera. Respond naturally."}}
    })}
});
```

### 流式 Token 生成

```cpp
conversation->SendMessageAsync(
    json{{"role", "user"}, {"content", "Write me a poem"}},
    [](absl::StatusOr<Message> message) {
        if (!message.ok()) return;
        auto& json_msg = std::get<JsonMessage>(*message);
        if (json_msg.is_null()) return;  // 流式输出完成
        for (const auto& content : json_msg["content"]) {
            std::cout << content["text"].get<std::string>() << std::flush;
        }
    }
);
engine->WaitUntilDone(absl::Minutes(5));
```

### C API（用于 FFI / HTTP 服务器）

文件：`/c/engine.h` —— 纯 C 接口，使用 `extern "C"` 链接。

```c
// 流式回调类型
typedef void (*LiteRtLmStreamCallback)(
    void* callback_data, const char* chunk, bool is_final, const char* error_msg);

// 关键函数
LiteRtLmEngine* litert_lm_engine_create(const LiteRtLmEngineSettings* settings);
LiteRtLmConversation* litert_lm_conversation_create(LiteRtLmEngine* engine, ...);

// 流式多模态消息
int litert_lm_conversation_send_message_stream(
    LiteRtLmConversation* conversation,
    const char* message_json,         // 包含多模态内容的 JSON 字符串
    const char* extra_context,
    LiteRtLmStreamCallback callback,
    void* callback_data);

// 在生成过程中取消
void litert_lm_conversation_cancel_process(LiteRtLmConversation* conversation);
```

## 关键头文件

| 头文件 | 用途 |
|--------|---------|
| `runtime/engine/engine.h` | Engine 和 Session 接口 |
| `runtime/engine/engine_factory.h` | EngineFactory |
| `runtime/engine/engine_settings.h` | EngineSettings、SessionConfig |
| `runtime/engine/io_types.h` | InputText、InputImage、InputAudio、Responses |
| `runtime/conversation/conversation.h` | Conversation（高层对话接口） |
| `runtime/conversation/io_types.h` | JsonMessage、Message、Preface |
| `runtime/executor/executor_settings_base.h` | Backend 枚举、ModelAssets |
| `c/engine.h` | 纯 C API，用于 FFI |

## 模型专属处理

LiteRT-LM 内置了专用的 `Gemma4DataProcessor`，自动处理以下内容：
- Gemma 4 Jinja 提示词模板
- 图像预处理（缩放、按配置的预算进行 tokenize）
- 音频预处理（Mel 频谱图提取）
- 工具调用 / 函数调用
- 约束解码

该处理器根据模型元数据自动选择，无需手动配置。

## 其他功能

- **推测解码**：`enable_speculative_decoding` 参数（v0.10.1）
- **约束解码**：正则表达式、JSON Schema、Lark 语法
- **函数调用**：内置自动执行
- **Jinja 提示词模板**：从模型元数据自动加载
- **对话历史**：自动增量式提示词渲染
- **文本评分**：通过 `run_text_scoring()` 获取对数似然分数

## 参考来源
- https://github.com/google-ai-edge/LiteRT-LM
- https://ai.google.dev/edge/litert-lm/overview
- HuggingFace: litert-community/gemma-4-E4B-it-litert-lm
