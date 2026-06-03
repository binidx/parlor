# Parlor Capabilities & Design Review

## Project Overview

Parlor is an on-device, real-time multimodal AI voice+vision assistant. User talks and shows camera; AI understands and responds with speech -- all locally, no cloud dependency.

**Stack:** FastAPI + WebSocket | Silero VAD (WASM) | LLM (LiteRT-LM / mlx-vlm) | ASR (Qwen3-ASR) | TTS (Kokoro / Qwen3-TTS)

---

## Capabilities

### Voice Chat
- Browser-side VAD (Silero via WASM) -- hands-free, no push-to-talk
- Barge-in: interrupt AI mid-sentence (800ms echo grace period)
- Sentence-level streaming TTS: audio starts before LLM finishes
- Gapless Web Audio API playback with PCM chunks over WebSocket
- Response length control: Short / Medium / Long

### Vision
- Continuous frame sampling: 8-frame ring buffer, 1.5s interval
- Keyframe selection: grayscale pixel-diff, up to 5 frames per turn
- JPEG at 320px width, quality 0.7
- Multi-image for MLX runtime; single-image for LiteRT-LM

### TTS (5 backends, auto-selected)
| Backend | Engine | Platform |
|---|---|---|
| Qwen3-TTS | mlx-audio | macOS (Apple Silicon) |
| Kokoro MLX | mlx-audio | macOS (Apple Silicon) |
| Kokoro ONNX | kokoro-onnx | Linux (CPU) |
| macOS `say` | system | macOS |
| Disabled | no-op | debug |

### ASR
- Qwen3-ASR via mlx-audio (0.6B / 1.7B, various quantizations)
- Auto-detects from `~/models/asr/`
- Can be disabled (`ASR_BACKEND=none`); Gemma 4 natively handles audio

### LLM (2 runtimes)
| Runtime | Model Format | Platform |
|---|---|---|
| LiteRTRuntime | `.litertlm` | macOS/Linux (GPU) |
| MlxRuntime | directory (config.json + safetensors) | macOS (Apple Silicon) |

### Frontend
- Single-file app (index.html): HTML + CSS + JS
- Waveform visualizer (40-bar frequency, Web Audio AnalyserNode)
- State-driven glow: green=listening, amber=processing, purple=speaking
- Speaking glow modulated by audio amplitude
- Settings panel: editable system prompt, persisted to localStorage
- Text input mode (toggle)

### API Endpoints
- `GET /` -- serves frontend
- `GET /config` -- model label, backend, system prompt default
- `WebSocket /ws` -- bidirectional audio/text/image streaming

### Configuration
- `.env` file with `python-dotenv`
- All model paths auto-detectable from `~/models/`
- Two launch scripts: `run-parlor.sh` (Gemma 4 + Kokoro), `run-parlor-qwen.sh` (Qwen stack)

---

## Design Review: Issues & Optimizations

### Priority 1 -- Should Fix

#### 1. XSS via `innerHTML` (index.html:1374)
`addMessage()` sets `div.innerHTML = text + ...`. LLM responses could contain arbitrary HTML.
**Fix:** Use `textContent` for message body, or sanitize input.

#### 2. Unbounded conversation history (server.py:664-668)
`MlxRuntime` appends every turn to `session["history"]` with no cap. Latency degrades over time, memory grows without bound.
**Fix:** Sliding window (cap at N recent turns) or summarization.

#### 3. LiteRT engine context manager leak (server.py:404)
`engine.__enter__()` called without matching `__exit__()`. GPU resources never released.
**Fix:** Add cleanup in lifespan shutdown or use proper `with` semantics.

#### 4. Temp directory leak (server.py:201-217)
`prepare_mlx_model_path` creates temp dirs via `mkdtemp` but never cleans them up.
**Fix:** Register `atexit` cleanup or use fixed temp path.

### Priority 2 -- Should Improve

#### 5. WebSocket reconnect without backoff (index.html:847)
`onclose` retries every 2s indefinitely. Hammers server when down.
**Fix:** Exponential backoff (2s, 4s, 8s, max 30s) + "Reconnecting..." UI.

#### 6. Missing error handling in TTS load (tts.py:242-244)
`ONNXBackend()` failure crashes server at startup. `MLXBackend` failure is caught but ONNX is not.
**Fix:** Wrap in try/except, fall back to `DisabledBackend`.

#### 7. Port default inconsistency (server.py:961 vs run-parlor.sh:35)
`server.py` defaults to 8000, shell scripts default to 9091-9096.
**Fix:** Align defaults or document clearly.

#### 8. Duplicate sentence splitting logic (server.py:49,65)
`SENTENCE_SPLIT_RE` and `STREAMING_SENTENCE_RE` serve similar purposes with subtly different patterns. `split_sentences()` only used in non-streaming path.
**Fix:** Consolidate into one implementation.

#### 9. `recv_task.cancel()` without await (server.py:956)
Cancelled task exception silently lost, may produce warnings.
**Fix:** `await recv_task` with `try/except CancelledError`.

#### 10. `LiteRTRuntime.infer` unsafe access (server.py:474)
`response["content"][0]["text"]` without bounds checking.
**Fix:** Defensive access with defaults.

### Priority 3 -- Nice to Have

#### 11. Base64 audio encoding overhead (server.py:828-831)
Every chunk base64-encoded as JSON text frame. ~33% size overhead + CPU cost.
**Fix:** Binary WebSocket frames (`ws.send_bytes`).

#### 12. Frame sampler runs during non-listening states (index.html:1125-1143)
JPEG encoding every 1.5s even when model is busy.
**Fix:** Skip encoding if state is not 'listening'.

#### 13. `ambientPhase` accumulates without bound (index.html:751)
After ~46 days, exceeds `Number.MAX_SAFE_INTEGER`.
**Fix:** `ambientPhase = (ambientPhase + 0.0001) % (Math.PI * 200)`.

#### 14. `streamSources` iteration during mutation (index.html:1306-1311)
`stopPlayback()` iterates `streamSources` while `onended` callbacks splice from same array.
**Fix:** Iterate over copy: `for (const src of [...streamSources])`.

#### 15. `float32ToWavBase64` slow string concatenation (index.html:1287-1302)
Per-sample loop with string concatenation. Slow for large audio.
**Fix:** Use `Uint8Array` + chunked `String.fromCharCode`.

#### 16. WebSocket `onmessage` no try/catch (index.html:849)
Malformed server message crashes handler.
**Fix:** Wrap `JSON.parse` in try/catch.

#### 17. `loadConfig` silently swallows errors (index.html:918-929)
User sees "Loading model..." forever on failure.
**Fix:** Show "Config unavailable" on error.

#### 18. No ARIA / screen reader support (index.html)
No `aria-label`, `role="log"`, or `aria-live` regions.
**Fix:** Add ARIA attributes to controls and transcript.

### Architecture Suggestions

| Item | Current | Suggested |
|---|---|---|
| Runtime interface | No base class, `isinstance` checks | `typing.Protocol` or ABC |
| Global state | Module-level `tts_backend`, `asr_backend` | `app.state` (FastAPI convention) |
| WebSocket handler | 230-line monolith | Extract `StreamingTTSHandler` / `BatchTTSHandler` |
| `.env` parsing | Duplicated in shell scripts + python-dotenv | Pick one mechanism |
| Python version | `>=3.12,<3.13` | Test and widen to include 3.13 |

### Undocumented Config

These env vars are read in code but missing from `.env.example`:
- `SYSTEM_PROMPT` (server.py:45)
- `LONG_MAX_TOKENS` (server.py:258)
- `MAX_TOKENS` (server.py:492)
- `TEMPERATURE` (server.py:493)
