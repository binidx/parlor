#!/bin/zsh
set -euo pipefail

# Edit this path when you want to switch models.
MODEL_PATH="${MODEL_PATH:-$HOME/models/llm/gemma-4-E4B-it-The-DECKARD-V2-Strong-HERETIC-UNCENSORED-Thinking-mxfp8-mlx}"
# Optional: point this at a local Kokoro model directory to avoid downloading TTS assets.
KOKORO_MODEL_PATH="${KOKORO_MODEL_PATH:-}"
# Optional: local ONNX Kokoro files.
KOKORO_ONNX_MODEL_PATH="${KOKORO_ONNX_MODEL_PATH:-}"
KOKORO_ONNX_VOICES_PATH="${KOKORO_ONNX_VOICES_PATH:-}"
# Optional: set to 1 if you want to test the LLM/video path without speech output.
DISABLE_TTS="${DISABLE_TTS:-}"
PORT="${PORT:-}"
DEFAULT_PORT=9091

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SRC_DIR="$SCRIPT_DIR/src"
DEFAULT_KOKORO_DIR="$SCRIPT_DIR/models/Kokoro-82M-bf16"
DEFAULT_TTS_DIR="$HOME/models/tts"

pick_port() {
  local candidate
  for candidate in "$DEFAULT_PORT" 9092 9093 9094 9095 9096; do
    if ! lsof -nP -iTCP:$candidate -sTCP:LISTEN >/dev/null 2>&1; then
      echo "$candidate"
      return 0
    fi
  done
  echo "No free port found in 9091-9096" >&2
  return 1
}

if [[ -z "$PORT" ]]; then
  PORT="$(pick_port)"
fi

if [[ -z "$KOKORO_MODEL_PATH" && -d "$DEFAULT_TTS_DIR/Kokoro-82M-bf16" ]]; then
  KOKORO_MODEL_PATH="$DEFAULT_TTS_DIR/Kokoro-82M-bf16"
fi

if [[ -z "$KOKORO_MODEL_PATH" ]]; then
  for candidate in "$DEFAULT_TTS_DIR"/Kokoro* "$DEFAULT_TTS_DIR"/kokoro*; do
    if [[ -d "$candidate" && -f "$candidate/config.json" ]]; then
      KOKORO_MODEL_PATH="$candidate"
      break
    fi
  done
fi

if [[ -z "$KOKORO_MODEL_PATH" && -d "$DEFAULT_KOKORO_DIR" && -f "$DEFAULT_KOKORO_DIR/config.json" ]]; then
  KOKORO_MODEL_PATH="$DEFAULT_KOKORO_DIR"
fi

if [[ -z "$KOKORO_ONNX_MODEL_PATH" && -f "$DEFAULT_TTS_DIR/kokoro-v1.0.onnx" ]]; then
  KOKORO_ONNX_MODEL_PATH="$DEFAULT_TTS_DIR/kokoro-v1.0.onnx"
fi

if [[ -z "$KOKORO_ONNX_VOICES_PATH" && -f "$DEFAULT_TTS_DIR/voices-v1.0.bin" ]]; then
  KOKORO_ONNX_VOICES_PATH="$DEFAULT_TTS_DIR/voices-v1.0.bin"
fi

cd "$SRC_DIR"

export MODEL_PATH
export PORT
export UV_CACHE_DIR="${UV_CACHE_DIR:-.uv-cache}"
[[ -n "$KOKORO_MODEL_PATH" ]] && export KOKORO_MODEL_PATH
[[ -n "$KOKORO_ONNX_MODEL_PATH" ]] && export KOKORO_ONNX_MODEL_PATH
[[ -n "$KOKORO_ONNX_VOICES_PATH" ]] && export KOKORO_ONNX_VOICES_PATH
[[ -n "$DISABLE_TTS" ]] && export DISABLE_TTS

if [[ -z "$KOKORO_MODEL_PATH" && -n "$KOKORO_ONNX_MODEL_PATH" && -n "$KOKORO_ONNX_VOICES_PATH" ]]; then
  export KOKORO_ONNX=1
fi

echo "Starting Parlor"
echo "MODEL_PATH=$MODEL_PATH"
echo "PORT=$PORT"
[[ -n "$KOKORO_MODEL_PATH" ]] && echo "KOKORO_MODEL_PATH=$KOKORO_MODEL_PATH"
[[ -n "$KOKORO_ONNX_MODEL_PATH" ]] && echo "KOKORO_ONNX_MODEL_PATH=$KOKORO_ONNX_MODEL_PATH"
[[ -n "$KOKORO_ONNX_VOICES_PATH" ]] && echo "KOKORO_ONNX_VOICES_PATH=$KOKORO_ONNX_VOICES_PATH"
[[ -n "$DISABLE_TTS" ]] && echo "DISABLE_TTS=$DISABLE_TTS"

exec uv run server.py
