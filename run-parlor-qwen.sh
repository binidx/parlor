#!/bin/zsh
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-$HOME/models/llm/Qwen3.6-A3B-opus-mxfp4-h}"
PORT="${PORT:-9091}"

ASR_BACKEND="${ASR_BACKEND:-qwen}"
ASR_LANGUAGE="${ASR_LANGUAGE:-chinese}"
ASR_MODEL_PATH="${ASR_MODEL_PATH:-}"

TTS_MODEL_PATH="${TTS_MODEL_PATH:-}"
QWEN_TTS_LANGUAGE="${QWEN_TTS_LANGUAGE:-chinese}"
QWEN_TTS_SPEAKER="${QWEN_TTS_SPEAKER:-serena}"
QWEN_TTS_INSTRUCT="${QWEN_TTS_INSTRUCT:-自然、清晰、亲切，适合中文日常对话。}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SRC_DIR="$SCRIPT_DIR/src"
DEFAULT_ASR_DIR="$HOME/models/asr/Qwen3-ASR-0.6B"
DEFAULT_ASR_DIR_MLX="$HOME/models/asr/Qwen3-ASR-0.6B-MLX-4bit"
DEFAULT_ASR_DIR_MLX_17="$HOME/models/asr/Qwen3-ASR-1.7B-MLX-8bit"
DEFAULT_ASR_DIR_4BIT="$HOME/models/asr/Qwen3-ASR-0.6B-4bit"
DEFAULT_ASR_DIR_8BIT="$HOME/models/asr/Qwen3-ASR-0.6B-8bit"
DEFAULT_TTS_DIR_SMALL="$HOME/models/tts/Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit"
DEFAULT_TTS_DIR="$HOME/models/tts/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit"

has_asr_preprocessor() {
  [[ -f "$1/preprocessor_config.json" ]]
}

has_qwen_tts_speech_tokenizer() {
  [[ -f "$1/speech_tokenizer/config.json" && -f "$1/speech_tokenizer/model.safetensors" ]]
}

if [[ -z "$ASR_MODEL_PATH" ]]; then
  if [[ -d "$DEFAULT_ASR_DIR_4BIT" ]] && has_asr_preprocessor "$DEFAULT_ASR_DIR_4BIT"; then
    ASR_MODEL_PATH="$DEFAULT_ASR_DIR_4BIT"
  elif [[ -d "$DEFAULT_ASR_DIR_8BIT" ]] && has_asr_preprocessor "$DEFAULT_ASR_DIR_8BIT"; then
    ASR_MODEL_PATH="$DEFAULT_ASR_DIR_8BIT"
  elif [[ -d "$DEFAULT_ASR_DIR_MLX" ]] && has_asr_preprocessor "$DEFAULT_ASR_DIR_MLX"; then
    ASR_MODEL_PATH="$DEFAULT_ASR_DIR_MLX"
  elif [[ -d "$DEFAULT_ASR_DIR_MLX_17" ]] && has_asr_preprocessor "$DEFAULT_ASR_DIR_MLX_17"; then
    ASR_MODEL_PATH="$DEFAULT_ASR_DIR_MLX_17"
  elif [[ -d "$DEFAULT_ASR_DIR" ]] && has_asr_preprocessor "$DEFAULT_ASR_DIR"; then
    ASR_MODEL_PATH="$DEFAULT_ASR_DIR"
  else
    ASR_MODEL_PATH="Qwen/Qwen3-ASR-0.6B"
  fi
fi

if [[ -z "$TTS_MODEL_PATH" ]]; then
  if [[ -d "$DEFAULT_TTS_DIR_SMALL" ]] && has_qwen_tts_speech_tokenizer "$DEFAULT_TTS_DIR_SMALL"; then
    TTS_MODEL_PATH="$DEFAULT_TTS_DIR_SMALL"
  elif [[ -d "$DEFAULT_TTS_DIR" ]] && has_qwen_tts_speech_tokenizer "$DEFAULT_TTS_DIR"; then
    TTS_MODEL_PATH="$DEFAULT_TTS_DIR"
  fi
fi

cd "$SRC_DIR"

export MODEL_PATH
export PORT
export UV_CACHE_DIR="${UV_CACHE_DIR:-.uv-cache}"
export ASR_BACKEND
export ASR_LANGUAGE
export ASR_MODEL_PATH
export TTS_MODEL_PATH
export QWEN_TTS_LANGUAGE
export QWEN_TTS_SPEAKER
export QWEN_TTS_INSTRUCT

echo "Starting Parlor (Qwen stack)"
echo "MODEL_PATH=$MODEL_PATH"
echo "PORT=$PORT"
echo "ASR_MODEL_PATH=$ASR_MODEL_PATH"
echo "TTS_MODEL_PATH=$TTS_MODEL_PATH"
echo "QWEN_TTS_LANGUAGE=$QWEN_TTS_LANGUAGE"
echo "QWEN_TTS_SPEAKER=$QWEN_TTS_SPEAKER"

exec uv run server.py
