"""Speech-to-text backends."""

import os
from pathlib import Path


def _auto_detect_asr_model() -> str:
    """Find a local Qwen3-ASR model directory, or fall back to HuggingFace."""
    home = Path.home()
    candidates = [
        home / "models" / "asr" / "Qwen3-ASR-0.6B-4bit",
        home / "models" / "asr" / "Qwen3-ASR-0.6B-8bit",
        home / "models" / "asr" / "Qwen3-ASR-0.6B-MLX-4bit",
        home / "models" / "asr" / "Qwen3-ASR-1.7B-MLX-8bit",
        home / "models" / "asr" / "Qwen3-ASR-0.6B",
    ]
    for d in candidates:
        if d.is_dir() and (d / "preprocessor_config.json").exists():
            return str(d)
    return "Qwen/Qwen3-ASR-0.6B"


class ASRBackend:
    """Unified speech-to-text interface."""

    def transcribe(self, audio_path: str) -> str:
        raise NotImplementedError


class QwenASRBackend(ASRBackend):
    """Qwen3-ASR via mlx-audio."""

    def __init__(self):
        from mlx_audio.stt import load as load_stt_model

        model_ref = os.environ.get("ASR_MODEL_PATH", "").strip()
        if not model_ref:
            model_ref = _auto_detect_asr_model()
            os.environ["ASR_MODEL_PATH"] = model_ref
            print(f"ASR: auto-detected model at {model_ref}")

        self._model = load_stt_model(os.path.expanduser(model_ref))
        self._language = os.environ.get("ASR_LANGUAGE", "chinese").strip() or "chinese"
        self._max_tokens = int(os.environ.get("ASR_MAX_TOKENS", "4096"))
        self._system_prompt = os.environ.get("ASR_SYSTEM_PROMPT", "").strip() or None

    def transcribe(self, audio_path: str) -> str:
        result = self._model.generate(
            audio_path,
            language=self._language,
            max_tokens=self._max_tokens,
            system_prompt=self._system_prompt,
        )
        text = getattr(result, "text", "")
        return text.strip()


def load() -> ASRBackend | None:
    """Load ASR backend if configured."""
    backend = os.environ.get("ASR_BACKEND", "").strip().lower()
    model_path = os.environ.get("ASR_MODEL_PATH", "").strip()
    if not backend:
        backend = "qwen" if model_path else "none"
    if backend in {"", "none", "off", "disabled"}:
        print("ASR: disabled")
        return None
    if backend == "qwen":
        asr_backend = QwenASRBackend()
        print("ASR: qwen3_asr")
        return asr_backend
    raise ValueError(f"Unsupported ASR_BACKEND: {backend}")
