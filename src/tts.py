"""TTS backends for local voice output."""

import json
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


def _is_apple_silicon() -> bool:
    return sys.platform == "darwin" and platform.machine() == "arm64"


class TTSBackend:
    """Unified TTS interface."""

    sample_rate: int = 24000

    def generate(self, text: str, voice: str = "af_heart", speed: float = 1.1) -> np.ndarray:
        raise NotImplementedError


class DisabledBackend(TTSBackend):
    """No-op backend for debugging model startup without audio output."""

    def generate(self, text: str, voice: str = "af_heart", speed: float = 1.1) -> np.ndarray:
        return np.zeros(0, dtype=np.float32)


class MLXBackend(TTSBackend):
    """Kokoro via mlx-audio."""

    def __init__(self):
        from mlx_audio.tts.generate import load_model

        model_ref = os.environ.get("KOKORO_MODEL_PATH", "mlx-community/Kokoro-82M-bf16")
        self._model_path = Path(os.path.expanduser(model_ref))
        self._model = load_model(str(self._model_path))
        self.sample_rate = self._model.sample_rate
        self._default_voice = self._resolve_voice_path("af_heart")
        # Warmup: triggers pipeline init (phonemizer, spacy, etc.)
        list(self._model.generate(text="Hello", voice=self._default_voice, speed=1.0))

    def _resolve_voice_path(self, voice: str) -> str:
        if self._model_path.is_dir():
            direct = self._model_path / f"{voice}.safetensors"
            if direct.exists():
                return str(direct)
        return voice

    def generate(self, text: str, voice: str = "af_heart", speed: float = 1.1) -> np.ndarray:
        results = list(
            self._model.generate(
                text=text,
                voice=self._resolve_voice_path(voice),
                speed=speed,
            )
        )
        return np.concatenate([np.array(r.audio) for r in results])


class ONNXBackend(TTSBackend):
    """Kokoro ONNX backend (CPU)."""

    def __init__(self):
        import kokoro_onnx
        from huggingface_hub import hf_hub_download

        model_path = os.environ.get("KOKORO_ONNX_MODEL_PATH", "").strip()
        voices_path = os.environ.get("KOKORO_ONNX_VOICES_PATH", "").strip()

        if model_path and voices_path:
            model_path = os.path.expanduser(model_path)
            voices_path = os.path.expanduser(voices_path)
        else:
            model_path = hf_hub_download("fastrtc/kokoro-onnx", "kokoro-v1.0.onnx")
            voices_path = hf_hub_download("fastrtc/kokoro-onnx", "voices-v1.0.bin")

        self._model = kokoro_onnx.Kokoro(model_path, voices_path)
        self.sample_rate = 24000

    def generate(self, text: str, voice: str = "af_heart", speed: float = 1.1) -> np.ndarray:
        pcm, _sr = self._model.create(text, voice=voice, speed=speed)
        return pcm


class SystemSayBackend(TTSBackend):
    """macOS native TTS via `say`."""

    def __init__(self):
        import soundfile as sf

        self._sf = sf
        self._voice = os.environ.get("SYSTEM_TTS_VOICE", "Tingting").strip() or "Tingting"
        self._base_rate = int(os.environ.get("SYSTEM_TTS_RATE", "180"))
        # Warm up once and capture the actual sample rate returned by the voice.
        self.generate("你好。")

    def generate(self, text: str, voice: str = "Tingting", speed: float = 1.0) -> np.ndarray:
        del voice
        if not text.strip():
            return np.zeros(0, dtype=np.float32)

        rate = max(80, int(self._base_rate * speed))
        with tempfile.NamedTemporaryFile(delete=False, suffix=".aiff") as handle:
            audio_path = Path(handle.name)

        try:
            subprocess.run(
                ["say", "-v", self._voice, "-r", str(rate), "-o", str(audio_path), text],
                check=True,
                capture_output=True,
            )
            audio, sample_rate = self._sf.read(str(audio_path), dtype="float32")
            if getattr(audio, "ndim", 1) > 1:
                audio = audio.mean(axis=1)
            self.sample_rate = int(sample_rate)
            return np.asarray(audio, dtype=np.float32)
        finally:
            audio_path.unlink(missing_ok=True)


class QwenTTSBackend(TTSBackend):
    """Qwen3-TTS via mlx-audio."""

    def __init__(self):
        from mlx_audio.tts.generate import load_model

        model_ref = os.environ.get("TTS_MODEL_PATH", "").strip()
        if not model_ref:
            raise RuntimeError("TTS_MODEL_PATH is not set for Qwen TTS.")

        self._model = load_model(os.path.expanduser(model_ref))
        self.sample_rate = self._model.sample_rate
        self._language = os.environ.get("QWEN_TTS_LANGUAGE", "chinese").strip() or "chinese"
        self._speaker = os.environ.get("QWEN_TTS_SPEAKER", "serena").strip() or "serena"
        self._instruct = os.environ.get("QWEN_TTS_INSTRUCT", "").strip() or None
        self._voice_design = (
            os.environ.get(
                "QWEN_TTS_VOICE_DESIGN",
                "一个自然、清晰、温和的中文女声，适合日常对话。",
            ).strip()
            or "一个自然、清晰、温和的中文女声，适合日常对话。"
        )
        self._speed = float(os.environ.get("QWEN_TTS_SPEED", "1.0"))
        self._model_type = getattr(self._model.config, "tts_model_type", "base")
        # Warm up once to avoid first-response hitch.
        list(self._generate("你好。"))

    def _generate(self, text: str):
        kwargs = {
            "text": text,
            "lang_code": self._language,
            "speed": self._speed,
        }
        if self._model_type == "voice_design":
            kwargs["instruct"] = self._voice_design
        else:
            kwargs["voice"] = self._speaker
            if self._instruct:
                kwargs["instruct"] = self._instruct
        return self._model.generate(**kwargs)

    def generate(self, text: str, voice: str = "serena", speed: float = 1.0) -> np.ndarray:
        del voice, speed
        results = list(self._generate(text))
        return np.concatenate([np.array(r.audio) for r in results])


def _read_model_type(model_ref: str) -> str | None:
    try:
        config_path = Path(os.path.expanduser(model_ref)) / "config.json"
        if not config_path.exists():
            return None
        with config_path.open(encoding="utf-8") as handle:
            config = json.load(handle)
        return config.get("model_type")
    except Exception:
        return None


def load() -> TTSBackend:
    """Load the best available TTS backend for this platform."""
    if os.environ.get("DISABLE_TTS"):
        print("TTS: disabled by DISABLE_TTS")
        return DisabledBackend()

    backend_pref = os.environ.get("TTS_BACKEND", "").strip().lower()
    if backend_pref in {"say", "system", "macos"}:
        backend = SystemSayBackend()
        print(f"TTS: macOS say ({backend._voice}, sample_rate={backend.sample_rate})")
        return backend

    tts_model_path = os.environ.get("TTS_MODEL_PATH", "").strip()
    if tts_model_path and _read_model_type(tts_model_path) == "qwen3_tts":
        try:
            backend = QwenTTSBackend()
            print(f"TTS: qwen3_tts (sample_rate={backend.sample_rate})")
            return backend
        except Exception as exc:
            print(f"TTS: qwen3_tts unavailable ({exc})")
            if sys.platform == "darwin":
                backend = SystemSayBackend()
                print(f"TTS: macOS say fallback ({backend._voice}, sample_rate={backend.sample_rate})")
                return backend

    if _is_apple_silicon() and not os.environ.get("KOKORO_ONNX"):
        try:
            backend = MLXBackend()
            print(f"TTS: mlx-audio (Apple GPU, sample_rate={backend.sample_rate})")
            return backend
        except ImportError:
            print("TTS: mlx-audio not installed, falling back to kokoro-onnx")

    backend = ONNXBackend()
    print(f"TTS: kokoro-onnx (CPU, sample_rate={backend.sample_rate})")
    return backend
