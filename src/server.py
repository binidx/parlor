"""Parlor — on-device, real-time multimodal AI (voice + vision)."""

import asyncio
import base64
import binascii
import json
import os
import re
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse

import asr
import tts

from dotenv import load_dotenv
load_dotenv()

HF_REPO = "litert-community/gemma-4-E2B-it-litert-lm"
HF_FILENAME = "gemma-4-E2B-it.litertlm"

LITERT_SYSTEM_PROMPT = (
    "You are a friendly, conversational AI assistant. The user is talking to you "
    "through a microphone and showing you their camera. "
    "You MUST always use the respond_to_user tool to reply. "
    "First transcribe exactly what the user said, then write your response."
)

MLX_SYSTEM_PROMPT = (
    "You are a friendly, conversational AI assistant. The user is talking to you "
    "through a microphone and may be showing you their camera. "
    "When you answer, use exactly this format:\n"
    "TRANSCRIPTION: <exactly what the user said>\n"
    "RESPONSE: <your reply in 1-4 short sentences>"
)

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
STRUCTURED_REPLY_RE = re.compile(
    r"TRANSCRIPTION:\s*(.*?)\s*RESPONSE:\s*(.*)",
    re.DOTALL | re.IGNORECASE,
)
RESPONSE_ONLY_RE = re.compile(r"RESPONSE:\s*(.*)", re.DOTALL | re.IGNORECASE)
THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def split_sentences(text: str) -> list[str]:
    """Split text into sentences for streaming TTS."""
    parts = SENTENCE_SPLIT_RE.split(text.strip())
    return [s.strip() for s in parts if s.strip()]


def prettify_model_label(model_path: str, backend_name: str) -> str:
    name = Path(model_path).stem if Path(model_path).is_file() else Path(model_path).name
    match = re.search(r"gemma-4-(e\d+b)", name, re.IGNORECASE)
    if match:
        return f"Gemma 4 {match.group(1).upper()} ({backend_name})"
    return f"{name} ({backend_name})"


def resolve_model_path() -> str:
    path = os.path.expanduser(os.environ.get("MODEL_PATH", "").strip())
    if path:
        return path
    from huggingface_hub import hf_hub_download

    print(f"Downloading {HF_REPO}/{HF_FILENAME} (first run only)...")
    return hf_hub_download(repo_id=HF_REPO, filename=HF_FILENAME)


def prepare_mlx_model_path(model_path: str) -> str:
    """Patch local processor configs that declare unsupported video processors.

    Some Qwen VLM checkpoints include a `video_processor` entry in
    `processor_config.json`. `transformers` then attempts to import torchvision
    even when this app only sends text and still images. Mirror the model into a
    temporary directory and remove the video processor declaration so MLX can
    load without a PyTorch video stack.
    """
    path = Path(model_path)
    processor_config_path = path / "processor_config.json"
    if not path.is_dir() or not processor_config_path.exists():
        return model_path

    try:
        processor_config = json.loads(processor_config_path.read_text(encoding="utf-8"))
    except Exception:
        return model_path

    if "video_processor" not in processor_config:
        return model_path

    patched_dir = Path(tempfile.mkdtemp(prefix="parlor-mlx-model-"))
    for child in path.iterdir():
        target = patched_dir / child.name
        if child.name == "processor_config.json":
            continue
        target.symlink_to(child)

    processor_config.pop("video_processor", None)
    (patched_dir / "processor_config.json").write_text(
        json.dumps(processor_config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        "Patched processor_config for MLX load "
        f"(disabled video processor): {patched_dir}"
    )
    return str(patched_dir)


def read_model_type(model_path: str) -> str:
    config_path = Path(model_path) / "config.json"
    if not config_path.exists():
        return ""
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return ""
    return str(config.get("model_type", "")).strip().lower()


def read_processor_config(model_path: str) -> dict[str, Any]:
    config_path = Path(model_path) / "processor_config.json"
    if not config_path.exists():
        return {}
    try:
        return json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def response_length_instruction(msg: dict[str, Any]) -> str:
    length = str(msg.get("response_length", "medium") or "medium").strip().lower()
    if length == "short":
        return "Keep the reply brief: about 1 to 3 short sentences."
    if length == "long":
        return (
            "Give a full, detailed reply with useful depth. "
            "Do not artificially limit the number of sentences; expand as needed to answer well."
        )
    return "Give a moderately detailed reply: usually 3 to 5 sentences."


def response_max_tokens(msg: dict[str, Any], default_max_tokens: int) -> int:
    length = str(msg.get("response_length", "medium") or "medium").strip().lower()
    if length == "short":
        return min(default_max_tokens, 128)
    if length == "long":
        return max(default_max_tokens, int(os.environ.get("LONG_MAX_TOKENS", "4096")))
    return max(default_max_tokens, 256)


def describe_user_turn(msg: dict[str, Any]) -> str:
    explicit_text = str(msg.get("text", "") or "").strip()
    length_hint = response_length_instruction(msg)
    if explicit_text and msg.get("image") and not msg.get("audio"):
        return (
            f'The user said: "{explicit_text}". '
            "They are also showing their camera. Respond to what they said, referencing what you see if relevant. "
            f"{length_hint}"
        )
    if explicit_text and not msg.get("audio") and not msg.get("image"):
        return f"{explicit_text}\n\nAdditional instruction: {length_hint}"
    if msg.get("audio") and msg.get("image"):
        return (
            "The user just spoke to you while showing their camera. "
            "Respond to what they said, referencing what you see if relevant. "
            f"{length_hint}"
        )
    if msg.get("audio"):
        return f"The user just spoke to you. Respond to what they said. {length_hint}"
    if msg.get("image"):
        return f"The user is showing you their camera. Describe what you see. {length_hint}"
    return f"{msg.get('text', 'Hello!')}\n\nAdditional instruction: {length_hint}"


def build_turn_content(msg: dict[str, Any], audio_key: str, image_key: str) -> list[dict[str, str]]:
    content: list[dict[str, str]] = []
    if msg.get("audio"):
        content.append({"type": "audio", audio_key: msg["audio"]})
    if msg.get("image"):
        content.append({"type": "image", image_key: msg["image"]})
    content.append({"type": "text", "text": describe_user_turn(msg)})
    return content


def parse_structured_reply(text: str) -> tuple[str | None, str]:
    cleaned = text.replace('<|"|>', "").strip()
    cleaned = THINK_BLOCK_RE.sub("", cleaned).replace("</think>", "").strip()
    match = STRUCTURED_REPLY_RE.search(cleaned)
    if not match:
        response_only = RESPONSE_ONLY_RE.search(cleaned)
        if response_only:
            response = response_only.group(1).strip()
            return None, response or cleaned
        paragraphs = [part.strip() for part in re.split(r"\n\s*\n", cleaned) if part.strip()]
        if paragraphs:
            cjk_start = next((i for i, part in enumerate(paragraphs) if CJK_RE.search(part)), None)
            if cjk_start is not None:
                return None, "\n\n".join(paragraphs[cjk_start:])
        return None, cleaned

    transcription = match.group(1).strip()
    response = match.group(2).strip()
    return transcription or None, response or cleaned


def write_temp_blob(encoded: str, suffix: str) -> str:
    data = base64.b64decode(encoded)
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
        handle.write(data)
        return handle.name


class InferenceResult:
    def __init__(
        self,
        transcription: str | None,
        response: str,
        elapsed: float,
        aborted: bool = False,
    ):
        self.transcription = transcription
        self.response = response
        self.elapsed = elapsed
        self.aborted = aborted


def make_unsupported_video_processor(
    base_cls: type,
    merge_size: int = 2,
    temporal_patch_size: int = 2,
):
    """Build a minimal `BaseVideoProcessor` implementation for type checks."""

    class _UnsupportedVideoProcessor(base_cls):
        model_input_names = ["pixel_values_videos", "video_grid_thw"]

        def __init__(self):
            super().__init__()
            self.merge_size = merge_size
            self.temporal_patch_size = temporal_patch_size

        def preprocess(self, videos=None, **kwargs):
            raise NotImplementedError(
                "Video tensor input is not supported by this Parlor runtime. "
                "Send still camera frames via the image channel."
            )

    return _UnsupportedVideoProcessor()


class LiteRTRuntime:
    backend_name = "LiteRT-LM"

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.display_name = prettify_model_label(model_path, self.backend_name)
        self.engine = None
        self._litert_lm = None

    def load(self) -> None:
        import litert_lm

        self._litert_lm = litert_lm
        print(f"Loading {self.display_name} from {self.model_path}...")
        self.engine = litert_lm.Engine(
            self.model_path,
            backend=litert_lm.Backend.GPU,
            vision_backend=litert_lm.Backend.GPU,
            audio_backend=litert_lm.Backend.CPU,
        )
        self.engine.__enter__()
        print("Engine loaded.")

    def create_session(self) -> dict[str, Any]:
        tool_result: dict[str, str] = {}

        def respond_to_user(transcription: str, response: str) -> str:
            tool_result["transcription"] = transcription
            tool_result["response"] = response
            return "OK"

        conversation = self.engine.create_conversation(
            messages=[{"role": "system", "content": LITERT_SYSTEM_PROMPT}],
            tools=[respond_to_user],
        )
        conversation.__enter__()
        return {"conversation": conversation, "tool_result": tool_result}

    def close_session(self, session: dict[str, Any]) -> None:
        session["conversation"].__exit__(None, None, None)

    def infer(
        self,
        session: dict[str, Any],
        msg: dict[str, Any],
        interrupted: asyncio.Event,
    ) -> InferenceResult:
        content = build_turn_content(msg, audio_key="blob", image_key="blob")
        tool_result = session["tool_result"]

        t0 = time.time()
        tool_result.clear()
        response = session["conversation"].send_message({"role": "user", "content": content})
        llm_time = time.time() - t0

        if tool_result:
            transcription, text_response = parse_structured_reply(
                "TRANSCRIPTION: "
                f"{tool_result.get('transcription', '')}\n"
                "RESPONSE: "
                f"{tool_result.get('response', '')}"
            )
            print(f"LLM ({llm_time:.2f}s) [tool] heard: {transcription!r} -> {text_response}")
            return InferenceResult(transcription, text_response, llm_time)

        text_response = response["content"][0]["text"].strip()
        print(f"LLM ({llm_time:.2f}s) [no tool]: {text_response}")
        return InferenceResult(None, text_response, llm_time, aborted=interrupted.is_set())


class MlxRuntime:
    backend_name = "MLX"

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.display_name = prettify_model_label(model_path, self.backend_name)
        self._load_model_path = model_path
        self.model = None
        self.processor = None
        self.apply_chat_template = None
        self.stream_generate = None
        self.PromptCacheState = None
        self.VisionFeatureCache = None
        self.max_tokens = int(os.environ.get("MAX_TOKENS", "192"))
        self.temperature = float(os.environ.get("TEMPERATURE", "0.2"))

    def load(self) -> None:
        try:
            from mlx_vlm import load as mlx_load
            from mlx_vlm.generate import PromptCacheState, stream_generate
            from mlx_vlm.models.base import load_chat_template
            from mlx_vlm.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor
            from mlx_vlm.prompt_utils import apply_chat_template
            from mlx_vlm.tokenizer_utils import load_tokenizer
            from mlx_vlm.utils import StoppingCriteria, load_image_processor, load_model
            from mlx_vlm.vision_cache import VisionFeatureCache
            from transformers import AutoTokenizer
            from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import (
                Qwen2VLImageProcessorPil,
            )
            from transformers.models.qwen3_vl.video_processing_qwen3_vl import (
                Qwen3VLVideoProcessor,
            )
        except ImportError as exc:
            raise RuntimeError(
                "MLX model directory detected, but mlx-vlm is not installed. "
                "Run `uv sync` in src/ after pulling the latest changes."
            ) from exc

        self.apply_chat_template = apply_chat_template
        self.stream_generate = stream_generate
        self.PromptCacheState = PromptCacheState
        self.VisionFeatureCache = VisionFeatureCache

        self._load_model_path = prepare_mlx_model_path(self.model_path)
        model_type = read_model_type(self._load_model_path)

        print(f"Loading {self.display_name} from {self.model_path}...")
        if model_type == "qwen3_5_moe":
            model_dir = Path(self._load_model_path)
            self.model = load_model(model_dir)
            processor_config = read_processor_config(model_dir)
            image_processor_config = dict(processor_config.get("image_processor") or {})
            image_processor_config.pop("image_processor_type", None)
            video_processor_config = dict(processor_config.get("video_processor") or {})
            image_processor = Qwen2VLImageProcessorPil(**image_processor_config)
            video_processor_config.pop("video_processor_type", None)
            video_processor = Qwen3VLVideoProcessor(**video_processor_config)
            tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
            load_chat_template(tokenizer, model_dir)
            detokenizer_class = load_tokenizer(model_dir, return_tokenizer=False)
            original_check = Qwen3VLProcessor.check_argument_for_proper_class

            def _accept_runtime_components(self, argument_name, argument):
                del self, argument_name, argument
                return type(None)

            Qwen3VLProcessor.check_argument_for_proper_class = _accept_runtime_components
            try:
                self.processor = Qwen3VLProcessor(
                    image_processor=image_processor,
                    tokenizer=tokenizer,
                    video_processor=video_processor,
                    chat_template=getattr(tokenizer, "chat_template", None),
                )
            finally:
                Qwen3VLProcessor.check_argument_for_proper_class = original_check
            self.processor.detokenizer = detokenizer_class(tokenizer)
            eos_token_id = getattr(self.model.config, "eos_token_id", None)
            final_eos_token_ids = (
                eos_token_id
                if eos_token_id is not None
                else getattr(tokenizer, "eos_token_ids", [])
            )
            self.processor.tokenizer.stopping_criteria = StoppingCriteria(
                final_eos_token_ids,
                self.processor.tokenizer,
            )
        else:
            self.model, self.processor = mlx_load(self._load_model_path)
        print("MLX model loaded.")

    def create_session(self) -> dict[str, Any]:
        return {
            "history": [
                {"role": "system", "content": [{"type": "text", "text": MLX_SYSTEM_PROMPT}]}
            ]
        }

    def close_session(self, session: dict[str, Any]) -> None:
        return None

    def infer(
        self,
        session: dict[str, Any],
        msg: dict[str, Any],
        interrupted: asyncio.Event,
    ) -> InferenceResult:
        current_turn = {"role": "user", "content": build_turn_content(msg, "data", "data")}
        messages = session["history"] + [current_turn]

        prompt = self.apply_chat_template(
            self.processor,
            self.model.config,
            messages,
            num_images=1 if msg.get("image") else 0,
            num_audios=1 if msg.get("audio") else 0,
        )

        image_path = None
        audio_path = None
        chunks: list[str] = []
        aborted = False
        t0 = time.time()
        prompt_cache_state = self.PromptCacheState()
        vision_cache = self.VisionFeatureCache()
        max_tokens = response_max_tokens(msg, self.max_tokens)

        try:
            if msg.get("image"):
                image_path = write_temp_blob(msg["image"], ".jpg")
            if msg.get("audio"):
                audio_path = write_temp_blob(msg["audio"], ".wav")

            for chunk in self.stream_generate(
                self.model,
                self.processor,
                prompt,
                image=image_path,
                audio=audio_path,
                max_tokens=max_tokens,
                temperature=self.temperature,
                skip_special_tokens=True,
                prompt_cache_state=prompt_cache_state,
                vision_cache=vision_cache,
            ):
                if interrupted.is_set():
                    aborted = True
                    break
                chunks.append(chunk.text)
        finally:
            for path in (image_path, audio_path):
                if path:
                    Path(path).unlink(missing_ok=True)

        llm_time = time.time() - t0
        raw_response = "".join(chunks).strip()

        if aborted:
            print(f"LLM ({llm_time:.2f}s) [mlx] interrupted")
            return InferenceResult(None, raw_response, llm_time, aborted=True)

        transcription, text_response = parse_structured_reply(raw_response)
        print(f"LLM ({llm_time:.2f}s) [mlx] heard: {transcription!r} -> {text_response}")

        history_text = transcription or str(msg.get("text", "") or "").strip() or describe_user_turn(msg)
        session["history"].append({"role": "user", "content": [{"type": "text", "text": history_text}]})
        session["history"].append(
            {"role": "assistant", "content": [{"type": "text", "text": text_response}]}
        )
        return InferenceResult(transcription, text_response, llm_time)


def build_runtime(model_path: str) -> LiteRTRuntime | MlxRuntime:
    path = Path(model_path)
    if path.is_dir():
        config_path = path / "config.json"
        has_weights = any(path.glob("*.safetensors"))
        if config_path.exists() and has_weights:
            return MlxRuntime(model_path)
        raise ValueError(f"Unsupported model directory: {model_path}")
    if path.suffix == ".litertlm":
        return LiteRTRuntime(model_path)
    raise ValueError(f"Unsupported model file: {model_path}")


MODEL_PATH = resolve_model_path()
runtime = build_runtime(MODEL_PATH)
tts_backend = None
asr_backend = None


def load_models() -> None:
    global tts_backend, asr_backend
    runtime.load()
    asr_backend = asr.load()
    tts_backend = tts.load()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await asyncio.get_event_loop().run_in_executor(None, load_models)
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def root():
    return HTMLResponse(content=(Path(__file__).parent / "index.html").read_text())


@app.get("/config")
async def config():
    return JSONResponse(
        {
            "model_label": runtime.display_name,
            "backend": runtime.backend_name,
            "model_path": MODEL_PATH,
        }
    )


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()

    session = runtime.create_session()
    interrupted = asyncio.Event()
    msg_queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

    async def receiver():
        """Receive messages from WebSocket and route them."""
        try:
            while True:
                raw = await ws.receive_text()
                msg = json.loads(raw)
                if msg.get("type") == "interrupt":
                    interrupted.set()
                    print("Client interrupted")
                else:
                    await msg_queue.put(msg)
        except WebSocketDisconnect:
            await msg_queue.put(None)

    recv_task = asyncio.create_task(receiver())

    try:
        while True:
            msg = await msg_queue.get()
            if msg is None:
                break

            interrupted.clear()
            transcription = None
            runtime_msg = dict(msg)

            if msg.get("audio") and asr_backend is not None:
                audio_path = write_temp_blob(msg["audio"], ".wav")
                try:
                    transcription = await asyncio.get_event_loop().run_in_executor(
                        None, lambda p=audio_path: asr_backend.transcribe(p)
                    )
                finally:
                    Path(audio_path).unlink(missing_ok=True)
                runtime_msg.pop("audio", None)
                runtime_msg["text"] = transcription

            try:
                result = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: runtime.infer(session, runtime_msg, interrupted)
                )
            except (ValueError, binascii.Error) as exc:
                await ws.send_text(json.dumps({"type": "error", "text": str(exc)}))
                continue

            if transcription:
                result.transcription = transcription

            if result.aborted or interrupted.is_set():
                print("Interrupted after LLM, skipping response")
                continue

            reply = {"type": "text", "text": result.response, "llm_time": round(result.elapsed, 2)}
            if result.transcription:
                reply["transcription"] = result.transcription
            await ws.send_text(json.dumps(reply))

            tts_enabled = msg.get("tts", True)

            if interrupted.is_set() or not tts_enabled:
                print("Interrupted before TTS, skipping audio")
                continue

            sentences = split_sentences(result.response)
            if not sentences:
                sentences = [result.response]

            tts_start = time.time()

            await ws.send_text(
                json.dumps(
                    {
                        "type": "audio_start",
                        "sample_rate": tts_backend.sample_rate,
                        "sentence_count": len(sentences),
                    }
                )
            )

            for i, sentence in enumerate(sentences):
                if interrupted.is_set():
                    print(f"Interrupted during TTS (sentence {i + 1}/{len(sentences)})")
                    break

                pcm = await asyncio.get_event_loop().run_in_executor(
                    None, lambda s=sentence: tts_backend.generate(s)
                )

                if interrupted.is_set():
                    break

                pcm_int16 = (pcm * 32767).clip(-32768, 32767).astype(np.int16)
                await ws.send_text(
                    json.dumps(
                        {
                            "type": "audio_chunk",
                            "audio": base64.b64encode(pcm_int16.tobytes()).decode(),
                            "index": i,
                        }
                    )
                )

            tts_time = time.time() - tts_start
            print(f"TTS ({tts_time:.2f}s): {len(sentences)} sentences")

            if not interrupted.is_set():
                await ws.send_text(json.dumps({"type": "audio_end", "tts_time": round(tts_time, 2)}))

    except WebSocketDisconnect:
        print("Client disconnected")
    finally:
        recv_task.cancel()
        runtime.close_session(session)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
