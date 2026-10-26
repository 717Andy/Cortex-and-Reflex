"""Pluggable completions for ``VlmLabeler.complete``.

``HuggingFaceVlm`` runs a local vision-language model through transformers.
``OpenAICompatibleVlm`` posts to a chat-completions endpoint described by
environment variables. ``FakeVlm`` returns schema-valid JSON for tests and for
a CPU run that should not download a model. Transport retries live here.
Schema retries and the response cache live on ``BackendVlmLabeler``, because
they need the episode length.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from cortex.images import load_label_images
from cortex.vlm_schema import SCHEMA_VERSION, cover_response

FAKE_QUALITY_NOTE = "Placeholder label. No vision model looked at this episode."
HF_MODEL_ALIASES: dict[str, str] = {
    "smolvlm": "HuggingFaceTB/SmolVLM-256M-Instruct",
    "qwen2.5-vl": "Qwen/Qwen2.5-VL-3B-Instruct",
    "qwen2.5-vl-3b": "Qwen/Qwen2.5-VL-3B-Instruct",
}
_FRAMES = re.compile(r"Frames: (\d+) ")
_RETRY_STATUS = {408, 429, 500, 502, 503, 504}


class VlmBackendError(RuntimeError):
    """The backend could not produce a completion."""


class VlmTransportError(VlmBackendError):
    """An HTTP completion failed. ``retryable`` is false for caller errors."""

    def __init__(self, message: str, *, retryable: bool = True) -> None:
        super().__init__(message)
        self.retryable = retryable


class CompletionBackend(Protocol):
    """What ``VlmLabeler.complete`` delegates to."""

    model_id: str

    def complete(self, prompt: str, image_refs: tuple[str, ...]) -> str:
        """Return the raw model text for ``prompt`` and image refs."""


class ResponseCache:
    """Disk cache of accepted raw completions, keyed by model, prompt, and refs."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def make_key(self, model_id: str, prompt: str, image_refs: tuple[str, ...]) -> str:
        payload = {
            "schema": SCHEMA_VERSION,
            "model": model_id,
            "prompt": prompt,
            "images": [_ref_token(ref) for ref in image_refs],
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()

    def get(self, key: str) -> str | None:
        path = self._path(key)
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            return None
        raw = payload.get("raw") if isinstance(payload, dict) else None
        if not isinstance(raw, str) or not raw:
            return None
        return raw

    def put(self, key: str, raw: str) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps({"raw": raw}))
        temporary.replace(path)

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"


class FakeVlm:
    """Deterministic backend. ``script`` plays canned replies, one per call."""

    def __init__(
        self,
        model_id: str = "fake",
        *,
        confidence: float = 0.42,
        success: bool = False,
        quality_notes: str = FAKE_QUALITY_NOTE,
        script: tuple[str, ...] | None = None,
    ) -> None:
        if not model_id:
            raise ValueError("model_id must be non-empty")
        self.model_id = model_id
        self.confidence = confidence
        self.success = success
        self.quality_notes = quality_notes
        self.script = script
        self.calls = 0
        self.seen_prompts: list[str] = []

    def complete(self, prompt: str, image_refs: tuple[str, ...]) -> str:
        del image_refs
        self.calls += 1
        self.seen_prompts.append(prompt)
        if self.script is not None:
            index = min(self.calls, len(self.script)) - 1
            return self.script[index]
        match = _FRAMES.search(prompt)
        frames = int(match.group(1)) if match else 1
        payload = cover_response(
            frames,
            confidence=self.confidence,
            success=self.success,
            quality_notes=self.quality_notes,
        )
        return json.dumps(payload)


class HuggingFaceVlm:
    """Local open VLM via Hugging Face transformers.

    The default ids are SmolVLM-256M and Qwen2.5-VL-3B. Transformers is imported
    on the first ``complete`` call so CI can import this module without it.
    """

    def __init__(
        self,
        model_id: str,
        *,
        device: str = "cpu",
        max_new_tokens: int = 768,
    ) -> None:
        resolved = HF_MODEL_ALIASES.get(model_id, model_id)
        if not resolved:
            raise ValueError("model_id must be non-empty")
        if not device:
            raise ValueError("device must be non-empty")
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be >= 1")
        self.model_id = resolved
        self.device = device
        self.max_new_tokens = max_new_tokens
        self._model: Any = None
        self._processor: Any = None

    def complete(self, prompt: str, image_refs: tuple[str, ...]) -> str:
        model, processor = self._ensure_loaded()
        images = load_label_images(image_refs)
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for image in images:
            content.append({"type": "image", "image": image})
        messages = [{"role": "user", "content": content}]
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        moved = _move_inputs(inputs, getattr(model, "device", self.device))
        input_ids = moved["input_ids"]
        prompt_width = int(input_ids.shape[-1])
        generated = _generate(model, moved, self.max_new_tokens)
        decoded = processor.batch_decode(generated[:, prompt_width:], skip_special_tokens=True)
        if not decoded or not str(decoded[0]).strip():
            raise VlmBackendError(f"{self.model_id} returned an empty completion")
        return str(decoded[0])

    def _ensure_loaded(self) -> tuple[Any, Any]:
        if self._model is not None and self._processor is not None:
            return self._model, self._processor
        processor, model = _load_transformers(self.model_id, self.device)
        self._processor = processor
        self._model = model
        return model, processor


class OpenAICompatibleVlm:
    """POST ``{base_url}/chat/completions`` using the OpenAI vision message shape."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout_s: float = 120.0,
        transport_attempts: int = 3,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        if not model:
            raise ValueError("model must be non-empty")
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        if transport_attempts < 1:
            raise ValueError("transport_attempts must be >= 1")
        self.model_id = model
        self.base_url = _normalize_base_url(base_url)
        self.api_key = api_key
        self.timeout_s = timeout_s
        self.transport_attempts = transport_attempts
        self._sleep = sleep or _default_sleep

    @classmethod
    def from_env(
        cls,
        model: str | None = None,
        *,
        timeout_s: float = 120.0,
        transport_attempts: int = 3,
        sleep: Callable[[float], None] | None = None,
    ) -> OpenAICompatibleVlm:
        """Read ``CORTEX_VLM_*`` with ``OPENAI_*`` as a fallback."""
        base_url = os.environ.get("CORTEX_VLM_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
        api_key = os.environ.get("CORTEX_VLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
        resolved = model or os.environ.get("CORTEX_VLM_MODEL") or os.environ.get("OPENAI_MODEL")
        if not base_url:
            raise ValueError("Set CORTEX_VLM_BASE_URL or OPENAI_BASE_URL for the openai backend")
        if not resolved:
            raise ValueError("Set CORTEX_VLM_MODEL or OPENAI_MODEL, or pass --model")
        return cls(
            base_url=base_url,
            model=resolved,
            api_key=api_key or None,
            timeout_s=timeout_s,
            transport_attempts=transport_attempts,
            sleep=sleep,
        )

    def complete(self, prompt: str, image_refs: tuple[str, ...]) -> str:
        body = _chat_body(self.model_id, prompt, image_refs)
        last: VlmTransportError | None = None
        for attempt in range(self.transport_attempts):
            try:
                return self._post(body)
            except VlmTransportError as exc:
                last = exc
                if not exc.retryable or attempt + 1 == self.transport_attempts:
                    raise
                self._sleep(min(8.0, 0.5 * (2**attempt)))
        assert last is not None
        raise last

    def _post(self, body: dict[str, Any]) -> str:
        data = json.dumps(body).encode()
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=data,
            headers=_headers(self.api_key),
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise VlmTransportError(
                f"HTTP {exc.code}: {detail}",
                retryable=exc.code in _RETRY_STATUS,
            ) from exc
        except urllib.error.URLError as exc:
            raise VlmTransportError(f"request failed: {exc.reason}", retryable=True) from exc
        except TimeoutError as exc:
            raise VlmTransportError("request timed out", retryable=True) from exc
        try:
            payload = json.loads(raw.decode())
        except json.JSONDecodeError as exc:
            raise VlmTransportError("completion response was not JSON", retryable=True) from exc
        if not isinstance(payload, dict):
            raise VlmTransportError("completion response was not a JSON object", retryable=True)
        return _message_text(payload)


def _headers(api_key: str | None) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _chat_body(model: str, prompt: str, image_refs: tuple[str, ...]) -> dict[str, Any]:
    images = load_label_images(image_refs)
    if images:
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for image in images:
            content.append({"type": "image_url", "image_url": {"url": _jpeg_data_url(image)}})
        message: dict[str, Any] = {"role": "user", "content": content}
    else:
        message = {"role": "user", "content": prompt}
    return {"model": model, "temperature": 0, "messages": [message]}


def _jpeg_data_url(image: Any) -> str:
    import base64
    import io

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _message_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise VlmTransportError("response missing choices", retryable=True)
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise VlmTransportError("response missing message", retryable=True)
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(str(item["text"]))
        text = "\n".join(part for part in parts if part.strip())
        if text.strip():
            return text
    raise VlmTransportError("response content was empty", retryable=True)


def _normalize_base_url(base_url: str) -> str:
    text = base_url.strip().rstrip("/")
    if not text.startswith(("http://", "https://")):
        raise ValueError("base URL must start with http:// or https://")
    return text


def _default_sleep(seconds: float) -> None:
    import time

    time.sleep(seconds)


def _ref_token(ref: str) -> str:
    path = Path(ref)
    if path.is_file():
        stat = path.stat()
        return f"{ref}:{stat.st_size}:{stat.st_mtime_ns}"
    return ref


def _load_transformers(model_id: str, device: str) -> tuple[Any, Any]:
    try:
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except ImportError as exc:
        raise VlmBackendError(
            "The Hugging Face backend needs transformers>=4.51. "
            "Install it with: uv pip install 'transformers>=4.51' "
            "or pass --backend fake."
        ) from exc
    processor = AutoProcessor.from_pretrained(model_id)
    kwargs: dict[str, Any] = {}
    try:
        import torch

        kwargs["torch_dtype"] = torch.float32 if device == "cpu" else torch.float16
    except ImportError:
        pass
    model = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
    if hasattr(model, "to"):
        model.to(device)
    if hasattr(model, "eval"):
        model.eval()
    return processor, model


def _move_inputs(inputs: Any, device: Any) -> dict[str, Any]:
    mapping = dict(inputs)
    return {key: _to_device(value, device) for key, value in mapping.items()}


def _to_device(value: Any, device: Any) -> Any:
    if hasattr(value, "to"):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_device(item, device) for item in value]
    return value


def _generate(model: Any, inputs: dict[str, Any], max_new_tokens: int) -> Any:
    try:
        import torch
    except ImportError:
        return model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    with torch.inference_mode():
        return model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
