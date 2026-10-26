from __future__ import annotations

import io
import json
import os
import urllib.error
import urllib.request
from email.message import Message
from pathlib import Path

import pytest

from cortex.images import load_label_images
from cortex.vlm_backend import (
    FAKE_QUALITY_NOTE,
    FakeVlm,
    HuggingFaceVlm,
    OpenAICompatibleVlm,
    VlmBackendError,
    VlmTransportError,
)
from cortex.vlm_schema import cover_response


class _Body:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return self._payload

    def __enter__(self) -> _Body:
        return self

    def __exit__(self, *args: object) -> None:
        return None


def _http_error(code: int, body: bytes = b"nope") -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        "http://127.0.0.1/v1/chat/completions",
        code,
        "err",
        Message(),
        io.BytesIO(body),
    )


def test_openai_retries_retryable_status(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    def fake_urlopen(request: urllib.request.Request, timeout: float = 0) -> _Body:
        del timeout
        calls["n"] += 1
        assert request.get_header("Authorization") == "Bearer secret"
        if calls["n"] == 1:
            raise _http_error(503, b"busy")
        body = json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode()
        return _Body(body)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    backend = OpenAICompatibleVlm(
        base_url="http://127.0.0.1/v1",
        model="local-vlm",
        api_key="secret",
        transport_attempts=2,
        sleep=lambda _seconds: None,
    )
    assert backend.complete("label", ()) == "ok"
    assert calls["n"] == 2


def test_openai_does_not_retry_client_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    def fake_urlopen(request: urllib.request.Request, timeout: float = 0) -> _Body:
        del request, timeout
        calls["n"] += 1
        raise _http_error(400, b"bad request")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    backend = OpenAICompatibleVlm(
        base_url="http://127.0.0.1/v1",
        model="local-vlm",
        transport_attempts=3,
        sleep=lambda _seconds: None,
    )
    with pytest.raises(VlmTransportError, match="HTTP 400"):
        backend.complete("label", ())
    assert calls["n"] == 1


def test_openai_from_env_and_image_part(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pytest.importorskip("PIL")
    from PIL import Image

    image_path = tmp_path / "frame.png"
    Image.new("RGB", (2, 2), color=(255, 0, 0)).save(image_path)
    captured: list[bytes] = []

    def fake_urlopen(request: urllib.request.Request, timeout: float = 0) -> _Body:
        del timeout
        assert request.data is not None
        captured.append(request.data)
        body = json.dumps({"choices": [{"message": {"content": "noted"}}]}).encode()
        return _Body(body)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setenv("CORTEX_VLM_BASE_URL", "http://127.0.0.1/v1")
    monkeypatch.setenv("CORTEX_VLM_MODEL", "env-model")
    monkeypatch.delenv("CORTEX_VLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    backend = OpenAICompatibleVlm.from_env(sleep=lambda _seconds: None)
    assert backend.model_id == "env-model"
    assert backend.complete("see the frame", (str(image_path),)) == "noted"
    payload = json.loads(captured[0])
    content = payload["messages"][0]["content"]
    assert content[0]["type"] == "text"
    assert content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_openai_env_requires_a_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CORTEX_VLM_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    with pytest.raises(ValueError, match="CORTEX_VLM_BASE_URL"):
        OpenAICompatibleVlm.from_env(model="anything")


def test_hf_aliases_and_missing_transformers() -> None:
    assert HuggingFaceVlm("qwen2.5-vl").model_id == "Qwen/Qwen2.5-VL-3B-Instruct"
    backend = HuggingFaceVlm("smolvlm")
    assert backend.model_id == "HuggingFaceTB/SmolVLM-256M-Instruct"
    try:
        import transformers  # noqa: F401
    except ImportError:
        with pytest.raises(VlmBackendError, match="transformers"):
            backend.complete("hi", ())
        return
    pytest.skip("transformers is installed; the missing-package path runs when it is not")


def test_hf_complete_uses_an_injected_model() -> None:
    torch = pytest.importorskip("torch")
    backend = HuggingFaceVlm("smolvlm", max_new_tokens=8)

    class Processor:
        messages: list[object]

        def apply_chat_template(
            self, messages: list[object], **kwargs: object
        ) -> dict[str, object]:
            del kwargs
            self.messages = messages
            return {"input_ids": torch.ones(1, 3, dtype=torch.long)}

        def batch_decode(self, tokens: object, skip_special_tokens: bool = True) -> list[str]:
            del skip_special_tokens
            assert tuple(tokens.shape) == (1, 2)  # type: ignore[attr-defined]
            return ['{"segments":[]}']

    class Model:
        device = "cpu"

        def generate(self, **kwargs: object) -> object:
            assert kwargs["max_new_tokens"] == 8
            assert kwargs["do_sample"] is False
            return torch.ones(1, 5, dtype=torch.long)

    processor = Processor()
    backend._processor = processor
    backend._model = Model()
    text = backend.complete("label this", ())
    assert "segments" in text
    content = processor.messages[0]["content"]  # type: ignore[index]
    assert content[0]["text"] == "label this"


def test_load_label_images_reads_png_and_skips_missing(tmp_path: Path) -> None:
    pytest.importorskip("PIL")
    from PIL import Image

    path = tmp_path / "frame.png"
    Image.new("RGB", (4, 4), color=(0, 255, 0)).save(path)
    images = load_label_images((str(path), str(tmp_path / "missing.png")))
    assert len(images) == 1
    assert images[0].size == (4, 4)


def test_fake_backend_covers_short_episodes() -> None:
    backend = FakeVlm()
    raw = backend.complete("Frames: 1 at 30 fps.\n", ())
    payload = json.loads(raw)
    assert payload["quality_notes"] == FAKE_QUALITY_NOTE
    assert payload["segments"][0]["end_frame"] == 1
    assert cover_response(1, confidence=0.2, success=True, quality_notes="ok")["segments"]


@pytest.mark.slow
def test_optional_real_vlm_completion() -> None:
    if os.environ.get("CORTEX_RUN_SLOW_VLM") != "1":
        pytest.skip("optional real-model test; set CORTEX_RUN_SLOW_VLM=1")
    pytest.importorskip("transformers")
    backend = HuggingFaceVlm("HuggingFaceTB/SmolVLM-256M-Instruct", device="cpu", max_new_tokens=32)
    text = backend.complete('Reply with a JSON object {"ok": true}', ())
    assert text.strip()
