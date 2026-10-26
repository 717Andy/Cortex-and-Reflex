"""VLM labeling.

``complete`` is the backend hook. Prompt construction, schema checks, schema
retries, and the response cache stay here so every client returns the same
``EpisodeAnnotation``. Segment confidence is copied onto the coarse phases,
which is what the review queue already thresholds.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from cortex.vlm_backend import (
    CompletionBackend,
    FakeVlm,
    HuggingFaceVlm,
    OpenAICompatibleVlm,
    ResponseCache,
)
from cortex.vlm_schema import LabelParseError, check_frame_coverage, parse_vlm_payload
from shared.episodes import EpisodeRecord
from shared.labels import (
    EpisodeAnnotation,
    FailureEvent,
    FailureMode,
    FoldPhase,
    PhaseSpan,
    SegmentSpan,
    SubTask,
    segments_to_phases,
)

__all__ = [
    "BackendVlmLabeler",
    "LabelParseError",
    "VlmLabeler",
    "annotation_to_dict",
    "make_vlm_labeler",
]


class VlmLabeler(ABC):
    """Vision-language labeler. Subclasses implement ``complete``."""

    def __init__(self, model_id: str) -> None:
        if not model_id:
            raise ValueError("model_id must be non-empty")
        self.model_id = model_id

    @property
    def name(self) -> str:
        return f"vlm:{self.model_id}"

    def label(self, episode: EpisodeRecord) -> EpisodeAnnotation:
        prompt = self.build_prompt(episode)
        raw = self.complete(prompt, episode.image_refs)
        return self.parse_response(episode, raw)

    def build_prompt(self, episode: EpisodeRecord) -> str:
        segments = ", ".join(phase.value for phase in SubTask)
        failures = ", ".join(mode.value for mode in FailureMode)
        return (
            "You label one bimanual SO-101 towel-folding episode.\n"
            f"Task: {episode.task}\n"
            f"Episode: {episode.episode_index}\n"
            f"Frames: {episode.num_frames} at {episode.fps} fps.\n"
            f"Image refs, in order: {list(episode.image_refs) or '(none attached)'}.\n"
            "Local image files and decodable videos among those refs are attached "
            "in ref order (up to two frames per video). When none are attached, "
            "label from this description and lower your confidence.\n"
            f"Segment phases, in order when they occur: {segments}.\n"
            f"Failure modes, only when the video shows them: {failures}.\n"
            "Reply with one JSON object and no other text. "
            "The example shows the shape, not the frame counts for this episode.\n"
            '{"segments":[{"phase":"grasp","start_frame":0,"end_frame":1,'
            '"confidence":0.0},{"phase":"flatten","start_frame":1,"end_frame":2,'
            '"confidence":0.0},{"phase":"fold1","start_frame":2,"end_frame":3,'
            '"confidence":0.0},{"phase":"fold2","start_frame":3,"end_frame":4,'
            '"confidence":0.0},{"phase":"release","start_frame":4,"end_frame":5,'
            '"confidence":0.0}],"success":false,'
            '"quality_notes":"one sentence on fold quality",'
            '"failures":[{"mode":"missed_grasp","start_frame":0,"end_frame":1,'
            '"detail":"why"}]}\n'
            "Frame ends are exclusive. Cover every frame from 0 to "
            f"{episode.num_frames} with exactly one segment, no gaps or overlap. "
            "success is true only when the towel ends folded and released. "
            "quality_notes is one non-empty sentence. "
            "Use an empty failures list when the fold looks clean."
        )

    @abstractmethod
    def complete(self, prompt: str, image_refs: tuple[str, ...]) -> str:
        """Send ``prompt`` and image refs to the VLM and return the raw text."""

    def parse_response(self, episode: EpisodeRecord, raw: str) -> EpisodeAnnotation:
        payload = _extract_json(raw)
        if "segments" in payload:
            return self._from_schema(episode, payload)
        if "phases" in payload:
            return self._from_legacy(episode, payload)
        raise LabelParseError("JSON must contain segments")

    def _from_schema(self, episode: EpisodeRecord, payload: dict[str, Any]) -> EpisodeAnnotation:
        parsed = parse_vlm_payload(payload)
        ordered = check_frame_coverage(parsed, episode.num_frames)
        segments = tuple(
            SegmentSpan(item.phase, item.start_frame, item.end_frame, item.confidence)
            for item in ordered
        )
        failures = tuple(
            FailureEvent(item.mode, item.start_frame, item.end_frame, item.detail)
            for item in parsed.failures
        )
        return EpisodeAnnotation(
            episode_index=episode.episode_index,
            labeler=self.name,
            phases=segments_to_phases(segments),
            failures=failures,
            segments=segments,
            success=parsed.success,
            quality_notes=parsed.quality_notes,
        )

    def _from_legacy(self, episode: EpisodeRecord, payload: dict[str, Any]) -> EpisodeAnnotation:
        phases = _parse_phases(payload.get("phases"), episode.num_frames)
        failures = _parse_failures(payload.get("failures"), episode.num_frames)
        return EpisodeAnnotation(
            episode_index=episode.episode_index,
            labeler=self.name,
            phases=tuple(phases),
            failures=tuple(failures),
        )


class BackendVlmLabeler(VlmLabeler):
    """``complete`` calls the backend. ``label`` retries schema failures and caches hits."""

    def __init__(
        self,
        backend: CompletionBackend,
        *,
        attempts: int = 3,
        cache: ResponseCache | None = None,
    ) -> None:
        if attempts < 1:
            raise ValueError("attempts must be >= 1")
        super().__init__(backend.model_id)
        self.backend = backend
        self.attempts = attempts
        self.cache = cache

    def complete(self, prompt: str, image_refs: tuple[str, ...]) -> str:
        return self.backend.complete(prompt, image_refs)

    def label(self, episode: EpisodeRecord) -> EpisodeAnnotation:
        prompt = self.build_prompt(episode)
        cache_key = None
        if self.cache is not None:
            cache_key = self.cache.make_key(self.model_id, prompt, episode.image_refs)
            cached = self.cache.get(cache_key)
            if cached is not None:
                try:
                    return self.parse_response(episode, cached)
                except LabelParseError:
                    pass
        current = prompt
        last_error: LabelParseError | None = None
        for _attempt in range(self.attempts):
            raw = self.complete(current, episode.image_refs)
            try:
                annotation = self.parse_response(episode, raw)
            except LabelParseError as exc:
                last_error = exc
                current = (
                    f"{prompt}\n\nThe previous reply failed validation: {exc}. "
                    "Reply with one JSON object and no other text."
                )
                continue
            if self.cache is not None and cache_key is not None:
                self.cache.put(cache_key, raw)
            return annotation
        assert last_error is not None
        raise last_error


def make_vlm_labeler(
    backend: str,
    *,
    model: str | None = None,
    retries: int = 2,
    cache_dir: Path | None = None,
    device: str = "cpu",
    max_new_tokens: int = 768,
    timeout_s: float = 120.0,
) -> BackendVlmLabeler:
    """Build a labeler. ``retries`` is how many extra schema attempts follow the first."""
    if retries < 0:
        raise ValueError("retries must be >= 0")
    client = _make_backend(
        backend,
        model=model,
        device=device,
        max_new_tokens=max_new_tokens,
        timeout_s=timeout_s,
    )
    cache = None if cache_dir is None else ResponseCache(cache_dir)
    return BackendVlmLabeler(client, attempts=retries + 1, cache=cache)


def annotation_to_dict(annotation: EpisodeAnnotation) -> dict[str, Any]:
    """JSON-ready view of one annotation, including segments and the review fields."""
    return {
        "episode_index": annotation.episode_index,
        "labeler": annotation.labeler,
        "success": annotation.success,
        "quality_notes": annotation.quality_notes,
        "segments": [
            {
                "phase": segment.phase.value,
                "start_frame": segment.start_frame,
                "end_frame": segment.end_frame,
                "confidence": segment.confidence,
            }
            for segment in annotation.segments
        ],
        "phases": [
            {
                "phase": span.phase.value,
                "start_frame": span.start_frame,
                "end_frame": span.end_frame,
                "confidence": span.confidence,
            }
            for span in annotation.phases
        ],
        "failures": [
            {
                "mode": event.mode.value,
                "start_frame": event.start_frame,
                "end_frame": event.end_frame,
                "detail": event.detail,
            }
            for event in annotation.failures
        ],
    }


def _make_backend(
    backend: str,
    *,
    model: str | None,
    device: str,
    max_new_tokens: int,
    timeout_s: float,
) -> CompletionBackend:
    if backend == "fake":
        return FakeVlm(model or "fake")
    if backend == "hf":
        return HuggingFaceVlm(model or "smolvlm", device=device, max_new_tokens=max_new_tokens)
    if backend == "openai":
        return OpenAICompatibleVlm.from_env(model, timeout_s=timeout_s)
    raise ValueError(f"unknown VLM backend {backend!r}")


def _extract_json(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines)
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise LabelParseError("VLM response did not contain a JSON object")
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise LabelParseError("VLM response was not valid JSON") from exc
    if not isinstance(payload, dict):
        raise LabelParseError("VLM response JSON must be an object")
    return payload


def _parse_phases(value: object, num_frames: int) -> list[PhaseSpan]:
    if not isinstance(value, list) or not value:
        raise LabelParseError("phases must be a non-empty list")
    spans: list[PhaseSpan] = []
    for item in value:
        if not isinstance(item, dict):
            raise LabelParseError("each phase must be an object")
        try:
            phase = FoldPhase(str(item["phase"]))
            start = int(item["start_frame"])
            end = int(item["end_frame"])
            confidence = float(item["confidence"])
        except (KeyError, ValueError) as exc:
            raise LabelParseError(f"invalid phase entry: {item!r}") from exc
        if end > num_frames:
            raise LabelParseError(f"phase {phase} ends at {end}, episode has {num_frames} frames")
        spans.append(PhaseSpan(phase, start, end, confidence))
    return spans


def _parse_failures(value: object, num_frames: int) -> list[FailureEvent]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise LabelParseError("failures must be a list")
    events: list[FailureEvent] = []
    for item in value:
        if not isinstance(item, dict):
            raise LabelParseError("each failure must be an object")
        try:
            mode = FailureMode(str(item["mode"]))
            start = int(item["start_frame"])
            end = int(item["end_frame"])
            detail = str(item.get("detail", ""))
        except (KeyError, ValueError) as exc:
            raise LabelParseError(f"invalid failure entry: {item!r}") from exc
        if end > num_frames:
            raise LabelParseError(f"failure {mode} ends at {end}, episode has {num_frames} frames")
        events.append(FailureEvent(mode, start, end, detail))
    return events
