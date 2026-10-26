"""VLM labeling seam.

``complete`` is the only method a backend has to implement. Prompt construction
and JSON parsing stay here so every client returns the same ``EpisodeAnnotation``.
No network client ships in this package.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any

from shared.episodes import EpisodeRecord
from shared.labels import (
    EpisodeAnnotation,
    FailureEvent,
    FailureMode,
    FoldPhase,
    PhaseSpan,
)


class LabelParseError(ValueError):
    """The VLM response was not the expected phase/failure JSON object."""


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
        phases = ", ".join(phase.value for phase in FoldPhase)
        failures = ", ".join(mode.value for mode in FailureMode)
        return (
            "You label one bimanual SO-101 cloth-folding episode.\n"
            f"Task: {episode.task}\n"
            f"Episode: {episode.episode_index}\n"
            f"Frames: {episode.num_frames} at {episode.fps} fps.\n"
            f"Image refs, in order: {list(episode.image_refs) or '(none attached)'}.\n"
            f"Phases, in order when they occur: {phases}.\n"
            f"Failure modes, only when the video shows them: {failures}.\n"
            "Reply with one JSON object and no other text:\n"
            '{"phases":[{"phase":"grasp_corner","start_frame":0,"end_frame":1,'
            '"confidence":0.0}],"failures":[{"mode":"missed_grasp","start_frame":0,'
            '"end_frame":1,"detail":"why"}]}\n'
            "Frame ends are exclusive. Cover every frame with exactly one phase. "
            "Use an empty failures list when the fold looks clean."
        )

    @abstractmethod
    def complete(self, prompt: str, image_refs: tuple[str, ...]) -> str:
        """Send ``prompt`` and image refs to the VLM and return the raw text."""

    def parse_response(self, episode: EpisodeRecord, raw: str) -> EpisodeAnnotation:
        payload = _extract_json(raw)
        phases = _parse_phases(payload.get("phases"), episode.num_frames)
        failures = _parse_failures(payload.get("failures"), episode.num_frames)
        return EpisodeAnnotation(
            episode_index=episode.episode_index,
            labeler=self.name,
            phases=tuple(phases),
            failures=tuple(failures),
        )


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
