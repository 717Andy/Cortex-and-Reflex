"""Pydantic schema for a VLM fold label.

The model must return segments that tile the episode, a success bit, and a
quality note. Frame coverage is checked against the episode length after the
schema accepts the object.
"""

from __future__ import annotations

import math
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from shared.labels import FailureMode, SubTask


class LabelParseError(ValueError):
    """The VLM response was not the expected segment JSON object."""


SCHEMA_VERSION = "segments-v1"


class FailureSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: FailureMode
    start_frame: int = Field(ge=0)
    end_frame: int = Field(gt=0)
    detail: str = ""

    @field_validator("detail")
    @classmethod
    def _strip_detail(cls, value: str) -> str:
        return value.strip()

    def span_ok(self) -> bool:
        return self.end_frame > self.start_frame


class SegmentSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    phase: SubTask
    start_frame: int = Field(ge=0)
    end_frame: int = Field(gt=0)
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("confidence")
    @classmethod
    def _finite_confidence(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("confidence must be finite")
        return value


class VlmResponseSchema(BaseModel):
    """Structured reply from ``VlmLabeler.complete``."""

    model_config = ConfigDict(extra="forbid")

    segments: list[SegmentSchema] = Field(min_length=1)
    success: bool
    quality_notes: str
    failures: list[FailureSchema] = Field(default_factory=list)

    @field_validator("quality_notes")
    @classmethod
    def _notes(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("quality_notes must be non-empty")
        return text


def parse_vlm_payload(payload: dict[str, Any]) -> VlmResponseSchema:
    """Validate ``payload`` and raise ``LabelParseError`` with a short message."""
    try:
        parsed = VlmResponseSchema.model_validate(payload)
    except ValidationError as exc:
        raise LabelParseError(_format_validation(exc)) from exc
    for index, failure in enumerate(parsed.failures):
        if failure.end_frame <= failure.start_frame:
            raise LabelParseError(f"failures.{index}: end_frame must be greater than start_frame")
    for index, segment in enumerate(parsed.segments):
        if segment.end_frame <= segment.start_frame:
            raise LabelParseError(f"segments.{index}: end_frame must be greater than start_frame")
    return parsed


def check_frame_coverage(parsed: VlmResponseSchema, num_frames: int) -> list[SegmentSchema]:
    """Return segments sorted by start frame, or raise when they do not tile the episode."""
    if num_frames < 1:
        raise LabelParseError("episode has no frames")
    ordered = sorted(parsed.segments, key=lambda segment: (segment.start_frame, segment.end_frame))
    cursor = 0
    for segment in ordered:
        if segment.end_frame > num_frames:
            raise LabelParseError(
                f"segment {segment.phase.value} ends at {segment.end_frame}, "
                f"episode has {num_frames} frames"
            )
        if segment.start_frame != cursor:
            raise LabelParseError(
                f"segments must cover every frame without gaps or overlap; "
                f"expected start {cursor}, found {segment.phase.value} at {segment.start_frame}"
            )
        cursor = segment.end_frame
    if cursor != num_frames:
        raise LabelParseError(f"segments end at frame {cursor}, episode has {num_frames} frames")
    for failure in parsed.failures:
        if failure.end_frame > num_frames:
            raise LabelParseError(
                f"failure {failure.mode.value} ends at {failure.end_frame}, "
                f"episode has {num_frames} frames"
            )
    return ordered


def cover_segments(num_frames: int, confidence: float) -> list[dict[str, object]]:
    """Split ``num_frames`` across the sub-tasks, dropping empty pieces."""
    if num_frames < 1:
        raise ValueError("num_frames must be >= 1")
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be between 0 and 1")
    names = [phase.value for phase in SubTask]
    bounds = [index * num_frames // len(names) for index in range(len(names) + 1)]
    segments: list[dict[str, object]] = []
    for name, start, end in zip(names, bounds[:-1], bounds[1:], strict=True):
        if end > start:
            segments.append(
                {
                    "phase": name,
                    "start_frame": start,
                    "end_frame": end,
                    "confidence": confidence,
                }
            )
    if not segments:
        raise ValueError("num_frames must produce at least one segment")
    return segments


def cover_response(
    num_frames: int,
    *,
    confidence: float,
    success: bool,
    quality_notes: str,
    failures: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    """Schema-shaped object used by the fake backend and tests."""
    return {
        "segments": cover_segments(num_frames, confidence),
        "success": success,
        "quality_notes": quality_notes,
        "failures": [] if failures is None else failures,
    }


def _format_validation(exc: ValidationError) -> str:
    parts: list[str] = []
    for error in exc.errors():
        location = ".".join(str(item) for item in error["loc"]) or "response"
        parts.append(f"{location}: {error['msg']}")
        if len(parts) == 8:
            break
    return "VLM JSON failed schema validation: " + "; ".join(parts)
