"""Fold-phase and failure labels shared by labeling and later reward models.

Phases follow the towel (and later T-shirt) routine: grasp a corner, flatten,
fold, release. VLM labels use finer sub-tasks (grasp, flatten, fold1, fold2,
release) that map back onto those phases. Failure modes are the ones the data
engine is meant to mine.
``arm_collision`` and ``timeout`` are separate so a review queue can tell a
contact spike from an episode that simply ran long. ``drop`` and ``missed_box``
are likewise separate: the cloth left the hand, or the hand never opened over
the receiving area.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class FoldPhase(StrEnum):
    GRASP_CORNER = "grasp_corner"
    FLATTEN = "flatten"
    FOLD = "fold"
    RELEASE = "release"


class SubTask(StrEnum):
    """Fine segments a VLM assigns. The stub labeler still uses ``FoldPhase``."""

    GRASP = "grasp"
    FLATTEN = "flatten"
    FOLD1 = "fold1"
    FOLD2 = "fold2"
    RELEASE = "release"


# Coarse phase used by the review dashboard. fold1 and fold2 share ``fold``.
SUBTASK_TO_PHASE: dict[SubTask, FoldPhase] = {
    SubTask.GRASP: FoldPhase.GRASP_CORNER,
    SubTask.FLATTEN: FoldPhase.FLATTEN,
    SubTask.FOLD1: FoldPhase.FOLD,
    SubTask.FOLD2: FoldPhase.FOLD,
    SubTask.RELEASE: FoldPhase.RELEASE,
}


class FailureMode(StrEnum):
    MISSED_GRASP = "missed_grasp"
    CRUMPLED_CORNER = "crumpled_corner"
    MISALIGNED_FOLD = "misaligned_fold"
    DOUBLE_LAYER_GRASP = "double_layer_grasp"
    ARM_COLLISION = "arm_collision"
    TIMEOUT = "timeout"
    DROP = "drop"
    MISSED_BOX = "missed_box"


@dataclass(frozen=True)
class PhaseSpan:
    phase: FoldPhase
    start_frame: int
    end_frame: int
    confidence: float

    def __post_init__(self) -> None:
        if self.start_frame < 0 or self.end_frame <= self.start_frame:
            raise ValueError(
                f"{self.phase} span must satisfy 0 <= start < end, got "
                f"{self.start_frame}:{self.end_frame}"
            )
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")

    @property
    def num_frames(self) -> int:
        return self.end_frame - self.start_frame


@dataclass(frozen=True)
class SegmentSpan:
    """One VLM sub-task span. Frame ends are exclusive."""

    phase: SubTask
    start_frame: int
    end_frame: int
    confidence: float

    def __post_init__(self) -> None:
        if self.start_frame < 0 or self.end_frame <= self.start_frame:
            raise ValueError(
                f"{self.phase} segment must satisfy 0 <= start < end, got "
                f"{self.start_frame}:{self.end_frame}"
            )
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")


def segments_to_phases(segments: tuple[SegmentSpan, ...]) -> tuple[PhaseSpan, ...]:
    """Map fine segments onto the coarse phases the dashboard already counts."""
    return tuple(
        PhaseSpan(
            SUBTASK_TO_PHASE[segment.phase],
            segment.start_frame,
            segment.end_frame,
            segment.confidence,
        )
        for segment in segments
    )


@dataclass(frozen=True)
class FailureEvent:
    mode: FailureMode
    start_frame: int
    end_frame: int
    detail: str

    def __post_init__(self) -> None:
        if self.start_frame < 0 or self.end_frame <= self.start_frame:
            raise ValueError(
                f"{self.mode} event must satisfy 0 <= start < end, got "
                f"{self.start_frame}:{self.end_frame}"
            )


@dataclass(frozen=True)
class EpisodeAnnotation:
    episode_index: int
    labeler: str
    phases: tuple[PhaseSpan, ...]
    failures: tuple[FailureEvent, ...]
    segments: tuple[SegmentSpan, ...] = ()
    success: bool | None = None
    quality_notes: str = ""

    @property
    def min_phase_confidence(self) -> float:
        if not self.phases:
            return 0.0
        return min(span.confidence for span in self.phases)

    @property
    def mean_phase_confidence(self) -> float:
        if not self.phases:
            return 0.0
        return sum(span.confidence for span in self.phases) / len(self.phases)


def annotation_from_mapping(data: dict[str, object]) -> EpisodeAnnotation:
    """Rebuild an annotation from the JSON object ``annotation_to_dict`` writes."""
    episode_index = data.get("episode_index")
    labeler = data.get("labeler")
    if isinstance(episode_index, bool) or not isinstance(episode_index, int):
        raise ValueError("episode_index must be an int")
    if not isinstance(labeler, str) or not labeler:
        raise ValueError("labeler must be a non-empty string")
    success = data.get("success")
    if success is not None and not isinstance(success, bool):
        raise ValueError("success must be a bool or null")
    notes = data.get("quality_notes", "")
    if not isinstance(notes, str):
        raise ValueError("quality_notes must be a string")
    return EpisodeAnnotation(
        episode_index=episode_index,
        labeler=labeler,
        phases=_phase_spans(data.get("phases", ())),
        failures=_failure_events(data.get("failures", ())),
        segments=_segment_spans(data.get("segments", ())),
        success=success,
        quality_notes=notes,
    )


def _phase_spans(value: object) -> tuple[PhaseSpan, ...]:
    return tuple(
        PhaseSpan(
            phase=FoldPhase(str(item["phase"])),
            start_frame=_span_int(item, "start_frame"),
            end_frame=_span_int(item, "end_frame"),
            confidence=_span_float(item, "confidence"),
        )
        for item in _span_list(value, "phases")
    )


def _segment_spans(value: object) -> tuple[SegmentSpan, ...]:
    return tuple(
        SegmentSpan(
            phase=SubTask(str(item["phase"])),
            start_frame=_span_int(item, "start_frame"),
            end_frame=_span_int(item, "end_frame"),
            confidence=_span_float(item, "confidence"),
        )
        for item in _span_list(value, "segments")
    )


def _failure_events(value: object) -> tuple[FailureEvent, ...]:
    events: list[FailureEvent] = []
    for item in _span_list(value, "failures"):
        detail = item.get("detail", "")
        if not isinstance(detail, str):
            raise ValueError("failure detail must be a string")
        events.append(
            FailureEvent(
                mode=FailureMode(str(item["mode"])),
                start_frame=_span_int(item, "start_frame"),
                end_frame=_span_int(item, "end_frame"),
                detail=detail,
            )
        )
    return tuple(events)


def _span_list(value: object, name: str) -> list[dict[str, object]]:
    if value is None:
        return []
    if not isinstance(value, list | tuple):
        raise ValueError(f"{name} must be a list")
    items: list[dict[str, object]] = []
    for item in value:
        if not isinstance(item, dict):
            raise ValueError(f"each {name} entry must be an object")
        items.append(item)
    return items


def _span_int(item: dict[str, object], name: str) -> int:
    value = item.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an int")
    return value


def _span_float(item: dict[str, object], name: str) -> float:
    value = item.get(name)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be a number")
    return float(value)
