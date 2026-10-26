"""Fold-phase and failure labels shared by labeling and later reward models.

Phases follow the towel (and later T-shirt) routine: grasp a corner, flatten,
fold, release. Failure modes are the ones the data engine is meant to mine.
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

    @property
    def min_phase_confidence(self) -> float:
        if not self.phases:
            return 0.0
        return min(span.confidence for span in self.phases)
