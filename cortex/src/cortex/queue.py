"""Review queue for episodes a human (or a later VLM) should look at."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal

from shared.labels import EpisodeAnnotation


class SelectionReason(StrEnum):
    FAILURE = "failure"
    UNCERTAIN_PHASE = "uncertain_phase"
    MANUAL = "manual"


ItemStatus = Literal["pending", "reviewed", "skipped"]


@dataclass
class QueueItem:
    episode_index: int
    priority: int
    reason: SelectionReason
    detail: str
    status: ItemStatus = "pending"


@dataclass
class DataSelectionQueue:
    """Higher priority is reviewed first. Ties break toward the lower episode index."""

    items: list[QueueItem] = field(default_factory=list)

    def enqueue(
        self,
        episode_index: int,
        *,
        priority: int,
        reason: SelectionReason,
        detail: str,
    ) -> QueueItem:
        for item in self.items:
            if (
                item.status == "pending"
                and item.episode_index == episode_index
                and item.reason == reason
                and item.detail == detail
            ):
                return item
        item = QueueItem(
            episode_index=episode_index,
            priority=priority,
            reason=reason,
            detail=detail,
        )
        self.items.append(item)
        return item

    def pending(self) -> list[QueueItem]:
        ranked = [item for item in self.items if item.status == "pending"]
        ranked.sort(key=lambda item: (-item.priority, item.episode_index, item.detail))
        return ranked

    def pop_next(self) -> QueueItem | None:
        ranked = self.pending()
        if not ranked:
            return None
        return ranked[0]

    def mark(self, item: QueueItem, status: ItemStatus) -> None:
        if status == "pending":
            raise ValueError("mark() accepts reviewed or skipped")
        item.status = status


def enqueue_annotation(
    queue: DataSelectionQueue,
    annotation: EpisodeAnnotation,
    *,
    uncertain_below: float = 0.5,
) -> None:
    """Queue each failure, an unsuccessful VLM judgment, and low-confidence phases."""
    for event in annotation.failures:
        queue.enqueue(
            annotation.episode_index,
            priority=100,
            reason=SelectionReason.FAILURE,
            detail=event.mode.value,
        )
    if annotation.success is False:
        note = annotation.quality_notes.strip()
        detail = "success=false" if not note else f"success=false: {note}"
        if len(detail) > 200:
            detail = detail[:197] + "..."
        queue.enqueue(
            annotation.episode_index,
            priority=100,
            reason=SelectionReason.FAILURE,
            detail=detail,
        )
    if annotation.min_phase_confidence < uncertain_below:
        queue.enqueue(
            annotation.episode_index,
            priority=40,
            reason=SelectionReason.UNCERTAIN_PHASE,
            detail=f"min_confidence={annotation.min_phase_confidence:.2f}",
        )
