"""Text dashboard over labeled episodes and the review queue."""

from __future__ import annotations

from dataclasses import dataclass

from cortex.queue import DataSelectionQueue, SelectionReason
from shared.labels import EpisodeAnnotation, FailureMode, FoldPhase


@dataclass(frozen=True)
class Dashboard:
    episodes_labeled: int
    phase_counts: dict[str, int]
    failure_counts: dict[str, int]
    queue_pending: int
    queue_by_reason: dict[str, int]


def build_dashboard(
    annotations: list[EpisodeAnnotation],
    queue: DataSelectionQueue,
) -> Dashboard:
    phase_counts = {phase.value: 0 for phase in FoldPhase}
    failure_counts = {mode.value: 0 for mode in FailureMode}
    for annotation in annotations:
        seen_phases = {span.phase.value for span in annotation.phases}
        for phase in seen_phases:
            phase_counts[phase] = phase_counts.get(phase, 0) + 1
        for event in annotation.failures:
            failure_counts[event.mode.value] = failure_counts.get(event.mode.value, 0) + 1
    pending = queue.pending()
    by_reason = {reason.value: 0 for reason in SelectionReason}
    for item in pending:
        by_reason[item.reason.value] = by_reason.get(item.reason.value, 0) + 1
    return Dashboard(
        episodes_labeled=len(annotations),
        phase_counts=phase_counts,
        failure_counts=failure_counts,
        queue_pending=len(pending),
        queue_by_reason=by_reason,
    )


def render_dashboard(dashboard: Dashboard) -> str:
    lines = [
        "Cortex dashboard",
        f"episodes labeled: {dashboard.episodes_labeled}",
        f"queue pending: {dashboard.queue_pending}",
        "phases:",
    ]
    for phase in FoldPhase:
        lines.append(f"  {phase.value}: {dashboard.phase_counts.get(phase.value, 0)}")
    lines.append("failures:")
    for mode in FailureMode:
        lines.append(f"  {mode.value}: {dashboard.failure_counts.get(mode.value, 0)}")
    lines.append("queue:")
    for reason in SelectionReason:
        lines.append(f"  {reason.value}: {dashboard.queue_by_reason.get(reason.value, 0)}")
    return "\n".join(lines)
