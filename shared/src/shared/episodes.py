"""In-memory view of one bimanual episode.

Video decoding stays outside this type. Callers that already have joint
trajectories (a parquet sample, a teleop buffer, a test) build an
``EpisodeRecord`` and hand it to Cortex labeling or Reflex open-loop checks.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from shared.contract import ACTION_DIM


@dataclass(frozen=True)
class EpisodeRecord:
    episode_index: int
    task: str
    fps: int
    states: tuple[tuple[float, ...], ...]
    actions: tuple[tuple[float, ...], ...]
    image_refs: tuple[str, ...] = ()

    @property
    def num_frames(self) -> int:
        return len(self.states)


def make_episode_record(
    *,
    episode_index: int,
    task: str,
    fps: int,
    states: Sequence[Sequence[float]],
    actions: Sequence[Sequence[float]],
    image_refs: Sequence[str] = (),
) -> EpisodeRecord:
    state_matrix = _as_matrix(states, "states")
    action_matrix = _as_matrix(actions, "actions")
    if len(state_matrix) != len(action_matrix):
        raise ValueError(
            "states and actions have different lengths "
            f"({len(state_matrix)} vs {len(action_matrix)})"
        )
    if fps <= 0:
        raise ValueError("fps must be positive")
    if not task:
        raise ValueError("task must be non-empty")
    if episode_index < 0:
        raise ValueError("episode_index must be >= 0")
    return EpisodeRecord(
        episode_index=episode_index,
        task=task,
        fps=fps,
        states=state_matrix,
        actions=action_matrix,
        image_refs=tuple(image_refs),
    )


def _as_matrix(rows: Sequence[Sequence[float]], label: str) -> tuple[tuple[float, ...], ...]:
    if not rows:
        raise ValueError(f"{label} must contain at least one frame")
    matrix: list[tuple[float, ...]] = []
    for row_index, row in enumerate(rows):
        values = tuple(float(item) for item in row)
        if len(values) != ACTION_DIM:
            raise ValueError(f"{label}[{row_index}] has width {len(values)}, expected {ACTION_DIM}")
        matrix.append(values)
    return tuple(matrix)
