"""Hardware-facing protocols that stay empty until the station and sim exist.

Mid-November is the first time follower arms are expected to be on the desk.
These protocols are the seams teleop, sim, and eval will implement. Nothing
here talks to a serial port or a physics engine.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@runtime_checkable
class FoldSim(Protocol):
    """Cloth sim step API. ``task`` is ``towel`` or ``tshirt``."""

    def reset(self, task: str) -> dict[str, object]:
        """Reset and return the initial observation."""

    def step(self, action: Sequence[float]) -> dict[str, object]:
        """Apply a 12-DoF action and return the next observation."""


@runtime_checkable
class TeleopSession(Protocol):
    """Leader-arm session that will feed ``LeRobotDataset.add_frame`` later."""

    def start(self) -> None:
        """Open the leader arms and cameras."""

    def read_action(self) -> Sequence[float]:
        """Return the current 12-DoF leader target."""

    def stop(self) -> None:
        """Release the leader arms."""


@dataclass(frozen=True)
class EvalResult:
    policy: str
    task: str
    episodes: int
    successes: int | None
    notes: str

    @property
    def success_rate(self) -> float | None:
        if self.successes is None or self.episodes <= 0:
            return None
        return self.successes / self.episodes


@runtime_checkable
class EvalHarness(Protocol):
    """Robot or sim evaluation. Success counts stay ``None`` until a judge exists."""

    def evaluate(self, policy: str, *, task: str, episodes: int) -> EvalResult:
        """Run ``episodes`` trials of ``task`` with the named policy."""
