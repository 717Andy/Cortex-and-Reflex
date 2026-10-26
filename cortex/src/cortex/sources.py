"""Episode sources. Video decoding is a later implementation of this protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from shared.episodes import EpisodeRecord


@runtime_checkable
class EpisodeSource(Protocol):
    def __len__(self) -> int:
        """Number of episodes available to the labeler."""

    def load_episode(self, episode_index: int) -> EpisodeRecord:
        """Return joint trajectories for one episode."""


@dataclass(frozen=True)
class InMemoryEpisodeSource:
    episodes: tuple[EpisodeRecord, ...]

    def __len__(self) -> int:
        return len(self.episodes)

    def load_episode(self, episode_index: int) -> EpisodeRecord:
        return self.episodes[episode_index]
