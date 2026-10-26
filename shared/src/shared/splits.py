"""Episode splits produced by Cortex curation and consumed by Reflex training.

A split file lists the train episodes a LeRobot run should see and the held-out
episodes offline eval should score. ``train_episodes`` is the list passed as
``--dataset.episodes``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EpisodeScore:
    quality: float
    diversity: float
    reward: float | None


@dataclass(frozen=True)
class SplitComparison:
    selected_mean_quality: float
    random_mean_quality: float
    selected_mean_reward: float | None
    random_mean_reward: float | None
    train_overlap_with_random: float


@dataclass(frozen=True)
class EpisodeSplit:
    dataset_repo_id: str
    strategy: str
    seed: int
    train_fraction: float
    train_episodes: tuple[int, ...]
    held_out_episodes: tuple[int, ...]
    baseline_strategy: str
    baseline_train_episodes: tuple[int, ...]
    comparison: SplitComparison
    scores: dict[int, EpisodeScore]

    def episodes(self, which: str) -> tuple[int, ...]:
        if which == "train":
            return self.train_episodes
        if which == "held_out":
            return self.held_out_episodes
        raise ValueError("episodes split must be train or held_out")


def episodes_cli_value(episodes: Sequence[int]) -> str:
    """Draccus value for ``--dataset.episodes``."""
    return "[" + ",".join(str(index) for index in episodes) + "]"


def dataset_episodes_flag(path: Path, *, dataset_repo_id: str, which: str) -> str:
    """``--dataset.episodes=[...]`` for the train or held-out half of a split file."""
    split = load_episode_split(path)
    if split.dataset_repo_id != dataset_repo_id:
        raise ValueError(f"{path} is for {split.dataset_repo_id}, not {dataset_repo_id}")
    episodes = split.episodes(which)
    if not episodes:
        raise ValueError(f"{path} has no {which} episodes")
    return "--dataset.episodes=" + episodes_cli_value(episodes)


def save_episode_split(split: EpisodeSplit, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(split_to_dict(split), indent=2) + "\n")


def load_episode_split(path: Path) -> EpisodeSplit:
    loaded = json.loads(path.read_text())
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must be a JSON object")
    try:
        return split_from_dict(loaded)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}: {exc}") from exc


def split_to_dict(split: EpisodeSplit) -> dict[str, object]:
    scores: dict[str, dict[str, float | None]] = {}
    for index, score in sorted(split.scores.items()):
        scores[str(index)] = {
            "quality": score.quality,
            "diversity": score.diversity,
            "reward": score.reward,
        }
    return {
        "version": 1,
        "dataset_repo_id": split.dataset_repo_id,
        "strategy": split.strategy,
        "seed": split.seed,
        "train_fraction": split.train_fraction,
        "train_episodes": list(split.train_episodes),
        "held_out_episodes": list(split.held_out_episodes),
        "baseline": {
            "strategy": split.baseline_strategy,
            "train_episodes": list(split.baseline_train_episodes),
        },
        "comparison": {
            "selected_mean_quality": split.comparison.selected_mean_quality,
            "random_mean_quality": split.comparison.random_mean_quality,
            "selected_mean_reward": split.comparison.selected_mean_reward,
            "random_mean_reward": split.comparison.random_mean_reward,
            "train_overlap_with_random": split.comparison.train_overlap_with_random,
        },
        "scores": scores,
    }


def split_from_dict(data: Mapping[str, object]) -> EpisodeSplit:
    version = data.get("version", 1)
    if version != 1:
        raise ValueError(f"unsupported split version {version!r}")
    repo = data.get("dataset_repo_id")
    if not isinstance(repo, str) or not repo:
        raise ValueError("dataset_repo_id must be a non-empty string")
    strategy = data.get("strategy")
    if not isinstance(strategy, str) or not strategy:
        raise ValueError("strategy must be a non-empty string")
    seed = _int(data.get("seed"), "seed")
    fraction = _float(data.get("train_fraction"), "train_fraction")
    train = _ints(data.get("train_episodes"), "train_episodes")
    held = _ints(data.get("held_out_episodes"), "held_out_episodes")
    baseline = data.get("baseline")
    if not isinstance(baseline, dict):
        raise ValueError("baseline must be an object")
    baseline_strategy = baseline.get("strategy")
    if not isinstance(baseline_strategy, str) or not baseline_strategy:
        raise ValueError("baseline.strategy must be a non-empty string")
    baseline_train = _ints(baseline.get("train_episodes"), "baseline.train_episodes")
    comparison_raw = data.get("comparison")
    if not isinstance(comparison_raw, dict):
        raise ValueError("comparison must be an object")
    comparison = SplitComparison(
        selected_mean_quality=_float(
            comparison_raw.get("selected_mean_quality"), "selected_mean_quality"
        ),
        random_mean_quality=_float(
            comparison_raw.get("random_mean_quality"), "random_mean_quality"
        ),
        selected_mean_reward=_optional_float(
            comparison_raw.get("selected_mean_reward"), "selected_mean_reward"
        ),
        random_mean_reward=_optional_float(
            comparison_raw.get("random_mean_reward"), "random_mean_reward"
        ),
        train_overlap_with_random=_float(
            comparison_raw.get("train_overlap_with_random"), "train_overlap_with_random"
        ),
    )
    scores_raw = data.get("scores", {})
    if not isinstance(scores_raw, dict):
        raise ValueError("scores must be an object")
    scores: dict[int, EpisodeScore] = {}
    for key, value in scores_raw.items():
        if not isinstance(key, str) or not key.isdigit():
            raise ValueError("score keys must be episode indices")
        if not isinstance(value, dict):
            raise ValueError(f"scores.{key} must be an object")
        scores[int(key)] = EpisodeScore(
            quality=_float(value.get("quality"), f"scores.{key}.quality"),
            diversity=_float(value.get("diversity"), f"scores.{key}.diversity"),
            reward=_optional_float(value.get("reward"), f"scores.{key}.reward"),
        )
    return EpisodeSplit(
        dataset_repo_id=repo,
        strategy=strategy,
        seed=seed,
        train_fraction=fraction,
        train_episodes=train,
        held_out_episodes=held,
        baseline_strategy=baseline_strategy,
        baseline_train_episodes=baseline_train,
        comparison=comparison,
        scores=scores,
    )


def _ints(value: object, name: str) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    indices: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            raise ValueError(f"{name} must contain ints")
        if item < 0:
            raise ValueError(f"{name} must be non-negative")
        indices.append(item)
    if len(set(indices)) != len(indices):
        raise ValueError(f"{name} contains duplicates")
    return tuple(indices)


def _int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an int")
    return value


def _float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be a number")
    return float(value)


def _optional_float(value: object, name: str) -> float | None:
    if value is None:
        return None
    return _float(value, name)
