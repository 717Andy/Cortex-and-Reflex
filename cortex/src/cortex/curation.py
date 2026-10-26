"""Select a training subset from VLM labels, versus a random baseline.

Quality ranks episodes by phase confidence, success, and the absence of failure
tags. Diversity greedily covers distinct phase and failure patterns. Reward
ranks by an external fold-quality score (the Reflex reward model writes that
file). Every strategy also draws a seeded random subset of the same size so the
split records whether the selection beat chance.

The JSON split is what ``reflex.cli train`` reads as ``--dataset.episodes``.
"""

from __future__ import annotations

import json
import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from shared.contract import PUBLIC_TOWEL_REPO_ID
from shared.labels import (
    EpisodeAnnotation,
    FailureMode,
    FoldPhase,
    annotation_from_mapping,
)
from shared.splits import (
    EpisodeScore,
    EpisodeSplit,
    SplitComparison,
    save_episode_split,
)

STRATEGIES = ("quality", "diversity", "reward", "random")


@dataclass(frozen=True)
class CurationConfig:
    strategy: str = "quality"
    seed: int = 0
    train_fraction: float = 0.8
    min_confidence: float = 0.0
    require_success: bool = False
    dataset_repo_id: str = PUBLIC_TOWEL_REPO_ID

    def __post_init__(self) -> None:
        if self.strategy not in STRATEGIES:
            raise ValueError(f"strategy must be one of {STRATEGIES}")
        if not 0.0 < self.train_fraction <= 1.0:
            raise ValueError("train_fraction must be in (0, 1]")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be between 0 and 1")
        if not self.dataset_repo_id:
            raise ValueError("dataset_repo_id must be non-empty")


def load_annotations(path: Path) -> tuple[EpisodeAnnotation, ...]:
    annotations: list[EpisodeAnnotation] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"{path}:{line_number} must be a JSON object")
        try:
            annotations.append(annotation_from_mapping(payload))
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: {exc}") from exc
    if not annotations:
        raise ValueError(f"{path} has no annotations")
    return tuple(annotations)


def load_reward_scores(path: Path) -> dict[int, float]:
    """Read ``{"0": 0.8}`` or ``{"scores": {"0": 0.8}}``."""
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must be a JSON object")
    raw = payload.get("scores", payload)
    if not isinstance(raw, dict):
        raise ValueError(f"{path} scores must be an object")
    scores: dict[int, float] = {}
    for key, value in raw.items():
        if key == "scores":
            continue
        if not isinstance(key, str) or not key.isdigit():
            raise ValueError(f"{path} score keys must be episode indices")
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"{path} score for episode {key} must be a number")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{path} score for episode {key} must be finite")
        scores[int(key)] = number
    return scores


def label_quality(annotation: EpisodeAnnotation) -> float:
    """Higher is a cleaner supervision target. Range is 0 to 1."""
    confidence = annotation.mean_phase_confidence
    if annotation.success is True:
        confidence = min(1.0, confidence + 0.05)
    elif annotation.success is False:
        confidence *= 0.5
    if annotation.failures:
        confidence *= 0.5
    return confidence


def curate(
    annotations: Sequence[EpisodeAnnotation],
    config: CurationConfig,
    reward_scores: dict[int, float] | None = None,
) -> EpisodeSplit:
    """Rank episodes and cut a train / held-out split plus a random baseline."""
    rewards = {} if reward_scores is None else reward_scores
    if config.strategy == "reward" and not rewards:
        raise ValueError("reward curation needs a reward score for at least one episode")
    chosen = _filter_annotations(annotations, config)
    if not chosen:
        raise ValueError("no episodes left after the confidence and success filters")
    features = {item.episode_index: _feature(item) for item in chosen}
    qualities = {item.episode_index: label_quality(item) for item in chosen}
    diversity = {
        item.episode_index: _intrinsic_diversity(item.episode_index, features) for item in chosen
    }
    order = _order(
        [item.episode_index for item in chosen],
        config,
        qualities=qualities,
        features=features,
        rewards=rewards,
    )
    train, held = _cut(order, config.train_fraction)
    baseline_order = _shuffled([item.episode_index for item in chosen], config.seed)
    baseline_train, _baseline_held = _cut(baseline_order, config.train_fraction)
    scores = {
        index: EpisodeScore(
            quality=qualities[index],
            diversity=diversity[index],
            reward=rewards.get(index),
        )
        for index in qualities
    }
    comparison = _comparison(train, baseline_train, qualities, rewards)
    return EpisodeSplit(
        dataset_repo_id=config.dataset_repo_id,
        strategy=config.strategy,
        seed=config.seed,
        train_fraction=config.train_fraction,
        train_episodes=tuple(train),
        held_out_episodes=tuple(held),
        baseline_strategy="random",
        baseline_train_episodes=tuple(baseline_train),
        comparison=comparison,
        scores=scores,
    )


def write_split(split: EpisodeSplit, path: Path) -> None:
    save_episode_split(split, path)


def curation_config_from_mapping(data: dict[str, object]) -> CurationConfig:
    unknown = set(data) - set(CurationConfig.__dataclass_fields__)
    if unknown:
        raise ValueError(f"unknown curation keys: {sorted(unknown)}")
    defaults = CurationConfig()
    strategy = data.get("strategy", defaults.strategy)
    if not isinstance(strategy, str):
        raise ValueError("strategy must be a string")
    repo = data.get("dataset_repo_id", defaults.dataset_repo_id)
    if not isinstance(repo, str):
        raise ValueError("dataset_repo_id must be a string")
    return CurationConfig(
        strategy=strategy,
        seed=_int(data, "seed", defaults.seed),
        train_fraction=_float(data, "train_fraction", defaults.train_fraction),
        min_confidence=_float(data, "min_confidence", defaults.min_confidence),
        require_success=_bool(data, "require_success", defaults.require_success),
        dataset_repo_id=repo,
    )


def _filter_annotations(
    annotations: Sequence[EpisodeAnnotation], config: CurationConfig
) -> list[EpisodeAnnotation]:
    kept: list[EpisodeAnnotation] = []
    seen: set[int] = set()
    for annotation in annotations:
        if annotation.episode_index in seen:
            raise ValueError(f"duplicate episode_index {annotation.episode_index}")
        seen.add(annotation.episode_index)
        if config.require_success and annotation.success is not True:
            continue
        if label_quality(annotation) < config.min_confidence:
            continue
        kept.append(annotation)
    return kept


def _feature(annotation: EpisodeAnnotation) -> tuple[float, ...]:
    phases = [0.0] * len(FoldPhase)
    for span in annotation.phases:
        phases[list(FoldPhase).index(span.phase)] = 1.0
    failures = [0.0] * len(FailureMode)
    for event in annotation.failures:
        failures[list(FailureMode).index(event.mode)] = 1.0
    success = 1.0 if annotation.success is True else 0.0
    return tuple([*phases, *failures, success, annotation.mean_phase_confidence])


def _intrinsic_diversity(index: int, features: dict[int, tuple[float, ...]]) -> float:
    others = [other for other in features if other != index]
    if not others:
        return 0.0
    return sum(_l2(features[index], features[other]) for other in others) / len(others)


def _order(
    indices: list[int],
    config: CurationConfig,
    *,
    qualities: dict[int, float],
    features: dict[int, tuple[float, ...]],
    rewards: dict[int, float],
) -> list[int]:
    if config.strategy == "random":
        return _shuffled(indices, config.seed)
    if config.strategy == "quality":
        return sorted(indices, key=lambda index: (-qualities[index], index))
    if config.strategy == "reward":
        return sorted(indices, key=lambda index: (-rewards.get(index, float("-inf")), index))
    return _diversity_order(indices, qualities, features)


def _diversity_order(
    indices: list[int],
    qualities: dict[int, float],
    features: dict[int, tuple[float, ...]],
) -> list[int]:
    remaining = set(indices)
    first = min(remaining, key=lambda index: (-qualities[index], index))
    selected = [first]
    remaining.remove(first)
    while remaining:
        nxt = min(
            remaining,
            key=lambda index: (
                -min(_l2(features[index], features[chosen]) for chosen in selected),
                index,
            ),
        )
        selected.append(nxt)
        remaining.remove(nxt)
    return selected


def _cut(order: list[int], fraction: float) -> tuple[list[int], list[int]]:
    if len(order) == 1:
        return list(order), []
    count = round(len(order) * fraction)
    count = min(max(count, 1), len(order) - 1)
    train = sorted(order[:count])
    held = sorted(order[count:])
    return train, held


def _shuffled(indices: list[int], seed: int) -> list[int]:
    copied = list(indices)
    random.Random(seed).shuffle(copied)
    return copied


def _comparison(
    train: list[int],
    baseline: list[int],
    qualities: dict[int, float],
    rewards: dict[int, float],
) -> SplitComparison:
    overlap = len(set(train) & set(baseline)) / len(train) if train else 0.0
    return SplitComparison(
        selected_mean_quality=_mean([qualities[index] for index in train]),
        random_mean_quality=_mean([qualities[index] for index in baseline]),
        selected_mean_reward=_mean_reward(train, rewards),
        random_mean_reward=_mean_reward(baseline, rewards),
        train_overlap_with_random=overlap,
    )


def _mean_reward(indices: list[int], rewards: dict[int, float]) -> float | None:
    present = [rewards[index] for index in indices if index in rewards]
    if not present:
        return None
    return _mean(present)


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _l2(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right, strict=True)))


def _bool(data: dict[str, object], name: str, default: bool) -> bool:
    value = data.get(name, default)
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a bool")
    return value


def _int(data: dict[str, object], name: str, default: int) -> int:
    value = data.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an int")
    return value


def _float(data: dict[str, object], name: str, default: float) -> float:
    value = data.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be a number")
    return float(value)
