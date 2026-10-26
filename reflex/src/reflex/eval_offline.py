"""Offline comparison of ACT, SmolVLA, and pi0.5 on held-out episodes.

Action MSE and per-joint error use the recorded actions. The fold-quality
reward model, when a checkpoint is available, scores the same frames so the
report has an automatic success rate. No robot and no dataset download: the
CPU path synthesizes episodes that match the 12-DoF contract.
"""

from __future__ import annotations

import json
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import yaml

from reflex.reward import RewardTrainConfig, load_reward_config, load_reward_model
from reflex.vla_train import TASK_PROMPT, load_action_predictor
from shared.contract import (
    ACTION_DIM,
    BIMANUAL_JOINT_NAMES,
    CAMERA_FEATURE_KEYS,
    PUBLIC_TOWEL_REPO_ID,
)
from shared.splits import load_episode_split


@dataclass(frozen=True)
class EvalEpisode:
    episode_index: int
    states: tuple[tuple[float, ...], ...]
    actions: tuple[tuple[float, ...], ...]
    images: object


@dataclass(frozen=True)
class PolicyEval:
    name: str
    action_mse: float
    per_joint_mse: dict[str, float]
    reward_mean: float | None
    success_rate: float | None
    episodes: int

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "action_mse": self.action_mse,
            "per_joint_mse": self.per_joint_mse,
            "reward_mean": self.reward_mean,
            "success_rate": self.success_rate,
            "episodes": self.episodes,
        }


@dataclass(frozen=True)
class OfflineEvalReport:
    dataset_repo_id: str
    task_prompt: str
    train_episodes: tuple[int, ...]
    held_out_episodes: tuple[int, ...]
    policies: tuple[PolicyEval, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "dataset_repo_id": self.dataset_repo_id,
            "task_prompt": self.task_prompt,
            "train_episodes": list(self.train_episodes),
            "held_out_episodes": list(self.held_out_episodes),
            "policies": [item.to_dict() for item in self.policies],
        }


@dataclass(frozen=True)
class OfflineEvalConfig:
    dataset_repo_id: str = PUBLIC_TOWEL_REPO_ID
    policies: dict[str, Path] | None = None
    reward_config: Path | None = None
    reward_checkpoint: Path | None = None
    split_file: Path | None = None
    synthetic: bool = True
    synthetic_episodes: int = 6
    frames_per_episode: int = 4
    image_size: int = 16
    holdout_fraction: float = 0.2
    seed: int = 0
    success_threshold: float = 0.5
    output_dir: Path = Path("outputs/eval/offline")
    task_prompt: str = TASK_PROMPT

    def __post_init__(self) -> None:
        if not self.dataset_repo_id or not self.task_prompt.strip():
            raise ValueError("dataset_repo_id and task_prompt must be non-empty")
        if self.synthetic_episodes < 2 or self.frames_per_episode < 1 or self.image_size < 8:
            raise ValueError(
                "synthetic_episodes must be >= 2 and frames/image size must be positive"
            )
        if not 0.0 < self.holdout_fraction < 1.0:
            raise ValueError("holdout_fraction must be in (0, 1)")
        if not 0.0 < self.success_threshold < 1.0:
            raise ValueError("success_threshold must be between 0 and 1")
        if self.policies is not None and not self.policies:
            raise ValueError("policies must name at least one checkpoint")


def held_out_split(
    indices: Sequence[int], *, fraction: float, seed: int
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Deterministic train / held-out cut. Held-out is non-empty when len > 1."""
    ordered = list(indices)
    if len(ordered) < 2:
        return tuple(ordered), ()
    shuffled = ordered[:]
    random.Random(seed).shuffle(shuffled)
    count = round(len(shuffled) * fraction)
    count = min(max(count, 1), len(shuffled) - 1)
    held = tuple(sorted(shuffled[:count]))
    train = tuple(sorted(shuffled[count:]))
    return train, held


def load_offline_eval_config(path: Path) -> OfflineEvalConfig:
    loaded = yaml.safe_load(path.read_text())
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must be a mapping")
    try:
        return offline_eval_config_from_mapping(loaded)
    except ValueError as exc:
        raise ValueError(f"{path}: {exc}") from exc


def offline_eval_config_from_mapping(data: dict[str, object]) -> OfflineEvalConfig:
    unknown = set(data) - set(OfflineEvalConfig.__dataclass_fields__)
    if unknown:
        raise ValueError(f"unknown eval keys: {sorted(unknown)}")
    defaults = OfflineEvalConfig()
    repo = data.get("dataset_repo_id", defaults.dataset_repo_id)
    task = data.get("task_prompt", defaults.task_prompt)
    if not isinstance(repo, str) or not isinstance(task, str):
        raise ValueError("dataset_repo_id and task_prompt must be strings")
    output = data.get("output_dir", defaults.output_dir)
    if not isinstance(output, str | Path):
        raise ValueError("output_dir must be a path")
    return OfflineEvalConfig(
        dataset_repo_id=repo,
        policies=_policies(data.get("policies", defaults.policies)),
        reward_config=_optional_path(
            data.get("reward_config", defaults.reward_config), "reward_config"
        ),
        reward_checkpoint=_optional_path(
            data.get("reward_checkpoint", defaults.reward_checkpoint), "reward_checkpoint"
        ),
        split_file=_optional_path(data.get("split_file", defaults.split_file), "split_file"),
        synthetic=_bool(data, "synthetic", defaults.synthetic),
        synthetic_episodes=_int(data, "synthetic_episodes", defaults.synthetic_episodes),
        frames_per_episode=_int(data, "frames_per_episode", defaults.frames_per_episode),
        image_size=_int(data, "image_size", defaults.image_size),
        holdout_fraction=_float(data, "holdout_fraction", defaults.holdout_fraction),
        seed=_int(data, "seed", defaults.seed),
        success_threshold=_float(data, "success_threshold", defaults.success_threshold),
        output_dir=Path(output),
        task_prompt=task,
    )


def run_offline_eval(config: OfflineEvalConfig) -> OfflineEvalReport:
    if not config.policies:
        raise ValueError("policies is required to run offline eval")
    episodes = _synthetic_episodes(config) if config.synthetic else []
    if not episodes:
        raise ValueError(
            "offline eval needs episodes; set synthetic: true until hardware data is local"
        )
    by_index = {episode.episode_index: episode for episode in episodes}
    if config.split_file is not None:
        split = load_episode_split(config.split_file)
        train_ids, held_ids = split.train_episodes, split.held_out_episodes
    else:
        train_ids, held_ids = held_out_split(
            [episode.episode_index for episode in episodes],
            fraction=config.holdout_fraction,
            seed=config.seed,
        )
    missing = [index for index in held_ids if index not in by_index]
    if missing:
        raise ValueError(f"held-out episodes are not in the eval set: {missing}")
    held = [by_index[index] for index in held_ids]
    reward = _load_reward(config)
    results = tuple(
        _score_policy(name, path, held, config, reward) for name, path in config.policies.items()
    )
    return OfflineEvalReport(
        dataset_repo_id=config.dataset_repo_id,
        task_prompt=config.task_prompt,
        train_episodes=train_ids,
        held_out_episodes=held_ids,
        policies=results,
    )


def write_report(report: OfflineEvalReport, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "report.json"
    markdown_path = output_dir / "report.md"
    json_path.write_text(json.dumps(report.to_dict(), indent=2) + "\n")
    markdown_path.write_text(render_markdown(report) + "\n")
    return json_path, markdown_path


def render_markdown(report: OfflineEvalReport) -> str:
    lines = [
        "# Offline evaluation",
        "",
        f"Dataset: `{report.dataset_repo_id}`",
        f"Task: {report.task_prompt}",
        f"Train episodes: {list(report.train_episodes)}",
        f"Held-out episodes: {list(report.held_out_episodes)}",
        "",
        "| Policy | Action MSE | Reward mean | Success rate | Episodes |",
        "| --- | --- | --- | --- | --- |",
    ]
    for item in report.policies:
        reward = "—" if item.reward_mean is None else f"{item.reward_mean:.4f}"
        success = "—" if item.success_rate is None else f"{item.success_rate:.3f}"
        lines.append(
            f"| {item.name} | {item.action_mse:.6f} | {reward} | {success} | {item.episodes} |"
        )
    lines.extend(["", "## Per-joint MSE", ""])
    header = "| Joint | " + " | ".join(item.name for item in report.policies) + " |"
    rule = "| --- | " + " | ".join("---" for _ in report.policies) + " |"
    lines.extend([header, rule])
    for joint in BIMANUAL_JOINT_NAMES:
        cells = [f"{item.per_joint_mse[joint]:.6f}" for item in report.policies]
        lines.append(f"| {joint} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _score_policy(
    name: str,
    checkpoint: Path,
    held: Sequence[EvalEpisode],
    config: OfflineEvalConfig,
    reward: object | None,
) -> PolicyEval:
    predictor = load_action_predictor(checkpoint)
    squared = 0.0
    per_joint = [0.0] * ACTION_DIM
    frames = 0
    reward_scores: list[float] = []
    for episode in held:
        for frame, (state, action) in enumerate(zip(episode.states, episode.actions, strict=True)):
            predicted = predictor.predict(state, config.task_prompt, _frame_images(episode, frame))
            for index in range(ACTION_DIM):
                error = (predicted[index] - action[index]) ** 2
                per_joint[index] += error
                squared += error
            frames += 1
        if reward is not None:
            reward_scores.append(_score_reward(reward, episode, config))
    if frames == 0:
        raise ValueError(f"{name} held-out split has no frames")
    mean_reward = None if not reward_scores else sum(reward_scores) / len(reward_scores)
    success = None
    if reward_scores:
        hits = sum(1 for score in reward_scores if score >= config.success_threshold)
        success = hits / len(reward_scores)
    return PolicyEval(
        name=name,
        action_mse=squared / (frames * ACTION_DIM),
        per_joint_mse={
            joint: per_joint[index] / frames for index, joint in enumerate(BIMANUAL_JOINT_NAMES)
        },
        reward_mean=mean_reward,
        success_rate=success,
        episodes=len(held),
    )


def _synthetic_episodes(config: OfflineEvalConfig) -> list[EvalEpisode]:
    import torch

    episodes: list[EvalEpisode] = []
    for index in range(config.synthetic_episodes):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(config.seed + index)
        level = float(index + 1)
        states: list[tuple[float, ...]] = []
        actions: list[tuple[float, ...]] = []
        frames = []
        for _frame in range(config.frames_per_episode):
            state = tuple(level for _ in range(ACTION_DIM))
            action = tuple(value + 0.25 for value in state)
            states.append(state)
            actions.append(action)
            frames.append(
                torch.rand(
                    len(CAMERA_FEATURE_KEYS),
                    3,
                    config.image_size,
                    config.image_size,
                    generator=generator,
                )
            )
        episodes.append(
            EvalEpisode(
                episode_index=index,
                states=tuple(states),
                actions=tuple(actions),
                images=torch.stack(frames, dim=0),
            )
        )
    return episodes


def _frame_images(episode: EvalEpisode, frame: int) -> dict[str, object]:
    import torch

    images = episode.images
    if not isinstance(images, torch.Tensor):
        return {}
    view = images[frame]
    return {camera: view[index] for index, camera in enumerate(CAMERA_FEATURE_KEYS)}


def _load_reward(config: OfflineEvalConfig) -> object | None:
    if config.reward_checkpoint is None:
        return None
    reward_config = (
        load_reward_config(config.reward_config)
        if config.reward_config is not None
        else RewardTrainConfig()
    )
    return cast(object, load_reward_model(reward_config, config.reward_checkpoint))


def _score_reward(model: object, episode: EvalEpisode, config: OfflineEvalConfig) -> float:
    import torch

    images = episode.images
    if not isinstance(images, torch.Tensor):
        raise ValueError("reward scoring needs image tensors")
    if not callable(model):
        raise ValueError("reward model is not callable")
    with torch.no_grad():
        _quality, _success, _episode_quality, episode_logits = model(images.unsqueeze(0))
    probability = torch.sigmoid(episode_logits).reshape(-1)[0]
    del config
    return float(probability)


def _policies(value: object) -> dict[str, Path] | None:
    if value is None:
        return None
    if not isinstance(value, dict) or not value:
        raise ValueError("policies must be a mapping of name to checkpoint path")
    policies: dict[str, Path] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            raise ValueError("policy names must be strings")
        if not isinstance(item, str | Path) or not str(item):
            raise ValueError(f"policies.{key} must be a path")
        policies[key] = Path(item)
    return policies


def _optional_path(value: object, name: str) -> Path | None:
    if value is None:
        return None
    if isinstance(value, str | Path) and str(value):
        return Path(value)
    raise ValueError(f"{name} must be a path or null")


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
