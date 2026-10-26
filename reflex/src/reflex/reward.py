"""Fold-quality reward model and success detector.

The image tower is SigLIP 2 when ``encoder`` is ``siglip2``. CI uses ``tiny``,
a few-parameter stand-in with the same head, so the train and eval loop never
downloads weights. Frame targets come from Cortex VLM labels: episode success,
phase confidence, and failure spans. Episode scores pool the frame embeddings.

Metrics are AUROC and expected calibration error on the held-out episodes.
"""

from __future__ import annotations

import json
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml

from shared.contract import CAMERA_FEATURE_KEYS
from shared.labels import EpisodeAnnotation, annotation_from_mapping

_ENCODERS = {"tiny", "siglip2"}


@dataclass(frozen=True)
class RewardTrainConfig:
    encoder: str = "tiny"
    siglip_model: str = "google/siglip2-base-patch16-224"
    freeze_encoder: bool = True
    image_size: int = 32
    embed_dim: int = 32
    device: str = "cpu"
    seed: int = 0
    steps: int = 2
    batch_size: int = 4
    episodes: int = 8
    frames_per_episode: int = 4
    eval_fraction: float = 0.25
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    success_threshold: float = 0.5
    calibration_bins: int = 10
    resume: str = "auto"
    save_freq: int = 2
    output_dir: Path = Path("outputs/reward/fold_quality_cpu")
    synthetic: bool = True
    annotations: Path | None = None
    images_dir: Path | None = None
    allow_missing_images: bool = False
    cameras: tuple[str, ...] = CAMERA_FEATURE_KEYS

    def __post_init__(self) -> None:
        if self.encoder not in _ENCODERS:
            raise ValueError(f"encoder must be one of {sorted(_ENCODERS)}")
        if not self.siglip_model:
            raise ValueError("siglip_model must be non-empty")
        if not self.device:
            raise ValueError("device must be non-empty")
        if self.resume not in {"auto", "true", "false"}:
            raise ValueError("resume must be auto, true, or false")
        if self.steps < 1 or self.batch_size < 1 or self.frames_per_episode < 1:
            raise ValueError("steps, batch_size, and frames_per_episode must be >= 1")
        if self.image_size < 8 or self.embed_dim < 4:
            raise ValueError("image_size must be >= 8 and embed_dim must be >= 4")
        if not 0.0 <= self.eval_fraction < 1.0:
            raise ValueError("eval_fraction must be in [0, 1)")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay must be >= 0")
        if not 0.0 < self.success_threshold < 1.0:
            raise ValueError("success_threshold must be between 0 and 1")
        if self.calibration_bins < 1:
            raise ValueError("calibration_bins must be >= 1")
        if not 1 <= self.save_freq <= self.steps:
            raise ValueError("save_freq must satisfy 1 <= save_freq <= steps")
        if self.synthetic and self.episodes < 4:
            raise ValueError("synthetic training needs at least 4 episodes")
        if not self.synthetic and self.annotations is None:
            raise ValueError("annotations is required when synthetic is false")
        if set(self.cameras) != set(CAMERA_FEATURE_KEYS) or len(self.cameras) != len(
            CAMERA_FEATURE_KEYS
        ):
            raise ValueError(f"cameras must be {list(CAMERA_FEATURE_KEYS)}")


@dataclass(frozen=True)
class RewardMetrics:
    frame_auroc: float
    frame_ece: float
    episode_auroc: float
    episode_ece: float
    loss: float
    held_out_episodes: tuple[int, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "frame_auroc": self.frame_auroc,
            "frame_ece": self.frame_ece,
            "episode_auroc": self.episode_auroc,
            "episode_ece": self.episode_ece,
            "loss": self.loss,
            "held_out_episodes": list(self.held_out_episodes),
        }


@dataclass(frozen=True)
class RewardTrainResult:
    step: int
    resumed: bool
    output_dir: Path
    checkpoint_dir: Path
    metrics: RewardMetrics


@dataclass
class RewardExample:
    episode_index: int
    images: Any
    frame_quality: Any
    frame_success: Any
    episode_success: float


def frame_targets(
    annotation: EpisodeAnnotation, num_frames: int
) -> tuple[list[float], list[float]]:
    """Per-frame ``(quality, success)`` targets in ``[0, 1]``.

    A successful episode starts at 1. Phase spans replace quality with that
    span's confidence. Failure spans are hard negatives on both targets. An
    unsuccessful episode stays at 0 except where a failure span already put it.
    """
    if annotation.success is None:
        raise ValueError(f"episode {annotation.episode_index} has no success label")
    if num_frames < 1:
        raise ValueError("num_frames must be >= 1")
    positive = 1.0 if annotation.success else 0.0
    quality = [positive] * num_frames
    success = [positive] * num_frames
    if annotation.success:
        for span in annotation.phases:
            for frame in range(span.start_frame, min(span.end_frame, num_frames)):
                quality[frame] = span.confidence
    for event in annotation.failures:
        for frame in range(event.start_frame, min(event.end_frame, num_frames)):
            quality[frame] = 0.0
            success[frame] = 0.0
    return quality, success


def episode_success_target(annotation: EpisodeAnnotation) -> float:
    if annotation.success is None:
        raise ValueError(f"episode {annotation.episode_index} has no success label")
    return 1.0 if annotation.success else 0.0


def auroc(labels: Sequence[float], scores: Sequence[float]) -> float:
    """Area under the ROC curve. One-class inputs return 0.5."""
    if len(labels) != len(scores) or not labels:
        raise ValueError("labels and scores must be the same non-empty length")
    positives = sum(1 for label in labels if label >= 0.5)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return 0.5
    order = sorted(range(len(scores)), key=lambda index: scores[index])
    ranks = [0.0] * len(scores)
    cursor = 0
    while cursor < len(order):
        end = cursor
        while end + 1 < len(order) and scores[order[end + 1]] == scores[order[cursor]]:
            end += 1
        average = (cursor + 1 + end + 1) / 2.0
        for position in range(cursor, end + 1):
            ranks[order[position]] = average
        cursor = end + 1
    rank_sum = sum(ranks[index] for index, label in enumerate(labels) if label >= 0.5)
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def expected_calibration_error(
    labels: Sequence[float],
    probabilities: Sequence[float],
    *,
    bins: int = 10,
) -> float:
    """ECE of P(success). Bins compare predicted probability to the positive rate."""
    if len(labels) != len(probabilities) or not labels:
        raise ValueError("labels and probabilities must be the same non-empty length")
    if bins < 1:
        raise ValueError("bins must be >= 1")
    counts = [0] * bins
    positives = [0.0] * bins
    confidence = [0.0] * bins
    for label, probability in zip(labels, probabilities, strict=True):
        clipped = min(1.0, max(0.0, float(probability)))
        index = min(int(clipped * bins), bins - 1)
        counts[index] += 1
        positives[index] += 1.0 if label >= 0.5 else 0.0
        confidence[index] += clipped
    total = float(len(labels))
    error = 0.0
    for index, count in enumerate(counts):
        if count == 0:
            continue
        empirical = positives[index] / count
        mean_confidence = confidence[index] / count
        error += (count / total) * abs(empirical - mean_confidence)
    return error


def load_reward_config(path: Path) -> RewardTrainConfig:
    loaded = yaml.safe_load(path.read_text())
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must be a mapping")
    try:
        return reward_config_from_mapping(loaded)
    except ValueError as exc:
        raise ValueError(f"{path}: {exc}") from exc


def reward_config_from_mapping(data: dict[str, object]) -> RewardTrainConfig:
    unknown = set(data) - set(RewardTrainConfig.__dataclass_fields__)
    if unknown:
        raise ValueError(f"unknown reward keys: {sorted(unknown)}")
    defaults = RewardTrainConfig()
    encoder = data.get("encoder", defaults.encoder)
    if not isinstance(encoder, str):
        raise ValueError("encoder must be a string")
    model = data.get("siglip_model", defaults.siglip_model)
    if not isinstance(model, str):
        raise ValueError("siglip_model must be a string")
    device = data.get("device", defaults.device)
    if not isinstance(device, str):
        raise ValueError("device must be a string")
    output = data.get("output_dir", defaults.output_dir)
    if not isinstance(output, str | Path):
        raise ValueError("output_dir must be a path")
    cameras = data.get("cameras", list(defaults.cameras))
    if not isinstance(cameras, list) or not all(isinstance(item, str) for item in cameras):
        raise ValueError("cameras must be a list of strings")
    return RewardTrainConfig(
        encoder=encoder,
        siglip_model=model,
        freeze_encoder=_bool(data, "freeze_encoder", defaults.freeze_encoder),
        image_size=_int(data, "image_size", defaults.image_size),
        embed_dim=_int(data, "embed_dim", defaults.embed_dim),
        device=device,
        seed=_int(data, "seed", defaults.seed),
        steps=_int(data, "steps", defaults.steps),
        batch_size=_int(data, "batch_size", defaults.batch_size),
        episodes=_int(data, "episodes", defaults.episodes),
        frames_per_episode=_int(data, "frames_per_episode", defaults.frames_per_episode),
        eval_fraction=_float(data, "eval_fraction", defaults.eval_fraction),
        learning_rate=_float(data, "learning_rate", defaults.learning_rate),
        weight_decay=_float(data, "weight_decay", defaults.weight_decay),
        success_threshold=_float(data, "success_threshold", defaults.success_threshold),
        calibration_bins=_int(data, "calibration_bins", defaults.calibration_bins),
        resume=_resume(data.get("resume", defaults.resume)),
        save_freq=_int(data, "save_freq", defaults.save_freq),
        output_dir=Path(output),
        synthetic=_bool(data, "synthetic", defaults.synthetic),
        annotations=_optional_path(data.get("annotations", defaults.annotations), "annotations"),
        images_dir=_optional_path(data.get("images_dir", defaults.images_dir), "images_dir"),
        allow_missing_images=_bool(data, "allow_missing_images", defaults.allow_missing_images),
        cameras=tuple(cast(list[str], cameras)),
    )


def train_reward(config: RewardTrainConfig) -> RewardTrainResult:
    """Train the fold-quality head and write ``checkpoints/last``."""
    import torch

    torch.manual_seed(config.seed)
    examples = build_examples(config)
    train_examples, held_examples = split_examples(examples, config.eval_fraction)
    if not train_examples:
        raise ValueError("the train split is empty")
    model = build_reward_model(config)
    model.to(config.device)
    optimizer = torch.optim.Adam(
        [param for param in model.parameters() if param.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    resumed, start = _maybe_resume(config, model, optimizer)
    model.train()
    last_loss = 0.0
    for step in range(start + 1, start + config.steps + 1):
        batch = _batch(train_examples, config, step)
        last_loss = _step(model, optimizer, batch, config.device)
        if step % config.save_freq == 0 or step == start + config.steps:
            _save_checkpoint(config, model, optimizer, step, resumed=resumed, loss=last_loss)
    metrics = evaluate_model(model, held_examples or train_examples, config)
    metrics = RewardMetrics(
        frame_auroc=metrics.frame_auroc,
        frame_ece=metrics.frame_ece,
        episode_auroc=metrics.episode_auroc,
        episode_ece=metrics.episode_ece,
        loss=last_loss,
        held_out_episodes=metrics.held_out_episodes,
    )
    checkpoint = _checkpoint_dir(config.output_dir)
    (checkpoint / "metrics.json").write_text(json.dumps(metrics.to_dict(), indent=2) + "\n")
    _write_config(checkpoint / "config.json", config)
    return RewardTrainResult(
        step=start + config.steps,
        resumed=resumed,
        output_dir=config.output_dir,
        checkpoint_dir=checkpoint,
        metrics=metrics,
    )


def evaluate_reward(config: RewardTrainConfig, checkpoint: Path | None = None) -> RewardMetrics:
    """Score the held-out split with a saved reward checkpoint."""
    model = load_reward_model(config, checkpoint)
    examples = build_examples(config)
    _train_examples, held_examples = split_examples(examples, config.eval_fraction)
    return evaluate_model(model, held_examples or examples, config)


def build_examples(config: RewardTrainConfig) -> list[RewardExample]:
    if config.synthetic:
        return [
            _synthetic_example(config, episode_index) for episode_index in range(config.episodes)
        ]
    assert config.annotations is not None
    annotations = _read_annotations(config.annotations)
    return [_annotation_example(config, annotation) for annotation in annotations]


def split_examples(
    examples: Sequence[RewardExample], eval_fraction: float
) -> tuple[list[RewardExample], list[RewardExample]]:
    ordered = sorted(examples, key=lambda item: item.episode_index)
    if len(ordered) < 4 or eval_fraction == 0.0:
        return list(ordered), []
    count = round(len(ordered) * eval_fraction)
    count = min(max(count, 2), len(ordered) - 2)
    return list(ordered[:-count]), list(ordered[-count:])


def evaluate_model(
    model: Any, examples: Sequence[RewardExample], config: RewardTrainConfig
) -> RewardMetrics:
    import torch

    if not examples:
        raise ValueError("evaluate_model needs at least one episode")
    model.eval()
    frame_labels: list[float] = []
    frame_probs: list[float] = []
    episode_labels: list[float] = []
    episode_probs: list[float] = []
    losses: list[float] = []
    for example in examples:
        images = example.images.unsqueeze(0).to(config.device)
        with torch.no_grad():
            quality_logits, success_logits, _eq, episode_logits = model(images)
            loss = _loss(
                quality_logits,
                success_logits,
                episode_logits,
                example.frame_quality.unsqueeze(0).to(config.device),
                example.frame_success.unsqueeze(0).to(config.device),
                torch.tensor([example.episode_success], device=config.device),
            )
        success_prob = torch.sigmoid(success_logits).detach().cpu().reshape(-1)
        episode_prob = torch.sigmoid(episode_logits).detach().cpu().reshape(-1)
        frame_labels.extend(float(value) for value in example.frame_success.reshape(-1))
        frame_probs.extend(float(value) for value in success_prob)
        episode_labels.append(example.episode_success)
        episode_probs.append(float(episode_prob[0]))
        losses.append(float(loss.detach().cpu()))
    return RewardMetrics(
        frame_auroc=auroc(frame_labels, frame_probs),
        frame_ece=expected_calibration_error(
            frame_labels, frame_probs, bins=config.calibration_bins
        ),
        episode_auroc=auroc(episode_labels, episode_probs),
        episode_ece=expected_calibration_error(
            episode_labels, episode_probs, bins=config.calibration_bins
        ),
        loss=sum(losses) / len(losses),
        held_out_episodes=tuple(example.episode_index for example in examples),
    )


def build_reward_model(config: RewardTrainConfig) -> Any:
    encoder: Any
    if config.encoder == "tiny":
        encoder = TinyVisionEncoder(config.embed_dim)
        embed_dim = config.embed_dim
    else:
        encoder = Siglip2Encoder(
            config.siglip_model,
            freeze=config.freeze_encoder,
            image_size=config.image_size,
        )
        embed_dim = int(encoder.embed_dim)
    return FoldRewardModel(
        encoder,
        embed_dim,
        num_cameras=len(config.cameras),
        image_size=config.image_size,
    )


def load_reward_model(config: RewardTrainConfig, checkpoint: Path | None = None) -> Any:
    import torch

    directory = checkpoint or _checkpoint_dir(config.output_dir)
    weights = directory / "model.pt"
    if not weights.is_file():
        raise FileNotFoundError(f"reward checkpoint is missing {weights}")
    model = build_reward_model(config)
    payload = torch.load(weights, map_location=config.device, weights_only=False)
    model.load_state_dict(payload["model"])
    model.to(config.device)
    model.eval()
    return model


class TinyVisionEncoder:
    """Small conv encoder used in CI instead of SigLIP 2."""

    def __init__(self, embed_dim: int) -> None:
        import torch
        from torch import nn

        self.embed_dim = embed_dim
        self.net = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(8, embed_dim),
        )
        self._torch = torch

    def parameters(self) -> Any:
        return self.net.parameters()

    def to(self, device: str) -> TinyVisionEncoder:
        self.net.to(device)
        return self

    def state_dict(self) -> dict[str, Any]:
        return self.net.state_dict()

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.net.load_state_dict(state)

    def train(self, mode: bool = True) -> TinyVisionEncoder:
        self.net.train(mode)
        return self

    def eval(self) -> TinyVisionEncoder:
        return self.train(False)

    def __call__(self, images: Any) -> Any:
        return self.net(images)


class Siglip2Encoder:
    """Frozen-or-finetuned SigLIP 2 vision tower. Imported only on this path."""

    def __init__(self, model_name: str, *, freeze: bool, image_size: int) -> None:
        import torch

        try:
            from transformers import AutoModel
        except ImportError as exc:
            raise RuntimeError(
                "SigLIP 2 needs transformers. On the GPU machine: "
                "uv pip install 'transformers>=4.51'"
            ) from exc
        self.image_size = image_size
        self.freeze = freeze
        self._torch = torch
        self.vision = AutoModel.from_pretrained(model_name)
        hidden = getattr(getattr(self.vision, "config", None), "vision_config", None)
        if hidden is not None and hasattr(hidden, "hidden_size"):
            self.embed_dim = int(hidden.hidden_size)
        else:
            self.embed_dim = int(self.vision.config.hidden_size)
        if freeze:
            for param in self.vision.parameters():
                param.requires_grad_(False)

    def parameters(self) -> Any:
        return self.vision.parameters()

    def to(self, device: str) -> Siglip2Encoder:
        self.vision.to(device)
        return self

    def state_dict(self) -> dict[str, Any]:
        state = self.vision.state_dict()
        if not isinstance(state, dict):
            raise TypeError("SigLIP state_dict must be a mapping")
        return cast(dict[str, Any], state)

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.vision.load_state_dict(state)

    def train(self, mode: bool = True) -> Siglip2Encoder:
        self.vision.train(mode and not self.freeze)
        return self

    def eval(self) -> Siglip2Encoder:
        self.vision.eval()
        return self

    def __call__(self, images: Any) -> Any:
        torch = self._torch
        normalized = (images - 0.5) / 0.5
        if self.freeze:
            with torch.no_grad():
                return self._features(normalized)
        return self._features(normalized)

    def _features(self, images: Any) -> Any:
        if hasattr(self.vision, "get_image_features"):
            features = self.vision.get_image_features(pixel_values=images)
        else:
            vision = getattr(self.vision, "vision_model", self.vision)
            output = vision(pixel_values=images)
            features = getattr(output, "pooler_output", None)
            if features is None:
                features = output.last_hidden_state.mean(dim=1)
        if not hasattr(features, "shape"):
            features = features.pooler_output
        return features


class FoldRewardModel:
    """Per-frame quality/success heads plus an episode head on the mean frame."""

    def __init__(self, encoder: Any, embed_dim: int, *, num_cameras: int, image_size: int) -> None:
        from torch import nn

        self.encoder = encoder
        self.embed_dim = embed_dim
        self.num_cameras = num_cameras
        self.image_size = image_size
        fused = embed_dim * num_cameras
        self.frame_head = nn.Sequential(
            nn.Linear(fused, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 2),
        )
        self.episode_head = nn.Sequential(
            nn.Linear(fused, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 2),
        )
        self._modules = nn.ModuleDict(
            {"frame_head": self.frame_head, "episode_head": self.episode_head}
        )

    def parameters(self) -> list[Any]:
        params = list(self.encoder.parameters())
        params.extend(self._modules.parameters())
        return params

    def to(self, device: str) -> FoldRewardModel:
        self.encoder.to(device)
        self._modules.to(device)
        return self

    def train(self, mode: bool = True) -> FoldRewardModel:
        self.encoder.train(mode)
        self._modules.train(mode)
        return self

    def eval(self) -> FoldRewardModel:
        return self.train(False)

    def state_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"heads": self._modules.state_dict()}
        payload["encoder"] = self.encoder.state_dict()
        return payload

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._modules.load_state_dict(state["heads"])
        self.encoder.load_state_dict(state["encoder"])

    def __call__(self, images: Any) -> tuple[Any, Any, Any, Any]:
        import torch.nn.functional as functional

        if images.ndim != 6:
            raise ValueError("images must be (batch, time, cameras, 3, height, width)")
        batch, time, cameras, channels, _height, _width = images.shape
        if cameras != self.num_cameras or channels != 3:
            raise ValueError(
                f"expected {self.num_cameras} RGB cameras, got shape {tuple(images.shape)}"
            )
        flat = images.reshape(batch * time * cameras, channels, images.shape[-2], images.shape[-1])
        if flat.shape[-1] != self.image_size or flat.shape[-2] != self.image_size:
            flat = functional.interpolate(
                flat,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
        encoded = self.encoder(flat).reshape(batch, time, cameras * self.embed_dim)
        frame_logits = self.frame_head(encoded)
        episode_logits = self.episode_head(encoded.mean(dim=1))
        return (
            frame_logits[..., 0],
            frame_logits[..., 1],
            episode_logits[..., 0],
            episode_logits[..., 1],
        )


def _synthetic_example(config: RewardTrainConfig, episode_index: int) -> RewardExample:
    import torch

    success = episode_index % 2 == 0
    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.seed + episode_index)
    level = 0.85 if success else 0.15
    noise = torch.rand(
        config.frames_per_episode,
        len(config.cameras),
        3,
        config.image_size,
        config.image_size,
        generator=generator,
    )
    images = (level + 0.1 * noise).clamp(0.0, 1.0)
    target = 1.0 if success else 0.0
    frames = config.frames_per_episode
    return RewardExample(
        episode_index=episode_index,
        images=images,
        frame_quality=torch.full((frames,), target),
        frame_success=torch.full((frames,), target),
        episode_success=target,
    )


def _annotation_example(config: RewardTrainConfig, annotation: EpisodeAnnotation) -> RewardExample:
    import torch

    length = _annotation_length(annotation)
    indices = _sample_indices(length, config.frames_per_episode)
    quality, success = frame_targets(annotation, length)
    images = _load_or_fill_images(config, annotation, indices)
    return RewardExample(
        episode_index=annotation.episode_index,
        images=images,
        frame_quality=torch.tensor([quality[index] for index in indices], dtype=torch.float32),
        frame_success=torch.tensor([success[index] for index in indices], dtype=torch.float32),
        episode_success=episode_success_target(annotation),
    )


def _load_or_fill_images(
    config: RewardTrainConfig, annotation: EpisodeAnnotation, indices: list[int]
) -> Any:
    import torch

    cameras = list(config.cameras)
    frames: list[Any] = []
    for frame in indices:
        views: list[Any] = []
        for camera in cameras:
            path = (
                None
                if config.images_dir is None
                else (
                    config.images_dir / str(annotation.episode_index) / camera / f"{frame:06d}.png"
                )
            )
            if path is not None and path.is_file():
                views.append(_read_png(path, config.image_size))
            elif config.allow_missing_images or config.images_dir is None:
                if not config.allow_missing_images and config.images_dir is None:
                    raise FileNotFoundError(
                        "images_dir is required when synthetic is false. "
                        "Set allow_missing_images for a wiring check without pixels."
                    )
                generator = torch.Generator(device="cpu")
                generator.manual_seed(config.seed + annotation.episode_index * 1000 + frame)
                views.append(
                    torch.rand(3, config.image_size, config.image_size, generator=generator)
                )
            else:
                raise FileNotFoundError(f"missing reward image {path}")
        frames.append(torch.stack(views, dim=0))
    return torch.stack(frames, dim=0)


def _read_png(path: Path, image_size: int) -> Any:
    import numpy as np
    import torch
    from PIL import Image

    image = Image.open(path).convert("RGB").resize((image_size, image_size))
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def _annotation_length(annotation: EpisodeAnnotation) -> int:
    ends = [span.end_frame for span in annotation.phases]
    ends.extend(event.end_frame for event in annotation.failures)
    ends.extend(segment.end_frame for segment in annotation.segments)
    if not ends:
        raise ValueError(f"episode {annotation.episode_index} has no frame spans")
    return max(ends)


def _sample_indices(length: int, count: int) -> list[int]:
    if count == 1:
        return [0]
    if length <= count:
        indices = list(range(length))
        return indices + [length - 1] * (count - length)
    return [round(index * (length - 1) / (count - 1)) for index in range(count)]


def _read_annotations(path: Path) -> list[EpisodeAnnotation]:
    annotations: list[EpisodeAnnotation] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"{path}:{line_number} must be a JSON object")
        annotations.append(annotation_from_mapping(payload))
    if not annotations:
        raise ValueError(f"{path} has no annotations")
    return annotations


def _batch(
    examples: Sequence[RewardExample], config: RewardTrainConfig, step: int
) -> list[RewardExample]:
    order = list(range(len(examples)))
    random.Random(config.seed + step).shuffle(order)
    chosen = order[: min(config.batch_size, len(examples))]
    return [examples[index] for index in chosen]


def _step(model: Any, optimizer: Any, batch: Sequence[RewardExample], device: str) -> float:
    import torch

    images = torch.stack([example.images for example in batch]).to(device)
    frame_quality = torch.stack([example.frame_quality for example in batch]).to(device)
    frame_success = torch.stack([example.frame_success for example in batch]).to(device)
    episode_success = torch.tensor(
        [example.episode_success for example in batch], dtype=torch.float32, device=device
    )
    quality_logits, success_logits, _episode_quality, episode_logits = model(images)
    loss = _loss(
        quality_logits,
        success_logits,
        episode_logits,
        frame_quality,
        frame_success,
        episode_success,
    )
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return float(loss.detach().cpu())


def _loss(
    quality_logits: Any,
    success_logits: Any,
    episode_logits: Any,
    frame_quality: Any,
    frame_success: Any,
    episode_success: Any,
) -> Any:
    import torch
    import torch.nn.functional as functional

    quality = functional.mse_loss(torch.sigmoid(quality_logits), frame_quality)
    success = functional.binary_cross_entropy_with_logits(success_logits, frame_success)
    episode = functional.binary_cross_entropy_with_logits(episode_logits, episode_success)
    return quality + success + episode


def _maybe_resume(config: RewardTrainConfig, model: Any, optimizer: Any) -> tuple[bool, int]:
    import torch

    weights = _checkpoint_dir(config.output_dir) / "model.pt"
    if config.resume == "false":
        if weights.is_file():
            raise FileExistsError(
                f"{config.output_dir} already has a reward checkpoint. "
                "Re-run with resume auto or true, or choose a new output directory."
            )
        return False, 0
    if not weights.is_file():
        if config.resume == "true":
            raise FileNotFoundError(f"resume is true but {weights} does not exist")
        return False, 0
    payload = torch.load(weights, map_location=config.device, weights_only=False)
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    return True, int(payload["step"])


def _save_checkpoint(
    config: RewardTrainConfig,
    model: Any,
    optimizer: Any,
    step: int,
    *,
    resumed: bool,
    loss: float,
) -> None:
    import torch

    checkpoints = config.output_dir / "checkpoints"
    step_name = f"{step:06d}"
    directory = checkpoints / step_name
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"step": step, "model": model.state_dict(), "optimizer": optimizer.state_dict()},
        directory / "model.pt",
    )
    _write_config(directory / "config.json", config)
    _point_last(checkpoints, step_name)
    manifest = {
        "mode": "reward",
        "resumed": resumed,
        "step": step,
        "loss": loss,
        "encoder": config.encoder,
    }
    (config.output_dir / "launch_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def _checkpoint_dir(output_dir: Path) -> Path:
    return output_dir / "checkpoints" / "last"


def _point_last(checkpoints: Path, step_name: str) -> None:
    last = checkpoints / "last"
    if last.is_symlink():
        last.unlink()
    elif last.exists():
        raise FileExistsError(f"{last} exists and is not a symlink")
    last.symlink_to(step_name, target_is_directory=True)


def _write_config(path: Path, config: RewardTrainConfig) -> None:
    payload = {
        "encoder": config.encoder,
        "siglip_model": config.siglip_model,
        "freeze_encoder": config.freeze_encoder,
        "image_size": config.image_size,
        "embed_dim": config.embed_dim,
        "device": config.device,
        "seed": config.seed,
        "frames_per_episode": config.frames_per_episode,
        "synthetic": config.synthetic,
        "cameras": list(config.cameras),
        "success_threshold": config.success_threshold,
        "calibration_bins": config.calibration_bins,
        "eval_fraction": config.eval_fraction,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _resume(value: object) -> str:
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, str) and value in {"auto", "true", "false"}:
        return value
    raise ValueError("resume must be auto, true, or false")


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
