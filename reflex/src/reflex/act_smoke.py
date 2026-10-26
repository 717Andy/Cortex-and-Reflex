"""One CPU training step of a tiny ACT policy on synthetic bimanual observations.

The network is the LeRobot ACT implementation with a randomly initialized
ResNet-18 (no ImageNet download) and a small transformer. Images are 64x64 so
the step fits a CPU runner. This checks wiring, not fold quality.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from shared.contract import ACTION_DIM, CAMERA_FEATURE_KEYS, lerobot_feature_dict


@dataclass(frozen=True)
class ActSmokeConfig:
    steps: int = 1
    batch_size: int = 2
    image_height: int = 64
    image_width: int = 64
    chunk_size: int = 4
    dim_model: int = 64
    n_heads: int = 4
    dim_feedforward: int = 128
    n_encoder_layers: int = 1
    n_decoder_layers: int = 1
    use_vae: bool = False
    seed: int = 0
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.steps < 1:
            raise ValueError("steps must be >= 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.dim_model % self.n_heads != 0:
            raise ValueError("dim_model must be divisible by n_heads")
        if self.dim_model % 4 != 0:
            raise ValueError("dim_model must be divisible by 4 for ACT's 2D position embedding")
        if self.device != "cpu":
            raise ValueError("the smoke entry point is CPU-only")


@dataclass(frozen=True)
class ActSmokeResult:
    steps: int
    losses: tuple[float, ...]
    action_shape: tuple[int, ...]
    checkpoint_dir: Path
    device: str


def limit_torch_threads() -> None:
    """Keep a CPU smoke step from grabbing every core on a shared runner."""
    import torch

    threads = os.cpu_count() or 1
    torch.set_num_threads(min(4, threads))


def build_smoke_policy(config: ActSmokeConfig) -> tuple[Any, Any]:
    """Return a tiny CPU ``(policy, optimizer)`` pair for ``config``."""
    import torch
    from lerobot.configs.types import FeatureType
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.utils.feature_utils import dataset_to_policy_features

    limit_torch_threads()
    torch.manual_seed(config.seed)
    features = lerobot_feature_dict(
        image_shape=(config.image_height, config.image_width, 3),
    )
    policy_features = dataset_to_policy_features(features)
    output_features = {
        key: feature
        for key, feature in policy_features.items()
        if feature.type is FeatureType.ACTION
    }
    input_features = {
        key: feature for key, feature in policy_features.items() if key not in output_features
    }
    cfg = ACTConfig(
        input_features=input_features,
        output_features=output_features,
        chunk_size=config.chunk_size,
        n_action_steps=config.chunk_size,
        dim_model=config.dim_model,
        n_heads=config.n_heads,
        dim_feedforward=config.dim_feedforward,
        n_encoder_layers=config.n_encoder_layers,
        n_decoder_layers=config.n_decoder_layers,
        use_vae=config.use_vae,
        pretrained_backbone_weights=None,
        vision_backbone="resnet18",
        device=config.device,
        push_to_hub=False,
    )
    policy = ACTPolicy(cfg)
    policy.train()
    policy.to(config.device)
    optimizer = cfg.get_optimizer_preset().build(policy.parameters())
    return policy, optimizer


def run_act_smoke(output_dir: Path, config: ActSmokeConfig | None = None) -> ActSmokeResult:
    """Train ``config.steps`` of ACT and write a local checkpoint."""
    resolved = config or ActSmokeConfig()
    policy, optimizer = build_smoke_policy(resolved)
    losses, action_shape = train_smoke_steps(policy, optimizer, resolved)
    output_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(output_dir)
    return ActSmokeResult(
        steps=resolved.steps,
        losses=tuple(losses),
        action_shape=action_shape,
        checkpoint_dir=output_dir,
        device=resolved.device,
    )


def synthetic_act_batch(config: ActSmokeConfig) -> dict[str, Any]:
    """One random bimanual batch: 12-DoF state/action and the three contract cameras."""
    import torch

    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.seed)
    batch: dict[str, Any] = {
        "observation.state": torch.randn(config.batch_size, ACTION_DIM, generator=generator),
        "action": torch.randn(
            config.batch_size, config.chunk_size, ACTION_DIM, generator=generator
        ),
        "action_is_pad": torch.zeros(config.batch_size, config.chunk_size, dtype=torch.bool),
    }
    for key in CAMERA_FEATURE_KEYS:
        batch[key] = torch.rand(
            config.batch_size,
            3,
            config.image_height,
            config.image_width,
            generator=generator,
        )
    return batch


def train_smoke_steps(
    policy: Any, optimizer: Any, config: ActSmokeConfig
) -> tuple[list[float], tuple[int, ...]]:
    losses: list[float] = []
    last_batch: dict[str, Any] | None = None
    for _ in range(config.steps):
        batch = synthetic_act_batch(config)
        last_batch = batch
        loss, _loss_dict = policy.forward(batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        losses.append(float(loss.detach().cpu()))
    if last_batch is None:
        raise RuntimeError("smoke run produced no batch")
    policy.eval()
    policy.reset()
    observation = {key: last_batch[key][:1] for key in (*CAMERA_FEATURE_KEYS, "observation.state")}
    action = policy.select_action(observation)
    action_shape = tuple(int(dim) for dim in action.shape)
    if action_shape[-1] != ACTION_DIM:
        raise RuntimeError(
            f"ACT returned action shape {action_shape}, expected (..., {ACTION_DIM})"
        )
    return losses, action_shape
