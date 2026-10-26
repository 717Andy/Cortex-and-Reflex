"""Launch LeRobot 0.6 ACT training on the bimanual towel dataset.

The GPU config shells out to ``lerobot.scripts.lerobot_train``. Checkpoints land
in ``output_dir/checkpoints/last``. Run the same command again after a spot
preemption and the launcher adds ``--resume=true`` when that checkpoint exists.

``dry_run`` does not download the dataset or import the training extra. It takes
a few synthetic CPU steps with the 12-DoF / three-camera contract and writes the
same checkpoint layout so CI can prove resume.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import cast

import yaml

from reflex.act_smoke import (
    ActSmokeConfig,
    build_smoke_policy,
    train_smoke_steps,
)
from shared.contract import ACTION_DIM, CAMERA_FEATURE_KEYS
from shared.splits import dataset_episodes_flag

_WANDB_MODES = {"online", "offline", "disabled"}


class TrainLaunchError(RuntimeError):
    """The ACT launcher refused to start."""


@dataclass(frozen=True)
class ActPolicyTrainConfig:
    chunk_size: int = 100
    n_action_steps: int = 100
    dim_model: int = 512
    n_heads: int = 8
    dim_feedforward: int = 3200
    n_encoder_layers: int = 4
    n_decoder_layers: int = 1
    use_vae: bool = True
    latent_dim: int = 32
    n_vae_encoder_layers: int = 4
    dropout: float = 0.1
    kl_weight: float = 10.0
    vision_backbone: str = "resnet18"
    pretrained_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1"
    optimizer_lr: float = 1e-5
    optimizer_weight_decay: float = 1e-4
    optimizer_lr_backbone: float = 1e-5


@dataclass(frozen=True)
class WandbTrainConfig:
    enable: bool = False
    project: str = "cortex-reflex"
    entity: str | None = None
    mode: str | None = None
    run_id: str | None = None


@dataclass(frozen=True)
class ActTrainConfig:
    dataset_repo_id: str
    dataset_root: str | None = None
    revision: str | None = None
    job_name: str = "act_bimanual_so101_towel"
    output_dir: Path = Path("outputs/train/act_towel")
    device: str = "cuda"
    dry_run: bool = False
    seed: int = 1000
    batch_size: int = 8
    steps: int = 100_000
    num_workers: int = 4
    log_freq: int = 200
    save_freq: int = 2_000
    save_checkpoint: bool = True
    resume: str = "auto"
    push_to_hub: bool = False
    use_amp: bool = True
    episodes_file: Path | None = None
    episodes_split: str = "train"
    wandb: WandbTrainConfig = field(default_factory=WandbTrainConfig)
    policy: ActPolicyTrainConfig = field(default_factory=ActPolicyTrainConfig)
    cameras: tuple[str, ...] = CAMERA_FEATURE_KEYS
    action_dim: int = ACTION_DIM
    image_height: int = 480
    image_width: int = 640

    def __post_init__(self) -> None:
        if not self.dataset_repo_id:
            raise ValueError("dataset_repo_id must be non-empty")
        if not self.job_name:
            raise ValueError("job_name must be non-empty")
        if not self.device:
            raise ValueError("device must be non-empty")
        if self.resume not in {"auto", "true", "false"}:
            raise ValueError("resume must be auto, true, or false")
        if self.steps < 1 or self.batch_size < 1:
            raise ValueError("steps and batch_size must be >= 1")
        if self.num_workers < 0 or self.log_freq < 1:
            raise ValueError("num_workers must be >= 0 and log_freq must be >= 1")
        if self.save_checkpoint and not 1 <= self.save_freq <= self.steps:
            raise ValueError("save_freq must satisfy 1 <= save_freq <= steps when checkpointing")
        if self.push_to_hub:
            raise ValueError("push_to_hub must be false; checkpoints stay local for spot resume")
        if self.dry_run and self.device != "cpu":
            raise ValueError("dry_run is the CPU synthetic path; set device to cpu")
        if self.action_dim != ACTION_DIM:
            raise ValueError(f"action_dim must be {ACTION_DIM}")
        if set(self.cameras) != set(CAMERA_FEATURE_KEYS) or len(self.cameras) != len(
            CAMERA_FEATURE_KEYS
        ):
            raise ValueError(f"cameras must be {list(CAMERA_FEATURE_KEYS)}")
        if self.image_height < 1 or self.image_width < 1:
            raise ValueError("image_height and image_width must be positive")
        if self.episodes_split not in {"train", "held_out"}:
            raise ValueError("episodes_split must be train or held_out")
        if self.wandb.mode is not None and self.wandb.mode not in _WANDB_MODES:
            raise ValueError(f"wandb.mode must be one of {sorted(_WANDB_MODES)}")
        policy = self.policy
        if policy.n_action_steps < 1 or policy.n_action_steps > policy.chunk_size:
            raise ValueError("n_action_steps must satisfy 1 <= n_action_steps <= chunk_size")
        if policy.dim_model % policy.n_heads != 0 or policy.dim_model % 4 != 0:
            raise ValueError("dim_model must be divisible by n_heads and by 4")
        if self.dry_run and policy.n_action_steps != policy.chunk_size:
            raise ValueError("dry_run requires n_action_steps == chunk_size")
        if not policy.vision_backbone.startswith("resnet"):
            raise ValueError("vision_backbone must be a resnet variant")


@dataclass(frozen=True)
class ActTrainResult:
    mode: str
    resumed: bool
    step: int
    output_dir: Path
    checkpoint_dir: Path
    argv: tuple[str, ...]
    losses: tuple[float, ...]


def load_act_train_config(path: Path) -> ActTrainConfig:
    """Load a training YAML. Unknown keys are rejected."""
    loaded = yaml.safe_load(path.read_text())
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must be a mapping")
    try:
        return act_train_config_from_mapping(loaded)
    except ValueError as exc:
        raise ValueError(f"{path}: {exc}") from exc


def act_train_config_from_mapping(data: dict[str, object]) -> ActTrainConfig:
    unknown = set(data) - set(ActTrainConfig.__dataclass_fields__)
    if unknown:
        raise ValueError(f"unknown ACT train keys: {sorted(unknown)}")
    if "dataset_repo_id" not in data:
        raise ValueError("dataset_repo_id is required")
    defaults = ActTrainConfig(dataset_repo_id="placeholder")
    repo = data["dataset_repo_id"]
    if not isinstance(repo, str):
        raise ValueError("dataset_repo_id must be a string")
    dataset_root = _optional_str(data.get("dataset_root", defaults.dataset_root), "dataset_root")
    revision = _optional_str(data.get("revision", defaults.revision), "revision")
    job_name = data.get("job_name", defaults.job_name)
    if not isinstance(job_name, str):
        raise ValueError("job_name must be a string")
    output = data.get("output_dir", defaults.output_dir)
    if not isinstance(output, str | Path):
        raise ValueError("output_dir must be a path")
    device = data.get("device", defaults.device)
    if not isinstance(device, str):
        raise ValueError("device must be a string")
    resume = _resume_value(data.get("resume", defaults.resume))
    cameras = _cameras(data.get("cameras", list(defaults.cameras)))
    return ActTrainConfig(
        dataset_repo_id=repo,
        dataset_root=dataset_root,
        revision=revision,
        job_name=job_name,
        output_dir=Path(output),
        device=device,
        dry_run=_bool(data, "dry_run", defaults.dry_run),
        seed=_int(data, "seed", defaults.seed),
        batch_size=_int(data, "batch_size", defaults.batch_size),
        steps=_int(data, "steps", defaults.steps),
        num_workers=_int(data, "num_workers", defaults.num_workers),
        log_freq=_int(data, "log_freq", defaults.log_freq),
        save_freq=_int(data, "save_freq", defaults.save_freq),
        save_checkpoint=_bool(data, "save_checkpoint", defaults.save_checkpoint),
        resume=resume,
        push_to_hub=_bool(data, "push_to_hub", defaults.push_to_hub),
        use_amp=_bool(data, "use_amp", defaults.use_amp),
        episodes_file=_optional_path(
            data.get("episodes_file", defaults.episodes_file), "episodes_file"
        ),
        episodes_split=_episodes_split(data.get("episodes_split", defaults.episodes_split)),
        wandb=_wandb(data.get("wandb", {})),
        policy=_policy(data.get("policy", {})),
        cameras=cameras,
        action_dim=_int(data, "action_dim", defaults.action_dim),
        image_height=_int(data, "image_height", defaults.image_height),
        image_width=_int(data, "image_width", defaults.image_width),
    )


def lerobot_train_argv(config: ActTrainConfig, *, resume_config: Path | None = None) -> list[str]:
    """Arguments for ``python -m lerobot.scripts.lerobot_train``."""
    if resume_config is not None:
        args = [
            _flag("config_path", resume_config),
            _flag("resume", True),
            _flag("output_dir", config.output_dir),
            _flag("steps", config.steps),
            _flag("save_freq", config.save_freq),
            _flag("save_checkpoint", config.save_checkpoint),
            _flag("batch_size", config.batch_size),
            _flag("num_workers", config.num_workers),
            _flag("log_freq", config.log_freq),
            _flag("env_eval_freq", 0),
        ]
        args.extend(_wandb_flags(config))
        return args
    policy = config.policy
    args = [
        _flag("dataset.repo_id", config.dataset_repo_id),
        _flag("policy.type", "act"),
        _flag("policy.device", config.device),
        _flag("policy.push_to_hub", config.push_to_hub),
        _flag("policy.use_amp", config.use_amp),
        _flag("policy.n_obs_steps", 1),
        _flag("policy.chunk_size", policy.chunk_size),
        _flag("policy.n_action_steps", policy.n_action_steps),
        _flag("policy.dim_model", policy.dim_model),
        _flag("policy.n_heads", policy.n_heads),
        _flag("policy.dim_feedforward", policy.dim_feedforward),
        _flag("policy.n_encoder_layers", policy.n_encoder_layers),
        _flag("policy.n_decoder_layers", policy.n_decoder_layers),
        _flag("policy.use_vae", policy.use_vae),
        _flag("policy.latent_dim", policy.latent_dim),
        _flag("policy.n_vae_encoder_layers", policy.n_vae_encoder_layers),
        _flag("policy.dropout", policy.dropout),
        _flag("policy.kl_weight", policy.kl_weight),
        _flag("policy.vision_backbone", policy.vision_backbone),
        _flag("policy.pretrained_backbone_weights", policy.pretrained_backbone_weights),
        _flag("policy.optimizer_lr", policy.optimizer_lr),
        _flag("policy.optimizer_weight_decay", policy.optimizer_weight_decay),
        _flag("policy.optimizer_lr_backbone", policy.optimizer_lr_backbone),
        _flag("output_dir", config.output_dir),
        _flag("job_name", config.job_name),
        _flag("seed", config.seed),
        _flag("batch_size", config.batch_size),
        _flag("steps", config.steps),
        _flag("num_workers", config.num_workers),
        _flag("log_freq", config.log_freq),
        _flag("save_checkpoint", config.save_checkpoint),
        _flag("save_freq", config.save_freq),
        _flag("env_eval_freq", 0),
    ]
    if config.dataset_root:
        args.append(_flag("dataset.root", config.dataset_root))
    if config.revision:
        args.append(_flag("dataset.revision", config.revision))
    if config.episodes_file is not None:
        args.append(
            dataset_episodes_flag(
                config.episodes_file,
                dataset_repo_id=config.dataset_repo_id,
                which=config.episodes_split,
            )
        )
    args.extend(_wandb_flags(config))
    return args


def resolve_resume_config(config: ActTrainConfig) -> Path | None:
    """Return the LeRobot checkpoint config to resume, if this launch should resume."""
    path = _train_config_path(config.output_dir)
    if config.resume == "false":
        if path.is_file():
            raise FileExistsError(
                f"{config.output_dir} already has a checkpoint. "
                "Re-run with resume auto or true, or choose a new output directory."
            )
        return None
    if path.is_file():
        return path
    if config.resume == "true":
        raise FileNotFoundError(f"resume is true but {path} does not exist")
    return None


def launch_act_train(
    config: ActTrainConfig,
    *,
    runner: Callable[[list[str]], None] | None = None,
) -> ActTrainResult:
    """Run a dry-run step or exec LeRobot training."""
    resume_config = resolve_resume_config(config)
    argv = tuple(lerobot_train_argv(config, resume_config=resume_config))
    resumed = resume_config is not None
    if config.dry_run:
        return _run_dry(config, argv, resumed=resumed)
    if not resumed and config.output_dir.exists():
        raise FileExistsError(
            f"{config.output_dir} already exists. Pick a new directory or resume the checkpoint."
        )
    command = [sys.executable, "-m", "lerobot.scripts.lerobot_train", *argv]
    if resumed:
        _write_manifest(config.output_dir, mode="lerobot", resumed=True, step=None, argv=argv)
    if runner is None:
        _default_runner(command)
    else:
        runner(command)
    return ActTrainResult(
        mode="lerobot",
        resumed=resumed,
        step=0,
        output_dir=config.output_dir,
        checkpoint_dir=_train_config_path(config.output_dir).parent,
        argv=argv,
        losses=(),
    )


def _run_dry(config: ActTrainConfig, argv: tuple[str, ...], *, resumed: bool) -> ActTrainResult:
    import torch
    from lerobot.policies.act.modeling_act import ACTPolicy

    smoke = _smoke_config(config)
    checkpoints = config.output_dir / "checkpoints"
    previous = 0
    if resumed:
        state_path = checkpoints / "last" / "training_state.pt"
        if not state_path.is_file():
            raise FileNotFoundError(f"dry-run resume is missing {state_path}")
        policy = ACTPolicy.from_pretrained(checkpoints / "last" / _pretrained_dir_name())
        policy.train()
        optimizer = policy.config.get_optimizer_preset().build(policy.parameters())
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        previous = int(state["step"])
    else:
        policy, optimizer = build_smoke_policy(smoke)
    losses, _action_shape = train_smoke_steps(policy, optimizer, smoke)
    step = previous + smoke.steps
    step_name = f"{step:06d}"
    step_dir = checkpoints / step_name
    pretrained = step_dir / _pretrained_dir_name()
    pretrained.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(pretrained)
    torch.save(
        {"step": step, "optimizer": optimizer.state_dict()},
        step_dir / "training_state.pt",
    )
    (pretrained / "train_config.json").write_text(
        json.dumps(
            {
                "dry_run": True,
                "step": step,
                "dataset_repo_id": config.dataset_repo_id,
                "cameras": list(CAMERA_FEATURE_KEYS),
                "action_dim": ACTION_DIM,
                "lerobot_argv": list(argv),
            },
            indent=2,
        )
        + "\n"
    )
    _point_last(checkpoints, step_name)
    _write_manifest(config.output_dir, mode="dry-run", resumed=resumed, step=step, argv=argv)
    return ActTrainResult(
        mode="dry-run",
        resumed=resumed,
        step=step,
        output_dir=config.output_dir,
        checkpoint_dir=checkpoints / "last" / _pretrained_dir_name(),
        argv=argv,
        losses=tuple(losses),
    )


def _smoke_config(config: ActTrainConfig) -> ActSmokeConfig:
    policy = config.policy
    return ActSmokeConfig(
        steps=config.steps,
        batch_size=config.batch_size,
        image_height=config.image_height,
        image_width=config.image_width,
        chunk_size=policy.chunk_size,
        dim_model=policy.dim_model,
        n_heads=policy.n_heads,
        dim_feedforward=policy.dim_feedforward,
        n_encoder_layers=policy.n_encoder_layers,
        n_decoder_layers=policy.n_decoder_layers,
        use_vae=policy.use_vae,
        seed=config.seed,
        device="cpu",
    )


def _default_runner(command: list[str]) -> None:
    if importlib.util.find_spec("accelerate") is None:
        raise TrainLaunchError(
            "GPU ACT training needs the LeRobot training extra. On the GPU machine: "
            "uv pip install 'lerobot[training]'"
        )
    subprocess.run(command, check=True)


def _train_config_path(output_dir: Path) -> Path:
    return output_dir / "checkpoints" / "last" / _pretrained_dir_name() / "train_config.json"


def _pretrained_dir_name() -> str:
    from lerobot.utils.constants import PRETRAINED_MODEL_DIR

    return str(PRETRAINED_MODEL_DIR)


def _point_last(checkpoints: Path, step_name: str) -> None:
    last = checkpoints / "last"
    if last.is_symlink():
        last.unlink()
    elif last.exists():
        raise FileExistsError(f"{last} exists and is not a symlink")
    last.symlink_to(step_name, target_is_directory=True)


def _write_manifest(
    output_dir: Path,
    *,
    mode: str,
    resumed: bool,
    step: int | None,
    argv: tuple[str, ...],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "mode": mode,
        "resumed": resumed,
        "step": step,
        "argv": list(argv),
    }
    (output_dir / "launch_manifest.json").write_text(json.dumps(payload, indent=2) + "\n")


def _wandb_flags(config: ActTrainConfig) -> list[str]:
    flags = [_flag("wandb.enable", config.wandb.enable)]
    if not config.wandb.enable:
        return flags
    flags.append(_flag("wandb.project", config.wandb.project))
    if config.wandb.entity:
        flags.append(_flag("wandb.entity", config.wandb.entity))
    if config.wandb.mode:
        flags.append(_flag("wandb.mode", config.wandb.mode))
    if config.wandb.run_id:
        flags.append(_flag("wandb.run_id", config.wandb.run_id))
    return flags


def _flag(name: str, value: object) -> str:
    if isinstance(value, bool):
        text = "true" if value else "false"
    elif value is None:
        text = "null"
    else:
        text = str(value)
    return f"--{name}={text}"


def _policy(value: object) -> ActPolicyTrainConfig:
    if value is None:
        return ActPolicyTrainConfig()
    if not isinstance(value, dict):
        raise ValueError("policy must be a mapping")
    data = cast(dict[str, object], value)
    unknown = set(data) - set(ActPolicyTrainConfig.__dataclass_fields__)
    if unknown:
        raise ValueError(f"unknown policy keys: {sorted(unknown)}")
    defaults = ActPolicyTrainConfig()
    weights = data.get("pretrained_backbone_weights", defaults.pretrained_backbone_weights)
    if weights is not None and not isinstance(weights, str):
        raise ValueError("pretrained_backbone_weights must be a string or null")
    backbone = data.get("vision_backbone", defaults.vision_backbone)
    if not isinstance(backbone, str):
        raise ValueError("vision_backbone must be a string")
    return ActPolicyTrainConfig(
        chunk_size=_int(data, "chunk_size", defaults.chunk_size),
        n_action_steps=_int(data, "n_action_steps", defaults.n_action_steps),
        dim_model=_int(data, "dim_model", defaults.dim_model),
        n_heads=_int(data, "n_heads", defaults.n_heads),
        dim_feedforward=_int(data, "dim_feedforward", defaults.dim_feedforward),
        n_encoder_layers=_int(data, "n_encoder_layers", defaults.n_encoder_layers),
        n_decoder_layers=_int(data, "n_decoder_layers", defaults.n_decoder_layers),
        use_vae=_bool(data, "use_vae", defaults.use_vae),
        latent_dim=_int(data, "latent_dim", defaults.latent_dim),
        n_vae_encoder_layers=_int(data, "n_vae_encoder_layers", defaults.n_vae_encoder_layers),
        dropout=_float(data, "dropout", defaults.dropout),
        kl_weight=_float(data, "kl_weight", defaults.kl_weight),
        vision_backbone=backbone,
        pretrained_backbone_weights=weights,
        optimizer_lr=_float(data, "optimizer_lr", defaults.optimizer_lr),
        optimizer_weight_decay=_float(
            data, "optimizer_weight_decay", defaults.optimizer_weight_decay
        ),
        optimizer_lr_backbone=_float(data, "optimizer_lr_backbone", defaults.optimizer_lr_backbone),
    )


def _wandb(value: object) -> WandbTrainConfig:
    if value is None:
        return WandbTrainConfig()
    if not isinstance(value, dict):
        raise ValueError("wandb must be a mapping")
    data = cast(dict[str, object], value)
    unknown = set(data) - set(WandbTrainConfig.__dataclass_fields__)
    if unknown:
        raise ValueError(f"unknown wandb keys: {sorted(unknown)}")
    defaults = WandbTrainConfig()
    project = data.get("project", defaults.project)
    if not isinstance(project, str) or not project:
        raise ValueError("wandb.project must be a non-empty string")
    return WandbTrainConfig(
        enable=_bool(data, "enable", defaults.enable),
        project=project,
        entity=_optional_str(data.get("entity", defaults.entity), "wandb.entity"),
        mode=_optional_str(data.get("mode", defaults.mode), "wandb.mode"),
        run_id=_optional_str(data.get("run_id", defaults.run_id), "wandb.run_id"),
    )


def _cameras(value: object) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or not all(isinstance(item, str) for item in value):
        raise ValueError("cameras must be a list of strings")
    return tuple(cast(list[str], value))


def _resume_value(value: object) -> str:
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


def _episodes_split(value: object) -> str:
    if isinstance(value, str) and value in {"train", "held_out"}:
        return value
    raise ValueError("episodes_split must be train or held_out")


def _optional_str(value: object, name: str) -> str | None:
    if value is None:
        return None
    if isinstance(value, str) and value:
        return value
    raise ValueError(f"{name} must be a string or null")


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


def with_train_overrides(
    config: ActTrainConfig,
    *,
    output_dir: Path | None = None,
    steps: int | None = None,
    resume: str | None = None,
    wandb_enable: bool | None = None,
) -> ActTrainConfig:
    """Apply CLI overrides. ``replace`` re-runs validation."""
    wandb = config.wandb
    if wandb_enable is not None:
        wandb = replace(wandb, enable=wandb_enable)
    return replace(
        config,
        output_dir=config.output_dir if output_dir is None else output_dir,
        steps=config.steps if steps is None else steps,
        resume=config.resume if resume is None else resume,
        wandb=wandb,
    )
