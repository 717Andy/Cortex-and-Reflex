"""SmolVLA and pi0.5 fine-tune launcher for the bimanual towel contract.

GPU runs shell out to ``lerobot.scripts.lerobot_train`` with
``--policy.type`` and ``--policy.pretrained_path``. That loads base weights
while the dataset supplies ``observation.images.left``, ``right``, and ``top``.
Full fine-tune is the default: the vision encoder and the VLM are trainable.

``dry_run`` does not download SmolVLM or PaliGemma. It takes one synthetic step
on a 12-to-12 linear stand-in and writes the same checkpoint layout ACT uses,
so resume and offline eval can run on CPU.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, cast

import yaml

from reflex.act_train import TrainLaunchError, WandbTrainConfig
from shared.contract import ACTION_DIM, CAMERA_FEATURE_KEYS
from shared.splits import dataset_episodes_flag

POLICY_TYPES = ("smolvla", "pi05")
PRETRAINED_DEFAULTS = {
    "smolvla": "lerobot/smolvla_base",
    "pi05": "lerobot/pi05_base",
}
TASK_PROMPT = "Fold towel"


@dataclass(frozen=True)
class VlaPolicyConfig:
    pretrained_path: str
    freeze_vision_encoder: bool = False
    train_expert_only: bool = False
    train_state_proj: bool = True
    load_vlm_weights: bool = True
    chunk_size: int = 50
    n_action_steps: int = 50
    empty_cameras: int = 0
    num_steps: int = 10
    num_inference_steps: int = 10
    pad_language_to: str = "max_length"
    prefix_length: int = 0
    num_expert_layers: int = 0
    optimizer_lr: float = 1e-4
    optimizer_weight_decay: float = 1e-10
    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6
    dtype: str = "bfloat16"
    gradient_checkpointing: bool = True
    paligemma_variant: str = "gemma_2b"
    action_expert_variant: str = "gemma_300m"
    compile_model: bool = False
    use_amp: bool = False

    def __post_init__(self) -> None:
        if not self.pretrained_path:
            raise ValueError("pretrained_path must be non-empty")
        if self.n_action_steps < 1 or self.n_action_steps > self.chunk_size:
            raise ValueError("n_action_steps must satisfy 1 <= n_action_steps <= chunk_size")
        if self.empty_cameras < 0 or self.num_steps < 1 or self.num_inference_steps < 1:
            raise ValueError("empty_cameras must be >= 0 and sampling steps must be >= 1")
        if self.pad_language_to not in {"max_length", "longest"}:
            raise ValueError("pad_language_to must be max_length or longest")
        if self.dtype not in {"bfloat16", "float32"}:
            raise ValueError("dtype must be bfloat16 or float32")
        if self.paligemma_variant not in {"gemma_2b", "gemma_300m"}:
            raise ValueError("paligemma_variant must be gemma_2b or gemma_300m")
        if self.action_expert_variant not in {"gemma_2b", "gemma_300m"}:
            raise ValueError("action_expert_variant must be gemma_2b or gemma_300m")
        if self.optimizer_lr <= 0 or self.scheduler_decay_lr <= 0:
            raise ValueError("learning rates must be positive")
        if self.scheduler_warmup_steps < 0 or self.scheduler_decay_steps < 1:
            raise ValueError("scheduler steps are out of range")


@dataclass(frozen=True)
class VlaTrainConfig:
    policy_type: str
    dataset_repo_id: str
    dataset_root: str | None = None
    revision: str | None = None
    task_prompt: str = TASK_PROMPT
    job_name: str = "vla_bimanual_so101_towel"
    output_dir: Path = Path("outputs/train/vla_towel")
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
    episodes_file: Path | None = None
    episodes_split: str = "train"
    wandb: WandbTrainConfig = field(default_factory=WandbTrainConfig)
    policy: VlaPolicyConfig = field(
        default_factory=lambda: VlaPolicyConfig(pretrained_path=PRETRAINED_DEFAULTS["smolvla"])
    )
    cameras: tuple[str, ...] = CAMERA_FEATURE_KEYS
    action_dim: int = ACTION_DIM

    def __post_init__(self) -> None:
        if self.policy_type not in POLICY_TYPES:
            raise ValueError(f"policy_type must be one of {POLICY_TYPES}")
        if not self.dataset_repo_id or not self.job_name or not self.device:
            raise ValueError("dataset_repo_id, job_name, and device must be non-empty")
        if not self.task_prompt.strip():
            raise ValueError("task_prompt must be non-empty")
        if self.resume not in {"auto", "true", "false"}:
            raise ValueError("resume must be auto, true, or false")
        if self.episodes_split not in {"train", "held_out"}:
            raise ValueError("episodes_split must be train or held_out")
        if self.steps < 1 or self.batch_size < 1:
            raise ValueError("steps and batch_size must be >= 1")
        if self.num_workers < 0 or self.log_freq < 1:
            raise ValueError("num_workers must be >= 0 and log_freq must be >= 1")
        if self.save_checkpoint and not 1 <= self.save_freq <= self.steps:
            raise ValueError("save_freq must satisfy 1 <= save_freq <= steps when checkpointing")
        if self.push_to_hub:
            raise ValueError("push_to_hub must be false; checkpoints stay local for spot resume")
        if self.dry_run and self.device != "cpu":
            raise ValueError("dry_run is the CPU mock path; set device to cpu")
        if self.action_dim != ACTION_DIM:
            raise ValueError(f"action_dim must be {ACTION_DIM}")
        if set(self.cameras) != set(CAMERA_FEATURE_KEYS) or len(self.cameras) != len(
            CAMERA_FEATURE_KEYS
        ):
            raise ValueError(f"cameras must be {list(CAMERA_FEATURE_KEYS)}")
        if self.wandb.mode is not None and self.wandb.mode not in {"online", "offline", "disabled"}:
            raise ValueError("wandb.mode must be online, offline, or disabled")


@dataclass(frozen=True)
class VlaTrainResult:
    mode: str
    resumed: bool
    step: int
    output_dir: Path
    checkpoint_dir: Path
    argv: tuple[str, ...]
    loss: float | None


def load_vla_train_config(path: Path) -> VlaTrainConfig:
    loaded = yaml.safe_load(path.read_text())
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must be a mapping")
    try:
        return vla_train_config_from_mapping(loaded)
    except ValueError as exc:
        raise ValueError(f"{path}: {exc}") from exc


def vla_train_config_from_mapping(data: dict[str, object]) -> VlaTrainConfig:
    unknown = set(data) - set(VlaTrainConfig.__dataclass_fields__)
    if unknown:
        raise ValueError(f"unknown VLA train keys: {sorted(unknown)}")
    if "policy_type" not in data or "dataset_repo_id" not in data:
        raise ValueError("policy_type and dataset_repo_id are required")
    policy_type = data["policy_type"]
    repo = data["dataset_repo_id"]
    if not isinstance(policy_type, str) or not isinstance(repo, str):
        raise ValueError("policy_type and dataset_repo_id must be strings")
    if policy_type not in POLICY_TYPES:
        raise ValueError(f"policy_type must be one of {POLICY_TYPES}")
    defaults = VlaTrainConfig(policy_type="smolvla", dataset_repo_id="placeholder")
    job_name = data.get("job_name", defaults.job_name)
    device = data.get("device", defaults.device)
    task = data.get("task_prompt", defaults.task_prompt)
    if not isinstance(job_name, str) or not isinstance(device, str) or not isinstance(task, str):
        raise ValueError("job_name, device, and task_prompt must be strings")
    output = data.get("output_dir", defaults.output_dir)
    if not isinstance(output, str | Path):
        raise ValueError("output_dir must be a path")
    cameras = data.get("cameras", list(defaults.cameras))
    if not isinstance(cameras, list) or not all(isinstance(item, str) for item in cameras):
        raise ValueError("cameras must be a list of strings")
    return VlaTrainConfig(
        policy_type=policy_type,
        dataset_repo_id=repo,
        dataset_root=_optional_str(data.get("dataset_root", defaults.dataset_root), "dataset_root"),
        revision=_optional_str(data.get("revision", defaults.revision), "revision"),
        task_prompt=task,
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
        resume=_resume(data.get("resume", defaults.resume)),
        push_to_hub=_bool(data, "push_to_hub", defaults.push_to_hub),
        episodes_file=_optional_path(
            data.get("episodes_file", defaults.episodes_file), "episodes_file"
        ),
        episodes_split=_episodes_split(data.get("episodes_split", defaults.episodes_split)),
        wandb=_wandb(data.get("wandb", {})),
        policy=_policy(data.get("policy", {}), policy_type),
        cameras=tuple(cast(list[str], cameras)),
        action_dim=_int(data, "action_dim", defaults.action_dim),
    )


def lerobot_vla_argv(config: VlaTrainConfig, *, resume_config: Path | None = None) -> list[str]:
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
        _flag("policy.type", config.policy_type),
        _flag("policy.pretrained_path", policy.pretrained_path),
        _flag("policy.device", config.device),
        _flag("policy.push_to_hub", config.push_to_hub),
        _flag("policy.use_amp", policy.use_amp),
        _flag("policy.freeze_vision_encoder", policy.freeze_vision_encoder),
        _flag("policy.train_expert_only", policy.train_expert_only),
        _flag("policy.chunk_size", policy.chunk_size),
        _flag("policy.n_action_steps", policy.n_action_steps),
        _flag("policy.empty_cameras", policy.empty_cameras),
        _flag("policy.optimizer_lr", policy.optimizer_lr),
        _flag("policy.optimizer_weight_decay", policy.optimizer_weight_decay),
        _flag("policy.scheduler_warmup_steps", policy.scheduler_warmup_steps),
        _flag("policy.scheduler_decay_steps", policy.scheduler_decay_steps),
        _flag("policy.scheduler_decay_lr", policy.scheduler_decay_lr),
        _flag("policy.compile_model", policy.compile_model),
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
    if config.policy_type == "smolvla":
        args.extend(
            [
                _flag("policy.load_vlm_weights", policy.load_vlm_weights),
                _flag("policy.train_state_proj", policy.train_state_proj),
                _flag("policy.num_steps", policy.num_steps),
                _flag("policy.pad_language_to", policy.pad_language_to),
                _flag("policy.prefix_length", policy.prefix_length),
                _flag("policy.num_expert_layers", policy.num_expert_layers),
            ]
        )
    else:
        args.extend(
            [
                _flag("policy.dtype", policy.dtype),
                _flag("policy.gradient_checkpointing", policy.gradient_checkpointing),
                _flag("policy.num_inference_steps", policy.num_inference_steps),
                _flag("policy.paligemma_variant", policy.paligemma_variant),
                _flag("policy.action_expert_variant", policy.action_expert_variant),
            ]
        )
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


def resolve_vla_resume(config: VlaTrainConfig) -> Path | None:
    path = config.output_dir / "checkpoints" / "last" / "pretrained_model" / "train_config.json"
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


def launch_vla_train(
    config: VlaTrainConfig,
    *,
    runner: Callable[[list[str]], None] | None = None,
) -> VlaTrainResult:
    resume_config = resolve_vla_resume(config)
    argv = tuple(lerobot_vla_argv(config, resume_config=resume_config))
    resumed = resume_config is not None
    if config.dry_run:
        return _run_dry(config, argv, resumed=resumed)
    if not resumed and config.output_dir.exists():
        raise FileExistsError(
            f"{config.output_dir} already exists. Pick a new directory or resume the checkpoint."
        )
    command = [sys.executable, "-m", "lerobot.scripts.lerobot_train", *argv]
    _write_task_prompt(config)
    if runner is None:
        _default_runner(command)
    else:
        runner(command)
    return VlaTrainResult(
        mode="lerobot",
        resumed=resumed,
        step=0,
        output_dir=config.output_dir,
        checkpoint_dir=config.output_dir / "checkpoints" / "last" / "pretrained_model",
        argv=argv,
        loss=None,
    )


def write_mock_policy_checkpoint(
    directory: Path,
    *,
    policy_type: str,
    task_prompt: str = TASK_PROMPT,
    kind: str = "identity",
) -> Path:
    """Write a CPU linear checkpoint eval and distillation can load without a VLM."""
    import torch
    from torch import nn

    if policy_type not in POLICY_TYPES and policy_type != "act":
        raise ValueError(f"policy_type must be act, smolvla, or pi05, got {policy_type}")
    if kind not in {"identity", "zero"}:
        raise ValueError("kind must be identity or zero")
    module = nn.Linear(ACTION_DIM, ACTION_DIM)
    with torch.no_grad():
        module.weight.zero_()
        module.bias.zero_()
        if kind == "identity":
            module.weight.copy_(torch.eye(ACTION_DIM))
    pretrained = directory / "pretrained_model"
    pretrained.mkdir(parents=True, exist_ok=True)
    torch.save(module.state_dict(), pretrained / "model.pt")
    payload = {
        "mock": True,
        "policy_type": policy_type,
        "task_prompt": task_prompt,
        "cameras": list(CAMERA_FEATURE_KEYS),
        "action_dim": ACTION_DIM,
        "kind": kind,
    }
    (pretrained / "config.json").write_text(json.dumps(payload, indent=2) + "\n")
    (pretrained / "train_config.json").write_text(
        json.dumps({"mock": True, "policy_type": policy_type, "task_prompt": task_prompt}, indent=2)
        + "\n"
    )
    return pretrained


class LinearActionPredictor:
    """12-DoF linear map stored in a mock checkpoint."""

    def __init__(self, pretrained: Path) -> None:
        import torch
        from torch import nn

        config_path = pretrained / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"missing {config_path}")
        payload = json.loads(config_path.read_text())
        if not isinstance(payload, dict) or payload.get("mock") is not True:
            raise ValueError(f"{pretrained} is not a mock policy checkpoint")
        prompt = payload.get("task_prompt", TASK_PROMPT)
        self.task_prompt = prompt if isinstance(prompt, str) else TASK_PROMPT
        self.module = nn.Linear(ACTION_DIM, ACTION_DIM)
        state = torch.load(pretrained / "model.pt", map_location="cpu", weights_only=True)
        self.module.load_state_dict(state)
        self.module.eval()

    def predict(
        self,
        state: Sequence[float],
        task: str | None = None,
        images: dict[str, object] | None = None,
    ) -> tuple[float, ...]:
        import torch

        del task, images
        if len(state) != ACTION_DIM:
            raise ValueError(f"state width {len(state)} != {ACTION_DIM}")
        vector = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action = self.module(vector)[0]
        return tuple(float(value) for value in action)


def load_action_predictor(pretrained: Path) -> LinearActionPredictor | ActActionPredictor:
    config_path = pretrained / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"missing {config_path}")
    payload = json.loads(config_path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{config_path} must be a JSON object")
    if payload.get("mock") is True:
        return LinearActionPredictor(pretrained)
    policy_type = payload.get("type", payload.get("policy_type"))
    if policy_type == "act":
        return ActActionPredictor(pretrained)
    raise ValueError(
        f"{pretrained} is not a mock checkpoint or an ACT checkpoint. "
        "SmolVLA and pi0.5 CPU eval uses the dry-run mock weights."
    )


class ActActionPredictor:
    """One-step ACT ``select_action`` for a LeRobot checkpoint."""

    def __init__(self, pretrained: Path) -> None:
        from lerobot.policies.act.modeling_act import ACTPolicy

        self.policy = ACTPolicy.from_pretrained(pretrained)
        self.policy.eval()
        self.task_prompt = TASK_PROMPT

    def predict(
        self,
        state: Sequence[float],
        task: str | None = None,
        images: dict[str, object] | None = None,
    ) -> tuple[float, ...]:
        import torch

        del task
        if len(state) != ACTION_DIM:
            raise ValueError(f"state width {len(state)} != {ACTION_DIM}")
        observation: dict[str, object] = {
            "observation.state": torch.tensor(state, dtype=torch.float32).unsqueeze(0),
        }
        if images:
            for key, value in images.items():
                if isinstance(value, torch.Tensor):
                    observation[key] = value.unsqueeze(0) if value.ndim == 3 else value
        self.policy.reset()
        action = self.policy.select_action(observation)
        values = action.detach().cpu().reshape(-1)
        if values.shape[0] < ACTION_DIM:
            raise RuntimeError(f"ACT returned {tuple(values.shape)} values")
        return tuple(float(value) for value in values[:ACTION_DIM])


def _run_dry(config: VlaTrainConfig, argv: tuple[str, ...], *, resumed: bool) -> VlaTrainResult:
    import torch
    from torch import nn

    torch.manual_seed(config.seed)
    checkpoints = config.output_dir / "checkpoints"
    module = nn.Linear(ACTION_DIM, ACTION_DIM)
    optimizer = torch.optim.Adam(module.parameters(), lr=config.policy.optimizer_lr)
    previous = 0
    if resumed:
        state_path = checkpoints / "last" / "training_state.pt"
        if not state_path.is_file():
            raise FileNotFoundError(f"dry-run resume is missing {state_path}")
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        module.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        previous = int(state["step"])
    loss = torch.zeros(())
    for index in range(config.steps):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(config.seed + previous + index)
        batch = torch.randn(config.batch_size, ACTION_DIM, generator=generator)
        optimizer.zero_grad(set_to_none=True)
        prediction = module(batch)
        loss = torch.nn.functional.mse_loss(prediction, batch)
        cast(Any, loss).backward()
        optimizer.step()
    step = previous + config.steps
    step_name = f"{step:06d}"
    step_dir = checkpoints / step_name
    pretrained = step_dir / "pretrained_model"
    pretrained.mkdir(parents=True, exist_ok=True)
    torch.save(module.state_dict(), pretrained / "model.pt")
    torch.save(
        {"step": step, "optimizer": optimizer.state_dict(), "model": module.state_dict()},
        step_dir / "training_state.pt",
    )
    mock_config = {
        "mock": True,
        "policy_type": config.policy_type,
        "task_prompt": config.task_prompt,
        "cameras": list(CAMERA_FEATURE_KEYS),
        "action_dim": ACTION_DIM,
        "dry_run": True,
        "pretrained_path": config.policy.pretrained_path,
        "freeze_vision_encoder": config.policy.freeze_vision_encoder,
        "train_expert_only": config.policy.train_expert_only,
    }
    (pretrained / "config.json").write_text(json.dumps(mock_config, indent=2) + "\n")
    (pretrained / "train_config.json").write_text(
        json.dumps(
            {
                "dry_run": True,
                "step": step,
                "policy_type": config.policy_type,
                "task_prompt": config.task_prompt,
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
    _write_task_prompt(config)
    manifest = {
        "mode": "dry-run",
        "resumed": resumed,
        "step": step,
        "argv": list(argv),
        "task_prompt": config.task_prompt,
    }
    (config.output_dir / "launch_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return VlaTrainResult(
        mode="dry-run",
        resumed=resumed,
        step=step,
        output_dir=config.output_dir,
        checkpoint_dir=pretrained,
        argv=argv,
        loss=float(loss.detach().cpu()),
    )


def _default_runner(command: list[str]) -> None:
    if importlib.util.find_spec("accelerate") is None:
        raise TrainLaunchError(
            "GPU VLA training needs the LeRobot training extra. On the GPU machine: "
            "uv pip install 'lerobot[training]' and the smolvla or pi extra."
        )
    subprocess.run(command, check=True)


def _write_task_prompt(config: VlaTrainConfig) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "task_prompt.txt").write_text(config.task_prompt + "\n")


def _point_last(checkpoints: Path, step_name: str) -> None:
    last = checkpoints / "last"
    if last.is_symlink():
        last.unlink()
    elif last.exists():
        raise FileExistsError(f"{last} exists and is not a symlink")
    last.symlink_to(step_name, target_is_directory=True)


def _policy(value: object, policy_type: str) -> VlaPolicyConfig:
    if value is None:
        value = {}
    if not isinstance(value, dict):
        raise ValueError("policy must be a mapping")
    data = cast(dict[str, object], value)
    unknown = set(data) - set(VlaPolicyConfig.__dataclass_fields__)
    if unknown:
        raise ValueError(f"unknown policy keys: {sorted(unknown)}")
    default_path = PRETRAINED_DEFAULTS[
        policy_type if policy_type in PRETRAINED_DEFAULTS else "smolvla"
    ]
    defaults = VlaPolicyConfig(pretrained_path=default_path)
    path = data.get("pretrained_path", defaults.pretrained_path)
    if not isinstance(path, str):
        raise ValueError("pretrained_path must be a string")
    pad = data.get("pad_language_to", defaults.pad_language_to)
    dtype = data.get("dtype", defaults.dtype)
    paligemma = data.get("paligemma_variant", defaults.paligemma_variant)
    expert = data.get("action_expert_variant", defaults.action_expert_variant)
    if not all(isinstance(item, str) for item in (pad, dtype, paligemma, expert)):
        raise ValueError("policy string fields must be strings")
    return VlaPolicyConfig(
        pretrained_path=path,
        freeze_vision_encoder=_bool(data, "freeze_vision_encoder", defaults.freeze_vision_encoder),
        train_expert_only=_bool(data, "train_expert_only", defaults.train_expert_only),
        train_state_proj=_bool(data, "train_state_proj", defaults.train_state_proj),
        load_vlm_weights=_bool(data, "load_vlm_weights", defaults.load_vlm_weights),
        chunk_size=_int(data, "chunk_size", defaults.chunk_size),
        n_action_steps=_int(data, "n_action_steps", defaults.n_action_steps),
        empty_cameras=_int(data, "empty_cameras", defaults.empty_cameras),
        num_steps=_int(data, "num_steps", defaults.num_steps),
        num_inference_steps=_int(data, "num_inference_steps", defaults.num_inference_steps),
        pad_language_to=str(pad),
        prefix_length=_int(data, "prefix_length", defaults.prefix_length),
        num_expert_layers=_int(data, "num_expert_layers", defaults.num_expert_layers),
        optimizer_lr=_float(data, "optimizer_lr", defaults.optimizer_lr),
        optimizer_weight_decay=_float(
            data, "optimizer_weight_decay", defaults.optimizer_weight_decay
        ),
        scheduler_warmup_steps=_int(
            data, "scheduler_warmup_steps", defaults.scheduler_warmup_steps
        ),
        scheduler_decay_steps=_int(data, "scheduler_decay_steps", defaults.scheduler_decay_steps),
        scheduler_decay_lr=_float(data, "scheduler_decay_lr", defaults.scheduler_decay_lr),
        dtype=str(dtype),
        gradient_checkpointing=_bool(
            data, "gradient_checkpointing", defaults.gradient_checkpointing
        ),
        paligemma_variant=str(paligemma),
        action_expert_variant=str(expert),
        compile_model=_bool(data, "compile_model", defaults.compile_model),
        use_amp=_bool(data, "use_amp", defaults.use_amp),
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


def _wandb_flags(config: VlaTrainConfig) -> list[str]:
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


def with_vla_overrides(
    config: VlaTrainConfig,
    *,
    output_dir: Path | None = None,
    steps: int | None = None,
    resume: str | None = None,
    wandb_enable: bool | None = None,
) -> VlaTrainConfig:
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


def _resume(value: object) -> str:
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, str) and value in {"auto", "true", "false"}:
        return value
    raise ValueError("resume must be auto, true, or false")


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
