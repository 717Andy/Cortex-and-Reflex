"""Distill a teacher action chunk into a small folding student.

The GPU student is sized for 50-150M parameters and an inference budget of
30 Hz under 6 GB. CI trains and benches the tiny CPU student. The teacher is
either a scripted 12-DoF map or a mock checkpoint's action outputs. A real
pi0.5 or SmolVLA teacher is the same interface once its actions are the
checkpoint's linear stand-in or an ``actions.pt`` tensor.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import torch
import yaml
from torch import nn

from reflex.vla_train import TASK_PROMPT, LinearActionPredictor
from shared.contract import ACTION_DIM, CAMERA_FEATURE_KEYS

TARGET_HZ = 30.0
TARGET_VRAM_BYTES = 6 * 1024**3
PARAM_BUDGET = (50_000_000, 150_000_000)


class DistillError(RuntimeError):
    """The distillation loop cannot start."""


@dataclass(frozen=True)
class StudentArch:
    image_size: int = 64
    patch_size: int = 16
    dim_model: int = 768
    n_heads: int = 12
    n_layers: int = 10
    dim_feedforward: int = 3072
    chunk_size: int = 50
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.image_size < self.patch_size or self.image_size % self.patch_size != 0:
            raise ValueError("image_size must be a positive multiple of patch_size")
        if self.dim_model < 8 or self.dim_model % self.n_heads != 0:
            raise ValueError("dim_model must be divisible by n_heads")
        if self.n_heads < 1 or self.n_layers < 1 or self.dim_feedforward < 1:
            raise ValueError("n_heads, n_layers, and dim_feedforward must be >= 1")
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


@dataclass(frozen=True)
class DistillConfig:
    dataset_repo_id: str
    task_prompt: str = TASK_PROMPT
    teacher_kind: str = "scripted"
    teacher_checkpoint: Path | None = None
    device: str = "cpu"
    seed: int = 0
    steps: int = 1
    batch_size: int = 2
    learning_rate: float = 1e-3
    kd_weight: float = 1.0
    gt_weight: float = 0.0
    save_freq: int = 1
    resume: str = "auto"
    output_dir: Path = Path("outputs/distill/student_cpu")
    student: StudentArch = StudentArch(
        image_size=16,
        patch_size=8,
        dim_model=32,
        n_heads=4,
        n_layers=1,
        dim_feedforward=64,
        chunk_size=4,
    )
    cameras: tuple[str, ...] = CAMERA_FEATURE_KEYS
    action_dim: int = ACTION_DIM

    def __post_init__(self) -> None:
        if not self.dataset_repo_id or not self.task_prompt.strip() or not self.device:
            raise ValueError("dataset_repo_id, task_prompt, and device must be non-empty")
        if self.teacher_kind not in {"scripted", "checkpoint"}:
            raise ValueError("teacher_kind must be scripted or checkpoint")
        if self.teacher_kind == "checkpoint" and self.teacher_checkpoint is None:
            raise ValueError("teacher_checkpoint is required when teacher_kind is checkpoint")
        if self.teacher_kind == "scripted" and self.teacher_checkpoint is not None:
            raise ValueError("scripted teachers do not take a checkpoint")
        if self.resume not in {"auto", "true", "false"}:
            raise ValueError("resume must be auto, true, or false")
        if self.steps < 1 or self.batch_size < 1:
            raise ValueError("steps and batch_size must be >= 1")
        if self.learning_rate <= 0 or self.kd_weight < 0 or self.gt_weight < 0:
            raise ValueError("learning_rate must be positive and loss weights must be >= 0")
        if self.kd_weight == 0 and self.gt_weight == 0:
            raise ValueError("set kd_weight or gt_weight")
        if not 1 <= self.save_freq <= self.steps:
            raise ValueError("save_freq must satisfy 1 <= save_freq <= steps")
        if self.action_dim != ACTION_DIM:
            raise ValueError(f"action_dim must be {ACTION_DIM}")
        if set(self.cameras) != set(CAMERA_FEATURE_KEYS) or len(self.cameras) != len(
            CAMERA_FEATURE_KEYS
        ):
            raise ValueError(f"cameras must be {list(CAMERA_FEATURE_KEYS)}")


@dataclass(frozen=True)
class DistillResult:
    step: int
    resumed: bool
    loss: float
    output_dir: Path
    checkpoint_dir: Path
    parameters: int


@dataclass(frozen=True)
class BenchResult:
    parameters: int
    hertz: float
    milliseconds: float
    vram_bytes: int
    meets_hz: bool
    meets_vram: bool
    device: str
    tokens: int

    def to_dict(self) -> dict[str, object]:
        return {
            "parameters": self.parameters,
            "hertz": self.hertz,
            "milliseconds": self.milliseconds,
            "vram_bytes": self.vram_bytes,
            "meets_hz": self.meets_hz,
            "meets_vram": self.meets_vram,
            "device": self.device,
            "tokens": self.tokens,
            "target_hz": TARGET_HZ,
            "target_vram_bytes": TARGET_VRAM_BYTES,
        }


class Teacher(Protocol):
    def predict_chunk(self, state: torch.Tensor) -> torch.Tensor:
        """Return ``(batch, chunk, 12)`` teacher actions."""
        ...


def student_tokens(arch: StudentArch) -> int:
    side = arch.image_size // arch.patch_size
    return 3 * side * side + 1


def estimate_student_parameters(arch: StudentArch) -> int:
    """Parameter count of ``StudentPolicy`` with PyTorch's default biases."""
    dim = arch.dim_model
    feedforward = arch.dim_feedforward
    per_layer = 4 * dim * dim + 2 * dim * feedforward + feedforward + 9 * dim
    patch = dim * 3 * arch.patch_size * arch.patch_size + dim
    cameras = 3 * dim
    state = dim * ACTION_DIM + dim
    head = dim * (arch.chunk_size * ACTION_DIM) + arch.chunk_size * ACTION_DIM
    final_norm = 2 * dim
    return arch.n_layers * per_layer + patch + cameras + state + head + final_norm


def estimate_inference_vram_bytes(arch: StudentArch, parameters: int | None = None) -> int:
    """fp16 weights plus a residual-stream activation allowance."""
    count = estimate_student_parameters(arch) if parameters is None else parameters
    weights = count * 2
    activations = arch.n_layers * student_tokens(arch) * arch.dim_model * 32
    return weights + activations


def in_parameter_budget(parameters: int) -> bool:
    low, high = PARAM_BUDGET
    return low <= parameters <= high


class StudentPolicy(nn.Module):
    """Image-patch transformer that emits a 12-DoF action chunk."""

    def __init__(self, arch: StudentArch) -> None:
        super().__init__()
        self.arch = arch
        self.patch = nn.Conv2d(
            3, arch.dim_model, kernel_size=arch.patch_size, stride=arch.patch_size
        )
        self.cam_embed = nn.Parameter(torch.zeros(3, arch.dim_model))
        self.state_proj = nn.Linear(ACTION_DIM, arch.dim_model)
        layer = nn.TransformerEncoderLayer(
            d_model=arch.dim_model,
            nhead=arch.n_heads,
            dim_feedforward=arch.dim_feedforward,
            dropout=arch.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=arch.n_layers, norm=nn.LayerNorm(arch.dim_model)
        )
        self.action_head = nn.Linear(arch.dim_model, arch.chunk_size * ACTION_DIM)

    def forward(self, images: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        batch, cameras, channels, _height, _width = images.shape
        if cameras != 3 or channels != 3:
            raise ValueError("images must be (batch, 3, 3, height, width)")
        flat = images.reshape(batch * cameras, channels, images.shape[-2], images.shape[-1])
        tokens = self.patch(flat).flatten(2).transpose(1, 2)
        tokens = tokens.reshape(batch, cameras, -1, self.arch.dim_model)
        tokens = tokens + self.cam_embed.view(1, cameras, 1, self.arch.dim_model)
        sequence = tokens.reshape(batch, cameras * tokens.shape[2], self.arch.dim_model)
        state_token = self.state_proj(state).unsqueeze(1)
        encoded = self.encoder(torch.cat((sequence, state_token), dim=1))
        actions = self.action_head(encoded[:, -1]).reshape(batch, self.arch.chunk_size, ACTION_DIM)
        if not isinstance(actions, torch.Tensor):
            raise TypeError("action head did not return a tensor")
        return actions


def count_parameters(policy: StudentPolicy) -> int:
    return sum(int(param.numel()) for param in policy.parameters())


def load_distill_config(path: Path) -> DistillConfig:
    loaded = yaml.safe_load(path.read_text())
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must be a mapping")
    try:
        return distill_config_from_mapping(loaded)
    except ValueError as exc:
        raise ValueError(f"{path}: {exc}") from exc


def distill_config_from_mapping(data: dict[str, object]) -> DistillConfig:
    unknown = set(data) - set(DistillConfig.__dataclass_fields__)
    if unknown:
        raise ValueError(f"unknown distill keys: {sorted(unknown)}")
    if "dataset_repo_id" not in data:
        raise ValueError("dataset_repo_id is required")
    repo = data["dataset_repo_id"]
    if not isinstance(repo, str):
        raise ValueError("dataset_repo_id must be a string")
    defaults = DistillConfig(dataset_repo_id="placeholder")
    task = data.get("task_prompt", defaults.task_prompt)
    teacher = data.get("teacher_kind", defaults.teacher_kind)
    device = data.get("device", defaults.device)
    if not isinstance(task, str) or not isinstance(teacher, str) or not isinstance(device, str):
        raise ValueError("task_prompt, teacher_kind, and device must be strings")
    output = data.get("output_dir", defaults.output_dir)
    if not isinstance(output, str | Path):
        raise ValueError("output_dir must be a path")
    cameras = data.get("cameras", list(defaults.cameras))
    if not isinstance(cameras, list) or not all(isinstance(item, str) for item in cameras):
        raise ValueError("cameras must be a list of strings")
    return DistillConfig(
        dataset_repo_id=repo,
        task_prompt=task,
        teacher_kind=teacher,
        teacher_checkpoint=_optional_path(
            data.get("teacher_checkpoint", defaults.teacher_checkpoint), "teacher_checkpoint"
        ),
        device=device,
        seed=_int(data, "seed", defaults.seed),
        steps=_int(data, "steps", defaults.steps),
        batch_size=_int(data, "batch_size", defaults.batch_size),
        learning_rate=_float(data, "learning_rate", defaults.learning_rate),
        kd_weight=_float(data, "kd_weight", defaults.kd_weight),
        gt_weight=_float(data, "gt_weight", defaults.gt_weight),
        save_freq=_int(data, "save_freq", defaults.save_freq),
        resume=_resume(data.get("resume", defaults.resume)),
        output_dir=Path(output),
        student=_student(data.get("student", {})),
        cameras=tuple(cast(list[str], cameras)),
        action_dim=_int(data, "action_dim", defaults.action_dim),
    )


def train_distill(config: DistillConfig) -> DistillResult:
    """One KD loop. Teacher actions are the regression target."""
    import torch

    torch.manual_seed(config.seed)
    student = StudentPolicy(config.student).to(config.device)
    teacher = _load_teacher(config)
    optimizer = torch.optim.Adam(student.parameters(), lr=config.learning_rate)
    resumed, start = _maybe_resume(config, student, optimizer)
    student.train()
    last_loss = 0.0
    for step in range(start + 1, start + config.steps + 1):
        images, state = _batch(config, step)
        with torch.no_grad():
            teacher_actions = teacher.predict_chunk(state)
        predicted = student(images, state)
        loss = config.kd_weight * torch.nn.functional.mse_loss(predicted, teacher_actions)
        if config.gt_weight:
            loss = loss + config.gt_weight * torch.nn.functional.mse_loss(
                predicted, teacher_actions
            )
        optimizer.zero_grad(set_to_none=True)
        cast(Any, loss).backward()
        optimizer.step()
        last_loss = float(loss.detach().cpu())
        if step % config.save_freq == 0 or step == start + config.steps:
            _save(config, student, optimizer, step, resumed=resumed, loss=last_loss)
    return DistillResult(
        step=start + config.steps,
        resumed=resumed,
        loss=last_loss,
        output_dir=config.output_dir,
        checkpoint_dir=config.output_dir / "checkpoints" / "last",
        parameters=count_parameters(student),
    )


def benchmark_student(
    arch: StudentArch,
    *,
    steps: int = 20,
    warmup: int = 2,
    device: str = "cpu",
    batch_size: int = 1,
) -> BenchResult:
    """Time a batch-1 forward and estimate inference VRAM."""
    import torch

    if steps < 1 or warmup < 0 or batch_size < 1:
        raise ValueError("steps and batch_size must be >= 1 and warmup must be >= 0")
    policy = StudentPolicy(arch).to(device).eval()
    parameters = count_parameters(policy)
    images, state = _synthetic_batch(arch, batch_size, device, seed=0)
    with torch.inference_mode():
        for _ in range(warmup):
            policy(images, state)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(steps):
            policy(images, state)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
    hertz = steps / elapsed if elapsed > 0 else float("inf")
    vram = estimate_inference_vram_bytes(arch, parameters)
    if device.startswith("cuda"):
        vram = max(vram, int(torch.cuda.max_memory_allocated()))
    return BenchResult(
        parameters=parameters,
        hertz=hertz,
        milliseconds=1000.0 * elapsed / steps,
        vram_bytes=vram,
        meets_hz=hertz >= TARGET_HZ,
        meets_vram=vram < TARGET_VRAM_BYTES,
        device=device,
        tokens=student_tokens(arch),
    )


class ScriptedTeacher:
    def __init__(self, chunk_size: int) -> None:
        self.chunk_size = chunk_size

    def predict_chunk(self, state: torch.Tensor) -> torch.Tensor:
        return state[:, None, :].expand(-1, self.chunk_size, -1).contiguous()


class CheckpointTeacher:
    def __init__(self, pretrained: Path, chunk_size: int) -> None:
        actions = pretrained / "actions.pt"
        self.chunk_size = chunk_size
        self._actions: torch.Tensor | None = None
        self._linear: LinearActionPredictor | None = None
        if actions.is_file():
            loaded = torch.load(actions, map_location="cpu", weights_only=True)
            if not isinstance(loaded, torch.Tensor):
                raise DistillError("actions.pt must be a tensor")
            self._actions = loaded
            return
        config_path = pretrained / "config.json"
        if config_path.is_file():
            payload = json.loads(config_path.read_text())
            if isinstance(payload, dict) and payload.get("mock") is True:
                self._linear = LinearActionPredictor(pretrained)
                return
        raise DistillError(
            f"{pretrained} has no mock policy and no actions.pt. "
            "Export teacher action chunks to actions.pt, or use a CPU mock checkpoint."
        )

    def predict_chunk(self, state: torch.Tensor) -> torch.Tensor:
        if self._actions is not None:
            if self._actions.ndim not in {1, 2} or self._actions.shape[-1] != ACTION_DIM:
                raise DistillError("actions.pt must have shape (chunk, 12) or (12,)")
            chunk = self._actions
            if chunk.ndim == 1:
                chunk = chunk.unsqueeze(0).expand(self.chunk_size, -1)
            chunk = chunk.to(state.device)
            return chunk.unsqueeze(0).expand(state.shape[0], -1, -1).contiguous()
        assert self._linear is not None
        rows = []
        for row in state.detach().cpu():
            rows.append(self._linear.predict(tuple(float(value) for value in row)))
        actions = torch.tensor(rows, dtype=torch.float32, device=state.device)
        return actions[:, None, :].expand(-1, self.chunk_size, -1).contiguous()


def _load_teacher(config: DistillConfig) -> Teacher:
    if config.teacher_kind == "scripted":
        return ScriptedTeacher(config.student.chunk_size)
    assert config.teacher_checkpoint is not None
    if not config.teacher_checkpoint.exists():
        raise DistillError(f"teacher checkpoint does not exist: {config.teacher_checkpoint}")
    return CheckpointTeacher(config.teacher_checkpoint, config.student.chunk_size)


def _batch(config: DistillConfig, step: int) -> tuple[torch.Tensor, torch.Tensor]:
    return _synthetic_batch(
        config.student, config.batch_size, config.device, seed=config.seed + step
    )


def _synthetic_batch(
    arch: StudentArch, batch_size: int, device: str, *, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    images = torch.rand(batch_size, 3, 3, arch.image_size, arch.image_size, generator=generator).to(
        device
    )
    state = torch.randn(batch_size, ACTION_DIM, generator=generator).to(device)
    if not isinstance(images, torch.Tensor) or not isinstance(state, torch.Tensor):
        raise TypeError("synthetic batch must be tensors")
    return images, state


def _maybe_resume(
    config: DistillConfig, student: StudentPolicy, optimizer: object
) -> tuple[bool, int]:
    import torch

    weights = config.output_dir / "checkpoints" / "last" / "model.pt"
    if config.resume == "false":
        if weights.is_file():
            raise FileExistsError(
                f"{config.output_dir} already has a student checkpoint. "
                "Re-run with resume auto or true, or choose a new output directory."
            )
        return False, 0
    if not weights.is_file():
        if config.resume == "true":
            raise FileNotFoundError(f"resume is true but {weights} does not exist")
        return False, 0
    payload = torch.load(weights, map_location=config.device, weights_only=False)
    student.load_state_dict(payload["model"])
    if not hasattr(optimizer, "load_state_dict"):
        raise DistillError("optimizer cannot load state")
    optimizer.load_state_dict(payload["optimizer"])
    return True, int(payload["step"])


def _save(
    config: DistillConfig,
    student: StudentPolicy,
    optimizer: object,
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
    if not hasattr(optimizer, "state_dict"):
        raise DistillError("optimizer has no state_dict")
    torch.save(
        {"step": step, "model": student.state_dict(), "optimizer": optimizer.state_dict()},
        directory / "model.pt",
    )
    (directory / "config.json").write_text(
        json.dumps(
            {
                "task_prompt": config.task_prompt,
                "parameters": count_parameters(student),
                "estimated_parameters": estimate_student_parameters(config.student),
                "teacher_kind": config.teacher_kind,
                "cameras": list(CAMERA_FEATURE_KEYS),
                "action_dim": ACTION_DIM,
            },
            indent=2,
        )
        + "\n"
    )
    _point_last(checkpoints, step_name)
    manifest = {"mode": "distill", "resumed": resumed, "step": step, "loss": loss}
    (config.output_dir / "launch_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def _point_last(checkpoints: Path, step_name: str) -> None:
    last = checkpoints / "last"
    if last.is_symlink():
        last.unlink()
    elif last.exists():
        raise FileExistsError(f"{last} exists and is not a symlink")
    last.symlink_to(step_name, target_is_directory=True)


def _student(value: object) -> StudentArch:
    if value is None:
        value = {}
    if not isinstance(value, dict):
        raise ValueError("student must be a mapping")
    data = cast(dict[str, object], value)
    unknown = set(data) - set(StudentArch.__dataclass_fields__)
    if unknown:
        raise ValueError(f"unknown student keys: {sorted(unknown)}")
    defaults = StudentArch()
    return StudentArch(
        image_size=_int(data, "image_size", defaults.image_size),
        patch_size=_int(data, "patch_size", defaults.patch_size),
        dim_model=_int(data, "dim_model", defaults.dim_model),
        n_heads=_int(data, "n_heads", defaults.n_heads),
        n_layers=_int(data, "n_layers", defaults.n_layers),
        dim_feedforward=_int(data, "dim_feedforward", defaults.dim_feedforward),
        chunk_size=_int(data, "chunk_size", defaults.chunk_size),
        dropout=_float(data, "dropout", defaults.dropout),
    )


def _optional_path(value: object, name: str) -> Path | None:
    if value is None:
        return None
    if isinstance(value, str | Path) and str(value):
        return Path(value)
    raise ValueError(f"{name} must be a path or null")


def _resume(value: object) -> str:
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, str) and value in {"auto", "true", "false"}:
        return value
    raise ValueError("resume must be auto, true, or false")


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
