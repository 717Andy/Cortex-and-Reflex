"""LeRobot v3 feature contract for a 12-DoF bimanual SO-101 episode.

The public towel datasets record two follower arms and three cameras under the
LeRobotDataset v3 layout (``meta/info.json``, parquet shards, MP4 shards).
SO-101 and the earlier SO-100 share this joint vector, so Hub datasets often
label the robot ``bi_so100_follower`` even when the hardware is SO-101.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

ACTION_DIM = 12

LEFT_ARM_JOINTS: tuple[str, ...] = (
    "left_shoulder_pan.pos",
    "left_shoulder_lift.pos",
    "left_elbow_flex.pos",
    "left_wrist_flex.pos",
    "left_wrist_roll.pos",
    "left_gripper.pos",
)
RIGHT_ARM_JOINTS: tuple[str, ...] = (
    "right_shoulder_pan.pos",
    "right_shoulder_lift.pos",
    "right_elbow_flex.pos",
    "right_wrist_flex.pos",
    "right_wrist_roll.pos",
    "right_gripper.pos",
)
BIMANUAL_JOINT_NAMES: tuple[str, ...] = LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS

JOINT_INDEX: dict[str, int] = {name: index for index, name in enumerate(BIMANUAL_JOINT_NAMES)}
LEFT_GRIPPER = JOINT_INDEX["left_gripper.pos"]
RIGHT_GRIPPER = JOINT_INDEX["right_gripper.pos"]
LEFT_WRIST_ROLL = JOINT_INDEX["left_wrist_roll.pos"]
RIGHT_WRIST_ROLL = JOINT_INDEX["right_wrist_roll.pos"]
LEFT_SHOULDER_LIFT = JOINT_INDEX["left_shoulder_lift.pos"]
RIGHT_SHOULDER_LIFT = JOINT_INDEX["right_shoulder_lift.pos"]
LEFT_ELBOW = JOINT_INDEX["left_elbow_flex.pos"]
RIGHT_ELBOW = JOINT_INDEX["right_elbow_flex.pos"]

# Semantic role -> LeRobot feature key. Wrist cameras are left/right; the
# overhead view is stored as ``top`` in the public SO-101 datasets.
CAMERA_ROLES: dict[str, str] = {
    "left_wrist": "observation.images.left",
    "right_wrist": "observation.images.right",
    "overhead": "observation.images.top",
}
CAMERA_FEATURE_KEYS: tuple[str, ...] = tuple(CAMERA_ROLES.values())

INDEX_FEATURES: dict[str, str] = {
    "timestamp": "float32",
    "frame_index": "int64",
    "episode_index": "int64",
    "index": "int64",
    "task_index": "int64",
}

DATA_PATH_TEMPLATE = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
VIDEO_PATH_TEMPLATE = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
CODEBASE_VERSION_PREFIX = "v3"

PUBLIC_TOWEL_REPO_ID = "humjie/bimanual-so101-fold-towel"
DEFAULT_IMAGE_SHAPE: tuple[int, int, int] = (480, 640, 3)


class Severity(StrEnum):
    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True)
class Issue:
    severity: Severity
    code: str
    message: str


@dataclass
class ValidationReport:
    issues: list[Issue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(issue.severity == Severity.ERROR for issue in self.issues)

    def error(self, code: str, message: str) -> None:
        self.issues.append(Issue(Severity.ERROR, code, message))

    def warning(self, code: str, message: str) -> None:
        self.issues.append(Issue(Severity.WARNING, code, message))

    def extend(self, other: ValidationReport) -> None:
        self.issues.extend(other.issues)

    def errors(self) -> list[Issue]:
        return [issue for issue in self.issues if issue.severity == Severity.ERROR]

    def __str__(self) -> str:
        if not self.issues:
            return "validation ok"
        return "\n".join(
            f"[{issue.severity}] {issue.code}: {issue.message}" for issue in self.issues
        )


class DatasetContractError(Exception):
    """Raised when a dataset does not satisfy the bimanual SO-101 contract."""

    def __init__(self, report: ValidationReport) -> None:
        self.report = report
        super().__init__(str(report))


class FeatureSpec(BaseModel):
    model_config = ConfigDict(extra="allow")

    dtype: str
    shape: tuple[int, ...]
    names: list[str] | None = None
    info: dict[str, Any] | None = None

    @field_validator("shape", mode="before")
    @classmethod
    def _coerce_shape(cls, value: object) -> tuple[int, ...]:
        if isinstance(value, tuple | list):
            return tuple(int(item) for item in value)
        raise ValueError("shape must be a sequence of ints")


class DatasetInfo(BaseModel):
    """Subset of ``meta/info.json`` that defines the episode contract."""

    model_config = ConfigDict(extra="allow")

    codebase_version: str
    robot_type: str
    total_episodes: int = Field(ge=0)
    total_frames: int = Field(ge=0)
    total_tasks: int = Field(ge=0)
    chunks_size: int = Field(gt=0)
    fps: int = Field(gt=0)
    splits: dict[str, str]
    data_path: str
    video_path: str
    features: dict[str, FeatureSpec]
    data_files_size_in_mb: int | None = None
    video_files_size_in_mb: int | None = None


def make_dataset_info(
    *,
    episodes: int = 1,
    frames: int = 30,
    fps: int = 30,
    tasks: int = 1,
    robot_type: str = "bi_so101_follower",
    image_shape: tuple[int, int, int] = DEFAULT_IMAGE_SHAPE,
    video_dtype: str = "video",
) -> DatasetInfo:
    """Build a contract-valid ``DatasetInfo`` for tests and local recordings."""
    names = list(BIMANUAL_JOINT_NAMES)
    features: dict[str, FeatureSpec] = {
        "action": FeatureSpec(dtype="float32", shape=(ACTION_DIM,), names=names),
        "observation.state": FeatureSpec(dtype="float32", shape=(ACTION_DIM,), names=names),
    }
    for key in CAMERA_FEATURE_KEYS:
        features[key] = FeatureSpec(
            dtype=video_dtype,
            shape=image_shape,
            names=["height", "width", "channels"],
        )
    for key, dtype in INDEX_FEATURES.items():
        features[key] = FeatureSpec(dtype=dtype, shape=(1,), names=None)
    return DatasetInfo(
        codebase_version="v3.0",
        robot_type=robot_type,
        total_episodes=episodes,
        total_frames=frames,
        total_tasks=tasks,
        chunks_size=1000,
        fps=fps,
        splits={"train": f"0:{episodes}"},
        data_path=DATA_PATH_TEMPLATE,
        video_path=VIDEO_PATH_TEMPLATE,
        features=features,
    )


def lerobot_feature_dict(
    *,
    image_shape: tuple[int, int, int] = DEFAULT_IMAGE_SHAPE,
    video_dtype: str = "video",
) -> dict[str, dict[str, Any]]:
    """Feature dictionary accepted by ``dataset_to_policy_features``."""
    info = make_dataset_info(image_shape=image_shape, video_dtype=video_dtype)
    encoded: dict[str, dict[str, Any]] = {}
    for key, spec in info.features.items():
        if key in INDEX_FEATURES:
            continue
        encoded[key] = {
            "dtype": spec.dtype,
            "shape": spec.shape,
            "names": spec.names,
        }
    return encoded


def validate_dataset_info(info: DatasetInfo) -> ValidationReport:
    """Check joint names, cameras, and the v3 path templates."""
    report = ValidationReport()
    if not info.codebase_version.startswith(CODEBASE_VERSION_PREFIX):
        report.error(
            "codebase_version",
            f"expected a LeRobot v3 dataset, found {info.codebase_version!r}",
        )
    if "so" not in info.robot_type.lower():
        report.warning(
            "robot_type",
            f"robot_type {info.robot_type!r} does not look like an SO-10x follower; "
            "joint names are still checked exactly",
        )
    if info.data_path != DATA_PATH_TEMPLATE:
        report.error("data_path", f"expected {DATA_PATH_TEMPLATE}, found {info.data_path}")
    if info.video_path != VIDEO_PATH_TEMPLATE:
        report.error("video_path", f"expected {VIDEO_PATH_TEMPLATE}, found {info.video_path}")
    _validate_split(info, report)
    _validate_vector_feature(info, "action", report)
    _validate_vector_feature(info, "observation.state", report)
    for key in CAMERA_FEATURE_KEYS:
        _validate_camera(info, key, report)
    for key, dtype in INDEX_FEATURES.items():
        _validate_index_feature(info, key, dtype, report)
    return report


def validate_feature_stats(stats: dict[str, Any]) -> ValidationReport:
    """Check ``meta/stats.json`` vector widths for joints and cameras."""
    report = ValidationReport()
    if not isinstance(stats, dict):
        report.error("stats_type", "stats.json must be a JSON object")
        return report
    for key in ("action", "observation.state"):
        _validate_stat_block(stats, key, ACTION_DIM, report)
    for key in CAMERA_FEATURE_KEYS:
        _validate_stat_block(stats, key, 3, report)
    return report


def _validate_split(info: DatasetInfo, report: ValidationReport) -> None:
    train = info.splits.get("train")
    if train is None:
        report.error("splits", "missing the train split")
        return
    start_text, sep, end_text = train.partition(":")
    if sep != ":" or not start_text.isdigit() or not end_text.isdigit():
        report.error("splits", f"train split {train!r} is not 'start:end'")
        return
    start, end = int(start_text), int(end_text)
    if start != 0 or end != info.total_episodes:
        report.error(
            "splits",
            f"train split {train!r} does not cover episodes 0:{info.total_episodes}",
        )


def _validate_vector_feature(info: DatasetInfo, key: str, report: ValidationReport) -> None:
    spec = info.features.get(key)
    if spec is None:
        report.error("missing_feature", f"missing {key}")
        return
    if spec.dtype != "float32":
        report.error("feature_dtype", f"{key} dtype is {spec.dtype}, expected float32")
    if spec.shape != (ACTION_DIM,):
        report.error("feature_shape", f"{key} shape is {spec.shape}, expected {(ACTION_DIM,)}")
    if spec.names != list(BIMANUAL_JOINT_NAMES):
        report.error(
            "joint_names",
            f"{key} names must be the 12 left_/right_ SO-101 joints in arm order",
        )


def _validate_camera(info: DatasetInfo, key: str, report: ValidationReport) -> None:
    spec = info.features.get(key)
    if spec is None:
        report.error("missing_camera", f"missing camera feature {key}")
        return
    if spec.dtype not in {"video", "image"}:
        report.error("camera_dtype", f"{key} dtype is {spec.dtype}, expected video or image")
    if len(spec.shape) != 3 or spec.shape[2] != 3 or spec.shape[0] <= 0 or spec.shape[1] <= 0:
        report.error("camera_shape", f"{key} shape {spec.shape} is not (height, width, 3)")
    if spec.names not in (None, ["height", "width", "channels"]):
        report.error("camera_names", f"{key} names must be height, width, channels")


def _validate_index_feature(
    info: DatasetInfo, key: str, dtype: str, report: ValidationReport
) -> None:
    spec = info.features.get(key)
    if spec is None:
        report.error("missing_feature", f"missing index feature {key}")
        return
    if spec.dtype != dtype:
        report.error("feature_dtype", f"{key} dtype is {spec.dtype}, expected {dtype}")
    if spec.shape != (1,):
        report.error("feature_shape", f"{key} shape is {spec.shape}, expected (1,)")


def _validate_stat_block(
    stats: dict[str, Any], key: str, width: int, report: ValidationReport
) -> None:
    block = stats.get(key)
    if not isinstance(block, dict):
        report.error("missing_stats", f"stats.json is missing {key}")
        return
    for moment in ("min", "max", "mean", "std"):
        value = block.get(moment)
        if _outer_length(value) != width:
            report.error("stat_width", f"{key}.{moment} must be a length-{width} vector")
            continue
        if not _leaves_are_finite(value):
            report.error("stat_finite", f"{key}.{moment} contains a non-finite value")


def _outer_length(value: object) -> int | None:
    if isinstance(value, list):
        return len(value)
    return None


def _leaves_are_finite(value: object) -> bool:
    if isinstance(value, bool) or value is None:
        return False
    if isinstance(value, int | float):
        return math.isfinite(float(value))
    if isinstance(value, list):
        return all(_leaves_are_finite(item) for item in value)
    return False
