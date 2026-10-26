"""Bimanual SO-101 station templates and a mock teleop / deploy loop.

Ports stay null until the arms are enumerated. ``render_record_command`` prints
the ``lerobot-record`` invocation with ``<unset>`` where a device path is still
missing. ``MockBimanualRobot`` and ``MockLeaderTeleop`` exercise the same
12-DoF contract without hardware.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, cast

import yaml

from shared.contract import ACTION_DIM, CAMERA_FEATURE_KEYS
from shared.episodes import EpisodeRecord, make_episode_record
from shared.interfaces import TeleopSession


@dataclass(frozen=True)
class CameraTemplate:
    feature_key: str
    index_or_path: str | None
    width: int
    height: int
    fps: int

    def __post_init__(self) -> None:
        if self.feature_key not in CAMERA_FEATURE_KEYS:
            raise ValueError(f"unknown camera feature {self.feature_key}")
        if self.width < 1 or self.height < 1 or self.fps < 1:
            raise ValueError("camera width, height, and fps must be positive")


@dataclass(frozen=True)
class StationConfig:
    robot_type: str
    robot_id: str
    left_arm_port: str | None
    right_arm_port: str | None
    leader_type: str
    leader_id: str
    leader_left_port: str | None
    leader_right_port: str | None
    cameras: dict[str, CameraTemplate]
    repo_id: str
    single_task: str
    fps: int
    num_episodes: int
    image_height: int
    image_width: int

    def __post_init__(self) -> None:
        if self.robot_type != "bi_so_follower":
            raise ValueError("robot_type must be bi_so_follower")
        if self.leader_type != "bi_so_leader":
            raise ValueError("leader_type must be bi_so_leader")
        if not self.robot_id or not self.leader_id or not self.repo_id or not self.single_task:
            raise ValueError("ids, repo_id, and single_task must be non-empty")
        if self.fps < 1 or self.num_episodes < 1:
            raise ValueError("fps and num_episodes must be positive")
        if self.image_height < 1 or self.image_width < 1:
            raise ValueError("image size must be positive")
        keys = {camera.feature_key for camera in self.cameras.values()}
        if keys != set(CAMERA_FEATURE_KEYS):
            raise ValueError(f"cameras must cover {list(CAMERA_FEATURE_KEYS)}")

    @property
    def ports_ready(self) -> bool:
        ports = (
            self.left_arm_port,
            self.right_arm_port,
            self.leader_left_port,
            self.leader_right_port,
        )
        if any(port is None or not port.strip() for port in ports):
            return False
        return all(camera.index_or_path for camera in self.cameras.values())


@dataclass(frozen=True)
class DeployResult:
    steps: int
    task_prompt: str
    actions: tuple[tuple[float, ...], ...]


class DeployPolicy(Protocol):
    def predict(self, state: Sequence[float], task: str) -> Sequence[float]:
        """Return a 12-DoF action for the current observation."""
        ...


@dataclass
class MockBimanualRobot:
    """Follower stand-in. State advances by 0.1 each observation."""

    image_size: int = 8
    seed: int = 0
    step_index: int = 0
    actions: list[tuple[float, ...]] = field(default_factory=list)

    def reset(self) -> None:
        self.step_index = 0
        self.actions.clear()

    def get_observation(self) -> dict[str, object]:
        level = 0.1 * self.step_index
        state = tuple(level for _ in range(ACTION_DIM))
        images = {key: (3, self.image_size, self.image_size) for key in CAMERA_FEATURE_KEYS}
        return {"state": state, "images": images, "task_cameras": list(CAMERA_FEATURE_KEYS)}

    def send_action(self, action: Sequence[float]) -> None:
        if len(action) != ACTION_DIM:
            raise ValueError(f"action width {len(action)} != {ACTION_DIM}")
        self.actions.append(tuple(float(value) for value in action))
        self.step_index += 1


class MockLeaderTeleop:
    """Scripted leader that satisfies ``TeleopSession``."""

    def __init__(self) -> None:
        self._index = 0
        self._started = False

    def start(self) -> None:
        self._started = True
        self._index = 0

    def read_action(self) -> tuple[float, ...]:
        if not self._started:
            raise RuntimeError("start the teleop session before reading an action")
        action = [0.0] * ACTION_DIM
        action[5] = float(self._index)
        action[11] = float(self._index)
        self._index += 1
        return tuple(action)

    def stop(self) -> None:
        self._started = False


class StateCopyPolicy:
    """Deployment stand-in: echo the observed state as the action."""

    def predict(self, state: Sequence[float], task: str) -> tuple[float, ...]:
        del task
        if len(state) != ACTION_DIM:
            raise ValueError(f"state width {len(state)} != {ACTION_DIM}")
        return tuple(float(value) for value in state)


def load_station(teleop_path: Path, record_path: Path) -> StationConfig:
    teleop = _mapping(teleop_path)
    record = _mapping(record_path)
    try:
        return station_from_mappings(teleop, record)
    except ValueError as exc:
        raise ValueError(f"{teleop_path} / {record_path}: {exc}") from exc


def station_from_mappings(
    teleop: Mapping[str, object], record: Mapping[str, object]
) -> StationConfig:
    cameras_raw = teleop.get("cameras")
    if not isinstance(cameras_raw, dict):
        raise ValueError("cameras must be a mapping")
    cameras: dict[str, CameraTemplate] = {}
    for role, spec in cameras_raw.items():
        if not isinstance(role, str) or not isinstance(spec, dict):
            raise ValueError("each camera must be a named mapping")
        cameras[role] = _camera(spec)
    return StationConfig(
        robot_type=_text(teleop, "robot_type"),
        robot_id=_text(teleop, "robot_id"),
        left_arm_port=_optional_text(teleop.get("left_arm_port"), "left_arm_port"),
        right_arm_port=_optional_text(teleop.get("right_arm_port"), "right_arm_port"),
        leader_type=_text(teleop, "leader_type"),
        leader_id=_text(teleop, "leader_id"),
        leader_left_port=_optional_text(teleop.get("leader_left_port"), "leader_left_port"),
        leader_right_port=_optional_text(teleop.get("leader_right_port"), "leader_right_port"),
        cameras=cameras,
        repo_id=_text(record, "repo_id"),
        single_task=_text(record, "single_task"),
        fps=_int_value(record.get("fps"), "fps"),
        num_episodes=_int_value(record.get("num_episodes"), "num_episodes"),
        image_height=_int_value(record.get("image_height"), "image_height"),
        image_width=_int_value(record.get("image_width"), "image_width"),
    )


def render_record_command(station: StationConfig) -> str:
    """``lerobot-record`` command. Missing ports render as ``<unset>``."""

    def port(value: str | None) -> str:
        return value if value else "<unset>"

    lines = [
        "lerobot-record",
        f"--robot.type={station.robot_type}",
        f"--robot.id={station.robot_id}",
        f"--robot.left_arm_config.port={port(station.left_arm_port)}",
        f"--robot.right_arm_config.port={port(station.right_arm_port)}",
        f"--teleop.type={station.leader_type}",
        f"--teleop.id={station.leader_id}",
        f"--teleop.left_arm_config.port={port(station.leader_left_port)}",
        f"--teleop.right_arm_config.port={port(station.leader_right_port)}",
        f"--dataset.repo_id={station.repo_id}",
        f"--dataset.single_task={station.single_task}",
        f"--dataset.num_episodes={station.num_episodes}",
        f"--dataset.fps={station.fps}",
        "--dataset.push_to_hub=false",
    ]
    for role, camera in station.cameras.items():
        index = camera.index_or_path or "<unset>"
        lines.append(
            f"--camera.{role}={camera.feature_key}:{index}:{camera.width}x{camera.height}@{camera.fps}"
        )
    if not station.ports_ready:
        lines.append("# ports are unset; fill them in before recording")
    return " \\\n  ".join(lines)


def run_inference_loop(
    robot: MockBimanualRobot,
    policy: DeployPolicy,
    *,
    steps: int,
    task_prompt: str,
) -> DeployResult:
    if steps < 1:
        raise ValueError("steps must be >= 1")
    if not task_prompt.strip():
        raise ValueError("task_prompt must be non-empty")
    robot.reset()
    actions: list[tuple[float, ...]] = []
    for _ in range(steps):
        observation = robot.get_observation()
        state = observation["state"]
        if not isinstance(state, tuple):
            raise RuntimeError("mock observation state must be a tuple")
        action = tuple(float(value) for value in policy.predict(state, task_prompt))
        robot.send_action(action)
        actions.append(action)
    return DeployResult(steps=steps, task_prompt=task_prompt, actions=tuple(actions))


def record_mock_episode(
    robot: MockBimanualRobot,
    teleop: TeleopSession,
    *,
    frames: int,
    task: str,
    fps: int = 30,
    episode_index: int = 0,
) -> EpisodeRecord:
    if frames < 1:
        raise ValueError("frames must be >= 1")
    robot.reset()
    teleop.start()
    states: list[Sequence[float]] = []
    actions: list[Sequence[float]] = []
    try:
        for _ in range(frames):
            observation = robot.get_observation()
            state = observation["state"]
            if not isinstance(state, tuple):
                raise RuntimeError("mock observation state must be a tuple")
            action = teleop.read_action()
            states.append(state)
            actions.append(action)
            robot.send_action(action)
    finally:
        teleop.stop()
    return make_episode_record(
        episode_index=episode_index,
        task=task,
        fps=fps,
        states=states,
        actions=actions,
        image_refs=CAMERA_FEATURE_KEYS,
    )


def _mapping(path: Path) -> dict[str, object]:
    loaded = yaml.safe_load(path.read_text())
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must be a mapping")
    return cast(dict[str, object], loaded)


def _camera(spec: Mapping[str, object]) -> CameraTemplate:
    feature = spec.get("feature_key")
    if not isinstance(feature, str):
        raise ValueError("feature_key must be a string")
    return CameraTemplate(
        feature_key=feature,
        index_or_path=_optional_text(spec.get("index_or_path"), "index_or_path"),
        width=_int_value(spec.get("width"), "width"),
        height=_int_value(spec.get("height"), "height"),
        fps=_int_value(spec.get("fps"), "fps"),
    )


def _text(data: Mapping[str, object], name: str) -> str:
    value = data.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _optional_text(value: object, name: str) -> str | None:
    if value is None:
        return None
    if isinstance(value, str) and value:
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    raise ValueError(f"{name} must be a string or null")


def _int_value(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an int")
    return value
