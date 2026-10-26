"""Phase labeling and kinematic failure mining.

``StubLabeler`` splits an episode into four equal phases and applies kinematic
rules so the selection queue and dashboard have a stable contract before a VLM
is attached. The rules are not a cloth perception model. Gripper sign is not
assumed: a "close" is a change in gripper position, not a calibrated unit.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from shared.contract import (
    LEFT_ELBOW,
    LEFT_GRIPPER,
    LEFT_SHOULDER_LIFT,
    LEFT_WRIST_ROLL,
    RIGHT_ELBOW,
    RIGHT_GRIPPER,
    RIGHT_SHOULDER_LIFT,
    RIGHT_WRIST_ROLL,
)
from shared.episodes import EpisodeRecord
from shared.labels import (
    EpisodeAnnotation,
    FailureEvent,
    FailureMode,
    FoldPhase,
    PhaseSpan,
)


@runtime_checkable
class FoldLabeler(Protocol):
    """Anything that turns an episode into phases and failure events."""

    name: str

    def label(self, episode: EpisodeRecord) -> EpisodeAnnotation:
        """Label one episode."""


@dataclass(frozen=True)
class MinerConfig:
    """Thresholds in the raw ``.pos`` units stored by LeRobot."""

    gripper_motion_min: float = 5.0
    oscillation_flips: int = 4
    align_delta_min: float = 20.0
    elbow_still_max: float = 3.0
    lift_min: float = 10.0
    spike_min: float = 25.0
    timeout_frames: int = 900
    drop_delta: float = 5.0
    stub_confidence: float = 0.25


def miner_config_from_mapping(data: Mapping[str, object]) -> MinerConfig:
    """Build a miner config from a YAML mapping. Unknown keys are rejected."""
    unknown = set(data) - set(MinerConfig.__dataclass_fields__)
    if unknown:
        raise ValueError(f"unknown miner keys: {sorted(unknown)}")
    defaults = MinerConfig()
    integer_fields = {"oscillation_flips", "timeout_frames"}
    resolved: dict[str, int | float] = {}
    for name in MinerConfig.__dataclass_fields__:
        raw = data.get(name, getattr(defaults, name))
        if name in integer_fields:
            if isinstance(raw, bool) or not isinstance(raw, int):
                raise ValueError(f"{name} must be an int")
            resolved[name] = raw
        else:
            if isinstance(raw, bool) or not isinstance(raw, int | float):
                raise ValueError(f"{name} must be a number")
            resolved[name] = float(raw)
    return MinerConfig(
        gripper_motion_min=float(resolved["gripper_motion_min"]),
        oscillation_flips=int(resolved["oscillation_flips"]),
        align_delta_min=float(resolved["align_delta_min"]),
        elbow_still_max=float(resolved["elbow_still_max"]),
        lift_min=float(resolved["lift_min"]),
        spike_min=float(resolved["spike_min"]),
        timeout_frames=int(resolved["timeout_frames"]),
        drop_delta=float(resolved["drop_delta"]),
        stub_confidence=float(resolved["stub_confidence"]),
    )


class StubLabeler:
    """Equal-time phases plus the kinematic miner. Confidence stays low on purpose."""

    name = "stub"

    def __init__(self, config: MinerConfig | None = None) -> None:
        self.config = config or MinerConfig()

    def label(self, episode: EpisodeRecord) -> EpisodeAnnotation:
        phases = _equal_phases(episode.num_frames, self.config.stub_confidence)
        failures = _mine(episode, phases, self.config)
        return EpisodeAnnotation(
            episode_index=episode.episode_index,
            labeler=self.name,
            phases=tuple(phases),
            failures=tuple(failures),
        )


def _equal_phases(num_frames: int, confidence: float) -> list[PhaseSpan]:
    phases = (
        FoldPhase.GRASP_CORNER,
        FoldPhase.FLATTEN,
        FoldPhase.FOLD,
        FoldPhase.RELEASE,
    )
    bounds = [index * num_frames // len(phases) for index in range(len(phases) + 1)]
    spans: list[PhaseSpan] = []
    for phase, start, end in zip(phases, bounds[:-1], bounds[1:], strict=True):
        if end > start:
            spans.append(PhaseSpan(phase, start, end, confidence))
    return spans


def _mine(
    episode: EpisodeRecord,
    phases: list[PhaseSpan],
    config: MinerConfig,
) -> list[FailureEvent]:
    by_phase = {span.phase: span for span in phases}
    failures: list[FailureEvent] = []
    grasp = by_phase.get(FoldPhase.GRASP_CORNER)
    flatten = by_phase.get(FoldPhase.FLATTEN)
    fold = by_phase.get(FoldPhase.FOLD)
    release = by_phase.get(FoldPhase.RELEASE)

    if episode.num_frames > config.timeout_frames:
        failures.append(
            FailureEvent(
                FailureMode.TIMEOUT,
                0,
                episode.num_frames,
                f"length {episode.num_frames} exceeds {config.timeout_frames} frames",
            )
        )

    spike_at = _action_spike_frame(episode.actions, config.spike_min)
    if spike_at is not None:
        failures.append(
            FailureEvent(
                FailureMode.ARM_COLLISION,
                spike_at,
                min(episode.num_frames, spike_at + 2),
                f"action step exceeded {config.spike_min}",
            )
        )

    if grasp is not None:
        left_grip = _column(episode.states, LEFT_GRIPPER, grasp)
        right_grip = _column(episode.states, RIGHT_GRIPPER, grasp)
        if (
            _span(left_grip) < config.gripper_motion_min
            and _span(right_grip) < config.gripper_motion_min
        ):
            failures.append(
                FailureEvent(
                    FailureMode.MISSED_GRASP,
                    grasp.start_frame,
                    grasp.end_frame,
                    "both grippers barely moved during grasp_corner",
                )
            )
        elif _double_layer(episode, grasp, config):
            failures.append(
                FailureEvent(
                    FailureMode.DOUBLE_LAYER_GRASP,
                    grasp.start_frame,
                    grasp.end_frame,
                    "both grippers moved while both elbows stayed still and both arms lifted",
                )
            )

    if flatten is not None and _wrist_flips(episode, flatten, config):
        failures.append(
            FailureEvent(
                FailureMode.CRUMPLED_CORNER,
                flatten.start_frame,
                flatten.end_frame,
                "wrist_roll oscillated during flatten",
            )
        )

    if fold is not None and _misaligned(episode, fold, config):
        failures.append(
            FailureEvent(
                FailureMode.MISALIGNED_FOLD,
                fold.start_frame,
                fold.end_frame,
                "left and right shoulder_lift diverged during fold",
            )
        )

    pre_release_end = release.start_frame if release is not None else episode.num_frames
    if _dropped(episode, pre_release_end, config):
        failures.append(
            FailureEvent(
                FailureMode.DROP,
                0,
                pre_release_end,
                "a gripper reversed direction before release",
            )
        )

    if release is not None and grasp is not None and _missed_box(episode, grasp, release, config):
        failures.append(
            FailureEvent(
                FailureMode.MISSED_BOX,
                release.start_frame,
                release.end_frame,
                "grippers moved during grasp and stayed still during release",
            )
        )
    return failures


def _double_layer(episode: EpisodeRecord, grasp: PhaseSpan, config: MinerConfig) -> bool:
    grippers_moved = all(
        _span(_column(episode.states, joint, grasp)) >= config.gripper_motion_min
        for joint in (LEFT_GRIPPER, RIGHT_GRIPPER)
    )
    elbows_still = all(
        _span(_column(episode.states, joint, grasp)) < config.elbow_still_max
        for joint in (LEFT_ELBOW, RIGHT_ELBOW)
    )
    arms_lifted = all(
        _span(_column(episode.states, joint, grasp)) >= config.lift_min
        for joint in (LEFT_SHOULDER_LIFT, RIGHT_SHOULDER_LIFT)
    )
    return grippers_moved and elbows_still and arms_lifted


def _wrist_flips(episode: EpisodeRecord, span: PhaseSpan, config: MinerConfig) -> bool:
    for joint in (LEFT_WRIST_ROLL, RIGHT_WRIST_ROLL):
        if _sign_flips(_column(episode.states, joint, span)) >= config.oscillation_flips:
            return True
    return False


def _misaligned(episode: EpisodeRecord, span: PhaseSpan, config: MinerConfig) -> bool:
    left = _column(episode.states, LEFT_SHOULDER_LIFT, span)
    right = _column(episode.states, RIGHT_SHOULDER_LIFT, span)
    if not left:
        return False
    mean_delta = abs(sum(a - b for a, b in zip(left, right, strict=True)) / len(left))
    return mean_delta > config.align_delta_min


def _dropped(episode: EpisodeRecord, end_frame: int, config: MinerConfig) -> bool:
    for joint in (LEFT_GRIPPER, RIGHT_GRIPPER):
        series = [frame[joint] for frame in episode.states[:end_frame]]
        if _fell_then_rose(series, config.drop_delta):
            return True
    return False


def _missed_box(
    episode: EpisodeRecord,
    grasp: PhaseSpan,
    release: PhaseSpan,
    config: MinerConfig,
) -> bool:
    grasped = any(
        _span(_column(episode.states, joint, grasp)) >= config.gripper_motion_min
        for joint in (LEFT_GRIPPER, RIGHT_GRIPPER)
    )
    released = any(
        _span(_column(episode.states, joint, release)) >= config.gripper_motion_min
        for joint in (LEFT_GRIPPER, RIGHT_GRIPPER)
    )
    return grasped and not released


def _column(states: tuple[tuple[float, ...], ...], joint: int, span: PhaseSpan) -> list[float]:
    return [frame[joint] for frame in states[span.start_frame : span.end_frame]]


def _span(values: list[float]) -> float:
    if not values:
        return 0.0
    return max(values) - min(values)


def _sign_flips(values: list[float], epsilon: float = 1e-3) -> int:
    flips = 0
    previous = 0
    last = values[0] if values else 0.0
    for value in values[1:]:
        delta = value - last
        last = value
        sign = 1 if delta > epsilon else -1 if delta < -epsilon else 0
        if sign and previous and sign != previous:
            flips += 1
        if sign:
            previous = sign
    return flips


def _fell_then_rose(values: list[float], delta: float) -> bool:
    if len(values) < 2:
        return False
    baseline = values[0]
    trough = values[0]
    fell = False
    for value in values[1:]:
        if not fell:
            if baseline - value >= delta:
                fell = True
                trough = value
            else:
                baseline = max(baseline, value)
            continue
        trough = min(trough, value)
        if value - trough >= delta:
            return True
    return False


def _action_spike_frame(actions: tuple[tuple[float, ...], ...], spike_min: float) -> int | None:
    for index in range(1, len(actions)):
        step = max(abs(a - b) for a, b in zip(actions[index], actions[index - 1], strict=True))
        if step > spike_min:
            return index
    return None
