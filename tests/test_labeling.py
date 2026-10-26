from cortex.labeling import MinerConfig, StubLabeler
from shared.episodes import EpisodeRecord, make_episode_record
from shared.labels import FailureMode, FoldPhase


def _clean(num_frames: int = 40) -> tuple[list[list[float]], list[list[float]]]:
    states = [[0.0] * 12 for _ in range(num_frames)]
    for frame in range(num_frames):
        states[frame][2] = float(frame)
        states[frame][5] = float(frame)
        states[frame][8] = float(frame)
        states[frame][11] = float(frame)
        states[frame][1] = 0.5 * frame
        states[frame][7] = 0.5 * frame
    actions = [row.copy() for row in states]
    return states, actions


def _record(states: list[list[float]], actions: list[list[float]]) -> EpisodeRecord:
    return make_episode_record(
        episode_index=3,
        task="Fold the towel",
        fps=30,
        states=states,
        actions=actions,
    )


def _modes(
    states: list[list[float]], actions: list[list[float]], config: MinerConfig | None = None
) -> set[FailureMode]:
    annotation = StubLabeler(config).label(_record(states, actions))
    return {event.mode for event in annotation.failures}


def test_clean_episode_has_four_phases_and_no_failures() -> None:
    states, actions = _clean()
    annotation = StubLabeler().label(_record(states, actions))
    assert [span.phase for span in annotation.phases] == list(FoldPhase)
    assert annotation.phases[0].start_frame == 0
    assert annotation.phases[-1].end_frame == 40
    assert annotation.failures == ()
    assert annotation.min_phase_confidence < 0.5


def test_failure_rules_fire_on_synthetic_signals() -> None:
    states, actions = _clean()
    states = [[0.0] * 12 for _ in range(40)]
    actions = [row.copy() for row in states]
    assert _modes(states, actions) == {FailureMode.MISSED_GRASP}

    states, actions = _clean()
    for frame in range(10, 20):
        states[frame][4] = 5.0 if frame % 2 == 0 else -5.0
    actions = [row.copy() for row in states]
    assert FailureMode.CRUMPLED_CORNER in _modes(states, actions)

    states, actions = _clean()
    for frame in range(20, 30):
        states[frame][7] = states[frame][1] + 40.0
    actions = [row.copy() for row in states]
    assert FailureMode.MISALIGNED_FOLD in _modes(states, actions)

    states, actions = _clean()
    for frame in range(10):
        states[frame][2] = 0.0
        states[frame][8] = 0.0
        states[frame][1] = float(frame) * 2.0
        states[frame][7] = float(frame) * 2.0
    actions = [row.copy() for row in states]
    assert FailureMode.DOUBLE_LAYER_GRASP in _modes(states, actions)

    states, actions = _clean()
    for frame in range(10):
        states[frame][5] = 50.0 - frame * 4.0
    for frame in range(10, 20):
        states[frame][5] = 10.0 + (frame - 10) * 4.0
    actions = [row.copy() for row in states]
    assert FailureMode.DROP in _modes(states, actions)

    states, actions = _clean()
    for frame in range(30, 40):
        states[frame][5] = states[29][5]
        states[frame][11] = states[29][11]
    actions = [row.copy() for row in states]
    assert FailureMode.MISSED_BOX in _modes(states, actions)

    states, actions = _clean()
    actions[6][0] = actions[5][0] + 40.0
    assert FailureMode.ARM_COLLISION in _modes(states, actions)

    states, actions = _clean()
    assert _modes(states, actions, MinerConfig(timeout_frames=10)) == {FailureMode.TIMEOUT}
