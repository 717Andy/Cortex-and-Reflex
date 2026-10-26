from collections.abc import Sequence

from shared.interfaces import EvalHarness, EvalResult, FoldSim, TeleopSession


class _Sim:
    def reset(self, task: str) -> dict[str, object]:
        return {"task": task}

    def step(self, action: Sequence[float]) -> dict[str, object]:
        return {"action_dim": len(action)}


class _Teleop:
    def start(self) -> None:
        return None

    def read_action(self) -> Sequence[float]:
        return (0.0,) * 12

    def stop(self) -> None:
        return None


class _Harness:
    def evaluate(self, policy: str, *, task: str, episodes: int) -> EvalResult:
        return EvalResult(policy, task, episodes, successes=1, notes="fixture")


def test_protocols_accept_local_stand_ins() -> None:
    assert isinstance(_Sim(), FoldSim)
    assert isinstance(_Teleop(), TeleopSession)
    assert isinstance(_Harness(), EvalHarness)
    result = _Harness().evaluate("act", task="towel", episodes=4)
    assert result.success_rate == 0.25
