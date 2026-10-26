from pathlib import Path

import pytest

from reflex.cli import main
from reflex.station import (
    MockBimanualRobot,
    MockLeaderTeleop,
    StateCopyPolicy,
    load_station,
    record_mock_episode,
    render_record_command,
    run_inference_loop,
)
from shared.contract import ACTION_DIM, CAMERA_FEATURE_KEYS
from shared.interfaces import TeleopSession

ROOT = Path(__file__).resolve().parents[1]
TELEOP = ROOT / "hardware" / "teleop_bimanual_so101.yaml"
RECORD = ROOT / "hardware" / "record_bimanual_so101.yaml"


def test_station_template_keeps_ports_unset() -> None:
    station = load_station(TELEOP, RECORD)
    assert station.robot_type == "bi_so_follower"
    assert station.leader_type == "bi_so_leader"
    assert station.single_task == "Fold towel"
    assert station.ports_ready is False
    keys = {camera.feature_key for camera in station.cameras.values()}
    assert keys == set(CAMERA_FEATURE_KEYS)
    command = render_record_command(station)
    assert "--robot.type=bi_so_follower" in command
    assert "--teleop.type=bi_so_leader" in command
    assert "observation.images.left" in command
    assert "observation.images.top" in command
    assert "<unset>" in command
    assert "Fold towel" in command


def test_mock_record_and_deploy_use_the_contract() -> None:
    teleop = MockLeaderTeleop()
    assert isinstance(teleop, TeleopSession)
    episode = record_mock_episode(
        MockBimanualRobot(),
        teleop,
        frames=4,
        task="Fold towel",
    )
    assert episode.num_frames == 4
    assert len(episode.actions[3]) == ACTION_DIM
    assert episode.actions[3][5] == 3.0
    assert episode.image_refs == CAMERA_FEATURE_KEYS
    result = run_inference_loop(
        MockBimanualRobot(), StateCopyPolicy(), steps=3, task_prompt="Fold towel"
    )
    assert result.actions[0] == (0.0,) * ACTION_DIM
    assert result.actions[1] == (0.1,) * ACTION_DIM
    assert result.actions[2] == (0.2,) * ACTION_DIM
    assert result.task_prompt == "Fold towel"


def test_record_plan_and_deploy_cli(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["record-plan", "--teleop", str(TELEOP), "--record", str(RECORD)]) == 0
    plan = capsys.readouterr().out
    assert "ports_ready=false" in plan
    assert "bi_so_follower" in plan
    assert main(["deploy", "--steps", "2", "--task", "Fold towel"]) == 0
    deployed = capsys.readouterr().out
    assert "deploy ok" in deployed
    assert "action_dim=12" in deployed
    assert main(["record-mock", "--frames", "3", "--task", "Fold towel"]) == 0
    recorded = capsys.readouterr().out
    assert "frames=3" in recorded
