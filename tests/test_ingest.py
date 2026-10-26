from pathlib import Path

import pytest

from cortex.ingest import load_bimanual_dataset
from shared.contract import PUBLIC_TOWEL_REPO_ID, DatasetContractError
from tests.support import write_local_dataset


def test_local_dataset_round_trip(tmp_path: Path) -> None:
    write_local_dataset(tmp_path, [10, 20, 15])
    loaded = load_bimanual_dataset("local/towel", root=tmp_path)
    assert loaded.report.ok
    assert len(loaded.canonical_episodes) == 3
    assert loaded.total_frames == 45
    assert loaded.tasks[0].task == "Fold the towel"
    assert loaded.orphan_episode_rows == 0


def test_orphan_episode_rows_are_warnings(tmp_path: Path) -> None:
    write_local_dataset(tmp_path, [10, 20], orphan=True)
    loaded = load_bimanual_dataset("local/towel", root=tmp_path)
    assert loaded.report.ok
    assert loaded.orphan_episode_rows == 1
    assert any(issue.code == "orphan_episode_rows" for issue in loaded.report.issues)
    assert [episode.episode_index for episode in loaded.canonical_episodes] == [0, 1]


def test_broken_joint_names_raise(tmp_path: Path) -> None:
    write_local_dataset(tmp_path, [8])
    info_path = tmp_path / "meta" / "info.json"
    text = info_path.read_text().replace("left_gripper.pos", "left_gripper.open")
    info_path.write_text(text)
    with pytest.raises(DatasetContractError) as caught:
        load_bimanual_dataset("local/towel", root=tmp_path)
    assert any(issue.code == "joint_names" for issue in caught.value.report.errors())


@pytest.mark.integration
def test_public_towel_dataset_matches_the_contract() -> None:
    loaded = load_bimanual_dataset(PUBLIC_TOWEL_REPO_ID)
    assert loaded.info.robot_type
    assert loaded.info.total_episodes == 120
    assert loaded.total_frames == 21717
    assert len(loaded.canonical_episodes) == 120
    assert loaded.orphan_episode_rows > 0
    assert "Fold towel" in {task.task for task in loaded.tasks}
    assert "observation.images.top" in loaded.info.features
