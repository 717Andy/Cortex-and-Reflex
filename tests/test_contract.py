import json
from pathlib import Path

import pytest
import yaml

from shared.contract import (
    ACTION_DIM,
    BIMANUAL_JOINT_NAMES,
    CAMERA_ROLES,
    PUBLIC_TOWEL_REPO_ID,
    DatasetInfo,
    make_dataset_info,
    validate_dataset_info,
    validate_feature_stats,
)
from shared.episodes import make_episode_record

FIXTURES = Path(__file__).parent / "fixtures"
ROOT = Path(__file__).resolve().parents[1]


def test_joint_vector_is_twelve_bimanual_names() -> None:
    assert len(BIMANUAL_JOINT_NAMES) == ACTION_DIM == 12
    assert BIMANUAL_JOINT_NAMES[:6] == tuple(
        name.replace("right_", "left_") for name in BIMANUAL_JOINT_NAMES[6:]
    )
    assert all(name.startswith("left_") for name in BIMANUAL_JOINT_NAMES[:6])
    assert all(name.startswith("right_") for name in BIMANUAL_JOINT_NAMES[6:])


def test_generated_info_passes() -> None:
    report = validate_dataset_info(make_dataset_info())
    assert report.ok, report


def test_public_info_fixture_passes() -> None:
    info = DatasetInfo.model_validate_json((FIXTURES / "bimanual_so101_info.json").read_text())
    report = validate_dataset_info(info)
    assert report.ok, report
    assert info.features["action"].names == list(BIMANUAL_JOINT_NAMES)
    assert info.total_episodes == 120
    assert info.fps == 30


def test_public_stats_fixture_passes() -> None:
    stats = json.loads((FIXTURES / "bimanual_so101_stats.json").read_text())
    report = validate_feature_stats(stats)
    assert report.ok, report


def test_swapped_joint_names_fail() -> None:
    info = make_dataset_info()
    names = list(BIMANUAL_JOINT_NAMES)
    names[0], names[1] = names[1], names[0]
    info.features["action"].names = names
    report = validate_dataset_info(info)
    assert not report.ok
    assert any(issue.code == "joint_names" for issue in report.errors())


def test_missing_overhead_camera_fails() -> None:
    info = make_dataset_info()
    del info.features["observation.images.top"]
    report = validate_dataset_info(info)
    assert any(issue.code == "missing_camera" for issue in report.errors())


def test_v2_codebase_fails() -> None:
    info = make_dataset_info()
    info.codebase_version = "v2.1"
    report = validate_dataset_info(info)
    assert any(issue.code == "codebase_version" for issue in report.errors())


def test_episode_record_rejects_the_wrong_width() -> None:
    with pytest.raises(ValueError, match="width"):
        make_episode_record(
            episode_index=0,
            task="Fold the towel",
            fps=30,
            states=[[0.0] * 11],
            actions=[[0.0] * 11],
        )


def test_dataset_config_matches_the_contract() -> None:
    loaded = yaml.safe_load(
        (ROOT / "configs" / "dataset" / "bimanual_so101_towel.yaml").read_text()
    )
    assert loaded["repo_id"] == PUBLIC_TOWEL_REPO_ID
    assert loaded["action_dim"] == ACTION_DIM
    assert loaded["cameras"] == CAMERA_ROLES
