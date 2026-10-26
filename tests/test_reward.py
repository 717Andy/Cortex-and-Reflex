from dataclasses import replace
from pathlib import Path

import pytest

from reflex.cli import main
from reflex.reward import (
    auroc,
    evaluate_reward,
    expected_calibration_error,
    frame_targets,
    load_reward_config,
    train_reward,
)
from shared.contract import CAMERA_FEATURE_KEYS
from shared.labels import EpisodeAnnotation, FailureEvent, FailureMode, FoldPhase, PhaseSpan

ROOT = Path(__file__).resolve().parents[1]
CPU_CONFIG = ROOT / "configs" / "reward" / "fold_quality_cpu.yaml"
GPU_CONFIG = ROOT / "configs" / "reward" / "fold_quality_gpu.yaml"


def test_frame_targets_use_phase_confidence_and_failure_spans() -> None:
    annotation = EpisodeAnnotation(
        episode_index=3,
        labeler="vlm",
        phases=(PhaseSpan(FoldPhase.FOLD, 0, 4, 0.8),),
        failures=(FailureEvent(FailureMode.DROP, 2, 4, "slip"),),
        success=True,
    )
    quality, success = frame_targets(annotation, 4)
    assert quality[:2] == [0.8, 0.8]
    assert success[:2] == [1.0, 1.0]
    assert quality[2:] == [0.0, 0.0]
    assert success[2:] == [0.0, 0.0]


def test_auroc_and_calibration() -> None:
    assert auroc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == pytest.approx(1.0)
    assert auroc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) == pytest.approx(0.0)
    assert auroc([0, 1], [0.4, 0.4]) == pytest.approx(0.5)
    assert auroc([1, 1], [0.2, 0.9]) == pytest.approx(0.5)
    assert expected_calibration_error([0, 0, 1, 1], [0.0, 0.0, 1.0, 1.0], bins=2) == pytest.approx(
        0.0
    )


def test_reward_configs() -> None:
    cpu = load_reward_config(CPU_CONFIG)
    gpu = load_reward_config(GPU_CONFIG)
    assert cpu.encoder == "tiny"
    assert cpu.synthetic is True
    assert cpu.device == "cpu"
    assert cpu.cameras == CAMERA_FEATURE_KEYS
    assert gpu.encoder == "siglip2"
    assert gpu.siglip_model == "google/siglip2-base-patch16-224"
    assert gpu.freeze_encoder is True
    assert gpu.synthetic is False
    assert gpu.device == "cuda"
    assert gpu.annotations is not None


def test_cpu_reward_learns_the_brightness_cue(tmp_path: Path) -> None:
    config = replace(
        load_reward_config(CPU_CONFIG),
        output_dir=tmp_path / "reward",
        steps=40,
        save_freq=40,
        learning_rate=0.05,
    )
    result = train_reward(config)
    metrics = result.metrics
    assert metrics.frame_auroc > 0.9
    assert metrics.episode_auroc > 0.9
    assert metrics.frame_ece < 0.2
    assert metrics.episode_ece < 0.2
    assert (result.checkpoint_dir / "model.pt").is_file()
    again = evaluate_reward(config, result.checkpoint_dir)
    assert again.frame_auroc == pytest.approx(metrics.frame_auroc)


def test_cpu_reward_cli_resumes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    output = tmp_path / "reward"
    args = ["reward-train", "--config", str(CPU_CONFIG), "--output", str(output), "--steps", "2"]
    assert main(args) == 0
    first = capsys.readouterr().out
    assert "resumed=false" in first
    assert "step=2" in first
    assert main(args) == 0
    second = capsys.readouterr().out
    assert "resumed=true" in second
    assert "step=4" in second
