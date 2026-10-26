import json
from dataclasses import replace
from pathlib import Path

import pytest

from reflex.act_train import lerobot_train_argv, load_act_train_config
from reflex.cli import main
from reflex.vla_train import launch_vla_train, lerobot_vla_argv, load_vla_train_config
from shared.contract import CAMERA_FEATURE_KEYS, PUBLIC_TOWEL_REPO_ID
from shared.splits import EpisodeScore, EpisodeSplit, SplitComparison, save_episode_split

ROOT = Path(__file__).resolve().parents[1]
SMOL_GPU = ROOT / "configs" / "policies" / "smolvla_towel_gpu.yaml"
SMOL_CPU = ROOT / "configs" / "policies" / "smolvla_towel_cpu_dry_run.yaml"
PI_GPU = ROOT / "configs" / "policies" / "pi05_towel_gpu.yaml"
PI_CPU = ROOT / "configs" / "policies" / "pi05_towel_cpu_dry_run.yaml"
ACT_GPU = ROOT / "configs" / "policies" / "act_towel_gpu.yaml"


def _split(path: Path) -> None:
    save_episode_split(
        EpisodeSplit(
            dataset_repo_id=PUBLIC_TOWEL_REPO_ID,
            strategy="quality",
            seed=0,
            train_fraction=0.5,
            train_episodes=(0, 2, 4),
            held_out_episodes=(1, 3),
            baseline_strategy="random",
            baseline_train_episodes=(1, 2, 3),
            comparison=SplitComparison(0.8, 0.4, None, None, 0.33),
            scores={0: EpisodeScore(0.8, 0.1, None)},
        ),
        path,
    )


def test_smolvla_and_pi05_are_full_finetunes() -> None:
    smol = load_vla_train_config(SMOL_GPU)
    pi = load_vla_train_config(PI_GPU)
    assert smol.policy_type == "smolvla"
    assert pi.policy_type == "pi05"
    assert smol.task_prompt == "Fold towel"
    assert pi.task_prompt == "Fold towel"
    assert smol.cameras == CAMERA_FEATURE_KEYS
    assert pi.action_dim == 12
    assert smol.policy.freeze_vision_encoder is False
    assert smol.policy.train_expert_only is False
    assert pi.policy.freeze_vision_encoder is False
    assert pi.policy.train_expert_only is False
    assert smol.policy.pretrained_path == "lerobot/smolvla_base"
    assert pi.policy.pretrained_path == "lerobot/pi05_base"
    assert smol.save_freq == 2000
    assert pi.save_freq == 2000
    smol_argv = " ".join(lerobot_vla_argv(smol))
    pi_argv = " ".join(lerobot_vla_argv(pi))
    assert "--policy.type=smolvla" in smol_argv
    assert "--policy.pretrained_path=lerobot/smolvla_base" in smol_argv
    assert "--policy.freeze_vision_encoder=false" in smol_argv
    assert "--policy.train_expert_only=false" in smol_argv
    assert "--policy.load_vlm_weights=true" in smol_argv
    assert "--policy.path=" not in smol_argv
    assert "--policy.type=pi05" in pi_argv
    assert "--policy.pretrained_path=lerobot/pi05_base" in pi_argv
    assert "--policy.dtype=bfloat16" in pi_argv
    assert "--policy.gradient_checkpointing=true" in pi_argv
    assert "--policy.load_vlm_weights=" not in pi_argv
    assert "--save_freq=2000" in pi_argv
    assert "--env_eval_freq=0" in smol_argv
    with pytest.raises(ValueError, match="dry_run"):
        replace(smol, dry_run=True)


def test_curated_episodes_reach_the_act_and_vla_argv(tmp_path: Path) -> None:
    split = tmp_path / "split.json"
    _split(split)
    act = replace(load_act_train_config(ACT_GPU), episodes_file=split, output_dir=tmp_path / "act")
    smol = replace(
        load_vla_train_config(SMOL_GPU), episodes_file=split, output_dir=tmp_path / "smol"
    )
    assert "--dataset.episodes=[0,2,4]" in lerobot_train_argv(act)
    assert "--dataset.episodes=[0,2,4]" in lerobot_vla_argv(smol)


def test_cpu_dry_runs_resume_without_downloading_weights(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    for config_path, family in ((SMOL_CPU, "smolvla"), (PI_CPU, "pi05")):
        output = tmp_path / family
        args = ["train", "--config", str(config_path), "--output", str(output)]
        assert main(args) == 0
        first = capsys.readouterr().out
        assert f"{family} dry-run ok" in first
        assert "resumed=false" in first
        assert "Fold towel" in first
        pretrained = output / "checkpoints" / "last" / "pretrained_model"
        saved = json.loads((pretrained / "config.json").read_text())
        assert saved["mock"] is True
        assert saved["task_prompt"] == "Fold towel"
        assert saved["cameras"] == list(CAMERA_FEATURE_KEYS)
        assert saved["freeze_vision_encoder"] is False
        assert (output / "task_prompt.txt").read_text().strip() == "Fold towel"
        assert main(args) == 0
        second = capsys.readouterr().out
        assert "resumed=true" in second
        assert "step=2" in second
        manifest = json.loads((output / "launch_manifest.json").read_text())
        assert manifest["step"] == 2


def test_real_launch_is_not_executed_in_the_dry_run(tmp_path: Path) -> None:
    config = replace(load_vla_train_config(PI_CPU), output_dir=tmp_path / "pi")
    seen: list[list[str]] = []
    result = launch_vla_train(config, runner=seen.append)
    assert result.mode == "dry-run"
    assert seen == []
