import json
import math
from dataclasses import replace
from pathlib import Path

import pytest

from reflex.act_train import launch_act_train, lerobot_train_argv, load_act_train_config
from reflex.cli import main
from shared.contract import CAMERA_FEATURE_KEYS, PUBLIC_TOWEL_REPO_ID

ROOT = Path(__file__).resolve().parents[1]
GPU_CONFIG = ROOT / "configs" / "policies" / "act_towel_gpu.yaml"
CPU_CONFIG = ROOT / "configs" / "policies" / "act_towel_cpu_dry_run.yaml"


def test_gpu_config_targets_the_towel_contract() -> None:
    config = load_act_train_config(GPU_CONFIG)
    assert config.dataset_repo_id == PUBLIC_TOWEL_REPO_ID
    assert config.device == "cuda"
    assert config.dry_run is False
    assert config.wandb.enable is False
    assert config.save_freq == 2000
    assert config.save_freq < config.steps
    assert config.action_dim == 12
    assert config.cameras == CAMERA_FEATURE_KEYS
    argv = " ".join(lerobot_train_argv(config))
    assert "--dataset.repo_id=humjie/bimanual-so101-fold-towel" in argv
    assert "--policy.type=act" in argv
    assert "--policy.device=cuda" in argv
    assert "--policy.use_amp=true" in argv
    assert "--policy.push_to_hub=false" in argv
    assert "--save_freq=2000" in argv
    assert "--env_eval_freq=0" in argv
    assert "--wandb.enable=false" in argv
    assert "ResNet18_Weights.IMAGENET1K_V1" in argv
    with pytest.raises(ValueError, match="dry_run"):
        replace(config, dry_run=True)


def test_resume_argv_points_at_the_spot_checkpoint(tmp_path: Path) -> None:
    config = replace(load_act_train_config(GPU_CONFIG), output_dir=tmp_path / "run")
    fresh = lerobot_train_argv(config)
    assert not any(part.startswith("--resume=") for part in fresh)
    checkpoint = (
        config.output_dir / "checkpoints" / "last" / "pretrained_model" / "train_config.json"
    )
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text("{}\n")
    resumed = lerobot_train_argv(config, resume_config=checkpoint)
    text = " ".join(resumed)
    assert f"--config_path={checkpoint}" in text
    assert "--resume=true" in text
    assert "--policy.chunk_size=" not in text
    seen: list[list[str]] = []
    result = launch_act_train(config, runner=seen.append)
    assert result.resumed is True
    assert seen[0][1:3] == ["-m", "lerobot.scripts.lerobot_train"]
    assert f"--config_path={checkpoint}" in seen[0]


def test_cpu_dry_run_config_is_tiny() -> None:
    config = load_act_train_config(CPU_CONFIG)
    assert config.dry_run is True
    assert config.device == "cpu"
    assert config.steps == 1
    assert config.policy.pretrained_backbone_weights is None
    assert config.policy.use_vae is False
    assert "--policy.pretrained_backbone_weights=null" in lerobot_train_argv(config)


@pytest.mark.smoke
def test_cpu_dry_run_resumes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    output = tmp_path / "dry"
    args = ["train", "--config", str(CPU_CONFIG), "--output", str(output)]
    assert main(args) == 0
    first = capsys.readouterr().out
    assert "resumed=false" in first
    assert "step=1" in first
    pretrained = output / "checkpoints" / "last" / "pretrained_model"
    assert (pretrained / "config.json").is_file()
    assert (pretrained / "train_config.json").is_file()
    assert (output / "checkpoints" / "last" / "training_state.pt").is_file()
    saved = json.loads((pretrained / "train_config.json").read_text())
    assert saved["dataset_repo_id"] == PUBLIC_TOWEL_REPO_ID
    assert saved["cameras"] == list(CAMERA_FEATURE_KEYS)
    assert saved["action_dim"] == 12

    assert main(args) == 0
    second = capsys.readouterr().out
    assert "resumed=true" in second
    assert "step=2" in second
    match_loss = second.rsplit("loss=", 1)[-1].split()[0]
    assert math.isfinite(float(match_loss))
    manifest = json.loads((output / "launch_manifest.json").read_text())
    assert manifest["resumed"] is True
    assert manifest["step"] == 2
    assert (output / "checkpoints" / "last").resolve().name == "000002"
