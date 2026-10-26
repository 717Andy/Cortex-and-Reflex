from dataclasses import replace
from pathlib import Path

from reflex.eval_offline import (
    held_out_split,
    load_offline_eval_config,
    render_markdown,
    run_offline_eval,
    write_report,
)
from reflex.reward import load_reward_config, train_reward
from reflex.vla_train import write_mock_policy_checkpoint

ROOT = Path(__file__).resolve().parents[1]


def test_held_out_split_is_deterministic() -> None:
    train, held = held_out_split(range(10), fraction=0.2, seed=1)
    assert held_out_split(range(10), fraction=0.2, seed=1) == (train, held)
    assert set(train).isdisjoint(held)
    assert set(train) | set(held) == set(range(10))
    assert held


def test_offline_report_ranks_identity_ahead_of_zero(tmp_path: Path) -> None:
    reward_cfg = replace(
        load_reward_config(ROOT / "configs" / "reward" / "fold_quality_cpu.yaml"),
        output_dir=tmp_path / "reward",
        steps=2,
        save_freq=2,
    )
    trained = train_reward(reward_cfg)
    policies = {
        "act": write_mock_policy_checkpoint(tmp_path / "act", policy_type="act", kind="identity"),
        "smolvla": write_mock_policy_checkpoint(
            tmp_path / "smolvla", policy_type="smolvla", kind="identity"
        ),
        "pi05": write_mock_policy_checkpoint(tmp_path / "pi05", policy_type="pi05", kind="zero"),
    }
    config = load_offline_eval_config(ROOT / "configs" / "eval" / "offline_cpu.yaml")
    config = replace(
        config,
        policies=policies,
        reward_config=ROOT / "configs" / "reward" / "fold_quality_cpu.yaml",
        reward_checkpoint=trained.checkpoint_dir,
        output_dir=tmp_path / "eval",
        synthetic_episodes=6,
    )
    report = run_offline_eval(config)
    by_name = {item.name: item for item in report.policies}
    assert set(by_name) == {"act", "smolvla", "pi05"}
    assert by_name["act"].action_mse == by_name["smolvla"].action_mse
    assert by_name["act"].action_mse < by_name["pi05"].action_mse
    assert by_name["act"].reward_mean is not None
    success = by_name["act"].success_rate
    assert success is not None
    assert 0.0 <= success <= 1.0
    assert "left_gripper.pos" in by_name["pi05"].per_joint_mse
    json_path, markdown_path = write_report(report, config.output_dir)
    text = json_path.read_text()
    assert '"action_mse"' in text
    rendered = markdown_path.read_text()
    assert rendered == render_markdown(report) + "\n"
    assert "Offline evaluation" in rendered
    assert "Fold towel" in rendered
