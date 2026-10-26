"""Reflex command line. CPU entry points do not download VLM or SigLIP weights."""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import yaml

from reflex.act_smoke import ActSmokeConfig, run_act_smoke
from reflex.act_train import (
    TrainLaunchError,
    launch_act_train,
    load_act_train_config,
    with_train_overrides,
)
from reflex.distill import (
    DistillError,
    benchmark_student,
    load_distill_config,
    train_distill,
)
from reflex.eval_offline import load_offline_eval_config, run_offline_eval, write_report
from reflex.registry import POLICIES
from reflex.reward import evaluate_reward, load_reward_config, train_reward
from reflex.station import (
    MockBimanualRobot,
    MockLeaderTeleop,
    StateCopyPolicy,
    load_station,
    record_mock_episode,
    render_record_command,
    run_inference_loop,
)
from reflex.vla_train import launch_vla_train, load_vla_train_config, with_vla_overrides


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="reflex", description="Reflex policy entry points")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("policies", help="list policy families and which ones run today")
    smoke = sub.add_parser("act-smoke", help="one tiny ACT step on CPU")
    smoke.add_argument("--output", type=Path, default=Path("outputs/act_smoke"))
    smoke.add_argument("--steps", type=int, default=None)
    smoke.add_argument("--config", type=Path, default=None)

    train = sub.add_parser("train", help="train ACT on the towel dataset, or run the CPU dry-run")
    train.add_argument("--config", type=Path, required=True)
    train.add_argument("--output", type=Path, default=None)
    train.add_argument("--steps", type=int, default=None)
    train.add_argument("--resume", choices=("auto", "true", "false"), default=None)
    train.add_argument("--wandb", action="store_true", help="enable Weights & Biases")
    train.add_argument("--no-wandb", action="store_true", help="force Weights & Biases off")

    reward_train = sub.add_parser("reward-train", help="train the fold-quality reward model")
    reward_train.add_argument("--config", type=Path, required=True)
    reward_train.add_argument("--output", type=Path, default=None)
    reward_train.add_argument("--steps", type=int, default=None)

    reward_eval = sub.add_parser("reward-eval", help="AUROC and calibration on the held-out split")
    reward_eval.add_argument("--config", type=Path, required=True)
    reward_eval.add_argument("--checkpoint", type=Path, default=None)

    offline = sub.add_parser("eval-offline", help="compare policy checkpoints on held-out episodes")
    offline.add_argument("--config", type=Path, required=True)

    distill = sub.add_parser("distill", help="distill teacher actions into the student policy")
    distill.add_argument("--config", type=Path, required=True)
    distill.add_argument("--output", type=Path, default=None)

    bench = sub.add_parser("bench", help="latency and VRAM estimate for the student")
    bench.add_argument("--config", type=Path, required=True)
    bench.add_argument("--steps", type=int, default=20)
    bench.add_argument(
        "--strict",
        action="store_true",
        help="exit 1 when the 30 Hz or 6 GB target is missed",
    )

    deploy = sub.add_parser("deploy", help="run the inference loop on the mock robot")
    deploy.add_argument("--steps", type=int, default=4)
    deploy.add_argument("--task", default="Fold towel")

    record_plan = sub.add_parser("record-plan", help="print the lerobot-record command")
    record_plan.add_argument(
        "--teleop",
        type=Path,
        default=Path("hardware/teleop_bimanual_so101.yaml"),
    )
    record_plan.add_argument(
        "--record",
        type=Path,
        default=Path("hardware/record_bimanual_so101.yaml"),
    )

    record_mock = sub.add_parser("record-mock", help="record one episode from the mock leader")
    record_mock.add_argument("--frames", type=int, default=4)
    record_mock.add_argument("--task", default="Fold towel")

    args = parser.parse_args(argv)
    if args.command == "policies":
        for entry in POLICIES.values():
            extra = entry.extra or "installed"
            print(f"{entry.name:10} {entry.status.value:8} extra={extra}  {entry.notes}")
        return 0
    if args.command == "act-smoke":
        config = _load_config(args.config)
        if args.steps is not None:
            config = ActSmokeConfig(
                steps=args.steps,
                batch_size=config.batch_size,
                image_height=config.image_height,
                image_width=config.image_width,
                chunk_size=config.chunk_size,
                dim_model=config.dim_model,
                n_heads=config.n_heads,
                dim_feedforward=config.dim_feedforward,
                n_encoder_layers=config.n_encoder_layers,
                n_decoder_layers=config.n_decoder_layers,
                use_vae=config.use_vae,
                seed=config.seed,
                device=config.device,
            )
        result = run_act_smoke(args.output, config)
        print(
            f"ACT smoke ok: steps={result.steps} loss={result.losses[-1]:.4f} "
            f"action_shape={result.action_shape} checkpoint={result.checkpoint_dir}"
        )
        return 0
    if args.command == "train":
        return _train(args)
    if args.command == "reward-train":
        return _reward_train(args)
    if args.command == "reward-eval":
        return _reward_eval(args)
    if args.command == "eval-offline":
        return _eval_offline(args)
    if args.command == "distill":
        return _distill(args)
    if args.command == "bench":
        return _bench(args)
    if args.command == "deploy":
        return _deploy(args)
    if args.command == "record-plan":
        return _record_plan(args)
    if args.command == "record-mock":
        return _record_mock(args)
    parser.error(f"unknown command {args.command}")
    return 2


def _train(args: argparse.Namespace) -> int:
    family = _policy_family(args.config)
    if family in {"smolvla", "pi05"}:
        return _train_vla(args)
    if family != "act":
        print(f"unknown policy_type {family}", file=sys.stderr)
        return 2
    if args.wandb and args.no_wandb:
        print("pass only one of --wandb and --no-wandb", file=sys.stderr)
        return 2
    try:
        config = load_act_train_config(args.config)
        wandb_enable: bool | None
        if args.wandb:
            wandb_enable = True
        elif args.no_wandb:
            wandb_enable = False
        else:
            wandb_enable = None
        config = with_train_overrides(
            config,
            output_dir=args.output,
            steps=args.steps,
            resume=args.resume,
            wandb_enable=wandb_enable,
        )
        result = launch_act_train(config)
    except (ValueError, FileExistsError, FileNotFoundError, TrainLaunchError) as exc:
        print(exc, file=sys.stderr)
        return 1
    except subprocess.CalledProcessError as exc:
        return exc.returncode or 1
    if result.mode == "dry-run":
        loss = result.losses[-1]
        print(
            f"ACT dry-run ok: resumed={str(result.resumed).lower()} step={result.step} "
            f"loss={loss:.4f} checkpoint={result.checkpoint_dir}"
        )
        return 0
    print(f"ACT train finished: resumed={str(result.resumed).lower()} output={result.output_dir}")
    return 0


def _train_vla(args: argparse.Namespace) -> int:
    if args.wandb and args.no_wandb:
        print("pass only one of --wandb and --no-wandb", file=sys.stderr)
        return 2
    try:
        config = load_vla_train_config(args.config)
        if args.wandb:
            wandb_enable: bool | None = True
        elif args.no_wandb:
            wandb_enable = False
        else:
            wandb_enable = None
        config = with_vla_overrides(
            config,
            output_dir=args.output,
            steps=args.steps,
            resume=args.resume,
            wandb_enable=wandb_enable,
        )
        result = launch_vla_train(config)
    except (ValueError, FileExistsError, FileNotFoundError, TrainLaunchError) as exc:
        print(exc, file=sys.stderr)
        return 1
    except subprocess.CalledProcessError as exc:
        return exc.returncode or 1
    if result.mode == "dry-run":
        loss = 0.0 if result.loss is None else result.loss
        print(
            f"{config.policy_type} dry-run ok: resumed={str(result.resumed).lower()} "
            f"step={result.step} loss={loss:.4f} task={config.task_prompt} "
            f"checkpoint={result.checkpoint_dir}"
        )
        return 0
    print(
        f"{config.policy_type} train finished: resumed={str(result.resumed).lower()} "
        f"output={result.output_dir}"
    )
    return 0


def _reward_train(args: argparse.Namespace) -> int:
    try:
        config = load_reward_config(args.config)
        if args.output is not None:
            config = replace(config, output_dir=args.output)
        if args.steps is not None:
            config = replace(config, steps=args.steps, save_freq=min(config.save_freq, args.steps))
        result = train_reward(config)
    except (ValueError, FileExistsError, FileNotFoundError, OSError) as exc:
        print(exc, file=sys.stderr)
        return 1
    metrics = result.metrics
    print(
        f"reward train ok: step={result.step} resumed={str(result.resumed).lower()} "
        f"loss={metrics.loss:.4f} frame_auroc={metrics.frame_auroc:.3f} "
        f"frame_ece={metrics.frame_ece:.3f} episode_auroc={metrics.episode_auroc:.3f} "
        f"episode_ece={metrics.episode_ece:.3f} checkpoint={result.checkpoint_dir}"
    )
    return 0


def _reward_eval(args: argparse.Namespace) -> int:
    try:
        config = load_reward_config(args.config)
        metrics = evaluate_reward(config, args.checkpoint)
    except (ValueError, FileNotFoundError, OSError) as exc:
        print(exc, file=sys.stderr)
        return 1
    print(
        f"reward eval: frame_auroc={metrics.frame_auroc:.3f} frame_ece={metrics.frame_ece:.3f} "
        f"episode_auroc={metrics.episode_auroc:.3f} episode_ece={metrics.episode_ece:.3f} "
        f"held_out={list(metrics.held_out_episodes)}"
    )
    return 0


def _eval_offline(args: argparse.Namespace) -> int:
    try:
        config = load_offline_eval_config(args.config)
        report = run_offline_eval(config)
        json_path, markdown_path = write_report(report, config.output_dir)
    except (ValueError, FileNotFoundError, OSError) as exc:
        print(exc, file=sys.stderr)
        return 1
    for item in report.policies:
        print(f"{item.name}: action_mse={item.action_mse:.6f} reward_mean={item.reward_mean}")
    print(f"wrote {json_path} and {markdown_path}")
    return 0


def _distill(args: argparse.Namespace) -> int:
    try:
        config = load_distill_config(args.config)
        if args.output is not None:
            config = replace(config, output_dir=args.output)
        result = train_distill(config)
    except (ValueError, FileExistsError, FileNotFoundError, DistillError, OSError) as exc:
        print(exc, file=sys.stderr)
        return 1
    print(
        f"distill ok: resumed={str(result.resumed).lower()} step={result.step} "
        f"loss={result.loss:.4f} params={result.parameters} checkpoint={result.checkpoint_dir}"
    )
    return 0


def _bench(args: argparse.Namespace) -> int:
    try:
        config = load_distill_config(args.config)
        result = benchmark_student(config.student, steps=args.steps, device=config.device)
    except (ValueError, DistillError, OSError) as exc:
        print(exc, file=sys.stderr)
        return 1
    print(
        f"bench: params={result.parameters} hz={result.hertz:.1f} "
        f"ms={result.milliseconds:.2f} vram_bytes={result.vram_bytes} "
        f"meets_hz={str(result.meets_hz).lower()} meets_vram={str(result.meets_vram).lower()}"
    )
    if args.strict and (not result.meets_hz or not result.meets_vram):
        return 1
    return 0


def _deploy(args: argparse.Namespace) -> int:
    try:
        result = run_inference_loop(
            MockBimanualRobot(),
            StateCopyPolicy(),
            steps=args.steps,
            task_prompt=args.task,
        )
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1
    print(
        f"deploy ok: steps={result.steps} task={result.task_prompt} "
        f"action_dim={len(result.actions[0])}"
    )
    return 0


def _record_plan(args: argparse.Namespace) -> int:
    try:
        station = load_station(args.teleop, args.record)
    except (ValueError, OSError) as exc:
        print(exc, file=sys.stderr)
        return 1
    print(render_record_command(station))
    print(f"ports_ready={str(station.ports_ready).lower()}")
    return 0


def _record_mock(args: argparse.Namespace) -> int:
    try:
        episode = record_mock_episode(
            MockBimanualRobot(),
            MockLeaderTeleop(),
            frames=args.frames,
            task=args.task,
        )
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1
    print(f"record-mock ok: frames={episode.num_frames} task={episode.task}")
    return 0


def _policy_family(path: Path) -> str:
    loaded = yaml.safe_load(path.read_text())
    if not isinstance(loaded, dict):
        raise SystemExit(f"{path} must be a mapping")
    family = loaded.get("policy_type", "act")
    if family is None:
        return "act"
    if not isinstance(family, str):
        raise SystemExit("policy_type must be a string")
    return family


def _load_config(path: Path | None) -> ActSmokeConfig:
    if path is None:
        root = _repo_root()
        if root is None:
            return ActSmokeConfig()
        path = root / "configs" / "policies" / "act_smoke.yaml"
        if not path.is_file():
            return ActSmokeConfig()
    loaded = yaml.safe_load(path.read_text())
    if not isinstance(loaded, dict):
        raise SystemExit(f"{path} must be a mapping")
    unknown = set(loaded) - set(ActSmokeConfig.__dataclass_fields__)
    if unknown:
        raise SystemExit(f"unknown ACT smoke keys in {path}: {sorted(unknown)}")
    return _config_from_mapping(loaded)


def _config_from_mapping(data: dict[str, object]) -> ActSmokeConfig:
    defaults = ActSmokeConfig()

    def _int(name: str, default: int) -> int:
        value = data.get(name, default)
        if isinstance(value, bool) or not isinstance(value, int):
            raise SystemExit(f"{name} must be an int")
        return value

    def _bool(name: str, default: bool) -> bool:
        value = data.get(name, default)
        if not isinstance(value, bool):
            raise SystemExit(f"{name} must be a bool")
        return value

    device = data.get("device", defaults.device)
    if not isinstance(device, str):
        raise SystemExit("device must be a string")
    return ActSmokeConfig(
        steps=_int("steps", defaults.steps),
        batch_size=_int("batch_size", defaults.batch_size),
        image_height=_int("image_height", defaults.image_height),
        image_width=_int("image_width", defaults.image_width),
        chunk_size=_int("chunk_size", defaults.chunk_size),
        dim_model=_int("dim_model", defaults.dim_model),
        n_heads=_int("n_heads", defaults.n_heads),
        dim_feedforward=_int("dim_feedforward", defaults.dim_feedforward),
        n_encoder_layers=_int("n_encoder_layers", defaults.n_encoder_layers),
        n_decoder_layers=_int("n_decoder_layers", defaults.n_decoder_layers),
        use_vae=_bool("use_vae", defaults.use_vae),
        seed=_int("seed", defaults.seed),
        device=device,
    )


def _repo_root() -> Path | None:
    current = Path.cwd().resolve()
    for candidate in (current, *current.parents):
        marker = candidate / "configs" / "policies" / "act_smoke.yaml"
        if marker.is_file():
            return candidate
    return None


if __name__ == "__main__":
    raise SystemExit(main())
