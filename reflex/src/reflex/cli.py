"""Reflex command line. ``act-smoke`` is the only training entry point that runs on CPU."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from reflex.act_smoke import ActSmokeConfig, run_act_smoke
from reflex.registry import POLICIES


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="reflex", description="Reflex policy entry points")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("policies", help="list policy families and which ones run today")
    smoke = sub.add_parser("act-smoke", help="one tiny ACT step on CPU")
    smoke.add_argument("--output", type=Path, default=Path("outputs/act_smoke"))
    smoke.add_argument("--steps", type=int, default=None)
    smoke.add_argument("--config", type=Path, default=None)

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
    parser.error(f"unknown command {args.command}")
    return 2


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
