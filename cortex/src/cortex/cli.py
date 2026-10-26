"""Command line for ingest, labeling, and the text dashboard."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import yaml

from cortex.curation import (
    CurationConfig,
    curate,
    curation_config_from_mapping,
    load_annotations,
    load_reward_scores,
    write_split,
)
from cortex.dashboard import render_dashboard
from cortex.dataset_episodes import records_from_dataset
from cortex.ingest import load_bimanual_dataset
from cortex.labeling import MinerConfig, StubLabeler, miner_config_from_mapping
from cortex.pipeline import run_labeling
from cortex.sources import InMemoryEpisodeSource
from cortex.vlm import annotation_to_dict, make_vlm_labeler
from cortex.vlm_backend import VlmBackendError
from cortex.vlm_schema import LabelParseError
from shared.contract import PUBLIC_TOWEL_REPO_ID, DatasetContractError
from shared.episodes import make_episode_record


@dataclass(frozen=True)
class VlmSettings:
    backend: str = "hf"
    model: str | None = "HuggingFaceTB/SmolVLM-256M-Instruct"
    retries: int = 2
    uncertain_below: float = 0.5
    max_new_tokens: int = 768
    response_cache: Path = Path("outputs/vlm_cache")
    device: str = "cpu"
    timeout_s: float = 120.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cortex", description="Cortex data engine")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="download dataset metadata and validate it")
    ingest.add_argument("--repo", default=None, help="Hugging Face dataset repo id")
    ingest.add_argument("--root", type=Path, default=None, help="local dataset directory")
    ingest.add_argument("--revision", default=None)
    ingest.add_argument("--cache-dir", type=Path, default=None)

    sub.add_parser("label-demo", help="label one synthetic episode with the stub labeler")

    label = sub.add_parser("label", help="label episodes with a VLM backend")
    label.add_argument("--dataset", default=None, help="Hugging Face dataset repo id")
    label.add_argument("--root", type=Path, default=None, help="local dataset directory")
    label.add_argument("--revision", default=None)
    label.add_argument("--hub-cache", type=Path, default=None, help="Hugging Face download cache")
    label.add_argument("--limit", type=int, required=True, help="how many episodes to label")
    label.add_argument("--backend", choices=("hf", "openai", "fake"), default=None)
    label.add_argument("--model", default=None, help="model id; openai falls back to env vars")
    label.add_argument("--retries", type=int, default=None, help="extra schema attempts")
    label.add_argument("--device", default=None, help="torch device for the hf backend")
    label.add_argument("--response-cache", type=Path, default=None)
    label.add_argument("--output", type=Path, default=Path("outputs/labels"))
    label.add_argument("--uncertain-below", type=float, default=None)
    label.add_argument(
        "--download-videos",
        action="store_true",
        help="also download video shards before labeling",
    )

    curate_cmd = sub.add_parser(
        "curate",
        help="select a train/held-out split from labels, versus a random baseline",
    )
    curate_cmd.add_argument("--annotations", type=Path, required=True)
    curate_cmd.add_argument("--output", type=Path, required=True)
    curate_cmd.add_argument("--config", type=Path, default=None)
    curate_cmd.add_argument("--strategy", choices=("quality", "diversity", "reward", "random"))
    curate_cmd.add_argument("--reward-scores", type=Path, default=None)
    curate_cmd.add_argument("--seed", type=int, default=None)
    curate_cmd.add_argument("--train-fraction", type=float, default=None)
    curate_cmd.add_argument("--dataset", default=None)
    curate_cmd.add_argument("--min-confidence", type=float, default=None)
    curate_cmd.add_argument("--require-success", action="store_true")

    args = parser.parse_args(argv)
    if args.command == "ingest":
        repo = args.repo or _dataset_repo()
        return _ingest(repo, args.root, args.revision, args.cache_dir)
    if args.command == "label-demo":
        return _label_demo()
    if args.command == "label":
        return _label(args)
    if args.command == "curate":
        return _curate(args)
    parser.error(f"unknown command {args.command}")
    return 2


def _ingest(repo: str, root: Path | None, revision: str | None, cache_dir: Path | None) -> int:
    try:
        loaded = load_bimanual_dataset(
            repo,
            root=root,
            revision=revision,
            cache_dir=cache_dir,
        )
    except DatasetContractError as exc:
        print(exc, file=sys.stderr)
        return 1
    print(loaded.summary())
    return 0


def _label_demo() -> int:
    config = _load_miner_config()
    states = [[0.0] * 12 for _ in range(40)]
    for frame in range(10):
        states[frame][5] = 50.0 - 4.0 * frame
        states[frame][11] = 50.0 - 4.0 * frame
    for frame in range(10, 40):
        states[frame][5] = 10.0
        states[frame][11] = 10.0
    actions = [list(row) for row in states]
    episode = make_episode_record(
        episode_index=0,
        task="Fold the towel",
        fps=30,
        states=states,
        actions=actions,
    )
    run = run_labeling(InMemoryEpisodeSource((episode,)), StubLabeler(config))
    annotation = run.annotations[0]
    print(f"labeler: {annotation.labeler}")
    for span in annotation.phases:
        print(f"phase {span.phase.value}: {span.start_frame}:{span.end_frame}")
    for event in annotation.failures:
        print(f"failure {event.mode.value}: {event.detail}")
    print(render_dashboard(run.dashboard))
    return 0


def _label(args: argparse.Namespace) -> int:
    settings = _load_vlm_settings()
    backend = args.backend or settings.backend
    retries = settings.retries if args.retries is None else args.retries
    uncertain_below = (
        settings.uncertain_below if args.uncertain_below is None else args.uncertain_below
    )
    device = args.device or settings.device
    response_cache = args.response_cache or settings.response_cache
    model = _label_model(backend, args.model, settings)
    repo = args.dataset or _dataset_repo()
    if args.limit < 1:
        print("--limit must be >= 1", file=sys.stderr)
        return 2
    try:
        root = _maybe_download_videos(args, repo)
        loaded = load_bimanual_dataset(
            repo,
            root=root,
            revision=args.revision,
            cache_dir=args.hub_cache,
        )
        available = len(loaded.canonical_episodes)
        limit = args.limit
        if limit > available:
            print(f"limit {limit} exceeds {available} episodes; labeling {available}")
            limit = available
        if limit < 1:
            print("dataset has no episodes to label", file=sys.stderr)
            return 1
        records = records_from_dataset(loaded, limit=limit)
        labeler = make_vlm_labeler(
            backend,
            model=model,
            retries=retries,
            cache_dir=response_cache,
            device=device,
            max_new_tokens=settings.max_new_tokens,
            timeout_s=settings.timeout_s,
        )
        run = run_labeling(
            InMemoryEpisodeSource(records),
            labeler,
            uncertain_below=uncertain_below,
        )
    except (DatasetContractError, LabelParseError, VlmBackendError, ValueError, OSError) as exc:
        print(exc, file=sys.stderr)
        return 1
    args.output.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(annotation_to_dict(annotation)) for annotation in run.annotations]
    (args.output / "annotations.jsonl").write_text("\n".join(lines) + "\n")
    dashboard = render_dashboard(run.dashboard)
    (args.output / "dashboard.txt").write_text(dashboard + "\n")
    for annotation in run.annotations:
        print(
            f"episode {annotation.episode_index}: success={annotation.success} "
            f"min_confidence={annotation.min_phase_confidence:.2f} "
            f"notes={annotation.quality_notes}"
        )
        for segment in annotation.segments:
            print(
                f"  {segment.phase.value}: {segment.start_frame}:{segment.end_frame} "
                f"confidence={segment.confidence:.2f}"
            )
    for item in run.queue.pending():
        print(f"queue episode {item.episode_index} {item.reason.value} {item.detail}")
    print(dashboard)
    return 0


def _curate(args: argparse.Namespace) -> int:
    try:
        loaded = _load_curation_config(args.config)
        config = CurationConfig(
            strategy=loaded.strategy if args.strategy is None else str(args.strategy),
            seed=loaded.seed if args.seed is None else args.seed,
            train_fraction=(
                loaded.train_fraction if args.train_fraction is None else args.train_fraction
            ),
            min_confidence=(
                loaded.min_confidence if args.min_confidence is None else args.min_confidence
            ),
            require_success=loaded.require_success or bool(args.require_success),
            dataset_repo_id=args.dataset or loaded.dataset_repo_id,
        )
        annotations = load_annotations(args.annotations)
        rewards = None if args.reward_scores is None else load_reward_scores(args.reward_scores)
        split = curate(annotations, config, rewards)
        write_split(split, args.output)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(exc, file=sys.stderr)
        return 1
    comparison = split.comparison
    print(
        f"curated {config.strategy}: train={list(split.train_episodes)} "
        f"held_out={list(split.held_out_episodes)} "
        f"quality={comparison.selected_mean_quality:.3f} "
        f"random_quality={comparison.random_mean_quality:.3f} "
        f"overlap={comparison.train_overlap_with_random:.3f}"
    )
    print(f"wrote {args.output}")
    return 0


def _load_curation_config(path: Path | None) -> CurationConfig:
    if path is None:
        root = _repo_root()
        if root is None:
            return CurationConfig()
        path = root / "configs" / "curation" / "quality.yaml"
        if not path.is_file():
            return CurationConfig()
    loaded = yaml.safe_load(path.read_text())
    if not isinstance(loaded, dict):
        raise SystemExit(f"{path} must be a mapping")
    try:
        return curation_config_from_mapping(loaded)
    except ValueError as exc:
        raise SystemExit(f"{path}: {exc}") from exc


def _label_model(backend: str, cli_model: str | None, settings: VlmSettings) -> str | None:
    if backend == "openai":
        return cli_model
    if backend == "fake":
        return cli_model or "fake"
    return cli_model or settings.model


def _maybe_download_videos(args: argparse.Namespace, repo: str) -> Path | None:
    root = cast(Path | None, args.root)
    if not args.download_videos:
        return root
    if root is not None:
        print("--download-videos ignored because --root is set", file=sys.stderr)
        return root
    from huggingface_hub import snapshot_download

    snapshot = snapshot_download(
        repo_id=repo,
        repo_type="dataset",
        revision=args.revision,
        cache_dir=None if args.hub_cache is None else str(args.hub_cache),
        allow_patterns=["meta/**", "videos/**"],
    )
    return Path(snapshot)


def _load_miner_config() -> MinerConfig:
    root = _repo_root()
    if root is None:
        return MinerConfig()
    path = root / "configs" / "labeling" / "stub_miner.yaml"
    if not path.is_file():
        return MinerConfig()
    loaded = yaml.safe_load(path.read_text())
    if not isinstance(loaded, dict):
        raise SystemExit(f"{path} must be a mapping")
    try:
        return miner_config_from_mapping(loaded)
    except ValueError as exc:
        raise SystemExit(f"{path}: {exc}") from exc


def _load_vlm_settings() -> VlmSettings:
    root = _repo_root()
    if root is None:
        return VlmSettings()
    path = root / "configs" / "labeling" / "vlm.yaml"
    if not path.is_file():
        return VlmSettings()
    loaded = yaml.safe_load(path.read_text())
    if not isinstance(loaded, dict):
        raise SystemExit(f"{path} must be a mapping")
    unknown = set(loaded) - set(VlmSettings.__dataclass_fields__)
    if unknown:
        raise SystemExit(f"unknown VLM keys in {path}: {sorted(unknown)}")
    try:
        return _vlm_settings_from_mapping(loaded)
    except ValueError as exc:
        raise SystemExit(f"{path}: {exc}") from exc


def _vlm_settings_from_mapping(data: dict[str, object]) -> VlmSettings:
    defaults = VlmSettings()
    backend = data.get("backend", defaults.backend)
    if backend not in {"hf", "openai", "fake"}:
        raise ValueError("backend must be hf, openai, or fake")
    model = data.get("model", defaults.model)
    if model is not None and not isinstance(model, str):
        raise ValueError("model must be a string or null")
    retries = _required_int(data, "retries", defaults.retries)
    max_new_tokens = _required_int(data, "max_new_tokens", defaults.max_new_tokens)
    uncertain = data.get("uncertain_below", defaults.uncertain_below)
    if isinstance(uncertain, bool) or not isinstance(uncertain, int | float):
        raise ValueError("uncertain_below must be a number")
    timeout = data.get("timeout_s", defaults.timeout_s)
    if isinstance(timeout, bool) or not isinstance(timeout, int | float):
        raise ValueError("timeout_s must be a number")
    device = data.get("device", defaults.device)
    if not isinstance(device, str) or not device:
        raise ValueError("device must be a non-empty string")
    cache = data.get("response_cache", defaults.response_cache)
    if not isinstance(cache, str | Path):
        raise ValueError("response_cache must be a path")
    if retries < 0 or max_new_tokens < 1 or float(timeout) <= 0:
        raise ValueError("retries, max_new_tokens, and timeout_s are out of range")
    if not 0.0 <= float(uncertain) <= 1.0:
        raise ValueError("uncertain_below must be between 0 and 1")
    return VlmSettings(
        backend=str(backend),
        model=model,
        retries=retries,
        uncertain_below=float(uncertain),
        max_new_tokens=max_new_tokens,
        response_cache=Path(cache),
        device=device,
        timeout_s=float(timeout),
    )


def _required_int(data: dict[str, object], name: str, default: int) -> int:
    value = data.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an int")
    return value


def _dataset_repo() -> str:
    root = _repo_root()
    if root is None:
        return PUBLIC_TOWEL_REPO_ID
    path = root / "configs" / "dataset" / "bimanual_so101_towel.yaml"
    loaded = yaml.safe_load(path.read_text())
    if not isinstance(loaded, dict) or not isinstance(loaded.get("repo_id"), str):
        raise SystemExit(f"{path} must set repo_id")
    return str(loaded["repo_id"])


def _repo_root() -> Path | None:
    current = Path.cwd().resolve()
    for candidate in (current, *current.parents):
        marker = candidate / "configs" / "labeling" / "stub_miner.yaml"
        if marker.is_file():
            return candidate
    return None


if __name__ == "__main__":
    raise SystemExit(main())
