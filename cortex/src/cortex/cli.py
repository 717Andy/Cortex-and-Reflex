"""Command line for ingest, the stub labeler, and the text dashboard."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from cortex.dashboard import render_dashboard
from cortex.ingest import load_bimanual_dataset
from cortex.labeling import MinerConfig, StubLabeler, miner_config_from_mapping
from cortex.pipeline import run_labeling
from cortex.sources import InMemoryEpisodeSource
from shared.contract import PUBLIC_TOWEL_REPO_ID, DatasetContractError
from shared.episodes import make_episode_record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cortex", description="Cortex data engine")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="download dataset metadata and validate it")
    ingest.add_argument("--repo", default=None, help="Hugging Face dataset repo id")
    ingest.add_argument("--root", type=Path, default=None, help="local dataset directory")
    ingest.add_argument("--revision", default=None)
    ingest.add_argument("--cache-dir", type=Path, default=None)

    sub.add_parser("label-demo", help="label one synthetic episode with the stub labeler")

    args = parser.parse_args(argv)
    if args.command == "ingest":
        repo = args.repo or _dataset_repo()
        return _ingest(repo, args.root, args.revision, args.cache_dir)
    if args.command == "label-demo":
        return _label_demo()
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
