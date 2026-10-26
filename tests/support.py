"""Builders for a tiny local LeRobot v3 dataset."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from shared.contract import ACTION_DIM, CAMERA_FEATURE_KEYS, make_dataset_info


def write_local_dataset(
    root: Path,
    lengths: list[int],
    *,
    orphan: bool = False,
    fps: int = 30,
    task: str = "Fold the towel",
) -> None:
    frames = sum(lengths)
    info = make_dataset_info(episodes=len(lengths), frames=frames, fps=fps, tasks=1)
    meta = root / "meta"
    episode_dir = meta / "episodes" / "chunk-000"
    episode_dir.mkdir(parents=True)
    (meta / "info.json").write_text(info.model_dump_json(indent=2))
    (meta / "stats.json").write_text(json.dumps(_stats()))
    pq.write_table(
        pa.table({"task_index": pa.array([0], type=pa.int64()), "task": pa.array([task])}),
        meta / "tasks.parquet",
    )
    rows: list[dict[str, object]] = []
    cursor = 0
    for index, length in enumerate(lengths):
        rows.append(_episode_row(index, length, cursor, fps, task))
        cursor += length
    if orphan:
        rows.append(_episode_row(0, 3, 10_000, fps, task))
    _write_episodes(episode_dir / "file-000.parquet", rows)


def _stats() -> dict[str, dict[str, list[float]]]:
    def block(width: int) -> dict[str, list[float]]:
        vector = [0.0] * width
        return {"min": vector, "max": vector, "mean": vector, "std": vector, "count": [1.0]}

    stats = {
        "action": block(ACTION_DIM),
        "observation.state": block(ACTION_DIM),
    }
    for camera in CAMERA_FEATURE_KEYS:
        stats[camera] = block(3)
    return stats


def _episode_row(index: int, length: int, start: int, fps: int, task: str) -> dict[str, object]:
    duration = length / fps
    row: dict[str, object] = {
        "episode_index": index,
        "length": length,
        "dataset_from_index": start,
        "dataset_to_index": start + length,
        "tasks": [task],
    }
    for camera in CAMERA_FEATURE_KEYS:
        row[f"videos/{camera}/from_timestamp"] = 0.0
        row[f"videos/{camera}/to_timestamp"] = duration
    return row


def _write_episodes(path: Path, rows: list[dict[str, object]]) -> None:
    columns: dict[str, pa.Array] = {
        "episode_index": pa.array([row["episode_index"] for row in rows], type=pa.int64()),
        "length": pa.array([row["length"] for row in rows], type=pa.int64()),
        "dataset_from_index": pa.array(
            [row["dataset_from_index"] for row in rows], type=pa.int64()
        ),
        "dataset_to_index": pa.array([row["dataset_to_index"] for row in rows], type=pa.int64()),
        "tasks": pa.array([row["tasks"] for row in rows], type=pa.list_(pa.string())),
    }
    for camera in CAMERA_FEATURE_KEYS:
        columns[f"videos/{camera}/from_timestamp"] = pa.array(
            [row[f"videos/{camera}/from_timestamp"] for row in rows], type=pa.float64()
        )
        columns[f"videos/{camera}/to_timestamp"] = pa.array(
            [row[f"videos/{camera}/to_timestamp"] for row in rows], type=pa.float64()
        )
    pq.write_table(pa.table(columns), path)
