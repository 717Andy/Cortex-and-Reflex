"""Pull LeRobot v3 metadata from the Hub and check it against the shared contract.

Videos and frame parquet stay on the Hub. ``meta/`` for the public towel set is
about a megabyte and is enough to confirm the 12-DoF joint vector, the three
cameras, the task list, and that episode rows reconstruct ``total_frames``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import snapshot_download

from shared.contract import (
    CAMERA_FEATURE_KEYS,
    PUBLIC_TOWEL_REPO_ID,
    DatasetContractError,
    DatasetInfo,
    ValidationReport,
    validate_dataset_info,
    validate_feature_stats,
)

_DURATION_TOLERANCE_S = 1e-3


@dataclass(frozen=True)
class TaskRecord:
    task_index: int
    task: str


@dataclass(frozen=True)
class EpisodeMeta:
    episode_index: int
    length: int
    dataset_from_index: int
    dataset_to_index: int
    tasks: tuple[str, ...]
    source: str
    video_duration_s: dict[str, float]
    video_from_s: dict[str, float] = field(default_factory=dict)
    video_chunk_index: dict[str, int] = field(default_factory=dict)
    video_file_index: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class LoadedDataset:
    repo_id: str
    root: Path
    info: DatasetInfo
    tasks: tuple[TaskRecord, ...]
    canonical_episodes: tuple[EpisodeMeta, ...]
    orphan_episode_rows: int
    report: ValidationReport

    @property
    def total_frames(self) -> int:
        return sum(episode.length for episode in self.canonical_episodes)

    def summary(self) -> str:
        task_text = ", ".join(record.task for record in self.tasks) or "(none)"
        lines = [
            f"repo: {self.repo_id}",
            f"root: {self.root}",
            f"robot_type: {self.info.robot_type}",
            f"codebase_version: {self.info.codebase_version}",
            f"fps: {self.info.fps}",
            f"episodes: {len(self.canonical_episodes)} / {self.info.total_episodes}",
            f"frames: {self.total_frames} / {self.info.total_frames}",
            f"tasks: {task_text}",
            f"orphan episode rows: {self.orphan_episode_rows}",
            str(self.report),
        ]
        return "\n".join(lines)


def load_bimanual_dataset(
    repo_id: str = PUBLIC_TOWEL_REPO_ID,
    *,
    root: Path | None = None,
    revision: str | None = None,
    cache_dir: Path | None = None,
) -> LoadedDataset:
    """Download ``meta/`` (or read a local checkout) and validate the contract.

    ``root`` points at a dataset directory that already contains ``meta/info.json``.
    Without it, metadata is fetched from the Hugging Face Hub and videos are left
    in place.
    """
    dataset_root = _materialize(repo_id, root=root, revision=revision, cache_dir=cache_dir)
    info = DatasetInfo.model_validate_json((dataset_root / "meta" / "info.json").read_text())
    report = validate_dataset_info(info)
    stats_path = dataset_root / "meta" / "stats.json"
    if stats_path.is_file():
        stats = json.loads(stats_path.read_text())
        report.extend(validate_feature_stats(stats))
    else:
        report.error("missing_stats", "meta/stats.json is missing")

    tasks = _read_tasks(dataset_root / "meta" / "tasks.parquet", report)
    if info.total_tasks != len(tasks):
        report.error(
            "task_count",
            f"info.total_tasks is {info.total_tasks} but tasks.parquet has {len(tasks)} rows",
        )
    known_tasks = {record.task for record in tasks}
    episodes = _read_episode_rows(dataset_root / "meta" / "episodes", report)
    canonical, orphan_count = _select_canonical(episodes, info, report)
    _validate_canonical(canonical, info, known_tasks, report)

    if not report.ok:
        raise DatasetContractError(report)
    return LoadedDataset(
        repo_id=repo_id,
        root=dataset_root,
        info=info,
        tasks=tuple(tasks),
        canonical_episodes=tuple(canonical),
        orphan_episode_rows=orphan_count,
        report=report,
    )


def _materialize(
    repo_id: str,
    *,
    root: Path | None,
    revision: str | None,
    cache_dir: Path | None,
) -> Path:
    if root is not None:
        if not (root / "meta" / "info.json").is_file():
            raise FileNotFoundError(f"No meta/info.json under {root}")
        return root
    snapshot = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        cache_dir=None if cache_dir is None else str(cache_dir),
        allow_patterns=["meta/**"],
    )
    return Path(snapshot)


def _read_tasks(path: Path, report: ValidationReport) -> list[TaskRecord]:
    if not path.is_file():
        report.error("missing_tasks", "meta/tasks.parquet is missing")
        return []
    table = pq.read_table(path)
    names = set(table.column_names)
    if "task" in names:
        texts = table.column("task").to_pylist()
    elif "__index_level_0__" in names:
        # LeRobot writes the task string as the pandas index of tasks.parquet.
        texts = table.column("__index_level_0__").to_pylist()
    else:
        report.error("task_column", "tasks.parquet has neither 'task' nor a pandas index")
        return []
    if "task_index" in names:
        indices = table.column("task_index").to_pylist()
    else:
        indices = list(range(len(texts)))
    records: list[TaskRecord] = []
    for index, text in zip(indices, texts, strict=True):
        if not isinstance(text, str) or not text:
            report.error("task_text", f"task row {index!r} is empty")
            continue
        records.append(TaskRecord(task_index=int(index), task=text))
    return records


def _read_episode_rows(episodes_dir: Path, report: ValidationReport) -> list[EpisodeMeta]:
    if not episodes_dir.is_dir():
        return []
    rows: list[EpisodeMeta] = []
    required = [
        "episode_index",
        "length",
        "dataset_from_index",
        "dataset_to_index",
        "tasks",
    ]
    for camera in CAMERA_FEATURE_KEYS:
        required.append(f"videos/{camera}/from_timestamp")
        required.append(f"videos/{camera}/to_timestamp")
    optional: list[str] = []
    for camera in CAMERA_FEATURE_KEYS:
        optional.append(f"videos/{camera}/chunk_index")
        optional.append(f"videos/{camera}/file_index")
    for path in sorted(episodes_dir.glob("**/*.parquet")):
        schema_names = set(pq.read_schema(path).names)
        missing = [column for column in required if column not in schema_names]
        if missing:
            report.error(
                "episode_columns",
                f"{path.name} is missing episode columns: {', '.join(missing)}",
            )
            continue
        present_optional = [column for column in optional if column in schema_names]
        try:
            table = pq.read_table(path, columns=[*required, *present_optional])
        except (OSError, pa.ArrowInvalid) as exc:
            report.error("episode_columns", f"{path.name} is missing episode columns: {exc}")
            continue
        data = table.to_pydict()
        for row_index in range(table.num_rows):
            task_cell = data["tasks"][row_index]
            task_values = tuple(str(item) for item in task_cell) if task_cell else ()
            durations = {
                camera: float(data[f"videos/{camera}/to_timestamp"][row_index])
                - float(data[f"videos/{camera}/from_timestamp"][row_index])
                for camera in CAMERA_FEATURE_KEYS
            }
            from_s = {
                camera: float(data[f"videos/{camera}/from_timestamp"][row_index])
                for camera in CAMERA_FEATURE_KEYS
            }
            chunk_index: dict[str, int] = {}
            file_index: dict[str, int] = {}
            for camera in CAMERA_FEATURE_KEYS:
                chunk_column = f"videos/{camera}/chunk_index"
                file_column = f"videos/{camera}/file_index"
                chunk = (
                    _optional_int(data[chunk_column][row_index]) if chunk_column in data else None
                )
                file_id = (
                    _optional_int(data[file_column][row_index]) if file_column in data else None
                )
                if chunk is not None:
                    chunk_index[camera] = chunk
                if file_id is not None:
                    file_index[camera] = file_id
            rows.append(
                EpisodeMeta(
                    episode_index=int(data["episode_index"][row_index]),
                    length=int(data["length"][row_index]),
                    dataset_from_index=int(data["dataset_from_index"][row_index]),
                    dataset_to_index=int(data["dataset_to_index"][row_index]),
                    tasks=task_values,
                    source=str(path.relative_to(episodes_dir)),
                    video_duration_s=durations,
                    video_from_s=from_s,
                    video_chunk_index=chunk_index,
                    video_file_index=file_index,
                )
            )
    return rows


def _optional_int(value: object) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float | str):
        return int(value)
    return None


def _select_canonical(
    rows: list[EpisodeMeta],
    info: DatasetInfo,
    report: ValidationReport,
) -> tuple[list[EpisodeMeta], int]:
    """Keep the episode rows that partition ``[0, total_frames)``.

    The public towel dataset appends extra episode parquet after the original
    recording. Those later rows repeat episode indices and point past
    ``total_frames``. The contiguous partition that matches ``info.json`` is the
    index LeRobot training should use; the rest are reported as orphans.
    """
    if info.total_episodes == 0:
        report.warning("empty_dataset", "dataset has zero episodes")
        return [], len(rows)

    if not rows:
        report.error("missing_episode_metadata", "meta/episodes has no parquet files")
        return [], 0

    invalid = [
        row
        for row in rows
        if row.dataset_to_index - row.dataset_from_index != row.length or row.length <= 0
    ]
    if invalid:
        report.error(
            "episode_length_mismatch",
            f"{len(invalid)} episode rows have length != dataset_to_index - dataset_from_index",
        )

    invalid_ids = {id(row) for row in invalid}
    usable = [row for row in rows if id(row) not in invalid_ids]
    chosen: list[EpisodeMeta] = []
    used: set[int] = set()
    cursor = 0
    while cursor < info.total_frames:
        options = [
            row
            for row in usable
            if row.dataset_from_index == cursor and row.episode_index not in used
        ]
        expected = len(chosen)
        preferred = [row for row in options if row.episode_index == expected]
        pool = preferred or options
        if not pool:
            report.error(
                "episode_index_inconsistent",
                f"no episode row continues the frame index at {cursor}",
            )
            return chosen, len(rows) - len(chosen)
        pool.sort(key=lambda row: (row.source, row.episode_index))
        picked = pool[0]
        chosen.append(picked)
        used.add(picked.episode_index)
        cursor = picked.dataset_to_index

    if cursor != info.total_frames or len(chosen) != info.total_episodes:
        report.error(
            "episode_index_inconsistent",
            f"canonical index has {len(chosen)} episodes ending at frame {cursor}, "
            f"info.json says {info.total_episodes} episodes and {info.total_frames} frames",
        )
    if used != set(range(info.total_episodes)):
        report.error(
            "episode_ids",
            "canonical episode indices are not exactly 0..total_episodes-1",
        )
    orphan_count = len(rows) - len(chosen)
    if orphan_count:
        report.warning(
            "orphan_episode_rows",
            f"ignored {orphan_count} episode rows that are outside the contiguous "
            f"0:{info.total_frames} index",
        )
    return chosen, orphan_count


def _validate_canonical(
    episodes: list[EpisodeMeta],
    info: DatasetInfo,
    known_tasks: set[str],
    report: ValidationReport,
) -> None:
    expected_duration_frames = info.fps
    for episode in episodes:
        if not episode.tasks:
            report.error("episode_task", f"episode {episode.episode_index} has no task string")
        unknown = [task for task in episode.tasks if task not in known_tasks]
        if unknown:
            report.error(
                "episode_task",
                f"episode {episode.episode_index} references unknown tasks {unknown}",
            )
        for camera, duration in episode.video_duration_s.items():
            expected = episode.length / expected_duration_frames
            if abs(duration - expected) > _DURATION_TOLERANCE_S:
                report.error(
                    "video_duration_mismatch",
                    f"episode {episode.episode_index} {camera} spans {duration:.4f}s, "
                    f"expected {expected:.4f}s from length/fps",
                )
