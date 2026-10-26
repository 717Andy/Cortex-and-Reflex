"""Build ``EpisodeRecord`` values from a dataset the metadata loader already checked.

Joint rows are read when ``data/`` parquet is present. Otherwise the record
keeps the episode length with zeros so a VLM can still label from video refs.
Video bytes stay optional: a missing shard becomes a stable ref in the prompt.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq

from cortex.ingest import EpisodeMeta, LoadedDataset
from shared.contract import ACTION_DIM, CAMERA_FEATURE_KEYS
from shared.episodes import EpisodeRecord, make_episode_record


def records_from_dataset(loaded: LoadedDataset, *, limit: int) -> tuple[EpisodeRecord, ...]:
    """Return the first ``limit`` canonical episodes as labeler inputs."""
    if limit < 1:
        raise ValueError("limit must be >= 1")
    chosen = loaded.canonical_episodes[:limit]
    return tuple(_one(loaded, episode) for episode in chosen)


def image_refs_for_episode(loaded: LoadedDataset, episode: EpisodeMeta) -> tuple[str, ...]:
    """Local video paths when the shard is on disk, otherwise a stable ref."""
    refs: list[str] = []
    for camera in CAMERA_FEATURE_KEYS:
        path = _video_file(loaded, episode, camera)
        if path is not None:
            refs.append(str(path))
            continue
        relative = _relative_video(loaded, episode, camera)
        if relative is not None:
            refs.append(f"missing:{relative}")
            continue
        refs.append(f"hf://datasets/{loaded.repo_id}/{camera}/episode-{episode.episode_index}")
    return tuple(refs)


def _one(loaded: LoadedDataset, episode: EpisodeMeta) -> EpisodeRecord:
    states, actions = _trajectories(loaded.root, episode)
    task = episode.tasks[0] if episode.tasks else "Fold the towel"
    return make_episode_record(
        episode_index=episode.episode_index,
        task=task,
        fps=loaded.info.fps,
        states=states,
        actions=actions,
        image_refs=image_refs_for_episode(loaded, episode),
    )


def _video_file(loaded: LoadedDataset, episode: EpisodeMeta, camera: str) -> Path | None:
    relative = _relative_video(loaded, episode, camera)
    if relative is not None:
        path = loaded.root / relative
        if path.is_file():
            return path
        return None
    directory = loaded.root / "videos" / camera
    if not directory.is_dir():
        return None
    matches = sorted(directory.glob("**/*.mp4"))
    if len(matches) == 1:
        return matches[0]
    return None


def _relative_video(loaded: LoadedDataset, episode: EpisodeMeta, camera: str) -> str | None:
    chunk = episode.video_chunk_index.get(camera)
    file_index = episode.video_file_index.get(camera)
    if chunk is None or file_index is None:
        return None
    return loaded.info.video_path.format(
        video_key=camera,
        chunk_index=chunk,
        file_index=file_index,
    )


def _trajectories(root: Path, episode: EpisodeMeta) -> tuple[list[list[float]], list[list[float]]]:
    data_dir = root / "data"
    files = sorted(data_dir.glob("**/*.parquet")) if data_dir.is_dir() else []
    if not files:
        return _blank(episode.length)
    states: list[list[float]] = []
    actions: list[list[float]] = []
    frames: list[int | None] = []
    for path in files:
        names = set(pq.read_schema(path).names)
        needed = {"observation.state", "action", "episode_index"}
        if not needed <= names:
            continue
        columns = ["observation.state", "action", "episode_index"]
        has_frame = "frame_index" in names
        if has_frame:
            columns.append("frame_index")
        table = pq.read_table(
            path,
            columns=columns,
            filters=[("episode_index", "==", episode.episode_index)],
        )
        data = table.to_pydict()
        frame_column = data["frame_index"] if has_frame else [None] * table.num_rows
        for state, action, frame in zip(
            data["observation.state"],
            data["action"],
            frame_column,
            strict=True,
        ):
            states.append([float(value) for value in state])
            actions.append([float(value) for value in action])
            frames.append(None if frame is None else int(frame))
    if not states:
        return _blank(episode.length)
    if all(frame is not None for frame in frames):
        order = sorted(range(len(frames)), key=lambda index: int(frames[index] or 0))
        states = [states[index] for index in order]
        actions = [actions[index] for index in order]
    if len(states) != episode.length:
        raise ValueError(
            f"episode {episode.episode_index} parquet has {len(states)} frames, "
            f"metadata says {episode.length}"
        )
    for index, (state, action) in enumerate(zip(states, actions, strict=True)):
        if len(state) != ACTION_DIM or len(action) != ACTION_DIM:
            raise ValueError(
                f"episode {episode.episode_index} frame {index} is not {ACTION_DIM}-wide"
            )
    return states, actions


def _blank(length: int) -> tuple[list[list[float]], list[list[float]]]:
    states = [[0.0] * ACTION_DIM for _ in range(length)]
    actions = [row.copy() for row in states]
    return states, actions
