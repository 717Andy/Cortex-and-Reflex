from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from cortex.dataset_episodes import records_from_dataset
from cortex.ingest import load_bimanual_dataset
from shared.contract import ACTION_DIM, CAMERA_FEATURE_KEYS
from tests.support import write_local_dataset


def test_video_refs_follow_the_v3_shard_path(tmp_path: Path) -> None:
    write_local_dataset(tmp_path, [6], video_files=True)
    loaded = load_bimanual_dataset("local/towel", root=tmp_path)
    records = records_from_dataset(loaded, limit=1)
    assert records[0].num_frames == 6
    assert records[0].states[0] == (0.0,) * ACTION_DIM
    for camera, ref in zip(CAMERA_FEATURE_KEYS, records[0].image_refs, strict=True):
        assert ref.endswith(f"videos/{camera}/chunk-000/file-000.mp4")
        assert Path(ref).is_file()


def test_parquet_trajectory_is_sorted_by_frame(tmp_path: Path) -> None:
    write_local_dataset(tmp_path, [2])
    data_dir = tmp_path / "data" / "chunk-000"
    data_dir.mkdir(parents=True)
    states = [[[1.0] * ACTION_DIM, [2.0] * ACTION_DIM], [[9.0] * ACTION_DIM, [8.0] * ACTION_DIM]]
    # Write frame 1 before frame 0 so the loader has to sort.
    table = pa.table(
        {
            "episode_index": pa.array([0, 0], type=pa.int64()),
            "frame_index": pa.array([1, 0], type=pa.int64()),
            "observation.state": pa.array(states[0], type=pa.list_(pa.float32())),
            "action": pa.array(states[1], type=pa.list_(pa.float32())),
        }
    )
    pq.write_table(table, data_dir / "file-000.parquet")
    loaded = load_bimanual_dataset("local/towel", root=tmp_path)
    record = records_from_dataset(loaded, limit=1)[0]
    assert record.states[0][0] == 2.0
    assert record.states[1][0] == 1.0
    assert record.actions[0][0] == 8.0
    assert record.actions[1][0] == 9.0
