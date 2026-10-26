# Bimanual SO-101 episode format

Episodes follow [LeRobotDataset v3.0](https://huggingface.co/docs/lerobot/en/lerobot-dataset-v3).
Cortex and Reflex share one contract, implemented in `shared.contract` and checked
by `validate_dataset_info`.

## Layout

```
meta/info.json
meta/stats.json
meta/tasks.parquet
meta/episodes/chunk-000/file-000.parquet
data/chunk-000/file-000.parquet
videos/observation.images.left/chunk-000/file-000.mp4
videos/observation.images.right/chunk-000/file-000.mp4
videos/observation.images.top/chunk-000/file-000.mp4
```

`codebase_version` must start with `v3`. Path templates are the v3 shard templates.
The loader in this repo downloads `meta/` only. Frame parquet and MP4 stay on the Hub
until a training run asks for them.

## Joints

`action` and `observation.state` are `float32` vectors of length 12. Names, in order:

| Index | Name |
| --- | --- |
| 0 | `left_shoulder_pan.pos` |
| 1 | `left_shoulder_lift.pos` |
| 2 | `left_elbow_flex.pos` |
| 3 | `left_wrist_flex.pos` |
| 4 | `left_wrist_roll.pos` |
| 5 | `left_gripper.pos` |
| 6 | `right_shoulder_pan.pos` |
| 7 | `right_shoulder_lift.pos` |
| 8 | `right_elbow_flex.pos` |
| 9 | `right_wrist_flex.pos` |
| 10 | `right_wrist_roll.pos` |
| 11 | `right_gripper.pos` |

Each arm is shoulder pan, shoulder lift, elbow flex, wrist flex, wrist roll, gripper.
Values are the raw LeRobot `.pos` units, not normalized radians.

## Cameras

| Role | Feature key | Shape |
| --- | --- | --- |
| Left wrist | `observation.images.left` | `(height, width, 3)` |
| Right wrist | `observation.images.right` | `(height, width, 3)` |
| Overhead | `observation.images.top` | `(height, width, 3)` |

The public towel dataset stores 480×640 AV1 video at 30 fps. `dtype` may be `video` or `image`.

## Robot type string

SO-101 and SO-100 share this 12-DoF vector. Hub recordings often say `bi_so100_follower`
or `bi_so_follower`. The contract checks the joint names and camera keys. A `robot_type`
that does not mention `so` is a warning.

## Labels

Labels are not columns in the LeRobot file yet. Cortex attaches them in memory:

- Phases: `grasp_corner`, `flatten`, `fold`, `release`
- Failures: `missed_grasp`, `crumpled_corner`, `misaligned_fold`, `double_layer_grasp`, `arm_collision`, `timeout`, `drop`, `missed_box`

`arm_collision` and `timeout` are separate events. `drop` and `missed_box` are separate events.

## Public dataset

`humjie/bimanual-so101-fold-towel` (120 episodes, 21,717 frames, task "Fold towel").
Its episode parquet contains extra rows past `total_frames`. The loader keeps the
contiguous partition that matches `meta/info.json` and reports the other rows as orphans.
