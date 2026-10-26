# Hardware

No arms are attached in this checkout. The station is planned for mid-November.

## Target cell

- Two SO-101 follower arms (6 DoF each: five joints plus gripper). Twelve DoF total.
- Two leader arms for teleoperation, added with the followers.
- Three cameras: left wrist, right wrist, and one overhead. LeRobot feature keys are
  `observation.images.left`, `observation.images.right`, and `observation.images.top`.
- Cloth tasks, in order: fold a towel, then fold a T-shirt.

SO-101 followers use the same joint vector as the SO-100 bimanual recordings already
on the Hub (`left_*` then `right_*`, gripper last on each arm). See `docs/data-format.md`.

## What is wired today

`shared.interfaces` defines `TeleopSession`, `FoldSim`, and `EvalHarness`.
`hardware/teleop_bimanual_so101.yaml` and `hardware/record_bimanual_so101.yaml`
are the record templates: robot `bi_so_follower`, leader `bi_so_leader`, cameras
`observation.images.left`, `right`, and `top` at 480×640 and 30 fps, task
`Fold towel`. Serial ports and camera indexes are null. Do not invent `/dev`
paths until those devices are enumerated.

```bash
uv run python -m reflex.cli record-plan
uv run python -m reflex.cli deploy
uv run python -m reflex.cli record-mock
```

`record-plan` prints the `lerobot-record` command and leaves missing ports as
`<unset>`. `deploy` steps `MockBimanualRobot` through a state-copy policy with
the task string `Fold towel`. `record-mock` records one episode from
`MockLeaderTeleop`. Neither command opens a serial port.

## First recording session

1. Bring up one follower, then the second, then the leaders.
2. Fill the ports and camera indexes in the two YAML templates, then run
   `record-plan` and use that command.
3. Point `python -m cortex.cli ingest --root <dataset>` at that directory before
   any training run.
