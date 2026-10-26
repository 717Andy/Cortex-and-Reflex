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

`shared.interfaces` defines `TeleopSession`, `FoldSim`, and `EvalHarness`. They are
protocols only. Serial ports, camera devices, and calibration files land here when
the hardware is on the bench. Do not invent port names in config until those devices
are enumerated.

## First recording session

1. Bring up one follower, then the second, then the leaders.
2. Record a handful of towel episodes with `lerobot-record` into the v3 layout.
3. Point `python -m cortex.cli ingest --root <dataset>` at that directory before
   any training run.
