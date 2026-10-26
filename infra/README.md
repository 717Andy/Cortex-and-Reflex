# Infra

CI is `.github/workflows/ci.yml`. It installs the uv workspace on Python 3.12,
then runs Ruff, mypy, and pytest. Pytest includes:

- schema and labeling tests, with no network
- an integration test that downloads `meta/` from `humjie/bimanual-so101-fold-towel`
- one ACT training step on CPU

Torch and torchvision resolve from the CPU wheel index
(`https://download.pytorch.org/whl/cpu`) so the runner does not pull CUDA builds.
The ACT smoke config uses a randomly initialized ResNet-18, a 64×64 image, and a
single optimizer step. It does not download ImageNet weights or dataset videos.

`pytest` also runs the ACT CPU dry-run in `configs/policies/act_towel_cpu_dry_run.yaml`.
That path does not download the towel videos. A real VLM test is marked `slow` and
skips unless `CORTEX_RUN_SLOW_VLM=1`. The Hugging Face backend is not installed in
this environment.

Real Diffusion Policy, SmolVLA, and pi0.5 runs need the matching LeRobot extras
and a GPU box. Those are registered in `reflex.registry` and are not installed here.
ACT GPU training uses `configs/policies/act_towel_gpu.yaml` and the `lerobot[training]`
extra on the rented machine, not in CI.

HF telemetry is disabled in the workflow (`HF_HUB_DISABLE_TELEMETRY=1`).
