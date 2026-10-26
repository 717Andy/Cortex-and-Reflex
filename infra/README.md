# Infra

CI is `.github/workflows/ci.yml`. It installs the uv workspace on Python 3.12,
then runs Ruff, mypy, and pytest. Pytest includes:

- schema, labeling, curation, and reward-metric tests, with no network
- an integration test that downloads `meta/` from `humjie/bimanual-so101-fold-towel`
- one ACT training step on CPU
- CPU dry-runs for ACT, SmolVLA, and pi0.5 (a 12-to-12 linear stand-in, no VLM weights)
- the tiny reward model, offline eval on mock checkpoints, the tiny distillation student, and the mock station

Torch and torchvision resolve from the CPU wheel index
(`https://download.pytorch.org/whl/cpu`) so the runner does not pull CUDA builds.
The ACT smoke config uses a randomly initialized ResNet-18, a 64×64 image, and a
single optimizer step. It does not download ImageNet weights or dataset videos.

`pytest` also runs the ACT CPU dry-run in `configs/policies/act_towel_cpu_dry_run.yaml`.
That path does not download the towel videos. A real VLM test is marked `slow` and
skips unless `CORTEX_RUN_SLOW_VLM=1`. The Hugging Face backend is not installed in
this environment.

CI does not download SigLIP 2, SmolVLM, or PaliGemma. The reward CPU config uses
a tiny conv encoder. SmolVLA and pi0.5 dry-runs do not import `transformers`.

Real SmolVLA and pi0.5 runs need `lerobot[smolvla]` or `lerobot[pi]`, plus
`lerobot[training]` for accelerate, and a CUDA torch build. Those extras are not
installed here. GPU configs live under `configs/policies/` and
`configs/reward/fold_quality_gpu.yaml`. Diffusion Policy is still planned.
ACT GPU training uses `configs/policies/act_towel_gpu.yaml` on the rented machine,
not in CI. Ruff in this workflow is `ruff check --no-cache` and
`ruff format --check --no-cache`.

HF telemetry is disabled in the workflow (`HF_HUB_DISABLE_TELEMETRY=1`).
