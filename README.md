# Cortex-and-Reflex

Pre-hardware foundation for a bimanual [SO-101](https://github.com/huggingface/lerobot) cloth-folding stack.
Cortex is the data engine. Reflex is the policy stack. Both sit on LeRobot 0.6.x
and the LeRobotDataset v3 episode format. Arms are expected mid-November; until
then the runnable path is a public towel dataset plus a CPU ACT smoke test.

The first task is folding a towel. The second is folding a T-shirt.

## Architecture

```mermaid
flowchart LR
  subgraph station [Station, mid-November]
    Arms[Two SO-101 followers]
    Cams[Wrist cameras and overhead]
    Leaders[Leader arms]
  end

  subgraph sharedPkg [shared]
    Schemas[12-DoF schemas]
    Teleop[Teleop protocol]
    Sim[Sim protocol]
    Harness[Eval harness]
  end

  subgraph cortexPkg [cortex]
    Ingest[Hub ingest]
    Label[Phase and failure labels]
    Queue[Selection queue]
    Dash[Dashboard]
  end

  subgraph reflexPkg [reflex]
    ACT[ACT]
    Diffusion[Diffusion Policy]
    Smol[SmolVLA]
    Pi[pi0.5]
  end

  Leaders --> Teleop
  Arms --> Teleop
  Cams --> Ingest
  Teleop --> Schemas
  Ingest --> Schemas
  Schemas --> Label
  Label --> Queue
  Queue --> Dash
  Schemas --> ACT
  ACT --> Harness
  Diffusion --> Harness
  Smol --> Harness
  Pi --> Harness
```

| Path | Role |
| --- | --- |
| `shared/` | Joint names, camera keys, episode records, label types, sim/teleop/eval protocols |
| `cortex/` | Metadata ingest, stub labeler, VLM backends, review queue, text dashboard |
| `reflex/` | Policy registry, ACT CPU smoke test, and the ACT training launcher |
| `configs/` | Dataset repo, labeler settings, ACT smoke and train hyperparameters |
| `docs/` | Episode format |
| `hardware/` | Station notes for the November bring-up |
| `infra/` | What CI runs without a GPU |
| `tests/` | Schema, ingest, labeling, and the ACT smoke step |

## Data contract

A frame is a 12-wide `float32` action, a 12-wide `float32` state, and three RGB cameras.
Joints use the `left_` / `right_` names from the public bimanual recordings
(`left_shoulder_pan.pos` through `right_gripper.pos`). Camera keys are
`observation.images.left`, `observation.images.right`, and `observation.images.top`
(left wrist, right wrist, overhead). Details and the v3 directory layout are in
[docs/data-format.md](docs/data-format.md).

Fold phases are `grasp_corner`, `flatten`, `fold`, and `release`. Failure tags are
`missed_grasp`, `crumpled_corner`, `misaligned_fold`, `double_layer_grasp`,
`arm_collision`, `timeout`, `drop`, and `missed_box`.

The checked-in loader reads
[`humjie/bimanual-so101-fold-towel`](https://huggingface.co/datasets/humjie/bimanual-so101-fold-towel)
(120 episodes, 30 fps, three cameras). It downloads `meta/` and checks it against
the contract. Video shards are left on the Hub.

## Setup

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest
```

`uv sync` installs the workspace, including a CPU build of PyTorch. The first
resolve is large because LeRobot depends on torch.

Ingest the public towel metadata:

```bash
uv run python -m cortex.cli ingest
```

Label one synthetic episode and print the dashboard:

```bash
uv run python -m cortex.cli label-demo
```

Run one ACT step on CPU and write a checkpoint under `outputs/act_smoke`:

```bash
uv run python -m reflex.cli act-smoke --steps 1
uv run python -m reflex.cli policies
```

Label a few towel episodes with the placeholder backend (no model download):

```bash
uv run python -m cortex.cli label \
  --dataset humjie/bimanual-so101-fold-towel \
  --limit 2 \
  --backend fake
```

## Labeling and policies

`StubLabeler` cuts each episode into four equal phases and applies kinematic
rules (gripper motion, wrist oscillation, left/right lift agreement, action
spikes, episode length). Phase confidence stays at 0.25, so every stub label
lands on the review queue. `VlmLabeler.complete()` is the backend hook.
`BackendVlmLabeler` validates the reply, retries schema failures, and caches
accepted JSON. Swap the labeler in `run_labeling` without changing the queue or
dashboard.

A VLM label is segments `grasp`, `flatten`, `fold1`, `fold2`, and `release`,
plus `success` and `quality_notes`. Segments tile the episode. Their confidence
is copied onto the coarse phases (`grasp` → `grasp_corner`, `fold1` and `fold2`
→ `fold`), so a minimum under 0.5 still enters the review queue as
`uncertain_phase`. `success: false` is queued as a failure. The checked-in
command is:

```bash
# Local SmolVLM or Qwen2.5-VL. transformers is not installed in CI.
uv pip install 'transformers>=4.51'
uv run python -m cortex.cli label \
  --dataset humjie/bimanual-so101-fold-towel \
  --limit 2 \
  --backend hf \
  --model HuggingFaceTB/SmolVLM-256M-Instruct

# Any OpenAI-compatible chat endpoint. OPENAI_BASE_URL / OPENAI_API_KEY /
# OPENAI_MODEL are accepted when the CORTEX_VLM_* variables are unset.
export CORTEX_VLM_BASE_URL=https://api.openai.com/v1
export CORTEX_VLM_API_KEY=your-key
export CORTEX_VLM_MODEL=gpt-4o-mini
uv run python -m cortex.cli label --limit 2 --backend openai
```

`--download-videos` fetches the mp4 shards before labeling. Without it, only
`meta/` is read and the prompt carries video refs. `--root` points at a dataset
directory that is already on disk. Replies are cached under `outputs/vlm_cache`.
Real-model tests are marked `slow` and skip unless `CORTEX_RUN_SLOW_VLM=1`.

`reflex.registry` lists ACT (ready), Diffusion Policy, SmolVLA, and pi0.5
(planned). The smoke test builds `ACTPolicy` from LeRobot 0.6.1 with
`pretrained_backbone_weights=None`, 64×64 images, and a 64-wide transformer, then
takes one optimizer step and calls `select_action`.

## Train ACT on a rented GPU

`configs/policies/act_towel_gpu.yaml` trains LeRobot 0.6 ACT on
`humjie/bimanual-so101-fold-towel`: 12-DoF actions and state, cameras
`observation.images.left`, `observation.images.right`, and `observation.images.top`.
Checkpoints go to `outputs/train/act_towel` every 2000 steps, which is enough to
restart a spot instance. W&B is off unless `wandb.enable` is true or you pass
`--wandb`.

The workspace lockfile pins CPU torch so CI stays small. On an A100, L40S, or
RTX 4090, install a CUDA torch build and the LeRobot training extra, then launch:

```bash
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install 'lerobot[training]'
uv run python -m reflex.cli train --config configs/policies/act_towel_gpu.yaml
```

Use the CUDA index that matches the driver if `cu128` does not. Re-run that
command after a preemption. When `outputs/train/act_towel/checkpoints/last`
exists, the launcher passes LeRobot `--resume=true` and that checkpoint's
`train_config.json`. On a 24 GB 4090, set `batch_size` to 2 if the run runs out
of memory.

Check the wiring on CPU before paying for the GPU. This does not download the
dataset:

```bash
uv run python -m reflex.cli train \
  --config configs/policies/act_towel_cpu_dry_run.yaml \
  --output outputs/train/act_towel_cpu_dry_run
```

The dry-run takes one synthetic step with the same three cameras and a 12-DoF
action, writes `checkpoints/last`, and resumes when you launch it again.

## Roadmap

1. **This checkout.** Monorepo, v3 schema, Hub metadata loader, stub labeler plus
   VLM interface, selection queue, text dashboard, ACT CPU smoke test, CI.
2. **Before the arms arrive.** VLM clients and the ACT GPU launcher are in this
   checkout. Review queued episodes against the public towel videos on a machine
   with the videos downloaded. Run the ACT config on a rented GPU. Keep Diffusion
   Policy, SmolVLA, and pi0.5 behind their extras until that machine exists.
3. **Mid-November station.** Two SO-101 followers, two leaders, three cameras.
   Implement `TeleopSession`, record towel episodes, and run `cortex ingest --root`
   on the new dataset before any long train.
4. **Towel policy.** Train ACT and one VLA baseline. Fill `EvalHarness` with a
   real success judge (fold flat, inside the target region).
5. **T-shirt.** Same 12-DoF contract, longer episodes, phase labels extended for
   sleeves. Reuse the failure taxonomy where it still applies.
6. **Reward model and distillation.** A fold-quality reward model, then distill
   pi0.5 into the small ACT student. Interfaces for that training stay out of
   this checkout until the towel policy has an eval number.

## License

Apache-2.0. LeRobot itself is Apache-2.0; this repo depends on it and does not
vendor it.
