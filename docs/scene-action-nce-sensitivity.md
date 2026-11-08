# Scene Action-NCE: ten-epoch sensitivity study

All 15 predeclared runs completed on 8 November 2026. Each final epoch-10
checkpoint was evaluated on the same 100 short recorded-goal tasks. The results
describe this grid and protocol; they do not establish comprehensive robustness.

## Results

Successes out of 100 (also percentages). Columns identify **training seeds**;
the evaluation seed is 43 throughout. Mean and sample SD summarize three
training runs, not 300 independent goals.

| Batch | Temperature | Seed 3072 | Seed 1 | Seed 2 | Mean ± sample SD (%) | Updates per run |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | .10 | 68 | 72 | 74 | 71.33 ± 3.06 | 137,950 |
| 128 | .10 | 66 | 69 | 74 | 69.67 ± 4.04 | 68,970 |
| 256 | .10 | 72 | 63 | 71 | 68.67 ± 4.93 | 34,480 |
| 128 | .05 | 75 | 70 | 69 | 71.33 ± 3.21 | 68,970 |
| 128 | .20 | 69 | 73 | 74 | 72.00 ± 2.65 | 68,970 |

The five observed means span 68.67–72.00%. At fixed batch 128, the three
temperature means span 69.67–72.00%. The three-seed design and single shared
goal sample do not support a significance or comprehensive robustness claim.
Batch comparisons also change optimizer updates because epochs are fixed;
they do not isolate the number of contrastive negatives. No setting or
checkpoint was selected from these results.

[Audited results](scene-action-nce-sensitivity-results.json) contain all 1,500
boolean outcomes, the common 100 task identifiers, final checkpoint SHA256s,
update counts, and unrounded summaries. The audit compared full resolved
training configurations after removing only seed, run identity, batch size
and temperature; evaluation configurations matched after removing only policy
and output path. Every run started at step zero without resume, reached epoch
10 with the expected updates, and exited successfully in training and evaluation.
All 15 task-index, episode-ID and start-step arrays matched exactly.

Training used source commit `8197969ce1e525547aa532d2484ad6dfde358996`, the shared
Scene HDF5 identified in the results JSON, and an RTX 5090 on physical GPU1.
The first evaluation also allocated EGL graphics on GPU0; its result was
retained. Later evaluations used explicit MuJoCo EGL device selection;
actual evaluation processes confirmed compute and graphics on GPU1. No scientific
configuration changed. The queue took 36.25 hours, including coordinated
resource holds; the 8 November training pause alone lasted 2,596.678 seconds.
Those pauses are wall time, not evidence of slower training. Final model
objects total about 1.04 GiB; study artifacts stayed below the 6 GiB budget.

## Proposed manuscript text

These are proposed replacements for author review; they have not been applied
to the paper.

**Main text:** “In a predeclared Scene short-goal sensitivity study, five
batch-size/temperature settings averaged 68.7–72.0% success across three
training seeds at the final tenth epoch (Appendix D). These results describe
this grid; fixed-epoch batch comparisons also change the number of optimizer
updates.”

**Appendix D:** “We trained Action-NCE from scratch for ten epochs at inverse
coefficient 0.30, varying batch size over {64, 128, 256} at temperature 0.10
and temperature over {0.05, 0.10, 0.20} at batch size 128. The shared default
cell gives five settings, each trained with seeds 3072, 1 and 2. We evaluated
only final epoch-10 checkpoints on the same 100 short recorded-goal tasks
(evaluation seed 43; goal offset 25, execution budget 50), using CEM with
300 samples, 30 iterations and 30 elites, and horizon/action-block/receding
horizon of 5. All predeclared runs completed. Table [above] reports every
seed and the mean ± sample SD across training seeds. The default mean was
69.67%; the five setting means ranged from 68.67% to 72.00%. The goals are
shared across models, not independent repetitions across seeds. Fixed epochs
give 137,950, 68,970 and 34,480 updates for batches 64, 128 and 256, respectively,
so the batch experiment does not isolate negative-set size. This single
short-goal protocol and three training seeds do not establish comprehensive
robustness or statistical significance. The three default Action-NCE
checkpoints are also used as controls in the hybrid study and are not
independent training replications.”

## Frozen protocol

Train from scratch for ten epochs, with the learning-rate schedule sized for
ten epochs. Disable early stopping. Evaluate only final epoch 10; never select
by validation or evaluation score. Do not resume the earlier one-epoch screen.

| Setting | Batch | Temperature | Inverse coefficient |
|---|---:|---:|---:|
| batch64 | 64 | .10 | .30 |
| default | 128 | .10 | .30 |
| batch256 | 256 | .10 | .30 |
| tau005 | 128 | .05 | .30 |
| tau020 | 128 | .20 | .30 |

Run settings in table order, each with training seeds 3072, 1, 2. Train all
15 checkpoints from scratch; do not resume the earlier screen. Use `lewm_masked_action_nce`, `data=scene`,
the existing 90/10 split, and all other configuration defaults.

Evaluate every checkpoint on the same 100 sampled tasks with evaluation seed
43, goal offset 25, execution budget 50, and environment batch 10. Use the
original `scene` evaluation configuration: CEM 300 samples, 30 iterations,
30 elites; planning horizon 5, action block 5, receding horizon 5. This is the
short recorded-goal protocol, not the official Scene task protocol.

Complete all settings and seeds regardless of scores. Infrastructure failures
halt execution for diagnosis; record failures and repairs. Do not alter the
protocol based on results. The saved row indices, episode IDs, and start steps
must match exactly across all evaluations. Recount all 100 boolean outcomes
and verify the saved configuration and final training step count before reporting.
Expected update counts are 137,950 / 68,970 / 34,480 for batches 64 / 128 / 256.

## Reproduction

For the reported experiment, use a separate checkout of source commit
`8197969ce1e525547aa532d2484ad6dfde358996`; do not change a running checkout.
Use the existing dataset cache and Python environment. `STABLEWM_HOME` points
to the cache containing `ogbench/visual_scene_play`; no dataset copy is needed.
Run from this checkout so `train.py`, `eval.py`, and local imports use this code.
The following commands reproduce one grid cell. Run all 15 cells in the order
above, changing `setting`, `batch`, `temperature`, and `train_seed` together.
Use a fresh `run` path for a new study; never run these commands against a live
or completed run directory.

```bash
export STABLEWM_HOME=/path/to/existing/cache
export PYTHONPATH="$PWD"
export CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl
# Match the EGL device UUID to the intended GPU before choosing this ordinal.
export MUJOCO_EGL_DEVICE_ID=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
python_bin=/path/to/existing/venv/bin/python
gpu_lock=/path/to/shared/gpu1.lock
setting=batch64
batch=64
temperature=0.10
train_seed=3072
run="rebuttal/scene_full10_${setting}_s${train_seed}"
mkdir -p progress/logs progress/evaluations/scene-full10

flock "$gpu_lock" "$python_bin" train.py \
  --config-name lewm_masked_action_nce data=scene \
  seed="$train_seed" subdir="$run" trainer.max_epochs=10 \
  num_workers=2 loader.prefetch_factor=2 wandb.enabled=false \
  early_stopping.enabled=false checkpoint.enabled=false \
  checkpoint.object_epoch_interval=10 logging.progress_refresh_steps=100 \
  loader.batch_size="$batch" loss.masked.action_nce_temperature="$temperature" \
  loss.masked.inverse_weight=0.30 \
  > "progress/logs/${setting}_s${train_seed}-train.log" 2>&1

# Run only after successful training and verification of the final object file.
flock "$gpu_lock" "$python_bin" eval.py --config-name scene \
  policy="$run/lewm_masked_action_nce_epoch_10" \
  eval.num_eval=100 eval.env_batch_size=10 seed=43 \
  output.filename="$PWD/progress/evaluations/scene-full10/${setting}_s${train_seed}.txt" \
  > "progress/logs/${setting}_s${train_seed}-eval.log" 2>&1
```

MuJoCo indexes `eglQueryDevicesEXT()` for `MUJOCO_EGL_DEVICE_ID`; do not assume
CUDA ordinals remap that list. Match EGL device UUIDs to `nvidia-smi` using
[EGL_EXT_device_persistent_id](https://registry.khronos.org/EGL/extensions/EXT/EGL_EXT_device_persistent_id.txt),
then verify a render process appears only on the intended physical GPU.
Machine-specific ordinals are not project defaults.

Execute long jobs in tmux. Acquire the shared GPU1 lock separately for each
train/evaluation job. Check GPU activity, other processes, disk, and logs before
launch. Use only GPU1 and leave other jobs alone. Retain only the final model
objects (about 71 MiB each, roughly 1.1 GiB total); optimizer checkpoint saving
is disabled. Keep total study artifacts below 6 GiB. Raw outputs stay local.

## Verification and reporting

The evaluator records sampled task identifiers in its output. Audit these
identifiers, resolved configurations, final checkpoint steps and all episode
outcomes before reporting results.

Report all 15 success counts and percentages, then the mean and sample standard
deviation across the three training seeds for each setting. Shared evaluation
goals are not independent repetitions across models. Fixed epochs change the
number of optimizer updates with batch size, so this does not isolate the
number of contrastive negatives. No significance, broad robustness, or
behavior-policy-shift conclusion follows from this study alone.

## Shared standalone Action-NCE controls

Sensitivity is the sole producer of the default Action-NCE checkpoints
(batch 128, temperature .10, coefficient .30; seeds 3072, 1, 2). The hybrid study
accepted the same three audited final epoch-10 checkpoints instead of training
separate default NCE replicas. These are shared baselines, not independent
training replications. The sensitivity grid, schedule, evaluation,
and final-checkpoint rule remain unchanged.

Each seed has a pre-training snapshot of its full resolved config (including
the dataset-derived action dimension), raw and normalized-AST source hashes,
Python/package versions, shared dataset SHA256 and file identity, GPU
model/ordinal/UUID/driver, and thread/environment settings. The producer
checks the live launch environment and final provenance against that snapshot.
A completion attestation requires training exit 0, fit_end at epoch 10 and
68,970 updates, no resume, and the final checkpoint's SHA256 and byte size.
The consumer independently validates the manifest and artifacts before use.

Only run identity (`subdir`, `wandb.config.id/name`) is normalized in config
comparison. Source AST comparison permits docstring-only differences.
Different physical GPU ordinals and import paths are recorded; GPU model,
software and thread controls must match. Provenance is not described as
byte-identical across studies. One-epoch screening checkpoints never qualify.

Producer checkpoints live at
`rebuttal/scene_full10_default_s<seed>/lewm_masked_action_nce_epoch_10_object.ckpt`
in the existing cache. Local atomic manifests are
`progress/evaluations/shared-nce-s<seed>-launch.json` (predeclared only) and
`shared-nce-s<seed>.json` (complete only after checks). No dataset or model
copies are needed. The fair-tuning study uses separate confirmations after coefficient selection.
