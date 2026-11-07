# Scene Action-NCE: ten-epoch sensitivity study

This protocol tests five settings across three training seeds. Complete-grid
results remain pending; partial runs do not establish robustness.

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
will consume the same audited final epoch-10 checkpoints rather than train
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
