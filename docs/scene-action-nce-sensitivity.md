# Scene Action-NCE: ten-epoch sensitivity study

## Status

As of 2026-11-07 13:43 UTC, **1/15 training runs and 0/15 evaluations** are
complete. Batch64, temperature .10, seed3072 finished all ten epochs and
137,950 updates at 12:01 UTC in 7,976 seconds (2h13m). The final checkpoint
is 74,247,854 bytes; CPU loading and finite-weight checks pass. Saved config
and events confirm the prescribed settings and training from scratch.
There is no planning success result yet.

The coordinator released the hold at 13:42 UTC and resumed the original queue.
The first evaluation has started; no duplicate jobs were launched. The delay
between first training completion and evaluation launch was about 1h42m.
Whole-study runtime is provisionally 37–47 hours, pending measured evaluation
and larger-batch throughput. No planning result is available yet.

Resource check found CUDA computation on GPU1 but MuJoCo EGL graphics memory
on GPU0 in the first evaluation. Installed MuJoCo selects rendering with
`MUJOCO_EGL_DEVICE_ID`, independently of CUDA visibility. EGL device 1's UUID
matches physical GPU1 on this workstation. Subsequent evaluations now have
that explicit setting; a locked render diagnostic is queued to verify actual
process/device allocation after the current evaluation. No result rerun is
planned solely for placement.

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

Run settings in table order, each with training seeds 3072, 1, 2. Retrain all
15 checkpoints; none is reused. Use `lewm_masked_action_nce`, `data=scene`,
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
# Verified on this workstation by EGL UUID vs nvidia-smi UUID; recheck elsewhere.
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
For an already-running queue whose environment cannot be changed, the same
supported variable can be set through Hydra's `hydra.job.env_set` in a local
configuration override before the next evaluation. Machine-specific ordinals
are not project defaults.

Execute long jobs in tmux. Acquire the shared GPU1 lock separately for each
train/evaluation job. Check GPU activity, other processes, disk, and logs before
launch. Use only GPU1 and leave other jobs alone. Retain only the final model
objects (about 71 MiB each, roughly 1.1 GiB total); optimizer checkpoint saving
is disabled. Keep total study artifacts below 6 GiB. Raw outputs stay local.

## Verification and reporting

The evaluator now persists sampled task identifiers without changing task
selection or planning. Existing planning-artifact and configuration tests pass
(12 tests). Full study integration and outcome audits are pending. Training
process, real optimizer progress, GPU placement and resource use were verified;
launch success alone was not treated as evidence of training completion.

Report all 15 success counts and percentages, then the mean and sample standard
deviation across the three training seeds for each setting. Shared evaluation
goals are not independent repetitions across models. Fixed epochs change the
number of optimizer updates with batch size, so this does not isolate the
number of contrastive negatives. No significance, broad robustness, or
behavior-policy-shift conclusion follows from this study alone.

## Proposed manuscript addition — pending results

Appendix D replacement template (fill only after all audits pass):

> We evaluated five predeclared batch-size/temperature settings at inverse-loss
> weight 0.30 after ten training epochs, using training seeds 3072, 1, and 2.
> Each final checkpoint used the same 100 short-goal Scene tasks (evaluation
> seed 43, 25-step goal offset, 50-step budget). Table [pending] reports every
> seed and the mean ± sample standard deviation across training seeds.
> Batch-size changes also change optimizer update count under this fixed-epoch
> protocol. These results concern this grid and short-goal protocol; they do
> not establish comprehensive robustness.

Main-text replacement template:

> Appendix D reports a ten-epoch, three-training-seed Scene sensitivity study
> across five predeclared batch-size/temperature settings on 100 shared
> short-goal tasks. [Add a narrowly scoped numerical finding after audit.]

These are proposed additions only. The paper, Google Doc, and Overleaf have
not been edited. The earlier one-epoch results must not be relabeled as these
pending ten-epoch results.
