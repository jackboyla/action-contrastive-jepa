# Fixed Action-NCE + SIGReg Scene study

## Status

As of 2026-11-07 12:12 UTC, hybrid seed 3072 completed ten epochs and
68,970 optimizer steps, including final validation. Its final epoch-10
checkpoint loads on CPU and all floating tensors are finite (311 state
tensors; 74,250,087 bytes). No planning or probe result is available.
The other two hybrid seeds, six fresh standalone baselines, 18 planning
evaluations and nine probes remain pending.

The first run took about 2 hours 10 minutes after acquiring GPU0.
The coordinated resource hold ended on November 7 at 13:42 UTC. The original
queues resumed without restarting training. At 13:43 UTC, hybrid seed 1 is
waiting for fair-tuning evaluation to release GPU0's lock. The follow-up
driver remains waiting for all three hybrid seeds.

Eight training runs remain (roughly 17–18 GPU-hours at measured throughput),
followed by planning and probes (roughly 6–8 hours based on prior runs).
Allow approximately 32–44 hours from release including shared-GPU waits;
this estimate depends on the other study's job durations.

The masked-transition implementation already adds SIGReg when its weight
is positive. The new config inherits Action-NCE and sets SIGReg to 0.09.
The existing Action-NCE config remains unchanged.

Verification: 27 masked-transition tests pass. The hybrid test composes the
real config and SIGReg, checks nonzero finite encoder gradients from each
term, and checks that their sum equals the total gradient. A live-run audit
also found finite logged metrics and positive contributions from both losses.
The partial audit is not evidence of planning performance.

## Frozen protocol

- Objectives: forward prediction + 0.30 Action-NCE + 0.09 SIGReg;
  standalone Action-NCE 0.30; standalone SIGReg 0.09.
- Training seeds: 3072, 1, 2. Ten full epochs from scratch. Evaluate epoch 10
  only; no early stopping, checkpoint selection or hybrid tuning.
- Batch 128, temperature 0.1, AdamW learning rate0.00005, weight decay0.001,
  train fraction0.9. Inherit remaining architecture, data and schedule
  settings from the respective configs.
- Fresh standalone runs use the same worker, checkpoint and stopping
  controls. Archived runs use different controls and are not reused.
- Each model gets 100 goals per protocol, evaluation seed 43, batch 10:
  short offset 25/budget 50 and harder offset 100/budget 100.
  CEM uses 300 candidates, 30 iterations, 30 elites; horizon 5,
  receding horizon 5 and action block 5. Audit start episode IDs and steps
  across methods within each protocol.
- Secondary probes use the existing Scene protocol: sample seed 20261106,
  6000 requested frames, episode-based 60/20/20 probe splits, standardized
  latents, ridge alpha selected on validation from
  0.01/0.1/1/10/100/1000, and 128 rollout windows.
  Report cube position/rotation, drawer, window and both buttons.
  These are probe-held-out episodes from the encoder's training collection.
- Report every seed, counts, mean and sample standard deviation. Shared
  goals are not independent model repetitions. This is a fixed-coefficient
  combination ablation, not a search for the optimal hybrid. No significance
  or robustness claim follows from this design.

## Reproduction commands

Run from this checkout. Use the shared dataset/cache and existing environment;
do not copy datasets. The active execution uses GPU 0 and acquires the shared
lock separately for every job. Do not launch these commands alongside the
already running queues.

```bash
export STABLEWM_HOME=/home/jack/workspace/Desktop/action-contrastive-jepa/.stable_worldmodel
export PYTHONPATH="$PWD"
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MUJOCO_GL=egl
study_python=/home/jack/workspace/Desktop/action-contrastive-jepa/.venv-rebuttal/bin/python
study_lock=/home/jack/workspace/Desktop/ten-percent-challenge/progress/gpu0.lock

# Repeat for seeds 3072, 1, 2. For baselines use lewm_masked_action_nce
# or lewm, with distinct subdirs.
flock "$study_lock" "$study_python" train.py \
  --config-name lewm_masked_action_nce_sigreg data=scene seed=3072 \
  subdir=rebuttal/hybrid_fixed_s3072 trainer.max_epochs=10 \
  num_workers=2 loader.prefetch_factor=2 wandb.enabled=false \
  early_stopping.enabled=false checkpoint.enabled=false \
  checkpoint.object_epoch_interval=10 logging.progress_refresh_steps=100

# Repeat for all nine final models and both protocols.
# Harder protocol: goal_offset_steps=100 eval_budget=100.
flock "$study_lock" "$study_python" eval.py --config-name scene \
  policy=rebuttal/hybrid_fixed_s3072/lewm_masked_action_nce_sigreg_epoch_10 \
  seed=43 eval.num_eval=100 eval.env_batch_size=10 \
  eval.goal_offset_steps=25 eval.eval_budget=50 \
  output.filename=progress/evaluations/hybrid_s3072_short.txt

OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 "$study_python" \
  -m pytest tests/test_masked_transition.py -q
```

Only final object checkpoints are requested. Budget: at most 6 GiB study
writes; retain final models. The continuation driver stops on job failure,
less than 10 GiB free disk or study checkpoint footprint over 5 GiB, and records
the failure without dropping trials. Original hybrid training has no
automatic resource threshold; scheduled monitoring checks it.

Local execution records live in the study worktree's progress directory:
run_hybrid.py owns the original three-seed queue; complete_hybrid.py waits
for it before baseline training, planning and probes. tmux sessions are
scene-hybrid-fixed and scene-hybrid-followup. audit_training.py checks live
configs and finite metrics. Raw records and machine-specific drivers remain
local; the commands above specify the reproducible training/planning jobs.
The probe script is currently local historical research tooling, not a
published supported interface.

## Conditional manuscript addition

Do not insert this paragraph until the complete result and sample-identity
audits pass:

> We evaluated a fixed combination of Action-NCE (coefficient 0.30) and
> SIGReg (0.09) on Scene, using three training seeds and final ten-epoch
> checkpoints. Against fresh matched standalone objectives, the combination
> achieved [audited short-goal results] and [audited harder-goal results]
> on 100 shared tasks per protocol. Static-factor probes showed [audited
> factor-specific findings]. This experiment tests the specified combination;
> it does not establish an optimal hybrid or robustness beyond these settings.

Replace brackets with all methods' audited results, regardless of direction.
Do not edit the manuscript from this branch.
