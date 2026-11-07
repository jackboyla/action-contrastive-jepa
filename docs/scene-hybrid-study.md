# Fixed Action-NCE + SIGReg Scene study

## Status

As of 2026-11-07 16:11 UTC, status is **Inprogress**, blocker **none**.
Training: **2/9 complete, 1 running, 6 pending**. Planning: **0/18**.
Probes: **0/9**. This PR is **draft, not merged**; the study is unfinished.

Hybrid seeds 3072 and 1 each completed ten epochs and 68,970 steps with
exit code 0. Both final checkpoints load on CPU and all floating tensors
are finite (311 state tensors each). Seed 1 finished at 15:58 UTC.
Seed 2 acquired GPU0 after the fair-tuning job and passed step 100/68,970
at 16:11 UTC. Estimated completion: 18:20–18:30 UTC.

No planning or probe results are available. Three SIGReg runs remain fresh.
Sensitivity is now the sole producer of the three default-NCE checkpoints;
none has yet been accepted. The consumer waits without holding a GPU lock.

### Protocol amendment — 2026-11-07 16:29 UTC

With parent authorization, use sensitivity's B128/temperature0.1/NCE0.30
final models for seeds3072,1,2 as the NCE baseline. This is a **shared
baseline, not an independent replication**. The fixed hybrid coefficients,
three seeds, epoch10 selection, planning tasks and probes remain unchanged.
Do not reuse fair-study selected models or one-epoch screening checkpoints.

Each seed requires a predeclared launch manifest and a complete producer
manifest, matching resolved scientific config, seed, epoch10,68,970updates,
exit0, software/thread settings, training-code AST and shared dataset
identity/hash. Check actual saved config, no-resume metadata, fit-end event,
checkpoint bytes and SHA256 independently. Raw source hashes, physical GPU
ordinal/UUID and checkout path remain recorded; only docstrings and run
identifiers may differ in code/config comparison. Both GPUs must have the
same model/driver. Missing or incomplete evidence waits; conflicting evidence
fails explicitly. Do not silently skip or substitute a run.

The idle follow-up driver was identity-checked and replaced to load the
consumer logic. Active hybrid training and the original three-seed driver
were untouched. The replacement removes all local NCE training launches.
Its gate passed CPU fixture checks for absent/incomplete evidence and
seed/update/hash/config/environment mismatches. No real NCE artifact has
passed the gate yet.

This avoids three duplicate training runs: about6.5GPU-hours at observed
throughput and about223MB of final checkpoints. Remaining wall time is
provisionally24–36hours from16:30UTC, depending on producer completion and
shared-GPU waits; it is not a completion guarantee.

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
- Fresh SIGReg runs and shared sensitivity NCE runs use the same worker,
  checkpoint and stopping controls. Archived runs are not reused.
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
> checkpoints. Against matched standalone objectives (with NCE checkpoints shared with
> the sensitivity study), the combination
> achieved [audited short-goal results] and [audited harder-goal results]
> on 100 shared tasks per protocol. Static-factor probes showed [audited
> factor-specific findings]. This experiment tests the specified combination;
> it does not establish an optimal hybrid or robustness beyond these settings.

Replace brackets with all methods' audited results, regardless of direction.
Do not edit the manuscript from this branch.
