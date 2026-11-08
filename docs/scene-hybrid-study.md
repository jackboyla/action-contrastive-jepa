# Fixed Action-NCE + SIGReg Scene study

This protocol tests a fixed combination against matched standalone objectives.
Planning and probe comparisons remain pending.

## Frozen protocol

- Objectives: forward prediction + 0.30 Action-NCE + 0.09 SIGReg;
  standalone Action-NCE 0.30; standalone SIGReg 0.09.
- Training seeds: 3072, 1, 2. Ten full epochs from scratch. Evaluate epoch 10
  only; no early stopping, checkpoint selection or hybrid tuning.
- Batch 128, temperature 0.1, AdamW learning rate 0.00005, weight decay 0.001,
  train fraction 0.9. Inherit remaining architecture, data and schedule
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

## Shared standalone Action-NCE controls

Use the sensitivity study's batch-128, temperature-0.1, coefficient-0.30 final
models for seeds 3072, 1 and 2. These are shared controls, not independent
training replications. Do not substitute fair-study selected models or
one-epoch screening checkpoints.

Each seed requires a predeclared launch manifest and a complete producer
manifest, matching resolved scientific config, seed, epoch 10, 68,970 updates,
exit 0, software/thread settings, training-code AST and shared dataset
identity/hash. Check actual saved config, no-resume metadata, fit-end event,
checkpoint bytes and SHA256 independently. Raw source hashes, physical GPU
ordinal/UUID and checkout path remain recorded; only docstrings and run
identifiers may differ in code/config comparison. Both GPUs must have the
same model/driver. Missing or incomplete evidence waits; conflicting evidence
fails explicitly. Do not silently skip or substitute a run.

## Reproduction commands

Run from this checkout. Use the shared dataset/cache and existing environment;
do not copy datasets. Choose an available GPU and acquire its shared lock separately for every job.
Use unique run and output paths to avoid duplicate or overwritten results.

```bash
export STABLEWM_HOME=/path/to/shared/.stable_worldmodel
export PYTHONPATH="$PWD"
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MUJOCO_GL=egl
study_python=/path/to/venv/bin/python
study_lock=/path/to/shared/gpu0.lock

# Repeat for seeds 3072, 1, 2. Use lewm with distinct subdirs for SIGReg.
# Reuse the audited sensitivity checkpoints for the standalone NCE baseline.
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
  output.filename="$PWD/progress/evaluations/hybrid_s3072_short.txt"

OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 "$study_python" \
  -m pytest tests/test_masked_transition.py -q
```

Run long jobs in tmux with logs. Retain final model objects and record failures
without dropping trials. Before reporting, audit all resolved configurations,
completed steps, checkpoint identities, matched goal/probe samples and outcomes.
Report all methods regardless of the direction of the result.

## Bounded evaluation scheduling trial

As of 2026-11-08 12:40 UTC, all nine final checkpoints, ten planning
results, and five probes have passed the local audits. The study is incomplete.
The default Action-NCE checkpoints are shared with the sensitivity study,
not independent training replications.

One authorized trial pairs the pending Action-NCE seed 1 short and hard
planning evaluations beneath a single exclusive GPU0 lock. Scientific commands,
checkpoints, seed 43, 100 tasks per protocol, batch size 10, and four CPU threads
per worker stay fixed. Later cells remain serial. The trial is waiting for the
lock; no throughput improvement has been measured.

`scripts/run_eval_pair.py SPEC.json` accepts a local JSON object with `lock`,
`cwd`, the full frozen `env`, `report`, and exactly two `jobs`. Each job supplies
an `argv` array and absolute `log` and `output` paths. Keep specs and reports
outside version control, particularly environment values. The runner acquires
the lock itself and keeps it until both processes exit, including when one fails.
Children inherit the lock descriptor. Existing trial receipts and nonempty
job logs prevent accidental reruns; successful outputs require audit before any
separate continuation resumes unfinished work.

The second worker starts only after at least 30 seconds and the first CEM solve,
verified physical GPU0 compute/graphics placement, at least 12 GiB GPU free
memory (8 GiB admission allowance plus 4 GiB reserve), and 16 GiB available
host RAM. The observed roughly 4 GiB single-worker usage is a snapshot, not a
peak bound. A failed admission check keeps the pair serial. The allowance cannot
guarantee a future peak; sampled free memory below 4 GiB flags the trial as risky.
The runner never kills a successful sibling to optimize throughput.

The local report retains launch/end times, exit codes, placement evidence,
sampled peak total GPU memory, and pair makespan. Samples occur every two seconds
only while the trial owns the lock; transient peaks can escape sampling.
Compare pair makespan and per-job elapsed times against the original serial
seed 3072 timings (449.789 s short, 925.199 s hard). This is an approximate
cross-seed scheduling comparison, not a paired speedup estimate. Do not expand
parallel scheduling without useful measured aggregate gain, safe headroom, and
matching config/task/outcome audits. Keep intentional pause and lock-wait time
separate from compute timing.
