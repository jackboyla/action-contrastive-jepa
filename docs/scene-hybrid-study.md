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

One trial pairs the Action-NCE seed 1 short and hard planning evaluations
beneath a single exclusive GPU0 lock. Checkpoints, evaluation seed 43,
100 tasks per protocol, batch size 10 and four CPU threads per worker stay
fixed. Later cells remain serial.

`scripts/run_eval_pair.py SPEC.json` accepts a local JSON object with `lock`,
`cwd`, the full frozen `env`, `report`, and exactly two `jobs`. Each job supplies
an `argv` array and absolute `log` and `output` paths. Keep specs and reports
outside version control, particularly environment values. The runner keeps the
lock until both processes exit, including when one fails; children inherit its
lock descriptor. Existing receipts and nonempty logs prevent automatic reruns.
Audit successful outputs before a separate continuation resumes unfinished work.

Admission requires the first CEM solve, at least 30 seconds of warmup, verified
physical GPU0 compute/graphics placement, 12 GiB free GPU memory (8 GiB allowance
for the second worker plus 4 GiB reserve), and 16 GiB available host RAM.
Otherwise the pair runs serially. The allowance is not a measured peak bound;
free GPU memory below 4 GiB flags the trial as risky. The runner does not kill
a successful sibling to improve throughput.

The report retains launch/end times, exit codes, placement evidence, sampled
peak total GPU memory and pair makespan. Two-second sampling runs only while
the trial holds the lock and can miss transient peaks. Receipt durations include
process startup; evaluator-reported durations exclude model/data setup. Use
matching timing boundaries for slowdown or speedup calculations. Comparisons
across training seeds remain exploratory. Further overlap requires useful
aggregate throughput, safe headroom and matching config/task/outcome audits;
keep lock waits and intentional pauses separate from compute time.
