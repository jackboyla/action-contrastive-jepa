# E1: Reacher with a 100-epoch learning-rate schedule

Compare Action-NCE and SIGReg after ten completed epochs while preserving the
released 100-epoch cosine horizon. Train six fresh models: each objective with
seeds 3072, 1 and 2. Evaluate only the final epoch-10 objects, without coefficient
tuning, checkpoint selection or outcome-based stopping. Historical SIGReg 76%
and inverse-MSE results lack the provenance needed to replace these controls.

Use `lewm_masked_action_nce` (inverse weight .30, temperature .10, SIGReg off)
and `lewm` (SIGReg weight .09). Retain their existing model paths, including the
Action-NCE inverse head and masked-transition predictor; this compares the two
configured methods, not an isolated loss swap on an identical model. Match
Reacher data, split seed, batch 128, AdamW, preprocessing, precision and runtime.
Both use `trainer.max_epochs=100`, `runtime.stop_after_epoch=10`, disabled early
stopping, and `checkpoint.object_epoch_interval=10`. Setting `max_epochs=10`
changes the learning-rate curve and does not implement this protocol.

Evaluate each model on the same 200 sampled tasks with seed 42, batch 10,
`qpos_match`, goal offset 25 and budget 50. Keep CEM at 300 samples, 30 iterations,
30 elites; horizon, receding horizon and action block all 5. Disable videos and
planning dumps. Export all task row indices, episode IDs, starting steps and
success booleans. Shared tasks are paired observations, not independent
repetitions across models or training seeds.

## Commands and audit

Use the existing environment and one persistent data cache containing
`dmc/reacher_random.h5`. The command list fixes run order: SIGReg then Action-NCE
for each seed 3072, 1, 2. Print it before launching:

```bash
python scripts/reacher_schedule.py --results progress/evaluations/e1
```

Before training, freeze the commands, resolved configs, source/package versions,
dataset hash, optimizer-step counts and 200 task identities:

```bash
python scripts/prepare_reacher_schedule.py --results progress/evaluations/e1 \
  --cache "$STABLEWM_HOME" --manifest progress/e1-manifest.json \
  --data-audit progress/evaluations/e1/data-audit.json
```

Preparation reads the dataset once for its SHA256 and refuses existing output
files or run directories. Set the training environment before preparation and
keep `STABLEWM_HOME` and all environment settings fixed across runs. Run the
printed commands from the repository root. Retain final checkpoint hashes,
resolved configs, events, raw evaluation files, exit statuses and runtime logs.
The audit rejects incomplete runs, changed configs, duplicate appended results,
wrong step counts or unequal task identities:

```bash
python scripts/reacher_schedule.py --results progress/evaluations/e1 \
  --audit progress/e1-manifest.json --cache "$STABLEWM_HOME" \
  --data-audit progress/evaluations/e1/data-audit.json
```

Report every seed, objective means and sample standard deviations, and paired
success differences. Report failed or incomplete runs explicitly; a failed run
is not a zero success rate. Do not infer the causal effect of schedule from
comparison with historical numbers under a different runtime. This experiment
alone does not establish general objective superiority.

## Local execution constraints

E1 started after the entire sensitivity study passed its 15-training/15-evaluation
artifact audit. Both local GPUs were authorized, with one independent stage per
GPU and a shared exclusive lock acquired separately for each job. Verify CUDA
and EGL placement on the assigned physical GPU. Preserve other studies, bound
experiment output to 6 GiB and retain at least 10 GiB free disk. Keep
machine-specific waiters, monitoring state and research logs under `progress/`,
outside commits.

All six runs completed and passed the full audit. See the
[results and limitations](reacher-schedule-results.md) and
[per-task outcomes and provenance](reacher-schedule-results.json).
