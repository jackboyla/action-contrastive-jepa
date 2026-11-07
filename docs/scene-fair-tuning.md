# Scene: equal-budget SIGReg and Action-NCE tuning

This ongoing study tests whether coefficient tuning changes the Scene comparison.
It is exploratory: tuning and final evaluation draw goals from the training
corpus, and the final evaluation seed has appeared in earlier experiments.
No result from the final comparison is available yet.

## Frozen protocol

The protocol was recorded before training on November 7, 2026. Training code is
commit `fd2739203122c13becbdd4f2db90ae393bfe17f1`.

| Setting | SIGReg | Action-NCE |
| --- | --- | --- |
| Training config | `lewm` | `lewm_masked_action_nce` |
| Coefficient key | `loss.sigreg.weight` | `loss.masked.inverse_weight` |
| Screening coefficients, in execution order | 0.09, 0.03, 0.27 | 0.30, 0.10, 0.90 |
| Default coefficient | 0.09 | 0.30 |
| Screen training seed | 3072 | 3072 |
| Screen duration | One full epoch, 6,897 updates | One full epoch, 6,897 updates expected; audit pending |
| Confirmation seeds | 3072, 1, 2 | 3072, 1, 2 |
| Confirmation duration | Ten full epochs | Ten full epochs |

The threefold grid covers each default and one value on either side. Both
objectives receive three trials, the same data split seed, batch size 128,
learning rate 5e-5, weight decay 0.001, and schedule duration. The objectives
retain their existing architectures and losses; equal update budgets do not
imply equal wall time or FLOPs. Action-NCE temperature stays at 0.1.

Select each objective's coefficient by successes on 50 short-goal validation
tasks, evaluation seed 44, goal offset 25 and action budget 50. Break ties by
preferring the default, then the smaller coefficient. Do not select by final
evaluation results. Retrain the selected settings from scratch for ten epochs;
use only epoch 10, with early stopping disabled. The one-epoch screen uses a
one-epoch learning-rate schedule, not the first epoch of the ten-epoch schedule.

Final evaluation uses seed 43 for every model: 50 short-goal tasks with offset
25/budget 50, and 100 harder-goal tasks with offset 100/budget 100. Both use
batch size 10, CEM 300 samples/30 iterations/30 elites, planning horizon 5,
receding horizon 5 and action block 5. Match sampled task rows across methods
and training seeds within each protocol. Training seeds are not independent
goal sets. No historical training runs are reused: the study budgets six
screening epochs and sixty confirmation epochs, plus 18 evaluations.

## Status and verified evidence

As of November 7, 2026, 10:55 UTC:

- SIGReg coefficient 0.09, seed 3072 completed its one-epoch screen: 6,897
  updates in 12 minutes 31 seconds, exit 0. Configuration, finite recorded
  metrics, and the sole epoch-1 object checkpoint passed the local audit.
- The queue and pending validation lock waiter are intentionally paused for
  a coordinated resource reservation. Current trainers finish untouched, then
  the challenge study uses both GPUs for about three hours (expected to start
  around 12:15 UTC). Resume requires an explicit release from the coordinator;
  this pause is not an experiment failure. No validation scores or selected
  coefficients are available.
- Five screen trainings, six confirmation trainings, and all 18 evaluations
  remain. No failures have been observed so far.
- Local study artifacts occupy about 69 MiB. Root disk has 43 GiB free.
  Provisional completion is November 8–9 morning UTC, including uncertain
  evaluation time and shared-GPU waits, now including the additional three-hour
  reservation. Timing will be revised after explicit release; this is not a deadline.

The goal audit reproduces `eval.py`'s seeded row selection. Validation versus
each final protocol shares zero exact start rows and zero exact goal rows,
but shares two source trajectories. Training uses a random split within the
same corpus. This is **not a held-out trajectory test**. The overlap audit
records all sampled row and trajectory IDs locally; matching these against
actual evaluation logs remains pending.

## Reproduce individual jobs

Use the repository's existing training/evaluation environment and prepared
`ogbench/visual_scene_play` dataset. The active study uses the main checkout's
`.venv-rebuttal` and `.stable_worldmodel`, without copying either. Run from the
checkout whose code you intend to evaluate. Set these paths for your machine:

```bash
export STUDY_PYTHON=/path/to/.venv-rebuttal/bin/python
export STABLEWM_HOME=/path/to/shared/.stable_worldmodel
export STUDY_LOCK=/path/to/shared/gpu0.lock
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 MUJOCO_GL=egl
export PYTHONPATH="$PWD"
mkdir -p progress/logs progress/evaluations
```

Exact first-screen arguments, apart from these machine-specific paths:

```bash
flock "$STUDY_LOCK" timeout 4h "$STUDY_PYTHON" train.py \
  --config-name lewm data=scene seed=3072 \
  subdir=rebuttal/fair_scene_20261107/screen_sigreg_0.09_s3072 \
  loss.sigreg.weight=0.09 trainer.max_epochs=1 num_workers=2 \
  loader.batch_size=128 loader.prefetch_factor=2 wandb.enabled=false \
  early_stopping.enabled=false checkpoint.enabled=false \
  checkpoint.object_epoch_interval=1 logging.progress_refresh_steps=100

flock "$STUDY_LOCK" timeout 4h "$STUDY_PYTHON" eval.py \
  --config-name scene \
  policy=rebuttal/fair_scene_20261107/screen_sigreg_0.09_s3072/lewm_epoch_1 \
  seed=44 eval.num_eval=50 eval.env_batch_size=10 \
  eval.goal_offset_steps=25 eval.eval_budget=50 \
  output.filename="$PWD/progress/evaluations/validation_sigreg_0.09.txt"
```

For each remaining screen substitute the table's config, coefficient key and
value, and a unique run/output name. For confirmation substitute the selected
coefficient and each training seed, set `trainer.max_epochs=10` and
`checkpoint.object_epoch_interval=10`, and evaluate the matching
`<config>_epoch_10` policy stem under both final protocols above. Never append
`_object.ckpt` to the policy argument. Evaluation appends to its output file;
use a new file for each run and do not combine retries silently.

Run long jobs in tmux with per-job logs. The active queue holds the shared
lock for each training or evaluation job, releasing it between jobs. It
uses only GPU0 and never interrupts other jobs. A four-hour timeout starts
only after lock acquisition. Check disk before each job: stop below 10 GiB
free or above 6 GiB of study artifacts. Retain only each run's final object,
with resumable/best checkpoint writes disabled. Treat failed or nonfinite
runs as failures to investigate, not candidates to drop. Record retries and
their extra compute. Do not start another copy of the active queue.

## Pending result audit and manuscript addition

Before reporting results, verify all resolved configs, completed update counts,
final checkpoint epochs, finite metrics, complete Boolean episode outcomes,
reported success rates, selection/tie breaking, and identical sampled tasks.
Report every screen and all three final training seeds, mean and sample SD;
do not infer significance or robustness from this design.

Conditional manuscript text, to fill only after the full audit:

> We compared SIGReg and Action-NCE after equal-budget coefficient screening
> on Scene. Each objective received three one-epoch trials with seed 3072;
> validation success selected one coefficient using a fixed tie rule. We then
> trained each selected setting for ten epochs with seeds 3072, 1, and 2 and
> evaluated the final checkpoints on matched short- and harder-goal tasks.
> [Insert all screening scores, selected coefficients, and final per-seed
> results with mean and sample SD.] Tuning is exploratory: validation and
> final goals share source trajectories with each other and the training
> corpus, and final goals had prior experimental exposure. Training-seed
> variation does not measure uncertainty over independent goal sets.

No paper, Google Doc, or Overleaf edits accompany this study PR. Scheduled
30-minute checks maintain the queue, audit results, and update this document
and the draft PR on material changes; monitoring ends after final delivery.
