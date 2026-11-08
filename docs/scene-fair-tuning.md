# Scene: equal-budget SIGReg and Action-NCE tuning

This study tests whether coefficient tuning changes the Scene comparison.
It is exploratory: tuning and final evaluation draw goals from the training
corpus, and the final evaluation seed has appeared in earlier experiments.
All six screening trials, six ten-epoch confirmations, and eighteen evaluations
completed on November 8, 2026. Both objectives selected their default coefficient.

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
| Screen duration | One full epoch, 6,897 updates | One full epoch, 6,897 updates |
| Confirmation seeds | 3072, 1, 2 | 3072, 1, 2 |
| Confirmation duration | Ten full epochs, 68,970 updates | Ten full epochs, 68,970 updates |

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

## Task overlap

The goal audit reproduces `eval.py`'s seeded row selection. Validation versus
each final protocol shares zero exact start rows and zero exact goal rows,
but shares two source trajectories. Training uses a random split within the
same corpus. This is **not a held-out trajectory test**. The overlap audit
records all sampled row and trajectory IDs. All eighteen evaluation logs
matched their protocol’s sampled start-row array exactly. Fixed offsets determine
the corresponding goal rows. The six final models used identical tasks within
each protocol; short and hard protocols use different tasks.

## Reproduce individual jobs

Use the repository's existing training/evaluation environment and prepared
`ogbench/visual_scene_play` dataset. The completed study used the main checkout's
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

Run long jobs in tmux with per-job logs. The study queue holds the shared
lock for each training or evaluation job, releasing it between jobs. It
uses only GPU0 and never interrupts other jobs. A four-hour timeout starts
only after lock acquisition. Check disk before each job: stop below 10 GiB
free or above 6 GiB of study artifacts. Retain only each run's final object,
with resumable/best checkpoint writes disabled. Treat failed or nonfinite
runs as failures to investigate, not candidates to drop. Record retries and
their extra compute. Do not start another copy of the active queue.

## Result audit and reporting

Before reporting results, verify all resolved configs, completed update counts,
final checkpoint epochs, finite metrics, complete Boolean episode outcomes,
reported success rates, selection/tie breaking, and identical sampled tasks.
Report every screen and all three final training seeds, mean and sample SD;
do not infer significance or robustness from this design.

## Audited results

Validation used one training seed (3072) and the same 50 seed-44 short goals for
every trial. These scores alone determined selection.

| Objective | Coefficient | Validation successes / 50 | Selected |
| --- | ---: | ---: | --- |
| SIGReg | 0.03 | 23 | |
| SIGReg | 0.09 | 23 | Yes, default tie-break |
| SIGReg | 0.27 | 21 | |
| Action-NCE | 0.10 | 25 | |
| Action-NCE | 0.30 | 26 | Yes |
| Action-NCE | 0.90 | 22 | |

Final results use only the selected settings' epoch-10 checkpoints. Each entry
is successful goals; the mean and sample SD are percentages across the three
training seeds, not uncertainty estimates from independent goal sets.

| Objective | Training seed | Short / 50 | Hard / 100 |
| --- | ---: | ---: | ---: |
| SIGReg 0.09 | 3072 | 29 | 15 |
| SIGReg 0.09 | 1 | 32 | 17 |
| SIGReg 0.09 | 2 | 29 | 21 |
| Action-NCE 0.30 | 3072 | 28 | 20 |
| Action-NCE 0.30 | 1 | 30 | 19 |
| Action-NCE 0.30 | 2 | 33 | 22 |

| Objective | Short mean ± sample SD (%) | Hard mean ± sample SD (%) |
| --- | ---: | ---: |
| SIGReg 0.09 | 60.00 ± 3.46 | 17.67 ± 3.06 |
| Action-NCE 0.30 | 60.67 ± 5.03 | 20.33 ± 1.53 |

Action-NCE's mean exceeds SIGReg's by 0.67 percentage points on short goals and
2.67 points on hard goals. The short-goal difference changes sign across training
seeds. These descriptive results do not establish significance or robustness.
The one-epoch screen selected both defaults; this limited grid and schedule do
not establish that either coefficient is optimal for ten-epoch training.

### Verification and compute accounting

The audit checked all twelve resolved training configs, epoch-end update counts,
final object filenames, finite logged metrics, all eighteen complete Boolean
outcome arrays and reported rates, validation-only selection, and actual sampled
task rows. Every process exited zero; no failed jobs, retries, dropped trials,
historical checkpoint substitutions, or best-checkpoint selection occurred.
MuJoCo emitted ignored EGL destructor exceptions during evaluation shutdown;
results were written and all expected outcomes were present before successful
process exit. These messages remain in the local logs.

The study executed 455,202 optimizer updates: 41,382 screening updates and
413,820 confirmation updates. Summed training epoch-end elapsed time was
50,524 seconds (14.03 hours). Summed evaluation elapsed time was 10,780.10 seconds
(2.99 hours), including an intentional 2,596.65-second pause in one evaluation;
subtracting that pause gives 8,183.45 seconds (2.27 hours). These elapsed times
exclude lock waits and are not GPU kernel timings. Shared GPU0 lock contention
and coordinated pauses extended the overall run across November 7–8. Retained
study training artifacts occupy about 0.84 GiB; datasets and environments were
shared. The experiment used no cloud jobs.

The execution checkout remained at reviewed head
`bdb04750813d43b114172052d8be23505bbf7f89`; the training/evaluation sources match
the frozen training-code commit above. Local evidence resides in the execution
checkout's `progress/evaluations/training-audit.json`, `final-audit-summary.json`,
`goal-overlap.json`, `selection.json`, and `final.json`, with exact job commands
and exit codes in `progress/logs/driver.log`. Checkpoints and resolved configs
remain under the shared cache's `rebuttal/fair_scene_20261107/` namespace.

## Proposed manuscript addition

“We screened three coefficients per objective using equal one-epoch budgets
(6,897 updates per trial), one training seed, and 50 validation goals. The
predeclared success/default/smaller-coefficient rule selected SIGReg 0.09 and
Action-NCE 0.30. Retraining each setting for ten epochs across three seeds and
evaluating matched goals yielded short-goal success rates of 60.00 ± 3.46% and
60.67 ± 5.03%, respectively, and harder-goal rates of 17.67 ± 3.06% and
20.33 ± 1.53% (mean ± sample SD across training seeds). This tuning study is
exploratory: validation and final goals share source trajectories and the
training corpus, and the final goal seed had prior exposure. The results do
not support a significance or robustness claim.”
