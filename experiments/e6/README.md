# E6: short-goal evaluation variability

Frozen before evaluation on 2026-11-07. `protocol.json` identifies the nine
original Scene epoch-10 checkpoints by SHA-256, checked against the original
checkpoint records published in PR #5. New tuning, sensitivity and hybrid
checkpoints cannot replace them. No archived run has complete matching
outcome/ID/config/runtime provenance established for reuse: all 27 are fresh.

Evaluate SIGReg, MTM-MSE and Action-NCE, each trained with seeds 3072, 1 and 2,
at evaluation seeds 42, 43 and 44. Each model evaluation uses 50 episodes.
The random control uses 200 episodes and evaluation seed 42. Goal sampling
uses the existing evaluator's sorted, without-replacement row selection.
Verify identical start/goal rows across all nine models within each evaluation
seed. The random 200 rows need not equal the models' 50 rows.

Primary protocol: future goal offset 25, interaction budget 50, CEM 300 samples,
30 iterations and 30 elites, planner batch size 1, horizon 5, receding horizon 5,
action block 5, environment batch size 10, 64px simulator renders and 224px model
input. The dataset loader uses an end-exclusive slice: offset 25 selects the
goal at start row + 24. Metadata records that consumed row; the evaluator's
offset argument and behavior remain unchanged.
Random action sampling uses seed `42 + batch_start` for each ten-episode
batch; simulator resets retain the dataset evaluator's behavior. These are
short-goal evaluations, not a repeat of the separate 100-offset/100-budget study.

Run from the repository root with authenticated Modal access and the original
assets in `multi-future-lewm-cache`:

```bash
nohup .venv/bin/modal run --detach modal_app.py::e6 --attempt e6-short-v2 --prior-gpu-seconds 600 > /tmp/e6-short-v2.log 2>&1 &
```

The CPU supervisor hashes the dataset, all nine checkpoints and their training
configs before allocating GPUs. It then runs at most two L4 jobs concurrently,
with no automatic retries and a two-hour timeout per job. The first two
predeclared evaluations count in the final matrix and estimate cost before
further launches. Each completed wave updates a conservative projection from
worker runtimes, excluding provider queue delays. If it exceeds 12 GPU-hours,
the remaining matrix stays pending. A separate six-hour wall-clock deadline
starts before the first GPU submission, includes queue time, and bounds total
GPU time at concurrency two, with shutdown allowance.
Infrastructure failures pause further launches and retain every attempted
outcome; no score controls scheduling. An attempt directory belongs to one
supervisor call. On preemption the same
call reattaches to saved worker call IDs, skips recorded completions and retains
its original deadline. A separate invocation cannot take over that directory.
A dispatch intent without a saved call ID is ambiguous: the supervisor fails
closed for manual reconciliation instead of spawning another job.
Every worker also receives the persisted deadline. It rejects late starts and
terminates its evaluator process group at that deadline, even if the supervisor
loses its dispatch receipt. The shutdown allowance covers worker startup and
teardown; the process timer does not measure provider billing overhead.

Modal provides bounded capacity when local GPUs are occupied. A failed or
budget-blocked run retains its completed cells; inspect them before planning
any continuation so they are not silently repeated.

Outputs live under `evaluations/<attempt>` on the volume: supervisor status,
call IDs, exact commands, raw logs/outcomes, resolved evaluation configs,
start/goal row IDs, episode IDs, success vectors, runtime, package versions,
GPU/driver details and source/checkpoint hashes. Retain all failures and all
27 seed cells plus the random control when publishing. Report means and sample
standard deviations separately across training seeds within each evaluation
seed and across evaluation seeds within each checkpoint. Shared tasks are not
independent training-seed repetitions; avoid significance claims from pooled
episodes. Fresh Modal runtime may differ from the archive: these results are
matched repeated evaluations, not replacements for archived headline numbers.

The command above reserves 600 GPU-seconds for prior-attempt uncertainty; this
is a budget allowance, not a claim of observed computation. Before a new
attempt, verify the previous app has no running or pending workers, retain its
raw receipts, and reconcile every completed cell. Never rerun accepted cells.
`--prior-gpu-seconds` reduces both the projection budget and the wall deadline.
The runner does not automatically restart incomplete worker evaluations.

After stopping and reconciling a failed app, `--resume-from <previous-attempt>`
starts a distinct attempt with the audited completed cells copied byte for byte.
It keeps their original commands, source hashes and task IDs, records the previous
status hash, and uses the earlier deadline. The prior-time reserve must cover
all earlier failed or uncertain GPU work; accepted worker runtimes are counted
separately. The supervisor rejects unrecorded completions and reduced reserves.
This option does not stop the previous app: verify its queued and running calls
are gone before launching. Incomplete cells remain in the failed attempt for
audit and run fresh only after this reconciliation.
