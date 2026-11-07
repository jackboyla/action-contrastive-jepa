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
input. Random action sampling uses seed `42 + batch_start` for each ten-episode
batch; simulator resets retain the dataset evaluator's behavior. These are
short-goal evaluations, not a repeat of the separate 100-offset/100-budget study.

Run from the repository root with authenticated Modal access and the original
assets in `multi-future-lewm-cache`:

```bash
nohup .venv/bin/modal run --detach modal_app.py::e6 > /tmp/e6-short-v1.log 2>&1 &
```

The CPU supervisor hashes the dataset, all nine checkpoints and their training
configs before allocating GPUs. It then runs at most two L4 jobs concurrently,
with no automatic retries and a two-hour timeout per job. The first two
predeclared evaluations count in the final matrix and estimate cost before
further launches. If their conservative projection exceeds 12 GPU-hours, the
remaining matrix stays pending. A six-hour wall-clock deadline from the first
allocation bounds total GPU time at concurrency two, with shutdown allowance.
Infrastructure failures pause further launches and retain every attempted
outcome; no score controls scheduling. The output directory must not already
exist, preventing accidental duplicate launches or overwrites.

Both local GPUs are occupied by higher-priority frozen studies. Modal is the
bounded fallback; no local queue or study is changed. If setup fails or the
budget is insufficient, report the blocker and wait for the existing GPU0
study to finish before any local fallback.

Outputs live under `evaluations/e6-short-v1` on the volume: supervisor status,
call IDs, exact commands, raw logs/outcomes, resolved evaluation configs,
start/goal row IDs, episode IDs, success vectors, runtime, package versions,
GPU/driver details and source/checkpoint hashes. Retain all failures and all
27 seed cells plus the random control when publishing. Report means and sample
standard deviations separately across training seeds within each evaluation
seed and across evaluation seeds within each checkpoint. Shared tasks are not
independent training-seed repetitions; avoid significance claims from pooled
episodes. Fresh Modal runtime may differ from the archive: these results are
matched repeated evaluations, not replacements for archived headline numbers.
