# E4: matched Reacher variance floor

Six fresh trainings pair seeds 3072, 1 and 2: inverse MSE weight 1 versus the
same model plus `VarianceReg(weight=0.05, gamma=0.1)`. Both have SIGReg weight
zero, batch size 128, AdamW at 5e-5 and weight decay 1e-3, bf16 mixed precision,
and a **10-epoch scheduler horizon**. Each completes 127,960 optimizer steps.
The first seed pair counts toward the six; there is no pilot discard, search,
checkpoint selection, historical baseline reuse, or E1 control substitution.

Both arms use `lewm_masked`; only the variance weight and output paths differ
within each seed. The loss is forward MSE + inverse MSE + weight ×
`mean(relu(0.1 - std(z.flatten(0, 1), dim=0)))`. This is sample standard
deviation over batch and time, with no epsilon or covariance term. The variance
term lies between 0 and 0.1 before weighting (at most 0.005 afterward). It does
not rescale the other losses. The existing runtime preprocessing, split,
optimizer, scheduler, initialization and loader seed conventions remain shared.
Initial model tensors, split indices, loader RNG and Torch RNG are hashed before
training. Actual initialization hashes must match within each pair.

Evaluate only the explicitly named `lewm_masked_epoch_10` checkpoint stem,
using evaluation seed 42, the 200 frozen start rows in `protocol.json`, batches
of 10, goal offset 25, interaction budget 50, and CEM 300 samples / 30 iterations /
30 elites. Horizon, receding horizon and action block are each 5; planner batch
size is 1. Metadata records actual first/last IDs, episode IDs, steps and qpos
from chunks returned to the evaluator, in addition to the existing row metadata.
All six runs must match these consumed tasks. Raw success booleans are retained.

Report each training seed, both arm means and sample standard deviations, and
paired differences. Shared episodes are not independent training repetitions.
Three seeds alone cannot establish superiority or a robust rescue. Preserve and
report failed seeds. Log existing embedding standard deviation, effective rank,
inverse margin, inverse baseline and component losses as continuous diagnostics;
there is no post-hoc binary collapse threshold. Check every scalar output and
backward gradient norm for finiteness, and every final checkpoint tensor.

## Execution

Run CPU checks before dispatch:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 .venv/bin/python -m pytest -q tests/test_e4.py tests/test_masked_transition.py tests/test_e6_evaluation.py tests/test_training_infra.py
```

When local GPUs are occupied, launch in a local tmux session with the canonical
Modal detach pattern:

```bash
nohup .venv/bin/modal run --detach modal_app.py::e4 --attempt e4-v1 > /tmp/e4-v1.log 2>&1 &
```

The CPU supervisor hashes `dmc/reacher_random.h5` in place on the existing
`multi-future-lewm-cache` volume and checks its schema and size against the
canonical local readiness evidence. No dataset download or copy occurs. It
freezes all six resolved train/eval configs and source hashes before dispatch.
Training and evaluation share the same Modal image and A100-40GB allocation,
with at most two E4 workers. E1 and E6 resources are independent and unchanged.

Each worker owns one training followed by its evaluation. Workers refuse existing
output directories; training resume and application retries are disabled. A
preempted training is a reported failure, never a silently restarted replicate.
The supervisor records dispatch intent before submission and call IDs immediately
afterward. The same supervisor call can reattach after preemption; a new owner or
an intent with no call ID fails closed. Completed cells are never redispatched.

The study caps allocation at 60 GPU-hours and output at 6 GiB. Each cell has a
1 GiB allowance including logs/checkpoint, with a 16 MiB shutdown margin.
A 24-hour wall deadline (including queue waits) and concurrency two bound even
unreceipted execution below 60 GPU-hours. Each worker has a 10-hour provider
limit and an earlier process deadline. No automatic retries or alternative GPUs
are allowed. Workers periodically commit progress and commit again on exit.

After each full pair, the supervisor projects total worker time using elapsed
worker runtimes, a 25% remaining-work margin and a 600-second startup/shutdown
reserve. Queue wait is not charged as observed worker runtime. A projection over
60 GPU-hours blocks further launches. The deadline remains authoritative if a
worker dies before recording its final runtime; reconcile provider receipts and
preserve uncertainty before any authorized continuation. Early throughput is
provisional; measure a full pair before giving a whole-study estimate.

Artifacts live in `e4/<attempt>` on the volume: `status.json`, `frozen.json`, and
one directory per cell with runtime/source/package hashes, GPU details, initial
hashes, raw train/eval logs, diagnostics, resolved configs, checkpoint, and task
outcomes. Local session logs and downloaded scratch evidence stay in `progress/`.
Publish the audited final scientific results in a focused artifact commit after
completion; protocol code alone is not a completed study.
