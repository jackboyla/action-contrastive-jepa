# Reacher: ten epochs under the released 100-epoch schedule

All six predeclared fresh trainings and evaluations completed on 11 November
2026. Under this shared schedule, SIGReg averaged **81.50% ± 1.00** success and
Action-NCE **67.83% ± 6.37**. The spreads are sample standard deviations across
three training seeds. SIGReg scored higher in each paired seed; the mean
Action-NCE minus SIGReg difference was **−13.67 percentage points**.

These results compare the two configured methods under the released schedule.
They do not establish either method's best schedule, general superiority, or
the causal effect of changing the schedule. The predictor paths and inverse
heads differ, so this is not an isolated loss swap.

## Every predeclared run

Each checkpoint used the same 200 tasks, evaluation seed 42 and environment
batch 10. No coefficient, checkpoint or outcome selection occurred.

| Training seed | SIGReg successes / 200 | SIGReg (%) | Action-NCE successes / 200 | Action-NCE (%) | NCE − SIGReg (pp) |
|---:|---:|---:|---:|---:|---:|
| 3072 | 161 | 80.5 | 142 | 71.0 | −9.5 |
| 1 | 163 | 81.5 | 121 | 60.5 | −21.0 |
| 2 | 165 | 82.5 | 144 | 72.0 | −10.5 |
| Mean ± sample SD | | 81.50 ± 1.00 | | 67.83 ± 6.37 | −13.67 |

| Training seed | Both succeed | SIGReg only | Action-NCE only | Both fail |
|---:|---:|---:|---:|---:|
| 3072 | 118 | 43 | 24 | 15 |
| 1 | 105 | 58 | 16 | 21 |
| 2 | 122 | 43 | 22 | 13 |

[Results JSON](reacher-schedule-results.json) retains all 1,200 Boolean outcomes
in the shared task order, exact start and consumed goal identities, checkpoint
and raw-artifact hashes, configurations, command arguments, timings and
unrounded summaries. These are 200 shared tasks and three training seeds,
not 1,200 independent task replications. No significance claim follows here.

## Protocol and provenance

Every training ran from scratch for ten complete epochs and 127,960 optimizer
updates, with `trainer.max_epochs=100` and `runtime.stop_after_epoch=10`.
The scheduler horizon was 1,279,600 updates. Early stopping and optimizer
checkpoint saving were disabled; each run produced one final epoch-10 model
object. SIGReg used weight .09; Action-NCE used inverse weight .30 and
temperature .10 with SIGReg off. Both used batch 128, AdamW at peak learning
rate 5e-5, weight decay .001, bf16 mixed precision and the existing 90/10 split.
The [protocol](reacher-schedule.md) provides reproduction and audit commands.

Evaluation used Reacher `qpos_match`, configured goal offset 25, budget 50,
CEM with 300 samples, 30 iterations and 30 elites, and planning horizon,
receding horizon and action block all 5. The installed loader uses an exclusive
slice `[start, start + 25)` and the world consumes its first and last rows:
the actual goal row is **start + 24**. All 200 consumed endpoints were checked
through the installed loader methods. Videos and planning dumps were disabled.

All runs executed source commit
`1d841c441588269954ab2664f6d70ebed50232b7`. The manifest recorded an earlier
checkout HEAD when frozen; its 45 source hashes match both the execution files
and the execution commit's Git blobs. The final audit reproduced every saved
result from its outcomes, matched full configurations and exact task identities,
verified no resume and all twelve successful stage exits, and rehashed the
complete 98,905,882,624-byte dataset. Its SHA256 remained
`85a7dddfa1801302abcb175a80a23bb69c78291dd977ce40d69aedcb9123da06`.
Runtime versions and dataset revision are recorded in the JSON.

Execution used two local RTX 5090 GPUs with one independent stage per GPU and
exclusive per-GPU locks. SIGReg ran on physical GPU1 and Action-NCE on GPU0;
historical process samples verify both CUDA and EGL placement for every
evaluation. The scheduler source hash is recorded in the JSON; subsequently
published lock-lifetime fixes were not deployed during this study. No failed
or retried cell was recorded. Execution ended at 07:27:25 UTC on 11 November;
the dispatcher and legacy parent exited naturally, with no E1 processes or
held locks remaining. Experiment outputs, including side artifacts, were
about 0.53 GiB at the final audit, below the 6 GiB limit, with over 371 GiB free.

## Relation to the original short-horizon results

Keep the original comparison alongside E1. The
[original README table](https://github.com/jackboyla/action-contrastive-jepa/blob/1d841c441588269954ab2664f6d70ebed50232b7/README.md#results)
reported Reacher SIGReg **68.8 ± 0.2%** and Action-NCE **68.3 ± 3.1%**.
Those historical figures and their reported spreads remain separate from
E1's sample-SD summaries. Recounting the archived short-horizon Action-NCE
outcomes gives 141, 141 and 128 successes out of 200 for seeds 3072, 1 and 2
(70.5%, 70.5%, 64.0%; mean 68.33%, sample SD 3.75%, population SD 3.06%).

The archived runs used a ten-epoch learning-rate horizon and identify policy
directories rather than hashed epoch objects. Runtime and provenance differ
from E1. Changing the horizon changes warm-up as well as cosine decay; this
historical comparison cannot isolate either effect. E1 supplements rather than
replaces the original result. A matched method-by-schedule experiment would
need fresh short-horizon controls and remains outside this study. The separate
historical SIGReg 76% checkpoint binding remains unresolved and is not used
as a substitute control.
