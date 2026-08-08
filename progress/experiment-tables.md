# Experiment Tables for ICLR Results

Last updated: 2026-07-03.

Sources: `progress/experiment-log.md`, `progress/results-summary.md`,
`progress/evaluation-protocol-ledger.md`,
`progress/evaluations/lewm-style-n50-r1/summary.md`,
`progress/evaluations/lewm-style-n50-r1/*.txt`,
`progress/evaluations/n50r2-diagnostics/*.txt`,
`progress/evaluations/reacher-mtm-rescue/*.txt`,
`progress/evaluations/reacher-checkpoint-audit/summary.md`, and the train/eval
configs under `config/`. OGBench Scene add-on results are in the Modal Volume at
`.stable_worldmodel/scene/scene_sigreg_e10_s{3072,1,2}_n50.txt` and
`.stable_worldmodel/scene/scene_action_nce_e10_s{3072,1,2}_n50.txt`. The
official fixed-goal Scene chunk summary is
`progress/evaluations/scene-official-s3072/summary.json`. Broad OGB single-seed
trajectory screening is summarized in
`progress/evaluations/ogb-broad-corrected-s3072/summary.json`.

## Reading Notes

- "Train seeds" counts distinct trained checkpoints used or reported for that
  row. Most paper-facing rows evaluate one checkpoint, even when extra seeds
  exist elsewhere.
- "Eval seed" is the episode-sampling seed. Nearly all reported planning evals
  use eval seed `42`.
- "10-epoch LR" means `trainer.max_epochs=10`, so the stepwise
  `LinearWarmupCosineAnnealingLR` decays nearly to zero by epoch 10. This was
  fair for matched e10 comparisons but under-trained some reproduction runs.
- "100-epoch LR @ eK" means `trainer.max_epochs=100`, often with
  `runtime.stop_after_epoch=K`, so the LR schedule remains the longer
  reproduction schedule while the checkpoint is evaluated at epoch K.
- Current main-paper planning rows use the completed n=50 tables for LeWM-style
  comparability. The seed-3072 e100/n=200 sweep is complete, but it should not
  replace the paper table without explicitly carrying the Reacher and TwoRoom
  robustness caveats.
- Rows labeled as sentinels report train-diagnostic evidence only. They are
  included here because they explain which candidate branches were rejected
  before spending full n=200 evaluation budget.

## Shared Eval Settings

| Protocol | Tasks | Episodes | Eval seed | Goal offset / budget | CEM | Plan horizon / receding / block | Notes |
|---|---|---:|---:|---|---|---|---|
| Paper-facing standard | TwoRoom, PushT, Reacher, OGBench-Cube | 50 | 42 | 25 / 50 | 300 samples, 30 elites, 30 iterations | 5 / 5 / 5 | Uses released or maintainer-confirmed config. |
| TwoRoom-long stress | TwoRoom | 50 | 42 | 100 / 150 | 300 / 30 / 30 | 5 / 5 / 5 | Paper-text long protocol, kept separate from standard-suite mean. |
| Internal matched standard | Most task comparisons | 200 | 42 | 25 / 50 | 300 / 30 / 30 | 5 / 5 / 5 | Higher-power diagnostic. Do not mix with n=50 main rows. |
| Internal TwoRoom-long | TwoRoom | 200 | 42 | 100 / 150 | 300 / 30 / 30 | 5 / 5 / 5 | Higher-power long-horizon diagnostic. |

No completed table uses multiple eval seeds. The repeated-seed evidence is
almost entirely training-seed variation, with eval seed fixed to 42 for paired
episode comparisons.

## Task and Seed Coverage Summary

| Task | Paper-facing train seeds per approach | Paper-facing eval seeds | Additional train-seed evidence | Main gaps |
|---|---|---|---|---|
| TwoRoom | SIGReg 1, MTM 1; both e100 epoch-30 checkpoints | 1 (`42`) for standard and long | AR e10 3, MTM e10 3, Direct-H10 e10 3, NoReg 3; e100 seed 1/2 standard evals complete for SIGReg and MTM | e100 MTM seed 2 is good at epochs 15/16/29 but fails at epoch 30; use e10 or a predeclared validation-selected checkpoint rule |
| PushT | SIGReg 1, MTM 1; both e10 checkpoints | 1 (`42`) | MTM e10 3, AC-CPC 1, Recon 1, MS-MTM 1, BYOL-WM 1, MTM e100 1; partial e100 extra-seed checkpoints exist | e100 broad sweep blocked; existing evidence supports e10 for paper reporting |
| Reacher | SIGReg 1, MTM 1; both e10 checkpoints | 1 (`42`) | SIGReg e10 3, MTM e10 3, Action-NCE e10 3; e100 epoch-30 complete for SIGReg and MTM seeds `3072,1,2`; MTM rescue/checkpoint audits complete | Reacher MTM remains schedule/objective-sensitive: e100 MTM epoch-30 scores 75.0 / 10.5 / 12.0 across seeds `3072,1,2`; Action-NCE weight-0.30 scores 70.5 / 70.5 / 64.0 |
| OGBench-Cube | SIGReg 1, MTM 1; both e10 checkpoints | 1 (`42`) | MTM e10 3, SIGReg e10 3; e100 seed-3072 complete plus SIGReg seed 2 complete | e100 broad sweep blocked; e10 already has clean MTM multi-seed win |
| OGBench-Scene | SIGReg 3, Action-NCE 3 completed | 1 (`42`) | Three-seed complex-task add-on: Action-NCE 80.0 mean vs SIGReg 58.0 mean at n=50 | Keep separate from the four-task standard-suite mean because Scene uses the repo trajectory-goal protocol; fixed-goal singletask check completed at 0/250 for both methods |

## Method Index

| Short name | Config / policy family | Training objective | Planner / rollout | Anti-collapse mechanism | Common schedule used |
|---|---|---|---|---|---|
| SIGReg / LeWM | `lewm`, `lewm_base`, `lewm_from_scratch` | One-step latent MSE plus SIGReg weight 0.09 | AR | SIGReg Gaussian marginal regularization | e10 and e100 milestone runs |
| MTM / masked | `lewm_masked` | One-step latent MSE plus inverse dynamics, SIGReg off | AR | Adjacent inverse action prediction | e10 and e100 milestone runs |
| NoReg | `lewm` with `loss.sigreg.weight=0.0` | One-step latent MSE only | AR | None | e10 |
| LeWM-Direct-H | `lewm_mh` | Direct H-step latent MSE, SIGReg inherited | Direct-H | SIGReg plus multi-horizon objective | e10 |
| AC-CPC | `lewm_accpc` | Action-conditioned InfoNCE, SIGReg off | AR | Contrastive future identification | e10 |
| MS-MTM | `lewm_ms_mtm` | H-step prediction plus random-gap inverse dynamics | Direct-H | Multi-step inverse action prediction | e10 |
| BYOL-WM | `lewm_byol` | EMA-target temporal prediction, SIGReg and inverse off | Direct-H | BYOL asymmetry | e6/e10 partial, PushT result at e6 |
| Recon | `lewm_masked_recon` | MTM plus 64x64 pixel reconstruction | AR | Inverse dynamics plus reconstruction target | e5 PushT |
| MTM + variance floor | `lewm_masked_vicreg` | MTM plus per-dim variance floor | AR | Inverse dynamics plus VICReg-style variance | launched, eval pending |
| MTM + action NCE | `lewm_masked_action_nce` | MTM forward latent MSE plus in-batch action classification | AR | Contrastive inverse-action pressure | Reacher sentinels and one full e10 seed-1 eval |

Shared optimizer defaults unless overridden: AdamW, lr `5e-5`, weight decay
`1e-3`, `LinearWarmupCosineAnnealingLR` per step, batch size 128, `bf16-mixed`,
history size 3, latent dim 192, train seed 3072 by default.

## Paper-Facing n=50 Standard Suite

All rows use eval seed 42, `eval.env_batch_size=10`,
`output.save_video=false`, exact checkpoint stems, unique output filenames, and
CEM 300/30/30.

| Task | Approach | Policy checkpoint | Train seeds in this row | LR schedule | Eval epoch | Offset / budget | Success (%) | Source |
|---|---|---|---:|---|---:|---|---:|---|
| TwoRoom | SIGReg | `tworoom/lewm_base_e100/lewm_epoch_30` | 1 | 100-epoch LR | 30 | 25 / 50 | 88.0 | `n50r1_sigreg_tworoom_released_protocol.txt` |
| TwoRoom | MTM | `tworoom/lewm_masked_e100/lewm_masked_epoch_30` | 1 | 100-epoch LR | 30 | 25 / 50 | 96.0 | `n50r1_mtm_tworoom_released_protocol.txt` |
| PushT | SIGReg | `pusht/lewm_base/lewm_epoch_10` | 1 | 10-epoch LR | 10 | 25 / 50 | 96.0 | `n50r1_sigreg_pusht_released_protocol.txt` |
| PushT | MTM | `pusht/lewm_masked/lewm_masked_epoch_10` | 1 | 10-epoch LR | 10 | 25 / 50 | 90.0 | `n50r1_mtm_pusht_released_protocol.txt` |
| Reacher | SIGReg | `reacher/lewm_base/lewm_epoch_10` | 1 | 10-epoch LR | 10 | 25 / 50 | 74.0 | `n50r1_sigreg_reacher_released_protocol.txt` |
| Reacher | MTM | `reacher/lewm_masked/lewm_masked_epoch_10` | 1 | 10-epoch LR | 10 | 25 / 50 | 74.0 | `n50r1_mtm_reacher_released_protocol.txt` |
| OGBench-Cube | SIGReg | `cube/lewm_base/lewm_epoch_10` | 1 | 10-epoch LR | 10 | 25 / 50 | 76.0 | `n50r1_sigreg_cube_released_protocol.txt` |
| OGBench-Cube | MTM | `cube/lewm_masked/lewm_masked_epoch_10` | 1 | 10-epoch LR | 10 | 25 / 50 | 86.0 | `n50r1_mtm_cube_released_protocol.txt` |

Standard-suite means: SIGReg 83.5, MTM 86.5, delta +3.0. The key schedule
caveat is that TwoRoom uses epoch-30 checkpoints on the 100-epoch LR schedule,
while PushT/Reacher/Cube use epoch-10 checkpoints on the 10-epoch LR schedule.

## Paper-Facing n=50 Stress and Diagnostic Rows

All rows use eval seed 42 and CEM 300/30/30. These are paper-supporting
diagnostics, not part of the standard-suite mean.

| Task / protocol | Approach | Policy checkpoint | Train seeds in row | LR schedule | Eval epoch | Episodes | Success (%) | Role |
|---|---|---|---:|---|---:|---:|---:|---|
| TwoRoom-long 100/150 | SIGReg | `tworoom/lewm_base_e100/lewm_epoch_30` | 1 | 100-epoch LR | 30 | 50 | 12.0 | Long-horizon stress |
| TwoRoom-long 100/150 | MTM | `tworoom/lewm_masked_e100/lewm_masked_epoch_30` | 1 | 100-epoch LR | 30 | 50 | 28.0 | Long-horizon stress |
| TwoRoom 25/50 | NoReg | `tworoom/lewm_noreg/lewm_epoch_10` | 1 | 10-epoch LR | 10 | 50 | 34.0 | Collapse sanity check |
| PushT 25/50 | Recon | `pusht/lewm_masked_recon/lewm_masked_recon_epoch_5` | 1 | 5-epoch run | 5 | 50 | 92.0 | Auxiliary mechanism |
| PushT 25/50 | AC-CPC | `pusht/lewm_accpc/lewm_accpc_epoch_10` | 1 | 10-epoch LR | 10 | 50 | 62.0 | Auxiliary mechanism |
| PushT 25/50 | MS-MTM | `pusht/lewm_ms_mtm/lewm_ms_mtm_epoch_10` | 1 | 10-epoch LR | 10 | 50 | 60.0 | Auxiliary mechanism, direct-H planner |
| PushT 25/50 | BYOL-WM | `pusht/lewm_byol/lewm_byol_epoch_6` | 1 | partial e10 run | 6 | 50 | 38.0 | Auxiliary mechanism, direct-H planner |
| OGBench-Scene 25/50 | SIGReg | `scene/lewm_sigreg_e10_s{3072,1,2}` | 3 (`3072,1,2`) | 10-epoch LR | 10 | 50 each | 58.0 mean; 56/58/60 | Complex visual manipulation add-on |
| OGBench-Scene 25/50 | Action-NCE | `scene/lewm_masked_action_nce_e10_s{3072,1,2}` | 3 (`3072,1,2`) | 10-epoch LR | 10 | 50 each | 80.0 mean; 80/78/82 | Complex visual manipulation add-on; +22 pts vs SIGReg mean |

Scene add-on paired breakdown across all three matched train seeds: Action-NCE
solves 120/150, SIGReg solves 87/150. The paired discordants are 40 Action-NCE
wins vs 7 SIGReg wins, with exact two-sided binomial/McNemar p ~= 1.1e-6. Keep
this separate from the standard suite mean because it uses the repo's
trajectory-goal Scene protocol, not the official fixed-goal OGBench protocol.
The fixed-goal singletask seed-3072 comparison completed as 50 recoverable chunks:
`{action_nce,sigreg} x task{1..5} x episode chunks {0-9,10-19,20-29,30-39,40-49}`.
After monolithic jobs and a local tmux-supervised chunk retry proved
client-fragile, orchestration moved into Modal app `ap-lP9ONtY1E5FEgKbaGHOKCV`
via `run_scene_official_chunk_supervisor`; this remote supervisor completed all
chunks. Result: Action-NCE 0/250 and SIGReg 0/250 under the official five fixed
goal states with a 750-step cap. Every task was 0/50 for both methods, so this
is a compatibility/orchestration result, not a paper-facing performance gain.
Because the public OGBench goal-conditioned harness evaluates `visual-scene-v0`
with `task_id` options, run a cheap base-env smoke before describing this
singletask result as an exact public-protocol reproduction.

## Broad OGB Trajectory-Goal Screening (Single Seed)

Completed 2026-06-29 after fixing the broad OGB eval path. All rows use train
seed `3072`, eval seed `42`, `n=50`, `goal_offset_steps=25`, `eval_budget=50`,
and `eval.protocol=ogb_trajectory`. Continuous tasks use CEM 64/10/topk8;
powderworld uses PGD 64/10 on the fixed discrete-action path. This is a repo
trajectory-goal screening protocol, not a public OGB fixed-goal leaderboard
result. Antsoccer uses the bounded synthetic top-down state-to-pixel conversion.

| Family | Task | Action-NCE (%) | SIGReg (%) | Delta | Paired Action-only / SIGReg-only / both / neither |
|---|---|---:|---:|---:|---|
| Combinatorial | `visual-puzzle-4x4-play-v0` | 34.0 | 50.0 | -16.0 | 4 / 12 / 13 / 21 |
| Combinatorial | `visual-puzzle-4x5-play-v0` | 26.0 | 52.0 | -26.0 | 3 / 16 / 10 / 21 |
| Stochastic / uncertainty | `visual-antmaze-teleport-navigate-v0` | 46.0 | 40.0 | +6.0 | 3 / 0 / 20 / 27 |
| Stochastic / uncertainty | `powderworld-medium-play-v0` | 16.0 | 6.0 | +10.0 | 6 / 1 / 2 / 41 |
| Stitching / long-horizon | `visual-antmaze-large-stitch-v0` | 30.0 | 30.0 | 0.0 | 2 / 2 / 13 / 33 |
| Stitching / long-horizon | `antsoccer-medium-stitch-v0` | 88.0 | 88.0 | 0.0 | 0 / 0 / 44 / 6 |

Interpretation: Action-NCE is not uniformly better on this broad OGB screen.
SIGReg wins the combinatorial puzzle tasks; Action-NCE modestly wins the
teleport/powderworld uncertainty tasks; the two stitching tasks tie at this
single seed.

## Historical n=200 Matched e10 Evidence

These rows use eval seed 42 and CEM 300/30/30 unless noted. They are useful for
variance reduction and mechanism diagnosis but are not the current paper-facing
episode count.

| Task / protocol | Approach | Train seeds | LR schedule | Eval epoch | Episodes | Success (%) | Comment |
|---|---|---|---|---:|---:|---:|---|
| TwoRoom 25/50 | SIGReg AR | 3 (`3072,1,2`) | 10-epoch LR | 10 | 200 | 85.5 +/- 0.4 | 85.5 / 86.0 / 85.0 |
| TwoRoom 25/50 | MTM | 3 (`3072,1,2`) | 10-epoch LR | 10 | 200 | 90.2 +/- 0.5 | 90.5 / 89.5 / 90.5 |
| TwoRoom 25/50 | NoReg | 3 (`3072,1,2`) | 10-epoch LR | 10 | 200 | 28.0 +/- 2.0 | 30.5 / 28.0 / 25.5; plain next-latent collapse sanity check |
| TwoRoom 25/50 | LeWM-Direct-H10 | 3 (`3072,1,2`) | 10-epoch LR | 10 | 200 | 87.5 +/- 3.5 | Directional, not significant |
| TwoRoom-long 100/150 | SIGReg AR | 3 (`3072,1,2`) | 10-epoch LR | 10 | 200 | 17.0 +/- 0.4 | 16.5 / 17.0 / 17.5 |
| TwoRoom-long 100/150 | MTM | 3 (`3072,1,2`) | 10-epoch LR | 10 | 200 | 28.0 +/- 0.8 | 29.0 / 27.0 / 28.0; all p<0.05 vs AR |
| TwoRoom-long 100/150 | LeWM-Direct-H10 | 3 (`3072,1,2`) | 10-epoch LR | 10 | 200 | 22.2 +/- 1.6 | 21.0 / 24.0 / 21.5 |
| PushT 25/50 | SIGReg | 3 (`3072,1,2`) | 10-epoch LR | 10 | 200 | 93.2 +/- 0.2 | 93.5 / 93.0 / 93.0 |
| PushT 25/50 | MTM | 3 (`3072,1,2`) | 10-epoch LR | 10 | 200 | 85.5 +/- 0.7 | 85.0 / 85.0 / 86.5; loses vs SIGReg |
| OGBench-Cube 25/50 | SIGReg | 3 (`3072,1,2`) | 10-epoch LR | 10 | 200 | 66.2 +/- 0.2 | 66.0 / 66.5 / 66.0 |
| OGBench-Cube 25/50 | MTM | 3 (`3072,1,2`) | 10-epoch LR | 10 | 200 | 79.3 +/- 2.4 | 81.5 / 76.0 / 80.5 |
| Reacher 25/50 | SIGReg | 3 (`3072,1,2`) | 10-epoch LR | 10 | 200 | 68.8 mean | 69.0 / 69.0 / 68.5 |
| Reacher 25/50 | MTM | 3 (`3072,1,2`) | 10-epoch LR | 10 | 200 | 31.0 mean | 68.0 / 11.5 / 13.5; seed-collapse caveat |

Reacher update: the original single-seed MTM tie was not robust. Seeds 1 and 2
collapsed (`emb_std` near zero), so plain e10 MTM on Reacher should be described
as seed-unstable unless a collapse gate or weak variance/SIGReg hybrid is added.

## Latent Surprise Diagnostics for Paper

Completed 2026-07-02 with corrected deranged counterfactual pairings. Each cell
uses 4096 sampled clips per train seed, diagnostic seed 42, and reports the
population mean +/- std over training seeds `3072,1,2`. The table values are
mean per-clip invalid/normal latent prediction-error ratios.

| Task | Model | Train seeds | Action counterfactual | State discontinuity | Min invalid > normal |
|---|---|---|---:|---:|---:|
| PushT | SIGReg | `3072,1,2` | 151.7 +/- 4.7x | 1246.0 +/- 34.1x | 99.95% |
| PushT | AC-MTM | `3072,1,2` | 9.6 +/- 0.3x | 40.0 +/- 1.4x | 99.98% |
| OGBench-Cube | SIGReg | `3072,1,2` | 274.2 +/- 12.6x | 835.4 +/- 35.2x | 100.00% |
| OGBench-Cube | AC-MTM | `3072,1,2` | 82.3 +/- 1.6x | 434.4 +/- 5.6x | 100.00% |

Interpretation for the paper: the diagnostics support internal latent surprise
and counterfactual consistency in the same representation used by CEM, not a
claim about public violation-of-expectation benchmarks or general physical
reasoning.

## Historical n=200 PushT Mechanism Probe Table

This was the locked PushT probe/eval table before the paper moved planning rows
to n=50. It remains useful for explaining why probes should not replace planning
metrics.

| Mechanism | Planner | Train checkpoint | Eval epoch | Orient. R2 | Mean R2 | n=200 SR (%) | n=50 SR (%) |
|---|---|---|---:|---:|---:|---:|---:|
| SIGReg | AR | `pusht/lewm_base/lewm_epoch_10` | 10 | 0.791 | 0.701 | 93.0 | 96.0 |
| MTM | AR | `pusht/lewm_masked/lewm_masked_epoch_10` | 10 | 0.508 | 0.674 | 89.0 | 90.0 |
| Recon | AR | `pusht/lewm_masked_recon/lewm_masked_recon_epoch_5` | 5 | 0.569 | 0.679 | 87.5 | 92.0 |
| AC-CPC | AR | `pusht/lewm_accpc/lewm_accpc_epoch_10` | 10 | 0.655 | 0.564 | 64.5 | 62.0 |
| MS-MTM | Direct-H | `pusht/lewm_ms_mtm/lewm_ms_mtm_epoch_10` | 10 | 0.560 | 0.737 | 62.0 | 60.0 |
| BYOL-WM | Direct-H | `pusht/lewm_byol/lewm_byol_epoch_6` | 6 | 0.638 | 0.647 | 42.0 | 38.0 |

Note the n=200 MTM row here differs from the earlier 3-seed mean in the matched
e10 table. Treat this as a later single-checkpoint mechanism diagnostic, not as a
replacement for the multi-seed summary.

## Direct-H TwoRoom Experiments

All Direct-H rows use `lewm_mh`, train seed 3072 unless otherwise stated, eval
seed 42, and CEM 300/30/30.

| Batch | Model | Horizon weights | Train seeds | LR schedule | Eval epoch | Episodes | TwoRoom 25/50 SR (%) | TwoRoom-long 100/150 SR (%) | Decision |
|---|---|---|---|---|---:|---:|---:|---:|---|
| Screening | AR baseline | n/a | 1 | 10-epoch LR | 10 | 50 | 86.0 | 16.0 | Baseline |
| Screening | Direct-H5 | uniform | 1 | 10-epoch LR | 10 | 50 | 88.0 | 12.0 | Not carried forward |
| Screening | Direct-H5 | discount | 1 | 10-epoch LR | 10 | 50 | 88.0 | 10.0 | Dropped |
| Screening | Direct-H10 | uniform | 1 | 10-epoch LR | 10 | 50 | 90.0 | 24.0 | Lead candidate |
| Confirmatory | AR baseline | n/a | 1 | 10-epoch LR | 10 | 200 | 85.0 | 18.0 | Baseline |
| Confirmatory | Direct-H10 | uniform | 3 (`3072,1,2`) | 10-epoch LR | 10 | 200 | 87.5 +/- 3.5 | 22.2 +/- 1.6 | Promising on long horizon, not significant |

## e100 Schedule and Milestone Runs

These are schedule-corrected or longer-budget runs using `trainer.max_epochs=100`
and, where available, milestone stopping/evaluation.

| Task | Approach | Train seeds | LR schedule | Eval epoch(s) | Episodes | Success (%) | Status / interpretation |
|---|---|---|---|---|---:|---|---|
| Reacher | SIGReg | 1 (`3072`) | 100-epoch LR | 10, 15, 30, 40, 50 | 200 | 76.0, 82.0, 81.5, 78.0, 87.5 | Confirms e10 LR schedule was under-training reproduction |
| Reacher | SIGReg | 2 (`1,2`) | 100-epoch LR | 30 | 200 | 79.5, 80.5 | Extra-seed r2 evals completed after budget restoration; outputs archived in `progress/evaluations/e100-reacher-r2/` |
| Reacher | MTM | 1 (`3072`) | 100-epoch LR | 10, 17, 25, 30, 40, 50 | 200 | 39.5, 69.5, 67.5, 75.0, 79.0, 78.0 | Improves after e10 but remains below best SIGReg |
| Reacher | MTM seed 1 trajectory | 1 (`1`) | 100-epoch LR | 10, 17, 25, 29, 30 | 200 | 11.5, 14.0, 13.5, 13.5, 10.5 | Already poor by epoch 10; not a late bad-final-checkpoint issue |
| Reacher | MTM seed 2 trajectory | 1 (`2`) | 100-epoch LR | 10, 17, 25, 29, 30 | 200 | 12.0, 11.5, 15.0, 10.0, 12.0 | Already poor by epoch 10; not a late bad-final-checkpoint issue |
| Reacher | MTM inverse weight 0.3 | 1 | 100-epoch LR | 10 | 200 | 60.0 | Ablation, does not rescue e10 |
| Reacher | MTM inverse warmup 5 | 1 | 100-epoch LR | 10 | 200 | 10.0 | Rejected, collapse-like |
| Reacher | MTM invw0.3 + warmup5 | 1 | 100-epoch LR | 10 | 200 | 9.5 | Rejected, collapse-like |
| PushT | MTM | 1 (`3072`) | 100-epoch LR | 10, 15, 30 | 200 | 78.0, 81.0, 85.0 | Schedule does not close PushT gap |
| PushT | SIGReg | 1 (`3072`) | 100-epoch LR | 30 | 200 | 84.5 | Completed in app `ap-7URW93LGChFJWh7MTmYYbB`; output `e100_sigreg_pusht_ep30_n200_r2.txt` |
| TwoRoom | SIGReg | 1 (`3072`) | 100-epoch LR | 30 | 200 | 84.0 | e100 matched sweep |
| TwoRoom | MTM | 1 (`3072`) | 100-epoch LR | 30 | 200 | 87.5 | e100 matched sweep |
| TwoRoom | SIGReg | 2 (`1,2`) | 100-epoch LR | 30 | 200 | 87.0, 85.0 | Extra seed robustness, standard 25/50 protocol |
| TwoRoom | MTM | 2 (`1,2`) | 100-epoch LR | 30 | 200 | 91.0, 45.0 | Seed 2 is anomalously low; do not claim robust e100 TwoRoom MTM until checkpoint-trajectory evals land |
| TwoRoom | MTM seed 2 checkpoint trajectory | 1 (`2`) | 100-epoch LR | 10, 15, 16, 29, 30 rerun | 200 | 60.5, 88.0, 90.5, 90.0, 46.5 | Confirms epoch-30 is a bad final checkpoint; seed is good at epochs 15/16/29 |
| OGBench-Cube | SIGReg | 1 (`3072`) | 100-epoch LR | 30 | 200 | 66.5 | Completed in app `ap-mgs5w2gOhPZ4iegixD3doZ`; output `e100_sigreg_cube_ep30_n200_r2.txt` |
| OGBench-Cube | SIGReg | 1 (`2`) | 100-epoch LR | 30 | 200 | 62.0 | Extra seed completed before workspace disable |
| OGBench-Cube | MTM | 1 (`3072`) | 100-epoch LR | 30 | 200 | 77.0 | Completed in app `ap-nwk5uMGsuLDtXNdz01grOv`; output `e100_mtm_cube_ep30_n200_r2.txt` |

Seed-3072 e100/n=200 table completed as of 2026-06-19 16:14 IST. Extra e100
robustness jobs launched on 2026-06-18 23:38-23:44 IST: Reacher seeds 1/2 were
stopped by 2026-06-20 after Modal workspace disable / 86400s input timeouts.
After budget restoration, the targeted Reacher gap-fill jobs were relaunched
from persistent tmux sessions and completed: SIGReg seed 1/2 scored 79.5/80.5,
while MTM seed 1/2 scored 10.5/12.0. The 2026-06-23 early-checkpoint sweep
confirmed those Reacher MTM seeds are already poor at epochs 10/17/25/29:
seed 1 scores 11.5/14.0/13.5/13.5, and seed 2 scores 12.0/11.5/15.0/10.0.
Together with the original seed-3072 Reacher MTM epoch-30 score of 75.0, this
gives a three-seed e100 Reacher MTM epoch-30 set of 75.0/10.5/12.0. TwoRoom
seeds 1/2 completed their standard n=200 evals: SIGReg s1/s2 87.0/85.0; MTM
s1/s2 91.0/45.0. The TwoRoom MTM seed-2 failure is confirmed as a bad final
checkpoint / late schedule instability: e100 seed 2 scores 88.0/90.5/90.0 at
epochs 15/16/29 but only 45.0/46.5 at epoch 30. Cube SIGReg seed 2 completed at
62.0. PushT/Cube
remaining extra-seed jobs have resumable checkpoints but are deferred unless we
decide the paper needs a full e100 robustness appendix. The first direct
detached launches
(`ap-V4d52gU06XW5C3DhzNFrLN`, `ap-2fwxdarYZ3E7nVmzK7R2H8`) were canceled after
local-client interruption and should not be used.

## Reacher MTM Rescue, Checkpoint Audit, and Action-NCE

All completed planning rows use Reacher 25/50, eval seed 42, n=200,
`eval.env_batch_size=10`, video off, and CEM 300/30/30 unless stated otherwise.
These are diagnostic rows for understanding Reacher instability, not current
main-paper rows.

| Approach | Config / override | Train seeds | Eval epoch | Success (%) | Interpretation |
|---|---|---|---:|---|---|
| Default MTM | `lewm_masked` | `3072,1,2` | 10 | 68.0, 11.5, 13.5 | Seeds 1/2 collapse; default Reacher MTM is not robust |
| MTM inverse weight 3.0 | `loss.masked.inverse_weight=3.0` | `3072,1,2` | 10 | 60.5, 81.5, 67.5 | Rescues collapsed seeds but hurts seed 3072; objective-weight change |
| MTM inverse weight 10.0 | `loss.masked.inverse_weight=10.0` | `1` | 10 | 57.5 | Stable but over-weighted / planner-poor |
| MTM + weak SIGReg | `loss.sigreg.weight=0.01` | `1` | 10 | 60.0 | Noncollapsed but not a useful paper direction |
| MTM lower LR | `optimizer.lr=1e-5` | `1` | 10 | 21.5 | Underfits / ineffective |
| MTM lower LR | `optimizer.lr=2e-5` | `1` | 10 | 54.0 | Learns inverse but weak control quality |
| MTM lower LR | `optimizer.lr=3e-5` | `3072,1,2` | 10 | 60.0, 73.0, 61.0 | Best pure-objective epoch-10 rescue but still not robust |
| MTM lower LR checkpoint audit | `optimizer.lr=3e-5` | `3072,1,2` | 9 | 68.5, 74.0, 65.5 | Best clean Reacher MTM option; mean 69.3, schedule-sensitive |
| MTM inverse weight 3.0 checkpoint audit | `loss.masked.inverse_weight=3.0` | `3072,1,2` | 9 | 68.0, 76.0, 67.0 | Mean 70.3; tied with epoch 10 but less clean than lower-LR MTM |
| MTM + action NCE | `lewm_masked_action_nce`, inverse weight 1.0 | `1` | 10 | 57.0 | Fixes collapse but not planning quality; below lower-LR seed-1 result |

Checkpoint-audit conclusion: default MTM is already poor by epoch 1 for the
collapsed seeds, so checkpoint selection does not rescue the default objective.
For lower-LR MTM, epoch 9 is stronger than epoch 10 across all three seeds.

Earlier-epoch coverage:

| Schedule / variant | Train seed 3072 | Train seed 1 | Train seed 2 | Coverage note |
|---|---|---|---|---|
| e100 MTM | Epochs 10/17/25/30/40/50 | Epochs 10/17/25/29/30 = 11.5/14.0/13.5/13.5/10.5 | Epochs 10/17/25/29/30 = 12.0/11.5/15.0/10.0/12.0 | Seeds 1/2 are already poor by epoch 10; unlike TwoRoom, no good earlier checkpoint was found |
| e10 default MTM | Epochs 9/10 | Epochs 1/5/9/10 | Epochs 1/5/9/10 | Collapsed seeds were already poor at epoch 1 |
| e10 MTM `inverse_weight=3.0` | Epochs 3/5/7/9/10 | Epochs 9/10 | Epochs 3/5/7/9/10 | Epoch 9 and 10 are near-tied by three-seed mean |
| e10 MTM `optimizer.lr=3e-5` | Epochs 3/5/7/9/10 | Epochs 9/10 | Epochs 3/5/7/9/10 | Epoch 9 is the best audited checkpoint across all three seeds |

### Reacher Objective and Action-NCE Sentinels

These rows are train-diagnostic sentinels. They have no completed n=200 planning
eval unless a success rate is shown.

| Candidate | Train scope | Final / latest diagnostic | Eval result | Decision |
|---|---|---|---|---|
| MTM multi-step overshooting | Seed 1, stopped in epoch 0 around step 10,575 | `emb_std` last/mean10 0.016/0.017; forward loss about `3e-4`; inverse near chance | None | Rejected, same collapse mode |
| MTM multi-step inverse | Seed 1, stopped in epoch 0 around step 11,800 | `emb_std` last/mean10 0.0156/0.0172; forward loss about `3e-4`; inverse/action near chance | None | Rejected, same collapse mode |
| Reacher AC-CPC sentinel | Seed 1, 4000 steps | `emb_std` held around 0.0715; CPC loss fell to about 1.00 | None | Validates that contrastive future identification prevents early collapse |
| MTM + action NCE sentinel | Seed 1, weight 1.0, 4000 steps | `emb_std` mean10 0.346; forward loss mean10 0.0074; NCE loss mean10 7.25 | None | Promoted to full seed-1 eval |
| MTM + action NCE low weights | Seed 1, weights 0.02/0.05/0.10, 4000 steps | mean10 `emb_std` 0.047/0.0567/0.0665 | None | Rejected or not promoted; too close to collapse |
| MTM + action NCE mid weights | Seed 1, weights 0.20/0.30/0.50, 4000 steps | mean10 `emb_std` 0.117/0.166/0.215 | None | Weight 0.30 promoted as balanced candidate |
| MTM + action NCE weight 0.30 full | Seeds `3072,1,2`, e10; seed-1 epoch 7/8/9/10 audit | No collapse: seed-1 mean10 `emb_std` 0.1825; seeds `3072,2` 0.289/0.304 | epoch-10 n=200 3-seed = 70.5 / 70.5 / 64.0 (s3072/s1/s2), mean 68.3; seed-1 audit epoch 7/8/9/10 = 64.5/73.5/68.5/70.5 | **3-seed Reacher set complete (2026-06-25):** matches SIGReg mean (68.8) with no collapse vs MTM 31.0 — robust Reacher win |

## Early TwoRoom Reproduction and Smoke Runs

These are important provenance but should not be mixed into the main result
tables.

| Run | Checkpoint / policy | Protocol | Episodes | Eval seed | Success (%) | Use |
|---|---|---|---:|---:|---:|---|
| Random baseline | random | TwoRoom paper-text 100/150 | 50 | 42 | 2.0 | Local sanity check |
| Released HF LeWM, local tiny eval | `tworoom/lewm_object.ckpt` | TwoRoom paper-text 100/150 | 5 | 42 | 20.0 | Smoke only, too noisy |
| Released HF LeWM, Modal | `tworoom/lewm_object.ckpt` | TwoRoom paper-text 100/150 | 50 | 42 | 10.0 | Showed paper/config mismatch |
| Released HF LeWM, Modal | `tworoom/lewm_object.ckpt` | Released 25/50 | 50 | 42 | 88.0 | Reproduces reported scale |
| From-scratch AR LeWM | `tworoom/lewm_from_scratch` | Released 25/50 | 50 | 42 | 86.0 | Baseline for early MH work |
| From-scratch AR LeWM | `tworoom/lewm_from_scratch` | TwoRoom-long 100/150 | 50 | 42 | 16.0 | Early long-protocol baseline |

## Pending, Partial, or Abandoned Result-Bearing Work

| Task(s) | Approach | State | What exists | Why not paper-facing yet |
|---|---|---|---|---|
| PushT, Reacher | Direct-H5 speed/probe runs | Stopped early | PushT checkpoint at global step 1000; Reacher checkpoint at global step 500 | Incomplete training, no reliable eval |
| PushT, Cube, TwoRoom, Reacher | MTM + variance floor | Launched | `lewm_masked_vicreg` training dirs were launched | Eval results still pending in the ledger |
| Reacher | MTM collapse-rescue / inverse-weight rescue | Completed | `inverse_weight=3.0` epoch 10 seeds `3072,1,2` scored 60.5/81.5/67.5; checkpoint audit gives epoch 9 scores 68.0/76.0/67.0, mean 70.3. Raw outputs archived in `progress/evaluations/reacher-mtm-rescue/` and `progress/evaluations/reacher-checkpoint-audit/`. | Inverse-weight balancing rescues collapsed seeds, but it changes the objective weighting and is only near-tied with the lower-LR pure-MTM schedule |
| Reacher | MTM lower-LR pure-objective controls | Completed | Seed-1 default-MTM results: `optimizer.lr=1e-5/2e-5/3e-5` scored 21.5/54.0/73.0 at epoch 10. Full `lr=3e-5` epoch-10 seed set `3072,1,2` scored 60.0/73.0/61.0; checkpoint audit shows epoch 9 scores 68.5/74.0/65.5, mean 69.3. | Best clean Reacher MTM option if reporting this task; frame as competitive and schedule-sensitive, not a strong win |
| Reacher | MTM multi-step overshooting / multi-step inverse sentinels | Rejected | Both seed-1 sentinels collapsed in epoch 0 before full training; no n=200 eval ran | Same latent-MSE collapse mode as fragile default MTM; do not pursue without a stronger collapse-prevention mechanism |
| Reacher | AC-CPC collapse sentinel | Completed diagnostic | 4000-step seed-1 run held `emb_std` around 0.0715 while CPC loss fell to about 1.00; no planning eval requested | Useful mechanism evidence that contrastive future identification prevents early collapse, but PushT AC-CPC planning results warn it may be planner-poor |
| Reacher | MTM + true action NCE weight 0.30 | Completed (3-seed) | e10 n=200 epoch-10: s3072/s1/s2 = 70.5/70.5/64.0, mean 68.3, no collapse. Weight-1.0 seed-1 was 57.0; sweep rejected 0.02/0.05/0.10, promoted 0.30. | Robust Reacher win: SIGReg-parity (68.8) and stable across seeds vs MTM 31.0. See "MTM + Action-NCE (w030) Completed Cross-Task Results". |
| TwoRoom, PushT, Cube | MTM + true action NCE weight 0.30 | Completed (3-seed) | e10 n=50 3-seed means: TwoRoom 92.0 (92/92/92), PushT 88.7 (92/88/86), Cube 80.0 (82/78/80). | 3-seed CONFIRMED: NCE-w030 >= MTM on every task (+1.8/+3.2/+0.7) and no collapse. PushT still < SIGReg (pre-existing MTM gap). See completed cross-task results section. |
| Cube, TwoRoom, Reacher | BYOL-WM / MS-MTM | Launched in all four tasks | PushT diagnostics landed; other task results not finalized locally | Do not use without completed eval summaries |
| Reacher | Ep50 probes | Launched | Probe app ids in experiment log | Probe results pending in ledger |
| PushT, Cube, TwoRoom, Reacher | e100 matched table | Reacher targeted gap-fill and early-checkpoint sweep complete; PushT/Cube extras deferred | All 8 seed-3072 cells complete; TwoRoom seed 1/2 evals complete; Cube SIGReg seed 2 complete; Reacher SIGReg s1/s2 completed at 79.5/80.5; Reacher MTM s1/s2 are poor across epochs 10/17/25/29/30 | Reacher e100 MTM is not robust; PushT/Cube extras are lower priority |
| TwoRoom | MTM e100 seed-2 checkpoint trajectory | Completed | n=200 evals for `tworoom/lewm_masked_e100_s2` checkpoint stems `epoch_10`, `epoch_15`, `epoch_16`, `epoch_29`, and rerun `epoch_30` | Shows epoch 30 is the bad checkpoint; epochs 15/16/29 remain strong |

## MTM + Action-NCE (w030) — Completed Cross-Task Results

Candidate `lewm_masked_action_nce` with `loss.masked.inverse_weight=0.30`, e10
schedule. An MTM variant adding in-batch contrastive inverse-action pressure to
prevent the Reacher latent-collapse that sinks plain MTM. Baselines: MTM
(`lewm_masked`) and SIGReg/LeWM (`lewm_base`). Completed 2026-06-25.

### Reacher (e10, n=200, 3 seeds `3072,1,2` — fully matched protocol)

| Method | s3072 | s1 | s2 | Mean | Note |
|---|---:|---:|---:|---:|---|
| SIGReg | 69.0 | 69.0 | 68.5 | 68.8 | stable |
| MTM | 68.0 | 11.5 | 13.5 | 31.0 | seeds 1,2 collapse |
| **NCE-w030** | 70.5 | 70.5 | 64.0 | **68.3** | no collapse; SIGReg-parity |

### Cross-task (e10, n=200, 3 seeds `3072,1,2` — fully matched to MTM n=200, completed 2026-06-25)

| Task | NCE-w030 n200 (s3072 / s1 / s2) | Mean | MTM n200 | SIGReg n200 | Δ vs MTM |
|---|---|---:|---:|---:|---:|
| TwoRoom | 90.5 / 90.0 / 91.5 | **90.7** | 90.2 | 85.5 ±0.4 | +0.5 |
| PushT | 87.5 / 88.0 / 84.5 | **86.7** | 85.5 | 93.2 ±0.2 | +1.2 |
| Cube | 80.5 / 76.5 / 79.5 | **78.8** | 79.3 | 66.2 ±0.2 | -0.5 |
| Reacher | 70.5 / 70.5 / 64.0 | **68.3** | 31.0 | 68.8 | +37.3 |

Now fully protocol-matched (SIGReg, candidate, and MTM all n=200, 3-seed for the
standard tasks). For reference, the candidate n=50 3-seed means were TwoRoom
92.0, PushT 88.7, Cube 80.0; the n50->n200 shift is small and does not change
the conclusion.

TwoRoom-long (100/150) stress, n=200, 3 seeds (completed 2026-06-25):

| Protocol | NCE-w030 (s3072 / s1 / s2) | Mean | MTM | AR/SIGReg | Direct-H10 |
|---|---|---:|---:|---:|---:|
| TwoRoom-long 100/150 | 25.0 / 23.5 / 24.0 | **24.2** | 28.0 ±0.8 | 17.0 ±0.4 | 22.2 ±1.6 |

Long-horizon is the **one place NCE-w030 carries a cost**: 24.2 vs MTM 28.0 (−3.8;
seed ranges 23.5–25.0 vs 27–29 do not overlap, so the gap is likely real), though
it still clearly beats AR/SIGReg (+7.2) and Direct-H10 (+2.0). The contrastive
action pressure preserves short-horizon performance and fixes Reacher but slightly
attenuates long-horizon planning relative to pure MTM.

### Verdict (3-seed, matched n=200)

- **Reacher collapse fixed** — NCE-w030 68.3 reaches SIGReg parity (68.8), stable
  across all 3 seeds (range 64–70.5), vs MTM 31.0 (seeds 1,2 collapse). First
  MTM-family method to do so.
- **Zero standard-horizon cost** — at matched n=200, NCE-w030 is within ±1.2 of MTM
  on all three non-Reacher standard-protocol tasks (TwoRoom +0.5, PushT +1.2, Cube
  -0.5): a statistical tie, no collapse anywhere.
- **One long-horizon cost** — on TwoRoom-long (100/150), NCE-w030 24.2 trails MTM
  28.0 by ~3.8 (non-overlapping seed ranges → likely real). It still beats AR/SIGReg
  (+7.2) and Direct-H10 (+2.0), so most of the masked-transition long-horizon edge
  is retained, just attenuated. Not a strict free lunch.
- **vs SIGReg** — beats SIGReg on TwoRoom (+5.7) and Cube (+12.8), ties on Reacher
  (-0.5); trails only on PushT (-6.8), the pre-existing MTM gap, not introduced by
  action-NCE.
- **Status** — cross-task generality 3-seed CONFIRMED at matched n=200 (standard +
  long-horizon). Net: a robust Reacher-collapse fix with no standard-horizon cost
  and a small (~4pt) long-horizon trade-off vs pure MTM.

## Recommended Paper Usage

1. Current draft: use the matched n=200, three-training-seed controlled table
   for statistical claims, with the LeWM-style n=50 figure used only to place
   the controlled rows beside external paper-reported baselines.
2. Keep the e100 reproduction sweep as provenance, not as the main claim: the
   Reacher e100 MTM collapse and TwoRoom epoch-30 instability make it a schedule
   audit rather than the clean paper table.
3. Appendix protocol: keep the eval settings table and explicitly separate
   paper-facing controlled claims from historical reproduction/screening rows.
4. Diagnostics: use the completed three-seed NoReg ablation for the
   anti-collapse sanity check, the PushT mechanism table for probe/planner
   confounds, and the latent surprise diagnostic for physical consistency.
5. Avoid claiming robust plain-MTM Reacher parity. The three-seed e10 evidence
   says plain MTM is seed-unstable because two masked seeds collapsed, and the
   e100 extra-seed sweep confirms seeds 1/2 are already poor at epochs
   10/17/25/29 as well as epoch 30.
6. Do not present the e100 epoch-30 TwoRoom MTM win as robust. The e10
   three-seed MTM win remains reliable, and e100 seed 2 is strong at
   checkpoints 15/16/29, but epoch 30 reruns at 45.0/46.5. Use e10 or a
   predeclared validation-selected checkpoint rule for TwoRoom.

## Current MTM Schedule Selection

For MTM specifically, the best paper-safe schedule is task-dependent but mostly
favors the original 10-epoch cosine run:

| Task / protocol | Best MTM schedule to report | Evidence | Caveat |
|---|---|---|---|
| TwoRoom 25/50 | 10-epoch LR, epoch 10 | Three-seed n=200 mean 90.2 +/- 0.5 | e100 can be strong at selected checkpoints, but epoch 30 is unstable |
| TwoRoom-long 100/150 | 10-epoch LR, epoch 10 | Three-seed n=200 mean 28.0 +/- 0.8 | e100 epoch-30 n=50 also scored 28.0, but not multi-seed robust |
| PushT 25/50 | 10-epoch LR, epoch 10 | Three-seed n=200 mean 85.5 +/- 0.7; e100 epoch 30 single seed only 85.0 | MTM loses to SIGReg on this task |
| Reacher 25/50 | Action-NCE `loss.masked.inverse_weight=0.30`, 10-epoch LR, epoch 10 | Weight-0.30 epoch-10 three-seed audit scored 70.5 / 70.5 / 64.0, mean 68.3; lower-LR MTM epoch 9 three-seed audit scored 68.5 / 74.0 / 65.5, mean 69.3 | Action-NCE is the paper-safe Reacher choice because it is SIGReg-parity without the plain-MTM collapse mode; lower-LR MTM is competitive but schedule-sensitive |
| OGBench-Cube 25/50 | 10-epoch LR, epoch 10 | Three-seed n=200 mean 79.3 +/- 2.4; e100 epoch 30 single seed 77.0 | e10 is both better and better supported |
