# Multi-Future LeWorldModel Progress

## Ongoing Instructions

- Keep this progress file updated as project decisions, implementation status, and outstanding tasks change.
- Preserve a concise conversation log so the research direction can be reconstructed after context loss.
- Treat the project as a research fork of LeWorldModel, not as a generic "predict more frames" variant.
- Keep the initial implementation controlled: deterministic direct multi-horizon prediction before multi-future diversity.
- Make generated prediction-visualization GIFs loop by default unless explicitly configured otherwise.
- Do not use nearest-neighbor frame retrieval for prediction visualization. Render predicted latents only with a trained decoder for that model/checkpoint; otherwise keep latent metrics and skip prediction-frame artifacts.
- Store stable-worldmodel datasets, checkpoints, decoders, evaluation outputs, and visualizations under the repository-local `.stable_worldmodel/` directory by default; keep that directory out of git.
- Launch Modal jobs with the blocking entrypoint plus the Modal CLI's global `run --detach` option; do not use entrypoint-level `--no-wait`.
- Keep `progress/experiment-tables.md` updated when paper-facing experiment rows, schedules, seeds, or eval protocols change.

## Vision

Build a research project around the claim that direct latent trajectory prediction can reduce long-horizon rollout drift in end-to-end JEPA world models, and that predicting multiple plausible future latent trajectories may further improve planning under ambiguous dynamics.

Working title:

**Reducing Latent Rollout Drift in End-to-End JEPA World Models via Multi-Horizon Prediction**

Upgrade path if the stochastic/diverse extension works:

**Multi-Future LeWorldModel: Direct Prediction of Diverse Future Latent Trajectories for Long-Horizon Planning**

## High-Level Requirements

- Reproduce the original LeWM autoregressive baseline.
- Add a deterministic direct multi-horizon predictor, `LeWM-Direct-H`.
- Run training and inference/evaluation on Modal when local compute is insufficient.
- Compare direct-horizon prediction against fair autoregressive baselines, not merely against a weaker one-step objective.
- Measure horizon-wise latent error as the first decisive result.
- Measure planning success as a function of planning horizon as the second decisive result.
- Add multi-future prediction only after deterministic direct-horizon prediction is working.

## Compute Plan

- Local machine profile observed 2026-05-25: Apple M2, 8 GB RAM, MPS available, no CUDA, and roughly 21 GiB free before dataset downloads. This is enough for TwoRoom smoke/paper-subset evaluation, but not enough for PushT/Reacher/Cube datasets without freeing space or using external storage.
- Default training GPU: single `A100-40GB` on Modal for early PushT/Reacher runs. It is only slightly more expensive than `L40S`, so it is likely more cost-efficient if it is even modestly faster.
- Paper-reproduction GPU: single `L40S`, used when we want hardware parity with the LeWM paper.
- Avoid multi-GPU training for the MVP unless single-GPU profiling shows a clear bottleneck and the training stack has been tested under DDP. Prefer running independent seeds/configs in parallel as separate one-GPU jobs.
- Avoid `A100-80GB`, `H100`, `H200`, and `B200` for routine runs unless memory pressure or benchmarked time-to-result justifies the higher hourly rate.
- Default evaluation GPU: single `L4`. Move to `L40S` only for planning sweeps where wall-clock time becomes the bottleneck.
- Random-policy or config-only evaluations should run CPU-only where possible.
- 2026-06-19 resource-planning note: On fixed 2 x 32 GB GPUs, current batch-128 training should be treated as one run per GPU, not two runs per GPU. The best local evidence is an L40S batch-128 speed probe at ~24.34 GiB allocated and batch-256 OOM at ~43.31 GiB. Four concurrent runs would require microbatching, likely `loader.batch_size<=64` with gradient accumulation to preserve the effective batch/LR schedule, and a short memory smoke before committing. H=10 direct-horizon batch-128 runs are known not to fit 32 GB.

## Project Plan

### Stage 1: Baseline Reproduction

- Train original LeWM on PushT.
- Verify existing planning evaluation on PushT.
- Repeat on Reacher after PushT is stable.
- Done: Add Modal remote train/eval/upload entrypoints.
- Done: Downloaded and extracted the TwoRoom dataset locally at `.stable_worldmodel/tworoom.h5` and converted the HF TwoRoom LeWM checkpoint to `.stable_worldmodel/tworoom/lewm_object.ckpt`.
- Done: Ran local TwoRoom eval smoke tests on MPS. Paper-aligned random baseline over 50 episodes produced 2% SR in 24s. Paper-aligned LeWM over 5 episodes produced 20% SR in 4m34s, which is too small/noisy to compare to the paper's 87% claim but shows the local runtime is high.
- Done: Ran checkpoint-based TwoRoom LeWM reproduction on Modal using the uploaded dataset/checkpoint, with `eval.num_eval=50`, `eval.env_batch_size=10`, `goal_offset_steps=100`, `eval_budget=150`, CEM horizon 5/action block 5, and video disabled. Result: 10% SR (5/50) in 1038s on A100-40GB eval. This is far below the paper's reported TwoRoom result and needs follow-up before treating the checkpoint reproduction as validated.
- Done: Investigated the TwoRoom discrepancy against upstream issues #38 and #72. The released HF checkpoint reproduces the public 10% failure under the paper-text `goal_offset_steps=100`, `eval_budget=150` protocol, but reproduces the reported-result scale under the upstream repo's current `goal_offset_steps=25`, `eval_budget=50` protocol: 88% SR (44/50) on Modal with `eval.env_batch_size=10`. `config/eval/tworoom.yaml` now tracks the released-checkpoint/report-reproducing protocol, while `config/eval/tworoom_long.yaml` preserves the paper-text long protocol.
- Done: Trained original autoregressive LeWM from scratch on TwoRoom through Modal under ticket `1779781949255`. Final run used Modal app `ap-T1PMgjnC4Clk5hzKLr01Id`, W&B `https://wandb.ai/jack-b/lewm/runs/tworoom_lewm_from_scratch`, output `tworoom/lewm_from_scratch`, staged `tworoom.h5` to local Modal disk, disabled early stopping for the paper-style 10-epoch budget, and finished 10 epochs / 51,380 steps. Final and best validation loss was `0.157995` at epoch 9.
- Done: Evaluated the trained-from-scratch TwoRoom LeWM on the report-reproducing `goal_offset_steps=25`, `eval_budget=50`, `eval.env_batch_size=10`, video-off protocol. Result: 86% SR (43/50) in 677.6s on L4 eval, saved to `/tworoom/lewm_from_scratch_tworoom_results.txt`. This reproduces the released-checkpoint result scale (previous HF checkpoint reproduction was 88%, 44/50).

### Stage 2: Deterministic Direct Multi-Horizon Prediction

- Done: Add a direct future latent predictor with inputs:
  - context embeddings: `B x C x D`
  - context action embeddings: `B x C x A_emb`
  - future action embeddings: `B x H x A_emb`
- Done: Output future embeddings: `B x H x D`.
- Done: Train with horizon-wise latent MSE plus SIGReg.
- Done: Add `config/train/lewm_mh.yaml` with `H = 5`, `history_size = 3`, uniform horizon weights.
- Done (screening pass, n=50): Modal `LeWM-Direct-H` horizon sweep (H=5 uniform, H=5 discount, H=10 uniform) vs AR on both protocols. MH-H10-uniform looked best (Protocol A 90% vs AR 86%; Long 24% vs AR 16%) but every delta was within noise at n=50 (paired McNemar p=0.69 / 0.45). Flagged for confirmation.
- Done (confirmatory run, n=200, 3 MH-H10 training seeds {3072,1,2} vs single AR checkpoint, eval seed=42, fully paired): see `progress/experiment-log.md` (2026-06-01 entry). Outcome — **MH-H10 is promising on the long-horizon regime but not a statistically established win**:
  - **Long protocol (100/150):** all 3 MH seeds beat AR — mean **22.2%** (21.0/24.0/21.5, sd 1.6) vs AR **18.0%**, +4.2 pts. The improvement *replicates across independent trainings*, but no single seed reaches significance (McNemar p=0.11–0.51).
  - **Protocol A (25/50):** MH ≈ AR — mean **87.5%** (86.0/91.5/85.0, sd 3.5) vs AR **85.0%**; only the high-variance seed-1 separates (p=0.024), the other two tie. No reliable short-horizon advantage.
  - The screening's headline gaps shrank with power (MH-s3072 Protocol A 90→86, Long 24→21), i.e. the n=50 result was inflated by noise + a single checkpoint. But the *pattern* — benefit concentrated on the long horizon where AR rollout drift is worst, absent on the short — matches the drift-reduction hypothesis.
- Next (Stage 3, not more planning episodes): test the mechanism directly — horizon-wise latent-error curves (MH vs AR rollout under recorded actions) and planning-success-vs-horizon sweeps — which discriminate the drift claim far more cheaply than pushing eval to n≥500. Carry MH-H10-uniform forward; H=5 discount arm dropped.
- Operational lessons logged for future Modal runs: launch **evals and long jobs** with global `modal run --detach` while keeping the entrypoint blocking; entrypoint-level `--no-wait` is unsafe because a spawn-and-return path can leave an app stopped with zero tasks. Treat the volume result file as source of truth, and write background monitor scripts with explicit arrays (zsh does not word-split unquoted `$VAR`).

### Stage 3: Fair Baselines and Metrics

- `LeWM-AR`: original one-step autoregressive LeWM.
- `LeWM-AR-MS`: original predictor trained/evaluated with multi-step rollout loss.
- `LeWM-Direct-H`: direct multi-horizon prediction.
- `LeWM-Direct-H + AR-cost`: direct training with the original autoregressive planning cost where applicable.
- Produce horizon-wise latent MSE/cosine-distance plots.
- Produce planning success versus planning horizon plots.
- Done: Add qualitative prediction visualization for dataset rollouts, including decoder-backed prediction renderings and horizon-wise latent MSE/cosine plots. Nearest-neighbor frame retrieval has been removed because it is not a valid visualization of predicted latents.

### Stage 4: Multi-Future Prediction

- Extend direct prediction to output `B x K x H x D`.
- Train with best-of-K latent trajectory loss.
- Add diversity regularization only after best-of-K is stable.

## Outstanding Tasks

- Done 2026-07-03 follow-up for Jack's seed-caveat and physical-understanding
  comments: completed TwoRoom-long SIGReg n=200 evals for seeds `3072,1,2`
  (`17.0+/-0.4`), completed NoReg n=200 controls for seeds `3072,1,2`
  (`28.0+/-2.0`), and ran latent surprise diagnostics on PushT and OGB-Cube
  for SIGReg and AC-MTM across seeds `3072,1,2`. The paper now removes the
  remaining one-seed caveats and includes measured latent surprise,
  action-counterfactual, and state-discontinuity evidence.
- Done 2026-07-02 paper polish pass: addressed Jack's ICLR-readiness comments in
  `paper/latex/iclr_main.tex`, removed all live `\jack{...}` / `\chris{...}`
  markup, added the Scene visual gap figure, clarified Action-NCE and AC-CPC,
  tightened the limitations/conclusion, and filled the missing SIGReg n=200
  baseline seeds for the controlled paper tables. Final SIGReg standard-task
  values are TwoRoom `85.5±0.4` (85.5/86.0/85.0), Reacher `68.8±0.2`,
  PushT `93.2±0.2` (93.5/93.0/93.0), and Cube `66.2±0.2` (66.0/66.5/66.0).
  Valid Modal apps: TwoRoom `ap-7NeURdrgy6QPgcZ0IxQkTD`,
  `ap-cy7fEpty1rVJDx9nde8JTf`, `ap-RTA2gqRSD8DNTAQetoU1K7`; PushT
  `ap-41hoTV7FgLSCaeqc9LeBx5`, `ap-xhIEQFw10pcOAC9vdXOZcz`; Cube
  `ap-gscZBvdE8fVhkE8vkC2DIb`, `ap-996NnMVPDXgWtMzvBsWWxy`. The initial
  backgrounded `nohup ... &` wrapper was reaped by this tool environment before
  usable logs were created, so valid reruns used foreground
  `.venv/bin/modal run --detach ...` sessions kept open to completion and
  verified through live progress plus saved Modal volume result files.
- Active 2026-06-30 OGB puzzle representation fix: added a loss-only
  in-batch `state_nce` term to masked-transition training. The small additive
  delta variant (`ogb_state_nce/puzzle_4x4_play/action_state_nce_s3072_e3`,
  app `ap-rLiKMWb4Po6Vw597CO9tCh`) was rejected: epoch-3 button-state mean R2
  was `-0.0854` despite healthy `validate/state_nce_acc=0.9135`. The
  replacement delta variant (`ogb_state_nce/puzzle_4x4_play/state_nce_replace_s3072_e3`,
  app `ap-1KHVGY0UjPWJsIrw1fX76Y`) was also rejected: epoch-3 button-state mean
  R2 was `-0.0563`, with weak validation state-NCE generalization
  (`validate/state_nce_acc=0.0370`, `validate/emb_std=0.0748`). Decision: do
  not run trajectory evals for either delta checkpoint. The next controlled
  step is now running: additive absolute state-NCE
  (`ogb_state_nce/puzzle_4x4_play/action_state_nce_abs_s3072_e3`, app
  `ap-vUb3SBagjFOwwavTj38rQ2`), changing only `loss.masked.state_nce.mode` from
  `delta` to `absolute`. Epoch-1 button-state probe was briefly positive
  (`PROBE_R2 MEAN = 0.0304`, probe app `ap-I3QfFmBIO6yWxeX4zIqKMy`) but did not
  hold: epoch 2 scored `-0.0394` and epoch 3 scored `-0.0722` despite healthy
  validation state-NCE metrics (`validate/state_nce_acc=0.7651`,
  `validate/emb_std=0.2659`). Decision: reject easy absolute in-batch negatives.
  Implemented the next controlled rung, hard-negative absolute state-NCE
  (`lewm_masked_action_state_hard_nce`, `hard_negatives=64`), which keeps the
  same loss family and no new module/labels but focuses contrast on currently
  confusable target states. Launched same 3-epoch puzzle-4x4 gate at
  `ogb_state_nce/puzzle_4x4_play/action_state_hard_nce_s3072_e3`, app
  `ap-qcDEiPUUKehwCZDn2mzQrU`; initial validation loss is on the expected
  `log(65)` hard-negative scale and first backward passed. Epoch 1 stayed
  noncollapsed (`validate/emb_std=0.3354`, `validate/state_nce_acc=0.4630`), but
  the button-state probe was not promising (`PROBE_R2 MEAN = -0.0039`, probe app
  `ap-3qcUEIEydUHGqrZ6LUU1oc`). Epoch 3 validation recovered numerically
  (`validate/emb_std=0.2683`, `validate/state_nce_acc=0.7216`), but the
  button-state probe still failed (`PROBE_R2 MEAN = -0.0681`, probe app
  `ap-QjpslS4ZFfEGPepTv9FACb`). Decision: reject additive hard-negative
  state-NCE and do not launch trajectory evals. Evidence now points away from
  "state identity contrast is sufficient"; next principled rung should make the
  auxiliary signal depend on long-horizon controllable endpoints, preferably
  endpoint Action-NCE rather than inverse-MSE. Implemented endpoint Action-NCE
  using the existing `HorizonInverseDynamics` path/config
  (`lewm_masked_endpoint_action_nce`); focused tests pass (`27 passed`). A first
  launch, app `ap-c5iKZB27TjDC60Ns41Vivm`, was stopped at step 486 because the
  config inherited 4-frame data windows and therefore only tested gaps 1-3.
  Fixed the config to load `history_size + horizon` frames while keeping
  `wm.num_preds=1`; the corrected h1-h5 sentinel
  (`ogb_endpoint_action_nce/puzzle_4x4_play/endpoint_action_nce_h5_s3072_e3`,
  app `ap-n4GhUjHZitzjIEvyysgdR7`) trained cleanly for 3 epochs and improved its
  own validation endpoint-action metric (`validate/inverse_acc=0.1325`,
  `validate/emb_std=0.3753`), but button probes failed: epoch 1
  `PROBE_R2 MEAN=-0.0021` and epoch 3 `PROBE_R2 MEAN=-0.0988`. Decision: reject
  endpoint Action-NCE for puzzle; no trajectory evals. The likely missing piece
  is not "more contrastive pressure" but an objective/sampling scheme that makes
  the latent variables with persistent button state unavoidable.
- Done 2026-06-26 CEM/test-time action semantics explainer: documented in the
  root `README.md` and paper appendix that CEM samples normalized flattened
  action blocks, refits a Gaussian to elite candidate sequences, and maps each
  `action_block=5` coarse transition to five ordered low-level simulator
  actions rather than one repeated action.
- Done 2026-06-26 README play-demo docs: added a user-facing Human play demo
  section to `README.md`, added reproducible generator
  `make_play_demo_gif.py`, generated `assets/play_demo.gif`, documented launch
  commands, global controls, task-specific controls, screenshots, self-tests,
  and fixed the Cube control conflicts by letting task-specific `Q` actions win
  over quit and moving gripper close from Space to `X`.
- Done 2026-06-25 paper revision: addressed all collaborator notes in
  `paper/latex/iclr_main.tex`, promoted action-NCE (`lewm_masked_action_nce`,
  inverse weight `0.30`) to the main method, retained inverse-MSE MTM as a
  non-contrastive inverse-regression ablation, and state explicitly that the
  inverse head and contrastive loss are training-only. Use the completed matched `n=200`,
  three-training-seed evidence as the controlled main table; keep the
  TwoRoom-long regression (24.2 vs 28.0 for inverse-MSE MTM) visible. Rebuilt
  the results figure, replaced the hero image with a training-vs-test TikZ
  diagram at the top of page 2, and verified the 10-page PDF with `pdflatex`
  plus rendered-page inspection.
- Done 2026-06-26 paper language cleanup: removed the paper-facing "original
  method works but is less stable" framing. AC-MTM now reads as the proposed
  method, while MTM-MSE is described scientifically as a non-contrastive
  inverse-regression ablation/baseline. Reinserted Chris's original comments as
  `\chris{...}` markers at the locations where the revised text addresses them,
  and rebuilt `paper/latex/iclr_main.pdf`.
- Done 2026-06-26 paper evaluation/layout cleanup: replaced the diagram with a
  regenerated original-style image-panel figure based on `images/hero_figure.png`
  that shows training-time action input, Action-NCE, CEM test-time action
  selection, and test-time inverse-branch removal. Removed the post-abstract
  blank gap while keeping the hero figure at the top of page 2. Restored the
  LeWM-style multi-baseline results figure with paper-reported
  PLDM/DINO-WM/GCBC/GCiQL/GCIVL/random baselines in the main results section,
  kept the main controlled table to AC-MTM vs SIGReg, and added the full
  SIGReg/MTM-MSE/AC-MTM three-way comparison table plus ablation chart to the
  appendix. Rebuilt and rendered the 11-page PDF for visual inspection.
- Done 2026-06-26 Figure 4 action-rollout redesign: replaced the thin
  action-block timeline in `paper/latex/iclr_main.tex` with
  `images/action-blocks-and-cem-rollout-explainer.png`, a richer diagram showing
  dataset action-block construction, CEM candidate rollouts/final-latent
  scoring, elite refit, low-level execution, and re-planning. Rebuilt
  `paper/latex/iclr_main.pdf` and rendered the affected appendix pages for
  visual inspection.
- Done 2026-06-27 OGBench Scene paper audit: interrogated the completed
  trajectory-goal Scene result from raw Modal result files, train configs,
  checkpoint states, metrics, and eval logs. Confirmed the matched n50/eval-seed
  42 comparison gives SIGReg `56/58/60` versus Action-NCE `80/78/82`, with 40
  Action-NCE-only wins versus 7 SIGReg-only wins across 150 paired episodes.
  Revised `paper/latex/iclr_main.tex` to distinguish confirmed facts from
  interpretation, to avoid public OGBench leaderboard language, and to add an
  appendix audit/alternative-explanations section. Rebuilt
  `paper/latex/iclr_main.pdf`.
- Get access to the full baseline checkpoint suite. The README/website Google Drive link is browser-visible but returned 404 to `gdown`/`curl` from the local CLI on 2026-05-25, so PLDM/DINO-WM/etc. were not downloaded.
- Decide whether to run the full 50-episode TwoRoom LeWM reproduction locally. Based on the 5-episode paper-aligned run, expect roughly 45 minutes on the local M2/MPS setup, with uncertain agreement until the full run is complete.
- Free disk or use external storage before attempting PushT locally. Current free space after TwoRoom is about 14 GiB; the PushT compressed dataset alone is about 12.2 GiB, and decompression would likely exceed available space. Reacher/Cube are larger.
- Decided to keep both `25/50` (Protocol A) and `100/150` (Long protocol) as separate evaluations for method comparisons. Confirmatory n=200 / 3-seed run (2026-06-01): MH-H10-uniform beats AR on the Long protocol across all 3 training seeds (mean 22.2% vs 18.0%) but ≈ AR on Protocol A; the long-horizon improvement is consistent but not individually significant (McNemar p=0.11–0.51). Treated as promising-not-confirmed; mechanism to be tested in Stage 3 rather than via larger planning evals.
- Done: Evaluated `tworoom/lewm_from_scratch` with `config/eval/tworoom_long.yaml` as the 100/150 long-goal stress test. Result: 16% SR (8/50).
- Run baseline LeWM training on PushT if checkpoint reproduction is not sufficient.
- Resume corrected `LeWM-Direct-H, H=5` training on PushT and Reacher after speed-probe policy is settled. The previous H100 apps are stopped: PushT `ap-NZxOY724QmJooHJLdrMhVb` with W&B run `pusht_lewm_direct_h5_fast` and checkpoint at global step 1000; Reacher `ap-Ng2vMTXVhDdYB5nRSdYXwl` with W&B run `reacher_lewm_direct_h5_fast` and checkpoint at global step 500.
- Done ticket `1779964126980`: first-class training logging with WandB via `$WANDB_API_KEY`, local `metrics.jsonl`/`events.jsonl` fallback, run metadata, timing/throughput/ETA/LR/grad-norm metrics, validation metrics, and checkpoint events.
- Done ticket `1779973200000`: reliable training resume with explicit resume config, compatible-run checks, full Lightning checkpoint loading (`weights_only=False` for PyTorch 2.6), RNG/W&B state persistence, atomic latest/best checkpoints, `checkpoint_state.json`, and periodic Modal Volume commits during training.
- Active ticket `1779967558491`: make training substantially faster through profiling-led, quality-preserving changes only. Implemented paper-aligned 10-epoch budget, explicit step-based scheduler config, bf16-mixed, local data staging, GPU-side image preprocessing, shm-safe DataLoader settings, early stopping, bounded `lewm_speed` probes, W&B/local timing observability, and CUDA memory metrics. Current probe evidence: L40S batch 128 completes at about 2.4 steps/s and 307 samples/s with negligible data wait; L40S batch 256 OOMs at about 43 GiB allocated; H100 batch 256 completed at about 2.17 steps/s and 555 samples/s but had significant data wait. H100 should not be the default probe GPU unless the larger memory/throughput is specifically being tested.
- Add a standalone horizon-wise latent prediction evaluation script.
- Add AR multi-step rollout-loss baseline.
- Add planning horizon sweep configs.
- Design a controlled "task-conditioned planning-time adaptation" experiment for
  PushT: use the current state, goal observation, candidate/action elite rollouts,
  and optionally online env observations during MPC to update either a lightweight
  latent metric/task head or a small adapter, while keeping a no-adaptation
  baseline and logging whether block pose/orientation decodability improves.
- Add a GC-IDM-style frozen-latent planner baseline inspired by Nguyen/Xu/Huang
  2026: extract embeddings from base/masked/direct checkpoints, train
  `(z_current, z_goal, remaining_horizon) -> next_action`, and evaluate as a
  replacement for CEM plus as a possible warm-start/proposal for CEM.
- Design, but do not conflate with the frozen-planner baseline, an end-to-end
  goal-conditioned inverse auxiliary loss for world-model training. This differs
  from current masked transition because it predicts the first action toward a
  distant sampled goal, not the action between adjacent latents.
- Retrain any previously generated auxiliary decoder artifacts with the author-aligned CLS decoder before using them for paper-quality decoded figures; legacy decoder checkpoints are no longer supported.
- Add multi-future predictor and losses after deterministic results are understood.
- Reacher correction from 2026-06-18: do not treat the old masked-transition 63-68% Reacher result as a robust tie. New masked seeds 1 and 2 collapsed (`emb_std` near zero) and scored 11.5% / 13.5%, while matched SIGReg seeds scored 69.0% / 68.5%. Add collapse prevention or an explicit collapse gate before using plain masked-transition Reacher claims.
- 2026-06-22 Reacher objective sentinels rejected: seed-1 standard e10 runs for multi-step latent overshooting (`reacher/mtm_overshoot_e10_s1`, app `ap-Sglp40SzY4Co2Z0Jrwch0i`) and multi-step inverse/action discrimination (`reacher/mtm_ms_inverse_e10_s1`, app `ap-DCoc1Ot1BhhYs1HRu1L0YR`) both collapsed during epoch 0. Final `mean10 train_step/emb_std` was about 0.017 for both while inverse/action loss stayed near chance (~1.0). Both apps were stopped before eval; do not launch seeds 3072/2 for these arms without a more fundamental objective change or collapse gate.
- Done 2026-06-22 Reacher mechanism sentinel: short `lewm_accpc` seed-1 run `reacher/accpc_collapse_sentinel_s1_4k` (app `ap-eVLarM0CAl64w8R3YLH6fI`) completed 4000 steps and did not collapse. `emb_std` increased from 0.0628 to 0.0715 and stayed there; CPC loss fell from 5.48 to 1.02. This validates the narrow hypothesis that same-encoder latent MSE is the early Reacher collapse driver. It does not yet validate planning quality; next minimum experiment should test a geometry-preserving hybrid or MSE-lite objective rather than another stronger inverse-MSE arm.
- Done 2026-06-22 Reacher targeted-fix sentinel: added `lewm_masked_action_nce`, which keeps standard MTM forward MSE but replaces inverse-action MSE with in-batch action discrimination. Short seed-1 run `reacher/mtm_action_nce_sentinel_s1_4k` (app `ap-7WCuUOLOVBjdKDI0SQAgel`) completed 4000 steps and did not collapse. Final `emb_std` was 0.3457 (mean10 0.3465), forward loss fell to 0.0071, and inverse NCE loss fell from 9.15 to 7.24. Promote this to a bounded seed-1 train-then-eval candidate before any multi-seed fanout.
- Done 2026-06-23 Reacher full candidate: `lewm_masked_action_nce` seed-1 e10 train-then-eval completed in `reacher/mtm_action_nce_e10_s1` (app `ap-3z4LVRSm4r0c8JR9hRs2t1`). It did not collapse: final train `emb_std` 0.2891 / mean10 0.2889. Reacher n=200 eval scored 57.0% (`113/200`) with output `reacher/mtm_action_nce_e10_s1_n200.txt`. Decision: action-NCE fixes collapse but is not good enough for multi-seed paper reporting; next work should target geometry/performance, not merely anti-collapse.
- Done 2026-06-23 action-NCE inverse-weight sweep: weights 0.02/0.05/0.10 were too close to collapse; 0.20 was borderline; 0.30 looked balanced (`emb_std` mean10 0.1656 at 4000 steps); 0.50 was safer but closer to weight-1.0 over-shaping. Full seed-1 candidate `loss.masked.inverse_weight=0.30` in `reacher/mtm_action_nce_w030_e10_s1` completed 10 epochs and remained noncollapsed (final `emb_std` 0.1826 / mean10 0.1825). Reacher n=200 eval scored 70.5% (`141/200`) with output `reacher/mtm_action_nce_w030_e10_s1_n200.txt`. Decision: weight 0.30 is a real improvement over action-NCE weight 1.0 (57.0%) and fixes collapse, but it is still one training seed; decide whether to fan out seeds `3072` and `2` or first audit earlier checkpoints.
- Active 2026-06-24 Reacher action-NCE weight-0.30 audit: seed-1 epoch 7/8/9 n=200 checkpoint evals completed at `64.5 / 73.5 / 68.5`; together with epoch 10 `70.5`, epoch 8 is the current best seed-1 checkpoint and exceeds the same-schedule SIGReg seed-1 result `69.0`. Checkpoint-audit apps are stopped: ep7 `ap-IjSQPih2XQCuIswhyaly2O`, ep8 `ap-1zWogr8NyS0pH1LeWYrJRP`, ep9 `ap-USO0WwaZWft3FxcraLwvxC`. Full train-then-eval fanout for seeds `3072` and `2` remains live in apps `ap-bav4j9lbktRcgGWjY4t486` and `ap-YkUFkYrajX6znb54NrUQdg`; both are beyond the 4000-step sentinel horizon and still healthy, with mean10 `emb_std` about `0.289` at step 5150 for seed 3072 and `0.304` at step 5500 for seed 2. Monitors will stop if mean10 `emb_std < 0.08` after 80 train-step rows.
- Active 2026-06-24 action-NCE cross-task generality check: launched `loss.masked.inverse_weight=0.30`, seed `3072`, e10 train-then-n50-eval runs for TwoRoom, PushT, and Cube. First direct background wrappers exited with zero-byte logs and no Modal apps, so the runs were relaunched in persistent tmux wrappers. Verified app IDs: TwoRoom `ap-8SglVluM3bxoLSCJ8bFqaP`, PushT `ap-ywPTjA7FCyE2HqE9IasNno`, Cube `ap-n2YqOQcdbNIf77oCZbGB8R`, each with `Tasks=1` and real train progress bars after first backward pass.
- Done 2026-06-27 OGBench Scene study: added first-class `scene` data/eval configs and a Modal conversion path for `visual-scene-play-v0` into `.stable_worldmodel/ogbench/visual_scene_play.h5`. The dataset has 1,000,000 transition rows at 64x64 RGB after dropping terminal observations. Fixed Scene eval rendering so the simulator renders 64x64 while the policy transform still receives 224x224 inputs. Modal random-policy smoke completed with `eval.num_eval=2`, `eval.env_batch_size=2`, `goal_offset_steps=25`, `eval_budget=50`, and expected 0% SR. Full seed-3072 e10 train-then-n50-eval runs completed via persistent tmux wrappers after direct background/interrupt launches proved unsafe: SIGReg `scene/lewm_sigreg_e10_s3072` app `ap-IexNJ27FhZGX5Yuy9pUiiu` scored 56.0% (28/50), Action-NCE `scene/lewm_masked_action_nce_e10_s3072` app `ap-YeuKRYAXXTvkcCnDGqbHCb` scored 80.0% (40/50). Paired discordants were 14 Action-NCE wins vs 2 SIGReg wins (exact McNemar/binomial p ~= 0.0042). Outputs: `.stable_worldmodel/scene/scene_sigreg_e10_s3072_n50.txt` and `.stable_worldmodel/scene/scene_action_nce_e10_s3072_n50.txt`.
- Done 2026-06-27 OGBench Scene robustness fanout: matched e10 train-then-n50-eval runs for SIGReg and Action-NCE completed for seeds `3072,1,2` under the repo's trajectory-goal Scene protocol. SIGReg scored `56.0 / 58.0 / 60.0` (mean 58.0), while Action-NCE scored `80.0 / 78.0 / 82.0` (mean 80.0), a three-seed mean delta of +22 pts. Combined paired discordants across 150 matched episodes were 40 Action-NCE wins vs 7 SIGReg wins (two-sided binomial/McNemar p ~= 1.1e-6). Outputs: `.stable_worldmodel/scene/scene_sigreg_e10_s{3072,1,2}_n50.txt` and `.stable_worldmodel/scene/scene_action_nce_e10_s{3072,1,2}_n50.txt`.
- 2026-06-27 public OGBench Scene comparison note: published `visual-scene-play-v0`
  overall scores use the official five fixed OGBench goals, averaged over four
  pixel-task seeds. Best-known overall public score found so far is HIQL at
  49 +/- 4 in the original OGBench table; newer SAW reports 47 +/- 6 and LPWM
  reports 40 +/- 1 overall, though LPWM is best on some individual goals. Our
  current Action-NCE 80.0% Scene result is under the repo's trajectory-goal MPC
  protocol (`goal_offset_steps=25`, `eval_budget=50`), so it should not be
  called public SOTA until we implement/run the official five-goal OGBench
  Visual Scene protocol.
- Done 2026-06-27 official OGBench Visual Scene eval plumbing: added
  `config/eval/scene_official.yaml` and an `official_scene` branch in `eval.py`
  that runs `visual-scene-singletask-task{1..5}-v0` fixed-goal envs with
  image goals, retaining the same `WorldModelPolicy`/CEM checkpoint path. Local
  random-policy task1 smoke passed. Action-NCE seed-3072 task1 ran cleanly on
  Modal at both a 75-step smoke cap and the official 750-step cap, scoring 0/1
  in both cases. This confirms compatibility/plumbing but suggests the current
  final-latent MPC setup is not solving the official fixed-goal task out of the
  box.
- Done 2026-06-28 OGBench Visual Scene fixed-goal singletask benchmark: after the
  monolithic fixed-goal jobs stopped around 14% from Modal client/network
  failures and the local tmux chunk retry remained client-fragile, orchestration
  moved into `run_scene_official_chunk_supervisor`, a Modal-resident supervisor
  app `ap-lP9ONtY1E5FEgKbaGHOKCV`. The run produced all 50 expected chunks for
  `{Action-NCE,SIGReg} x task{1..5} x episode starts {0,10,20,30,40}` and no
  longer depends on tmux or the laptop staying on. Fixed-goal singletask result:
  Action-NCE seed 3072 scored 0/250 and SIGReg seed 3072 scored 0/250; every
  task was 0/50 and every episode hit the 750-step cap. Treat this as a
  successful compatibility/orchestration result but not as evidence for a
  public OGBench leaderboard claim. Follow-up protocol check: run a cheap
  `visual-scene-v0` base-env `reset(options={"task_id": ...})` smoke, because
  the OGBench public goal-conditioned harness evaluates the base env by
  `task_id` while this run used the registered singletask envs. Summary
  artifact: `progress/evaluations/scene-official-s3072/summary.json`.
- Done 2026-06-29 broader OGB single-seed corrected trajectory screening:
  after billing was unblocked, ran `eval.protocol=ogb_trajectory` for SIGReg
  and Action-NCE on the six requested tasks from existing seed-3072 epoch-10
  checkpoints. Fixed powderworld's discrete-action planner path so it returns
  integer action ids and PGD warm-starts with `from_scalar=True`. Results
  (Action-NCE vs SIGReg): puzzle 4x4 `34/50`, puzzle 4x5 `26/52`, antmaze
  teleport `46/40`, powderworld `16/6`, antmaze large `30/30`, synthetic
  top-down antsoccer `88/88`. Treat this as a repo trajectory-goal screening
  result, not a public OGB fixed-goal leaderboard result. Summary artifact:
  `progress/evaluations/ogb-broad-corrected-s3072/summary.json`.
- Done 2026-06-24 collapse-mechanism writeup + diagnostics: documented *why* plain MTM collapses on Reacher specifically, and added gradient-free monitors to every `train.py` forward. Mechanism: the inverse term is a *conditional* anti-collapse mechanism vs SIGReg's *unconditional* `N(0,I)` floor — it goes slack when (a) the action is underdetermined from the latent endpoints so inverse MSE sits at the mean-action floor (Reacher's `frameskip:5` block + random-policy 2-link arm; matches the 2026-06-22 sentinel where inverse loss stuck ~1.0 while `emb_std`→0.017), and (b) the `LayerNorm` in `InverseDynamics` makes the inverse loss scale-invariant, so it cannot oppose the scale-shrink the (non-detached, LeWM-faithful) forward MSE rewards. action-NCE fixes it because its collapse floor is chance accuracy `1/N` (margin-independent, steep near collapse), not the mean-action MSE floor. New logged metrics (detached, ~one `DxD` covariance of overhead, off the loss graph): `effective_rank` (participation ratio `tr(C)²/‖C‖²_F`, catches *subspace* collapse that `emb_std` misses) on all 8 forwards, plus `inverse_margin`/`inverse_baseline` on the 4 inverse-bearing ones (MSE: mean-action-floor − loss; `action_nce`: `acc − 1/N`). Also wired into `lejepa`/`mh_lejepa` so SIGReg-vs-MTM geometry is directly comparable. Guarded by `tests/test_masked_transition.py::test_masked_forward_logs_collapse_diagnostics` and `::test_masked_forward_effective_rank_detects_collapse`; full mechanism in `paper/MaskedTransitionModel.md` ("Why it sometimes *doesn't*"). Use these to read the live seed-3072/seed-2 fanout: `effective_rank`↓ or `inverse_margin`→0 flags trouble before eval SR does.
- Done 2026-06-25 baked the tuned action-NCE weight into config: `config/train/lewm_masked_action_nce.yaml` now sets `loss.masked.inverse_weight: 0.30` as its default (previously inherited 1.0 from `lewm_masked` and applied via CLI override). The canonical action-NCE config is now the candidate itself; sweep with `loss.masked.inverse_weight=<x>` on the CLI. In-flight/finished runs that passed the override explicitly are unaffected (override equals the new default). Verified via `hydra.compose` (resolves to 0.30, SIGReg off) and `tests/test_config_sanity.py`.
- Future Reacher evals should use exact checkpoint stems and unique filenames, not directory policies plus default `dmc_results.txt`: e.g. `policy=reacher/lewm_masked_s1/lewm_masked_epoch_10`, `output.save_video=false`, `eval.env_batch_size=10`, and `output.filename=<unique>.txt`.
- Monitor the matched e100 epoch-30 SIGReg vs MTM sweep launched on 2026-06-18. Seed-3072 n=200 table is complete: PushT SIGReg 84.5%, PushT MTM 85.0%, TwoRoom SIGReg 84.0%, TwoRoom MTM 87.5%, Reacher SIGReg 83.5%, Reacher MTM 77.0%, Cube SIGReg 66.5%, Cube MTM 77.0%. Earlier direct detached retry apps `ap-V4d52gU06XW5C3DhzNFrLN` and `ap-2fwxdarYZ3E7nVmzK7R2H8` were canceled after local-client interruption and should not be used.
- E100 multi-seed robustness batch launched on 2026-06-18: TwoRoom standard n=200 seed 1/2 evals completed: SIGReg 87.0/85.0, MTM 91.0/45.0. Follow-up checkpoint-trajectory evals for `tworoom/lewm_masked_e100_s2` completed: epoch 10/15/16/29/30-rerun scored 60.5/88.0/90.5/90.0/46.5, confirming a bad final checkpoint / late e100-schedule instability rather than a bad seed or eval bug. Modal now appears budget/workspace-blocked (`workspace ... is disabled`) and several long jobs hit the 86400s input timeout. Reacher SIGReg seed 1 reached epoch 30 but its n=200 eval timed out before writing a result; Reacher SIGReg seed 2 and MTM seeds 1/2 stopped just before epoch 30. PushT/Cube extra seeds have partial resumable checkpoints, with Cube SIGReg seed 2 completed at 62.0.
- Done: Completed the LeWM-style `n=50` retry batch launched on 2026-06-18 after the GitHub issue audit. Use `progress/evaluation-protocol-ledger.md` and `progress/evaluations/lewm-style-n50-r1/summary.md` as the protocol/result source of truth. Ignore the first `n50_*` launch attempt because it was cancelled after local Modal client disconnects. Final `n50r1_*` results: TwoRoom SIGReg/MTM 88/96, PushT 96/90, Reacher 74/74, OGBench-Cube 76/86; TwoRoom-long 100/150 stress 12/28.
- Done: Completed `n50r2` diagnostics for paper consistency. TwoRoom NoReg scored 34.0% versus SIGReg/MTM 88/96; PushT diagnostics scored Recon 92.0, AC-CPC 62.0, MS-MTM 60.0, and BYOL-WM 38.0 under the same `n=50`, seed-42 protocol. These results support keeping alternate mechanisms as diagnostic rows rather than claiming they replace SIGReg.
- For final paper robustness, target 3 training seeds (`3072,1,2`) per task/method at e100 epoch 30, with n=200 eval seed 42. The extra-seed batch is now launched/queued; remaining manual eval work after training includes TwoRoom-long evals for the new TwoRoom seed checkpoints and the Cube SIGReg seed-3072 eval once its epoch-30 checkpoint lands.
- Done: Added `progress/experiment-tables.md` as the consolidated experiment inventory for the ICLR results section. It separates paper-facing `n=50` rows from historical/internal `n=200` diagnostics, records train/eval seed counts, eval settings, LR schedule, checkpoint epoch, pending cells, and Reacher collapse caveats.

## Conversation Log

- 2026-06-27: Completed the OGBench visual Scene study for the more complex
  `visual-scene-play-v0` task. Added the Scene HDF5 conversion, train/eval
  configs, Modal dataset staging, Scene config tests, and an eval render-size
  fix for 64x64 simulator frames. Under matched seed-3072 e10 training and
  n50/eval-seed-42 evaluation, Action-NCE beat SIGReg 80.0% vs 56.0% with 14
  paired wins and 2 paired losses.
- 2026-06-27: Launched Scene robustness fanout for train seeds `1` and `2`,
  covering SIGReg and Action-NCE under the same e10/n50/eval-seed-42 protocol.
  All four Modal apps are live with one task and have printed first-epoch train
  progress; results will complete the 3-training-seed Scene comparison.
- 2026-06-27: Checked public OGBench Visual Scene numbers. The original
  OGBench five-goal protocol reports HIQL as the best overall baseline at
  49 +/- 4 on `visual-scene-play-v0`; SAW and LPWM do not exceed that overall,
  though LPWM reports 100 +/- 0 on task1 and 89 +/- 9 on task3. Current repo
  Scene eval uses sampled future visual goals, so a public-SOTA comparison
  requires adding the official fixed-goal Visual Scene eval.
- 2026-06-27: Launched the full official OGBench Visual Scene fixed-goal
  benchmark for matched seed-3072 Action-NCE and SIGReg checkpoints. The run
  uses all five official `visual-scene-singletask-task{1..5}-v0` goals, 50
  episodes per task, and the 750-step cap. Both Modal apps were verified live
  with checkpoint load/env creation/progress logs and `Tasks=1`; outputs are
  pending because the full sweep is long.
- 2026-06-27: Added and smoke-tested the official fixed-goal Visual Scene eval.
  Official Scene uses a 750-step cap for every task. Primitive task complexity
  is roughly task1=2, task2=6, task3=4, task4=4, task5=8 object-level subtasks.
  The first Action-NCE seed-3072 official task1 one-episode runs scored 0/1 at
  both 75 and 750 steps, so the evaluator is working but the current planner
  does not immediately transfer to official fixed-goal success.
- 2026-06-27: Audited the completed OGBench Scene trajectory-goal paper result
  because the Action-NCE margin over SIGReg looked unusually large. Raw result
  files, configs, checkpoint states, train metrics, and eval logs support the
  matched comparison; no obvious episode mismatch, checkpoint mismatch, or
  privileged-state planner leak was found. The paper now frames the gap as a
  robust trajectory-goal planning result with plausible geometry/dynamics
  explanations and explicit remaining checks, not as a public OGBench
  leaderboard claim.
- 2026-06-25: Began revising the ICLR manuscript around action-contrastive
  masked transition modeling. Paper-facing interpretation: action-NCE fixes the
  Reacher seed-collapse of inverse-MSE MTM (68.3% vs 31.0% mean, three seeds)
  while remaining within 1.2 points on TwoRoom, PushT, and Cube at matched
  `n=200`; MTM-MSE is retained as a non-contrastive inverse-regression ablation
  with a roughly four-point advantage on TwoRoom-long. The training-only
  inverse branch is discarded for evaluation, leaving the LeWM encoder,
  forward predictor, and CEM planning path unchanged.
- 2026-05-22: Project reframed as **Multi-Future LeWorldModel**. Core hypothesis: direct multi-horizon latent prediction should reduce autoregressive rollout drift, with multi-future prediction as the later novel extension. First milestone is deterministic `LeWM-Direct-H` with `H=5`, `history_size=3`, PushT first, then Reacher.
- 2026-05-22: Added the deterministic direct-horizon code path: `FutureQueryPredictor`, `JEPA.predict_future`, `JEPA.rollout_direct`, `mh_lejepa_forward`, and `config/train/lewm_mh.yaml`. Static Python compilation passes, but runtime shape checks could not run because the current shell lacks `torch`.
- 2026-05-22: Added Modal support via `modal_app.py`. Remote training and evaluation use a persistent Modal Volume mounted under the repository-local `.stable_worldmodel/` path; WandB is disabled by default for remote training unless explicitly enabled.
- 2026-05-25: Installed `uv` under `~/.local/bin`, created `.venv` with Python 3.10.19, and installed the training/test stack. Local macOS setup required preinstalling `gym==0.21.0` with older pip/setuptools/wheel/packaging and putting the `swig` executable from the Python `swig` package on `PATH` for `box2d-py`.
- 2026-05-25: Added pytest sanity coverage for `FutureQueryPredictor`, `JEPA.predict_future`, `JEPA.rollout_direct`, and config mode selection. Verified `pytest -q` passes with 6 tests, `py_compile` passes for main scripts, `modal_app` imports, and Hydra composes `lewm_mh` with `data=pusht` and `wandb.enabled=false`.
- 2026-05-25: Settled initial Modal GPU policy: `A100-40GB` for routine single-GPU training, `L40S` for paper-parity runs, `L4` for normal evaluation, and independent one-GPU jobs instead of multi-GPU training until profiling justifies DDP.
- 2026-05-25: Investigated local reproduction feasibility. Patched eval to support `auto` device resolution (`cuda`/`mps`/`cpu`), macOS-friendly default `MUJOCO_GL`, and configurable video saving. Patched `JEPA.get_cost` to cast floating tensors to `float32` before moving them to MPS. Updated TwoRoom eval defaults to the paper-aligned `goal_offset_steps=100` and `eval_budget=150`.
- 2026-05-25: Downloaded TwoRoom from Hugging Face by streaming `tworoom.tar.zst` directly into `.stable_worldmodel/tworoom.h5` (11.9 GiB extracted). Downloaded `quentinll/lewm-tworooms` weights/config and converted to `tworoom/lewm_object.ckpt`; conversion required remapping older Hugging Face ViT key names to the installed transformers key names.
- 2026-05-25: Local eval results: random TwoRoom with paper settings (`num_eval=50`) produced 2% SR in 24s. LeWM TwoRoom with paper settings (`num_eval=5`) produced 20% SR in 4m34s; the result is not enough to judge the paper's 87% claim and suggests a full local 50-episode LeWM run would take on the order of 45 minutes.
- 2026-05-25: Started implementation of qualitative model prediction visualization. Planned output is a reusable script that loads a task config, checkpoint, and offline dataset; rolls out predicted latents under recorded actions; compares them against target latents with horizon-wise MSE/cosine metrics; and renders grids/videos using nearest-neighbor dataset frames as a decoder-free visual proxy.
- 2026-05-25: Implemented `visualize_predictions.py` and `config/visualize/predictions.yaml`. The script supports the eval tasks by name, auto-infers AR/direct rollout mode, context size, horizon, dataset, frameskip, and image size where possible, and writes `rollout_*.png`, optional `rollout_*.gif`, `horizon_errors.csv`, `horizon_errors.png`, `metadata.json`, and `index.html`.
- 2026-05-25: Verified visualization with local TwoRoom smoke tests on `tworoom/lewm`, including nearest-neighbor grid rendering and GIF generation. Also verified `pytest -q` passes with 6 tests and `py_compile` passes for `jepa.py`, `module.py`, `train.py`, `eval.py`, `modal_app.py`, and `visualize_predictions.py`.
- 2026-05-26: Discussed trust limits of decoder-free nearest-neighbor visualization. Latent MSE/cosine curves are the reliable signal; retrieved frames are qualitative aids and can hide off-manifold or ambiguous predicted latents. A paper-style decoder would be an auxiliary diagnostic trained from frozen LeWM embeddings to pixels, not part of the planning model.
- 2026-05-26: Started adding auxiliary decoder training as a separate frozen-world-model path, with a local Hydra script and Modal entrypoint. The decoder should not be attached to the main LeWM training loss unless explicitly running an ablation, because the LeWM paper reports worse PushT planning when reconstruction loss is jointly optimized.
- 2026-05-26: Implemented auxiliary decoder support: `decoder.py`, `train_decoder.py`, `config/decoder/train.yaml`, Modal `decoder` entrypoint, and `visualize_predictions.py decoder.path=...` rendering. Added `max_samples` for fast diagnostic decoder runs on large datasets. Local smoke trained one-batch/subset TwoRoom decoders and verified decoded visualization output. Added decoder tests; `pytest -q` now passes with 8 tests and `py_compile` passes for main scripts plus decoder files.
- 2026-05-26: Compared decoder path in `~/Desktop/thepuzzler/le-wm`. It is a stronger plain-PyTorch decoder training harness with top-k reconstruction loss, normalized-pixel targets, AMP, scheduler, clipping, resume, JSONL/TSV logging, and reconstruction plots, but it is coupled to a PushT-focused latent-action fork and single-file config. Best improvement path is to selectively port loss/training-harness ideas into our Hydra decoder path, not replace our implementation wholesale.
- 2026-05-26: Ported the useful decoder training-harness improvements into this repo: ImageNet-normalized target support, top-k pixel MSE loss, AMP on CUDA, gradient clipping, cosine scheduler, resume support, per-step JSONL/TSV metrics, training curves, epoch checkpoints, reconstruction previews, and decoder-output denormalization in `visualize_predictions.py`. Added Modal `train_with_decoder` as an opt-in way to produce a companion decoder after a world-model run without adding decoder cost to normal training, and kept partial train batches so tiny `max_samples` decoder smoke runs work with the default batch size. Verified with 9 passing tests, py_compile, local TwoRoom decoder smoke training, and decoded visualization smoke output.
- 2026-05-26: Implemented the planning-video figure path. `eval.py` can now run with `output.save_planning_artifacts=true`, wrapping the world-model policy to save the selected state-latent plan at every MPC replan into per-task `planning_artifacts.pt` files alongside rollout videos and metadata. Added `render_planning_videos.py` to decode those planning latents with a trained decoder into imagined trajectory videos, imagined-vs-real comparisons, replan grids, summaries, and an HTML index. Verified with 11 passing tests, py_compile, Hydra eval composition, and a synthetic planning-render smoke test.
- 2026-05-26: Started ticket `1779781920812-reproduce-baseline-results-using-given-checkpoints`. Current interpretation: evaluate provided LeWM-style checkpoint assets on Modal, train a frozen-latent decoder for the checkpoint, and render decoded open-loop prediction visualizations so the predictions can be inspected directly.
- 2026-05-26: Added Modal `visualize` and `reproduce_checkpoint` entrypoints. `visualize` runs `visualize_predictions.py` on the Modal Volume; `reproduce_checkpoint` chains checkpoint evaluation, optional diagnostic decoder training, and decoded prediction rendering. Local Modal auth works under profile `boylan-jack`; the Modal CLI is installed in `.venv`, not on the global shell `PATH`.
- 2026-05-26: Created Modal Volume `multi-future-lewm-cache` and uploaded local TwoRoom assets to it: `/tworoom.h5` and `/tworoom/lewm_object.ckpt`. First remote smoke attempt exposed a Modal 1.4 image-build constraint: `add_local_dir` must be the last image operation unless copied into the image. Patched `modal_app.py` so `workdir`/`env` are applied before mounting the local repo.
- 2026-05-26: Second remote smoke attempt exposed dependency drift: Modal pulled `stable-worldmodel==0.1.0`, while the local validated environment uses `stable-worldmodel==0.0.6` and `stable-pretraining==0.1.6`. The newer `World` path forwarded `history_size`/`frame_skip` into `TwoRoomEnv`, breaking eval. Pinned the Modal image to the local validated dependency versions for reproducible checkpoint evaluation.
- 2026-05-26: Pinning `stable-worldmodel==0.0.6` on Modal exposed the old `gym==0.21.0` packaging issue that was also seen locally. Switched the Modal image install path to explicit `pip` commands that first install the compatible `pip<24`, `setuptools==65.5.0`, `wheel<0.39`, and `packaging<22` toolchain, then install `gym==0.21.0`, then the pinned project runtime.
- 2026-05-26: Modal smoke for `tworoom/lewm` completed end to end: 2-episode eval, tiny one-epoch decoder, and decoded prediction visualization under `.stable_worldmodel/visualizations/tworoom/tworoom__lewm` in the Modal volume. Full 50-env eval attempts on L4 and A100 were stopped because the first CEM batch did not return promptly. Added `eval.env_batch_size` so a single 50-episode reproduction can preserve the sampled episode set while evaluating smaller vectorized batches and aggregating success.
- 2026-05-26: Completed Modal checkpoint reproduction for `tworoom/lewm` with `eval.num_eval=50`, `eval.env_batch_size=10`, and `output.save_video=false`. Result was 10% SR (5/50), with per-batch CEM solves around 32-37s and total eval time 1037.8s. Trained a 50k-sample, 3-epoch diagnostic decoder with best `val_loss=0.00636`, then rendered 8 decoded prediction rollouts. Downloaded outputs locally under `.stable_worldmodel/modal_outputs/tworoom_lewm/`, including `tworoom_results.txt`, `decoder/reconstruction_preview.png`, `decoder/metrics.json`, and `visualizations/tworoom__lewm/index.html`.
- 2026-05-26: Investigated the TwoRoom metric discrepancy in detail. Upstream issue #38 reports the same 10% SR when evaluating the HF checkpoint with `goal_offset_steps=100`, `eval_budget=150`; upstream issue #72 reports lower-than-paper TwoRoom/DMC results while PushT/OGB are similar. Confirmed our HF-to-object checkpoint conversion is exact (`303/303` keys, max absolute diff `0.0`) and that Modal is using the pinned local runtime. Batch size is only a secondary RNG/runtime factor: on the same 10 sampled `100/150` tasks, `eval.env_batch_size=10` gave 20% while `eval.env_batch_size=1` gave 30%. The decisive factor is the goal distance/budget: the same checkpoint scored 100% on a 10-task `25/50` probe and 88% (44/50) on the full 50-task `25/50` Modal run, matching the reported TwoRoom scale. Added `config/eval/tworoom_long.yaml` for the paper-text `100/150` protocol and restored `config/eval/tworoom.yaml` to the upstream/default `25/50` protocol.
- 2026-05-26: Completed ticket `1779819778148-use-author-given-decoder-arch-as-our-decoder`. Replaced the default auxiliary decoder with the author-provided CLS cross-attention decoder from the linked gist and Appendix D: one learned query per 16x16 output patch, CLS embedding as cross-attention key/value, residual MLP blocks, and linear RGB patch prediction. Decoder training now defaults to frozen-world-model, visualization-only RGB MSE. Verified with `PYTHONPATH=. .venv/bin/pytest -q` passing 14 tests, plus `py_compile` for decoder/visualization scripts.
- 2026-05-27: Updated `visualize_predictions.py` so generated rollout GIFs explicitly write GIF loop count `0`, which makes browser playback loop forever by default, and encode frame duration in milliseconds so `render.fps` is honored by the current Pillow-backed writer. Added `render.loop` to the visualization config and regression coverage for GIF loop/duration metadata.
- 2026-05-27: Moved the stable-worldmodel working directory into the repository at `.stable_worldmodel/` and added `project_paths.py` so local scripts default `STABLEWM_HOME` there unless explicitly overridden. Updated Modal to mount its persistent Volume at `/workspace/long-horizon-world-model/.stable_worldmodel`, ignored the cache in git and Modal image uploads, and documented the new storage convention.
- 2026-05-27: Removed the legacy auxiliary decoder implementations, config knobs, checkpoint-loading compatibility path, and tests. From this point on `decoder.py`, decoder training, visualization loading, and docs support only the author-given CLS cross-attention decoder aliases (`cls`, `cls_transformer`, `author_cls`). Verified with `PYTHONPATH=. .venv/bin/pytest -q` passing 16 tests and `py_compile` for the decoder/visualization scripts.
- 2026-05-27: Started ticket `1779885203491-reproduce-reported-results-across-all-tasks-using-given-checkpoints`. Initial implementation removes nearest-neighbor retrieval from prediction visualization, makes rollout grids/GIFs decoder-gated, adds `prepare_reported_assets.py` for Hugging Face dataset/checkpoint preparation inside the Modal Volume, and adds a Modal `reproduce_reported_results` entrypoint that sequentially evaluates the released LeWM checkpoint for TwoRoom, PushT, Reacher, and Cube, trains a decoder for each, and renders decoded predictions when the decoder exists.
- 2026-05-27: Fixed the ticket location/format by removing the mistaken `progress/tickets/1779885203491-...` file and updating the canonical `.yakanban/tickets/1779885203491-...md` ticket. Completed Modal asset preparation for PushT, Reacher, and Cube using released Hugging Face datasets/checkpoints, with prepared assets committed to the `multi-future-lewm-cache` Volume. Patched future asset-prep runs to silence curl progress output.
- 2026-05-27: Completed ticket `1779885203491-reproduce-reported-results-across-all-tasks-using-given-checkpoints`. Modal run `ap-8koN7GvBKFxtehVXch5yS8` reproduced TwoRoom and PushT, and run `ap-UYZYLZHyJ8NpzMaYiu3WO1` reproduced Reacher and Cube after pinning Modal MuJoCo to `3.8.1`. Scores: TwoRoom 92.0% SR (46/50), PushT 94.0% SR (47/50), Reacher 70.0% SR (35/50), Cube 70.0% SR (35/50). Trained 50k-sample, 3-epoch frozen-latent decoders for all four checkpoints and rendered decoder-backed prediction visualizations under `/visualizations/{tworoom,pusht,reacher,cube}/*__lewm` in the Modal Volume. Verified with `PYTHONPATH=. .venv/bin/pytest -q` passing 18 tests, `py_compile` for the main scripts, and `git diff --check`.
- 2026-05-28: Started ticket `1779953346468-train-multi-horizon-lewm-on-all-tasks`. Scope is deterministic `LeWM-Direct-H, H=5, history_size=3` training on PushT first, then Reacher, following `paper/MultiFutureLeWorldModel.md`; initial output should support horizon-wise latent error and planning comparisons against reproduced LeWM baselines.
- 2026-05-28: Implemented a Modal `train_multi_horizon_tasks` workflow that prepares datasets and trains deterministic `LeWM-Direct-H` across PushT/Reacher from a single detached remote function. Added `config/train/data/reacher.yaml`, corrected the DMC/Reacher training dataset to `dmc/reacher_random`, fixed README eval examples to load the trained policy directory, and added config sanity coverage. Verified with `PYTHONPATH=. .venv/bin/pytest -q` passing 19 tests, `py_compile`, Hydra composition for PushT/Reacher, and `git diff --check` on touched files. Launched full training: PushT app `ap-o4HIVkyqVvX8dc1yhHs8dP`, Reacher app `ap-ov5yXFXGJTXt7HcMVr4ZmS`.
- 2026-05-28: Continued monitoring `1779953346468` after launch. At `2026-05-28T08:35:32Z`, both Modal apps were still active after roughly one hour; filtered logs showed initial validation and first-backward success for both tasks, with no traceback, completion signal, or committed model artifacts yet.
- 2026-05-28: Investigated the Modal dashboard function-name mismatch for the two live `LeWM-Direct-H` runs. `run_train` and `run_train_multi_horizon_tasks` are different Modal wrappers, but both route into `python train.py --config-name=lewm_mh` when used for these runs. The meaningful run differences confirmed from logs are the Hydra data config and output subdir: PushT uses `data=pusht subdir=pusht/lewm_direct_h5`; Reacher uses `data=reacher subdir=reacher/lewm_direct_h5`.
- 2026-05-28: Root-caused the duplicate-training risk: the old `train_multi_horizon_tasks` wrapper defaulted to `tasks=pusht,reacher`, then a separate Reacher job was also launched. Removed the multi-task training wrapper and `train_with_decoder` launcher from `modal_app.py`; future world-model training should use only `modal_app.py::train` with explicit `--data` and `--subdir` for one dataset per app. Updated README examples and verified `py_compile modal_app.py` plus `tests/test_config_sanity.py`.
- 2026-05-28: Checked current Modal state after simplifying the entrypoints. Both live training apps had stopped. `ap-o4HIVkyqVvX8dc1yhHs8dP` ended with a Modal cancellation signal, and `ap-ov5yXFXGJTXt7HcMVr4ZmS` ended after a CLI stop/KeyboardInterrupt at epoch 0 with no checkpoint saved according to the CPUOffloadCallback log. The ticket remains in progress and needs explicit one-dataset relaunches through `modal_app.py::train`.
- 2026-05-28: Patched `modal_app.py::train` so PushT/Reacher datasets are copied from the Modal Volume to `/tmp/stable_worldmodel` before training and trained runs are copied back to the Volume after completion. Verified `py_compile`, import checks, scoped `git diff --check`, and `PYTHONPATH=. .venv/bin/pytest -q` passing 19 tests. Relaunched PushT as Modal app `ap-1PxOnUyF9eRogziCQK3ubx` and Reacher as `ap-3hH4wylEIBGTiYGn3dUJuV`; both logs show dataset staging to local disk.
- 2026-05-28: Improved the long-run Modal training path after observing that full epochs take roughly 11.3k steps and the initial staged runs were not resumable before process exit. `modal_app.py` now links staged-cache run directories back to the Modal Volume and commits every 300 seconds while `train.py` runs. Stopped the non-resumable PushT/Reacher apps before any checkpoints were saved, rejected an aggressive `num_workers=12`, `prefetch_factor=4` attempt because it exhausted Modal shared memory, and relaunched stable runs with `num_workers=8`, `prefetch_factor=2`: PushT `ap-r9bqChIx1CiJzOYcWC3EQl`, Reacher `ap-Ghya6XeWkjGKzdSnbQqkAg`.
- 2026-05-28: Took the new training-infrastructure work as YAKanban tickets. Expanded `1779964126980-improve-logging-make-it-more-verbose-and-add-wandb-logging` to cover WandB via `$WANDB_API_KEY`, local fallback logs, timing/throughput metrics, checkpoint events, and run identity. Created and took `1779973200000-make-training-reliably-resumable` for real checkpoint resume semantics across local and Modal jobs. Took `1779967558491-make-training-way-faster` with explicit constraints that speedups must be profiling-led and quality-preserving: no architecture cuts, dataset reduction, loss removal, or weaker evaluation.
- 2026-05-28: Implemented the training-infrastructure work end to end. `train.py` now uses W&B automatically from `$WANDB_API_KEY`, writes local `metrics.jsonl`/`events.jsonl`/`run_metadata.json`, records wall-clock timing/throughput/ETA/LR/grad-norm metrics, saves atomic latest/best checkpoints with `checkpoint_state.json`, resumes full Lightning state with PyTorch 2.6 `weights_only=False`, preserves RNG state, and refuses unsafe restarts in non-empty run directories. Modal forwards W&B credentials without printing them, links staged local output dirs back to the Volume, and commits run files during training.
- 2026-05-28: Implemented the first quality-preserving speed pass: corrected the stale `trainer.max_epochs=100` default to the paper-aligned 10 epochs, enabled bf16-mixed/H100 runs, moved raw HDF5 `uint8` image normalization/resizing onto the GPU, switched multiprocessing sharing to `file_system`, raised the loader to 8 workers with `prefetch_factor=4`, and enabled validation-based early stopping after a five-epoch minimum. Verified with `PYTHONPATH=. .venv/bin/pytest -q` passing 23 tests and `py_compile` for the touched training files.
- 2026-05-28: Relaunched corrected `LeWM-Direct-H, H=5` training on H100. Active apps: PushT `ap-NZxOY724QmJooHJLdrMhVb`, Reacher `ap-Ng2vMTXVhDdYB5nRSdYXwl`. W&B runs: `https://wandb.ai/jack-b/multi-future-lewm/runs/pusht_lewm_direct_h5_fast` and `https://wandb.ai/jack-b/multi-future-lewm/runs/reacher_lewm_direct_h5_fast`. PushT wrote `lewm_direct_h_weights.ckpt` at global step 1000; Reacher wrote `lewm_direct_h_weights.ckpt` at global step 500. Initial timing after GPU preprocessing is roughly 4-7 wall steps/sec with ETA still several hours, so speed ticket `1779967558491` remains open for measured batch-size or multi-GPU scaling rather than reducing model/data/loss quality.
- 2026-05-28: Stopped the long H100 training apps after the user clarified that speed work should churn through minute-scale experiments first. Checked upstream issue #52: it reports that the paper trains for 10 epochs while upstream `lewm.yaml` had `max_epochs=100`; our 10-epoch config is paper-aligned, but the stable-pretraining `LinearWarmupCosineAnnealingLR` uses the trainer's estimated stepping batches for warmup/cosine length, so changing epochs, max steps, or batch size changes the LR schedule and must be treated as an optimization change.
- 2026-05-28: Added `config/train/lewm_speed.yaml` as a bounded direct-horizon speed probe with W&B auto-enabled, local JSONL timing, no checkpoint/object overhead, `max_steps=300`, and restart protection disabled for unique probe subdirs. Added CUDA memory fields to the metrics stream and removed automatic Modal training retries so deterministic OOM configs do not repeat. Probe results: L40S batch 128 finished 300 PushT steps in about 2m09s at 2.40 steps/s and 307 samples/s; a 30-step memory smoke logged `gpu/max_memory_allocated_gib=24.34` of 44.39 GiB; L40S batch 256 OOMed at about 43.31 GiB allocated and the app was stopped. Earlier H100 batch 256 finished 300 steps in about 3m01s at 2.17 steps/s and 555 samples/s, with data wait dominating compute.
- 2026-05-29: Started from-scratch original LeWM training on TwoRoom via Modal app `ap-3GGJK2w3u9Yv1ubH98R8xd`, W&B run `tworoom_lewm_from_scratch`, output `tworoom/lewm_from_scratch`. Added TwoRoom to Modal local dataset staging and added sanity coverage for the TwoRoom train data config. The run uses `config-name=lewm data=tworoom`, L40S, bf16, 10 epochs, `early_stopping.enabled=false`, and staged local `/tmp/stable_worldmodel/tworoom.h5`. First metrics: step 1250, about 5.85 steps/s and 749 samples/s, checkpoint saved at step 1000.
- 2026-05-29: The first detached-with-wait training app was canceled when the local Modal client was interrupted; it saved a resumable checkpoint first. Relaunched with `modal run --detach ... --no-wait`, resumed cleanly from `lewm_weights.ckpt`, and completed the full TwoRoom from-scratch LeWM training as Modal app `ap-T1PMgjnC4Clk5hzKLr01Id`. Final metrics: 10 epochs, 51,380 steps, final/best `validate/loss=0.157995`, final/best `validate/pred_loss=0.007131`, final object `tworoom/lewm_from_scratch/lewm_epoch_10_object.ckpt`, latest/best weights saved.
- 2026-05-29: Evaluated `tworoom/lewm_from_scratch` with `config/eval/tworoom.yaml`, `eval.env_batch_size=10`, `output.save_video=false`, and `output.filename=lewm_from_scratch_tworoom_results.txt`. Modal eval app `ap-dTT7olN3v08Gq6Nvm3b3qx` on L4 produced 86% SR (43/50) in 677.6s. This matches the reported-result scale and is close to the released checkpoint reproduction result of 88% (44/50) under the same 25/50 protocol.
- 2026-06-05: Began paper framing for the masked transition modeling branch. Current recommended story: present masked transition modeling as a SIGReg-free world-model representation objective, emphasizing the mechanism (forward + inverse transition prediction prevents collapse), the confirmed gains on TwoRoom-Long and OGB-Cube, the Reacher tie, the PushT loss, and the `max_epochs`/cosine-LR reproduction caveat rather than claiming a universal replacement for SIGReg.
- 2026-06-05: Wrote a standalone LaTeX manuscript draft at `paper/MaskedTransitionModel.tex`, added `paper/masked_transition_refs.bib`, and compiled `paper/MaskedTransitionModel.pdf` with `latexmk -pdf`. The draft is 15 pages and includes embedded TikZ/PGFPlots figures, result tables, mechanism tables, limitations, and appendix tables. Remaining bibliography entries are explicitly marked as placeholders for later verification.
- 2026-06-05: Clarified the fairness framing: original LeWM already encodes actions and conditions the forward latent predictor on action embeddings, but it has no action-prediction/reconstruction loss. Masked transition therefore is not adding action access; it adds an extra supervised inverse-dynamics objective and removes SIGReg. The clean comparison should be described as "SIGReg marginal regularization vs inverse-dynamics anti-collapse, with identical action-conditioned forward planner," while noting that masked has an added training head/loss.
- 2026-06-05: Reacher `trainer.max_epochs=100` ep10 probe split the arms: baseline improved from 69.0% to 76.0%, confirming the LR-schedule reproduction diagnosis, while masked dropped from 68.0% to 39.5%. Current leading hypotheses: the SIGReg-free masked objective is sensitive to the sustained high LR because `inverse_weight=1.0` makes inverse dynamics a strong early encoder-shaping gradient; the inverse task may learn action-distinguishing but planner-poor Euclidean geometry without SIGReg's latent scale/shape constraint; and the old 10-epoch schedule may have accidentally acted like an annealed inverse regularizer. Next checks are ep20/30, `emb_std`/effective-rank, forward-vs-inverse loss balance, latent norm statistics, and ablations with inverse warmup or lower `inverse_weight`.
- 2026-06-05: Added `loss.masked.inverse_warmup_epochs` for masked transition training and logs `train_step/inverse_weight_effective`; default remains zero warmup and old behavior is unchanged. Validated with `pytest -q tests/test_masked_transition.py` (11 passed), `py_compile`, and Hydra config composition. Old-vs-current Reacher metrics from Modal show masked e100 keeps LR near 5e-5 and reaches lower `emb_std` by epoch 10 (about 0.075 vs 0.088 old) while inverse loss stays flat around 0.83, suggesting a latent-geometry/collapse-adjacent failure rather than inverse-loss convergence. Launched three Reacher e100 ablations: `reacher/lewm_masked_e100_invw03` (app `ap-PP2TJyWvHW5uzKmlHdkqWC`), `reacher/lewm_masked_e100_warm5` (app `ap-2TxPIgSHX5FDJqaOjoY9dB`), and `reacher/lewm_masked_e100_invw03_warm5` (app `ap-emKWMX0DTyr99XJPtM0wBz`). Also launched later-checkpoint current-run evals: masked e100 epoch 17 (app `ap-oVf28SmEs2tXdBjJyqvR3p`, output `lewm_masked_e100_ep17_reacher_n200.txt`) and base e100 epoch 15 (app `ap-7uwDVMsFaGLLk5evJBFkIp`, output `lewm_base_e100_ep15_reacher_n200.txt`).
- 2026-06-05: Stopped all active Modal apps to avoid unnecessary 100-epoch compute. Stopped original e100 apps `ap-8C3MaNgaNglBocVS24WU26` and `ap-RuaV1zvoPD7RkOn1XfLw5r`, plus ablation apps `ap-PP2TJyWvHW5uzKmlHdkqWC`, `ap-2TxPIgSHX5FDJqaOjoY9dB`, and `ap-emKWMX0DTyr99XJPtM0wBz`; verified all show `stopped` with zero tasks. Results landed before cleanup: base e100 epoch 15 evaluated at 82.0%, masked e100 epoch 17 recovered to 69.5%. Ablations reached epoch 6 and have resumable checkpoints/metrics, but no n=200 evals. Early ablation signal: `inverse_weight=0.3` remains non-collapsed enough to keep inverse loss near 0.83, while warmup-only and warmup+0.3 drive `emb_std` near zero and inverse loss stays near 1.0, so pure forward warmup appears unsafe. Fixed the local JSONL callback to retain `*_weight` / `*_weight_effective` metrics for future resumed runs.
- 2026-06-06: Added `runtime.stop_after_epoch` and a Modal `train_then_evaluate` entrypoint so resumed runs can keep `trainer.max_epochs=100` for the LR schedule, stop gracefully at a milestone, and automatically run n=200 Reacher eval on the target epoch checkpoint. Validated with `py_compile`, `pytest -q tests/test_masked_transition.py` (11 passed), Hydra compose with `runtime.stop_after_epoch=10`, and `modal run modal_app.py::train_then_evaluate --help`. Resumed ablations to stop/eval at epoch 10: `invw03` app `ap-6BvSItA3Jr6C4jwnPY2aYQ`, `warm5` app `ap-hVBT4CeDSYZQFkWqeuzaEm`, `invw03_warm5` app `ap-2kTU82KAkqYnFozLV2rJmH`. Resumed original e100 runs to stop/eval at epoch 30: masked app `ap-10eauXpGRa1L4nCTJRnQNT`, base app `ap-QzdGKaBe2Yix0tAPVTtI22`. Logs confirm checkpoint restore and training restart for at least `invw03`, `warm5`, and base e100.
- 2026-06-06: Added practical explanatory comments throughout `config/train/lewm.yaml`, especially around the `trainer.max_epochs` / cosine-LR coupling, `runtime.stop_after_epoch`, checkpoint/resume behavior, data loader knobs, W&B identity, world-model shape settings, predictor capacity, and SIGReg parameters. Verified Hydra composition for `lewm` and `lewm_masked` with Reacher still succeeds.
- 2026-06-06: Checked and stopped all active Modal apps; verified no active tasks remain. Milestone Reacher evals completed before cleanup: base e100 epoch 30 scored 81.5% (matches released 81.5%), masked e100 epoch 30 scored 75.0% (improved from 39.5% ep10-ish and 69.5% ep17, but still below baseline), `inverse_weight=0.3` epoch 10 scored 60.0%, warmup-only epoch 10 scored 10.0%, and `inverse_weight=0.3 + warmup5` epoch 10 scored 9.5%. Decision: pure forward warmup is rejected for Reacher; lower inverse weight alone is not enough at epoch 10; masked e100 keeps improving with training but remains behind SIGReg under the corrected schedule.
- 2026-06-07: Continued the two original Reacher e100 runs from epoch 30 to a bounded epoch-50 milestone, again preserving `trainer.max_epochs=100` and using `runtime.stop_after_epoch=50` plus automatic n=200 eval. Launched masked app `ap-AJgZPYxZkWbJegmQMs49vL` with expected output `reacher/lewm_masked_e100/lewm_masked_e100_ep50_reacher_n200.txt`, and base app `ap-lCuLsmXyuvCoY3leXNeSN0` with expected output `reacher/lewm_base_e100/lewm_base_e100_ep50_reacher_n200.txt`. Modal app list shows both active; masked and base logs both confirm checkpoint restore from epoch 30 and active epoch-31 training.
- 2026-06-08: Checked PushT masked under the corrected 100-epoch LR schedule after the user asked whether a better schedule should close the base gap. An active app `ap-QdSga0HL4l3vAyknyuZ5GW` is already training `pusht/lewm_masked_e100` around epoch 20/100. Completed milestone evals are 78.0% at epoch 10 and 81.0% at epoch 15, versus 85.0% for old masked and 93.5% for matched LeWM. Metrics show no obvious collapse (`emb_std` around 0.15), so the PushT miss looks more like objective mismatch/contact-dependent inverse dynamics than simply the old LR schedule. Recommendation: only evaluate bounded milestones and avoid launching another plain e100 duplicate.
- 2026-06-09: Inspected AC-CPC and confirmed it does not use masked transition's inverse-action prediction head. AC-CPC reuses LeWM's action-conditioned forward predictor, disables SIGReg, unit-normalizes latents, and trains an InfoNCE objective where predicted futures must identify their own true future among other trajectories' futures. Added `paper/ActionConditionedCPC.md` as the explainer for this approach.
- 2026-06-09: Discussed adding inverse dynamics to AC-CPC. Feasible design: add a separate `ac_cpc_inverse` mode/config that keeps InfoNCE forward training, wires `InverseDynamics`, adds `lambda_inv * inverse_loss`, and logs inverse metrics. Do not mutate plain `ac_cpc`; keeping contrastive-only and contrastive+inverse separate is required for an interpretable ablation.
- 2026-06-09: Clarified the "3 transitions per window" detail in AC-CPC training. With `history_size=3` and `num_preds=1`, each 4-frame sampled window yields three next-latent predictions, flattened into a `B*3` by `B*3` InfoNCE matrix. Same-window off-diagonal futures are masked, so each prediction competes against its own positive plus all futures from other sampled windows.
- 2026-06-09: Diagnosed the failed clean AC-CPC PushT run. It did not collapse (`cpc_loss` fell from about 6.10 to 0.05; `emb_std` stayed around 0.07) but scored only 62.5% vs LeWM 93.5% and masked 85.0%; paired eval shows AC-CPC solved zero episodes that LeWM failed. Leading cause: InfoNCE learned trajectory-identification structure without producing the balanced, physically decodable Euclidean latent geometry CEM needs. The full 3-way linear-probe table (base/masked/AC-CPC) is in `progress/results-summary.md`.
- 2026-06-11: Discussed a new hypothesis for beating the baseline: masked transition may learn a good controllable-agent world model but not a task-relevant world model for arbitrary test-time objectives, because inverse dynamics rewards state variables that explain action effects and can underweight weakly/indirectly controlled goal variables such as PushT block pose. Planning-time task adaptation is considered viable if scoped as fast task conditioning/adaptation rather than unrestricted model finetuning: first adapt the latent goal metric or a small adapter using `(current, goal, candidate rollouts, observed replans)` before attempting full world-model updates. Key risk is optimizer exploitation and latent drift during CEM; key test is whether adaptation improves PushT block pose/orientation probes and planning success without hurting already-strong tasks.
- 2026-06-11: Reviewed Nguyen/Xu/Huang 2026, "Latent Geometry Beyond Search: Amortizing Planning in World Models" and its GC-IDM code. The paper trains a small frozen-latent goal-conditioned inverse dynamics policy `(z_t, z_goal, remaining_horizon) -> a_t` on LeWM embeddings and uses closed-loop re-encoding every step to replace CEM. This is not already implemented here and is only superficially similar to our masked inverse head: masked transition is adjacent-pair `(z_t, z_{t+1}) -> a_t`, training-time anti-collapse, no goal/horizon, and the planner never calls it. Recommended next use: first add GC-IDM as a frozen-planner baseline/proposal mechanism for base/masked checkpoints; then separately test an end-to-end goal-conditioned inverse auxiliary, preserving world-model prediction/SIGReg or masked losses so the representation does not collapse into pure behavior cloning.
- 2026-06-11: Clarified prior "longer horizon action prediction" work. We have trained longer-horizon **latent** predictors (`LeWM-Direct-H`, `masked_horizon`, `BYOL-WM`) and adjacent inverse-dynamics heads that predict one normalized action block per latent transition. We have not yet trained a GC-IDM-style distant-goal action predictor `(z_t, z_goal, remaining_horizon) -> first action`; that remains a distinct planner/policy experiment.
- 2026-06-12: Simplified collaborator markup in `paper/latex/acl_latex.tex` to note-only commands: `\jack`, `\chris`, `\demian`, and `\lucas`, backed by a generic `\personnote` helper. Colors now use xcolor strings via `\colorlet` (for example `blue!65!black`) instead of RGB triples. Pushed the source edit to Overleaf, restored the missing root `README` support file, and verified with a clean pull that the remote has 8 files and the simplified macros. A first-pass `pdflatex` build succeeds; the full `latexmk` build still needs bibliography entries before it can complete.
- 2026-06-12: Replaced the project `CLAUDE.md` pointer file with a symlink to `AGENTS.md`, and replaced global `~/.claude/CLAUDE.md` with a symlink to `~/.codex/AGENTS.md`.
- 2026-06-16: Reviewed the ICLR draft title in `paper/latex/iclr_main.tex`. Current title foregrounds both the inverse-dynamics-vs-distribution-matching result and the linear-probe critique, but it is long. Candidate snappier framing should likely keep one memorable hook while the abstract/contributions carry the second claim.
- 2026-06-16: Brainstormed more tongue-in-cheek ICLR title candidates. Best direction should be playful without sounding unserious: use the Gaussian/SIGReg contrast, collapse avoidance, or probe critique as the joke, while retaining "JEPA world models" or "latent world models" for searchability.
- 2026-06-18: Investigated the alarming Reacher seed results. The low runs were `reacher/lewm_masked_s1` and `reacher/lewm_masked_s2`, scoring 11.5% and 13.5% at n=200, while paired SIGReg seeds scored 69.0% and 68.5%. Training metrics show real latent collapse in the masked seeds (`emb_std` `0.000119` / `0.000500`, forward loss near zero, inverse loss near one), not an eval harness bug. Also found a results-recording bug: concurrent default Reacher evals raced on shared `reacher/dmc_results.txt`; use unique filenames for future evals.
- 2026-06-18: Checked the matched e100 epoch-30 sweep. Existing-checkpoint evals completed at n=200: PushT MTM 85.0%, TwoRoom SIGReg 84.0%, TwoRoom MTM 87.5%, Reacher SIGReg 83.5%, Reacher MTM 77.0%. PushT SIGReg was preempted after saving at step 28,000; a detached resume `ap-BQI4AkN0lDB1adYe5eveDz` reached about step 28,725 but then received SIGTERM, so PushT SIGReg remains incomplete. Cube SIGReg and Cube MTM remain active at epochs 16 and 22.
- 2026-06-18: Relaunched PushT SIGReg e100 from the saved checkpoint using persistent tmux session `e100_pusht_sig`, because simple background/nohup relaunches were exiting silently. New app `ap-fmdghWRPkOM94m11MSsHjd` is active with one task and logs show live training at epoch 3 / step 28,175.
- 2026-06-18: Diagnosed the ICLR PDF font change after a title edit in `paper/latex/iclr_main.tex`. The source diff only changed `\title{...}`; the generated log changed from `pdflatex` to `xelatex`. With XeLaTeX, `\usepackage{times}` requests `TU/ptm` shapes that are unavailable and LaTeX substitutes Latin Modern, causing the document-wide font change. Build the ICLR source with `pdflatex`/`latexmk -pdf`, or use XeLaTeX-native font setup deliberately.
- 2026-06-18: Fixed the local VS Code/LaTeX Workshop recipe failure by making `.vscode/settings.json` self-contained: default recipe `latexmk`, explicit `latexmk` recipe, and explicit `latexmk`/`pdflatex`/`bibtex` tools. Verified JSON parsing and `latexmk -pdf` rebuild of `paper/latex/iclr_main.tex`; the log now starts with `pdfTeX` and no longer has the XeLaTeX `TU/ptm` fallback warnings.
- 2026-06-18: Replaced the first ICLR concept TikZ figure in `paper/latex/iclr_main.tex` with `paper/latex/images/hero_figure.png`, preserving the existing `fig:concept` caption and label.
- 2026-06-18: Moved the `fig:concept` hero figure source block to immediately after the abstract in `paper/latex/iclr_main.tex`.
- 2026-06-18: Made the post-abstract `fig:concept` hero figure non-floating with a local `\captionhere` helper so Introduction text cannot appear before it when the full-width image does not fit at the bottom of page 1.
- 2026-06-18: Rescaled the post-abstract `fig:concept` image to `0.90\linewidth`, which keeps the abstract and hero figure together on page 1 while the Introduction starts on page 2.
- 2026-06-18: Assessed the three live Modal training jobs. They are the matched e100 epoch-30 gap-fill runs: Cube MTM/masked (`ap-J1SlwTqAYYCmkHFEICgIQD`), Cube SIGReg/base (`ap-4DOTZ7f3m4riFkSz4Q3OX3`), and PushT SIGReg/base (`ap-fmdghWRPkOM94m11MSsHjd`). None has the needed epoch-30 checkpoint yet; killing is technically safe because latest weights are committed, but would leave the matched table incomplete.
- 2026-06-18: Created `progress/evaluation-protocol-ledger.md` for the LeWM-style n=50 comparison. Protocol decisions: TwoRoom main uses the maintainer-confirmed/released `25/50` protocol from lucas-maes/le-wm#38, TwoRoom-long keeps the paper-text `100/150` stress protocol, and Reacher/Cube main comparison uses released CEM `300/30/30` while noting the paper/config mismatch from issue #41. Launched a retry `n=50` eval matrix with exact checkpoint stems and unique filenames after the first backgrounded attempt was cancelled by local Modal client disconnects.
- 2026-06-18: Completed the `n50r1_*` eval matrix and copied raw results to `progress/evaluations/lewm-style-n50-r1/`. Main LeWM-style protocol: TwoRoom 88% SIGReg vs 96% MTM, PushT 96% vs 90%, Reacher 74% vs 74%, OGBench-Cube 76% vs 86%. TwoRoom-long 100/150 stress at n=50: 12% SIGReg vs 28% MTM. Updated the ICLR draft with a LeWM-style results figure, task-still figure, protocol audit appendix, long-horizon appendix table, expanded related work, and a cautious physics-understanding limitation.
- 2026-06-18: Removed the generated entrypoint-level `--no-wait` surface from `modal_app.py::{train,train_then_evaluate,evaluate,probe,probe_features}`. Current launch rule is blocking entrypoints plus global `.venv/bin/modal run --detach`; README and AGENTS now reflect that distinction.
- 2026-06-18: Standardized the ICLR paper on `n=50` planning results throughout for protocol clarity. Regenerated Figure 3 as a 2x2 LeWM-style comparison with larger labels, moved TwoRoom-long into the main result table, and updated the diagnostic PushT/NoReg rows from completed `n50r2` evaluations.
- 2026-06-18: Built `progress/experiment-tables.md`, a source-backed set of tables covering paper-facing `n=50` rows, diagnostic rows, historical `n=200` runs, Direct-H experiments, e100 milestone sweeps, early TwoRoom reproductions, and pending/partial runs. The file is intended as the provenance reference for tightening `paper/latex/iclr_main.tex`.
- 2026-06-18: Clarified final-result seed/eval strategy discussion. `n=50` means 50 sampled start/goal evaluation episodes; eval seeds choose episode sets, while training seeds create different checkpoints. Current recommendation is to prioritize multiple training seeds and paired larger episode sets (e.g. n=150-200) over treating `3 eval seeds x 50` as independent robustness evidence. Also checked live Modal state: PushT SIGReg e100 stopped at epoch 8, Cube SIGReg active at epoch 21, Cube MTM active at epoch 29.
- 2026-06-18: Decided not to switch `paper/latex/iclr_main.tex` to n=200 yet because the clean e100/n=200 SIGReg-vs-MTM table is incomplete. Target final paper direction is n=200 for our controlled table once all 8 e100 cells land, with n=50 retained for LeWM-style external comparability. Initial direct detached relaunches for Cube MTM (`ap-2fwxdarYZ3E7nVmzK7R2H8`) and PushT SIGReg (`ap-V4d52gU06XW5C3DhzNFrLN`) were canceled after local-client interruption. Relaunched both in persistent tmux sessions and verified real progress: Cube MTM epoch-30 n=200 eval `ap-nwk5uMGsuLDtXNdz01grOv` is at eval batch 2/20 with output `e100_mtm_cube_ep30_n200_r2.txt`; PushT SIGReg resume-to-epoch-30 plus n=200 eval `ap-7URW93LGChFJWh7MTmYYbB` is training at epoch 9 / step 124,725 with output `e100_sigreg_pusht_ep30_n200_r2.txt`. Cube SIGReg remains active in `ap-4DOTZ7f3m4riFkSz4Q3OX3` and still needs its epoch-30 eval after the checkpoint lands.
- 2026-06-18: Cube MTM e100 epoch-30 n=200 eval completed at 77.0% (`e100_mtm_cube_ep30_n200_r2.txt`). Launched the e100 epoch-30 robustness matrix for extra training seeds 1 and 2. Reacher apps are running: base s1 `ap-YjUwVFV7AFrwRwI55YXjsk`, base s2 `ap-2tsAmuzc4QiemOuZllWNgZ`, MTM s1 `ap-Myke76CHpd10OmIkIg4qE2`, MTM s2 `ap-sSi7JbJub9XST1tkBedavr`. TwoRoom apps are running: base s1 `ap-jh0UGytKoAyrDJGf3YlZnu`, base s2 `ap-nJx06nDyjkM1aLIcR4Wuaq`, MTM s1 `ap-JaSUDq483sn5lMx2pJ9EJo`, MTM s2 `ap-L8Uc5CwEF0kQwefc9IWIBb`. PushT/Cube extra-seed apps are created but queued at `Tasks=0`: PushT base s1/s2 `ap-CaTDh0n84uwBfA8xpjuLb2` / `ap-LAeeH6Q8AFOm7mgJU4ZDZr`, PushT MTM s1/s2 `ap-JAJPdjaycTT7GFpbzwG7jB` / `ap-dmEXS6IyjH5VduF8hSmItJ`, Cube base s1/s2 `ap-6rPc7wQSH4rVOIcmhD6u2k` / `ap-BWb9B1eZl1jNmbmhPCrv2X`, Cube MTM s1/s2 `ap-e9alMNWPtxiXTOmRiwU1zk` / `ap-8AUDCpDMPnJbMurxNpnj10`.
- 2026-06-19: Morning status: Cube SIGReg seed-3072 reached epoch 30 and wrote `cube/lewm_base_e100/lewm_epoch_30_object.ckpt`; launched its n=200 eval as `ap-mgs5w2gOhPZ4iegixD3doZ`, currently queued at `Tasks=0`. PushT SIGReg seed-3072 is still training around epoch 26. Reacher and TwoRoom seed 1/2 runs remain active with training logs; Cube SIGReg s2 is active around epoch 3; the other PushT/Cube seed 1/2 apps are still queued.
- 2026-06-19: Afternoon status: seed-3072 e100/n=200 table is complete. New results since morning: PushT SIGReg 84.5%, Cube SIGReg 66.5%. TwoRoom seed 1/2 standard n=200 evals also completed: SIGReg s1/s2 87.0/85.0, MTM s1/s2 91.0/45.0. The MTM seed-2 collapse/underperformance means TwoRoom e100 is not cleanly robust despite the seed-3072 win. Reacher seed 1/2 jobs are still running around epoch 20-21. PushT/Cube extra seeds are partially running, with PushT base s1 and MTM s2 still queued.
- 2026-06-20: After the Modal budget/workspace was restored, stopped stale broad-matrix apps with `Tasks=0` and relaunched only targeted Reacher gap-fill work from persistent tmux sessions. The r2 Reacher epoch-30 n=200 evals are complete and archived under `progress/evaluations/e100-reacher-r2/`: SIGReg seed 1/2 scored 79.5/80.5, while MTM seed 1/2 scored 10.5/12.0. Modal app list is empty and no tmux sessions remain active.
- 2026-06-20: Started a Reacher MTM collapse-rescue screen after confirming the failure is real encoder collapse (`emb_std` near zero for bad seeds). Patched `train.py` to allow optional MTM+weak-SIGReg hybrid ablations without changing pure MTM defaults; `tests/test_masked_transition.py` passes. Ten one-epoch screen jobs are live for variance floors, stronger inverse weight, weak hybrid SIGReg, and lower LR; promote any noncollapsed screen to full e10+n=200 seeds 1/2.
- 2026-06-20: Reacher rescue update: stopped all one-epoch screens after live metrics showed the variance/lower-LR variants were not learning inverse loss, while inverse-weight and weak-hybrid variants were. Promoted the cleanest pure-MTM signal, `loss.masked.inverse_weight=3.0`, to full e10+n=200 train-then-eval runs on seeds `3072,1,2`: apps `ap-RCksn1PAtRcqGexnedR2ha`, `ap-VqE5DexBFKRHmSftzizfJg`, and `ap-vCBxz6cvXn6WzGXtC99z7T`. Also launched seed-1 sentinels for `inverse_weight=10.0` (`ap-D0eLDTracUpnlcjktkcGAo`) and weak MTM+SIGReg `loss.sigreg.weight=0.01` (`ap-kwvnHENDeVLc0hg9qYM2PW`). All five full apps have `Tasks=1` and step-level training progress.
- 2026-06-20: Added pure-MTM lower-LR Reacher controls because LR may be the cleanest explanation/fix. Seed-1 full e10+n=200 train-then-eval apps are running with default MTM objective and `optimizer.lr=1e-5/2e-5/3e-5`: `ap-qPcUCjldA8MlxWeB9xxaAt`, `ap-lmBm268itga3ovNGlANt1x`, `ap-8eT4X70lKBonvnpzZFCJav`. Early metrics: `emb_std` 0.115-0.127 at step ~650-675, inverse loss still near 0.99; keep running to see whether lower LR rescues or only delays collapse.
- 2026-06-21: Reacher rescue first wave completed and raw outputs are archived in `progress/evaluations/reacher-mtm-rescue/`. Results: `inverse_weight=3.0` seeds `3072/1/2` scored 60.5/81.5/67.5; `inverse_weight=10.0` seed 1 scored 57.5; weak MTM+SIGReg 0.01 seed 1 scored 60.0; pure MTM lower LR seed 1 scored 21.5/54.0/73.0 for `1e-5/2e-5/3e-5`. The full `lr=3e-5` seed set scored 60.0/73.0/61.0 for seeds `3072/1/2`, so LR tuning helps but does not robustly rescue Reacher. Modal app list is empty and no tmux sessions remain active.
- 2026-06-21: Started a Reacher checkpoint-selection audit to test whether epoch 10 is missing better checkpoints. First wave evaluates epochs 3/5/7/9 at n=200 for `inv3` and `lr3` seeds `3072` and `2`. Ten evals are verified running with `Tasks=1`; six zero-task queued apps were stopped and will be relaunched after wave one finishes.
- 2026-06-21: Reacher checkpoint-selection audit update: first-wave completed values include `inv3/s3072` e3/e5 = 41.0/54.5, `inv3/s2` e3/e5/e7 = 55.0/63.5/53.0, `lr3/s3072` e5/e9 = 55.5/68.5, and `lr3/s2` e3/e9 = 1.5/65.5. The six deferred checkpoint evals were relaunched as `r-ckpt2-*` apps and verified live with `Tasks=1` plus `Evaluating batch` logs.
- 2026-06-21: Default Reacher MTM checkpoint audit launched after confirming epoch 1-10 checkpoints exist for `reacher/lewm_masked{,_s1,_s2}`. Metrics show collapsed seeds are already low-variance by epoch 1, so the narrow default audit is n=200 at seed 3072 epoch 9 and seeds 1/2 epochs 1, 5, 9. Seven `r-ckpt-def-*` apps are admitted with `Tasks=1`; awaiting rollout logs.
- 2026-06-22: Reacher checkpoint audit completed and archived under `progress/evaluations/reacher-checkpoint-audit/`. Default MTM is not rescued by earlier checkpoints. Best pure-MTM schedule audited is `optimizer.lr=3e-5` at epoch 9: seeds `3072/1/2` scored 68.5/74.0/65.5, mean 69.3, versus epoch-10 mean 64.7. Use this only as competitive/schedule-sensitive Reacher performance, not a strong MTM win.
- 2026-06-25: Audited the Reacher success condition in `play.py` after a visually close arm pose failed to trigger. There is no success-logic mismatch: both the installed environment and the demo require every joint angle to be within `0.05` rad, while the reported screenshot distance was `0.126` rad. The seed-0 self-test solved at `0.017` rad. Full-scale keyboard torque is easy to overshoot; reduce `mag` with `[` for fine control near the target.
- 2026-06-25: Fixed `play.py` repeatedly cycling tasks after `N`: Pygame's global key repeat queued additional `N` keydowns while slow environments such as Cube loaded. Repeated keydowns are now accepted only for action keys and Space; task switching and other one-shot controls fire once per physical press. Added `tests/test_play.py`; targeted suite passes 9/9.
- 2026-06-26: Added README documentation for the human-play demo with
  `assets/play_demo.gif` generated by `make_play_demo_gif.py`, setup notes,
  task launch commands, global controls, task-specific controls, and headless
  self-test commands. Also fixed Cube demo controls so `Q` can be used for z
  motion and `X` closes the gripper while Space remains the global no-op key.
- 2026-06-27: Promoted the three-seed OGBench Visual Scene trajectory-goal
  result into `paper/latex/iclr_main.tex` as a major selling point: abstract,
  introduction findings/contributions, task figure, a dedicated main-text Scene
  table, limitations, conclusion, and protocol appendix now report Action-NCE
  `80.0 +/- 2.0` vs SIGReg `58.0 +/- 2.0`, with 40 paired wins vs 7 losses
  across 150 matched episodes. Added `paper/latex/images/scene_still.png`.
- 2026-06-27: Relaunched the official fixed-goal OGBench Visual Scene benchmark
  using chunked tmux lanes after the monolithic jobs died around 14% from a
  Modal client DNS/network error before writing outputs. Added
  `eval.episode_start`, periodic eval-volume commits, and
  `scripts/run_scene_official_chunk_lane.sh`. Ten lanes are active:
  `{action_nce,sigreg} x task{1..5}`, each running five 10-episode chunks and
  verifying output files before advancing.
- 2026-06-28: Moved fixed-goal Scene chunk orchestration fully inside
  Modal with `run_scene_official_chunk_supervisor`, then confirmed it survived
  killing the local Modal CLI and completed all 50 chunk outputs. Aggregated
  fixed-goal singletask score is 0/250 for both Action-NCE and SIGReg, with
  every episode timing out at the 750-step cap. A direct local registry sanity
  check shows `visual-scene-v0` and `visual-scene-singletask-taskN-v0` expose
  matching task metadata, but their rendered goal images are not byte-identical;
  run one cheap base-env `task_id` smoke before calling the 0% result an exact
  public-protocol reproduction. The paper should continue to use the three-seed
  trajectory-goal Scene result as the main complex-task result.
- 2026-06-28: Started single-seed broader OGBench comparison for SIGReg
  (`lewm`) vs Action-NCE (`lewm_masked_action_nce`) on requested tasks
  `puzzle-4x4-play-v0`, `puzzle-4x5-play-v0`,
  `antmaze-teleport-navigate-v0`, `powderworld-medium-play-v0`,
  `antmaze-large-stitch-v0`, and `antsoccer-medium-stitch-v0`. Added generic
  OGB HDF5 conversion and online reset-goal eval plumbing. Visual aliases are
  used where OGB provides them; powderworld is converted from its first RGB
  channels and uses PGD for discrete planning; antsoccer renders pixels from
  qpos/qvel if the OGB NPZ exposes them. Launch target is one seed (`3072`) per
  method/task with train-then-online-eval. Supervisor app:
  `ap-t8vac1tgm8uhKScE0Olhv3`, verified detached with `Tasks=8` and live OGB
  dataset download progress.
- 2026-06-28: Status check for broader OGBench supervisor
  `ap-t8vac1tgm8uhKScE0Olhv3`: app remains `ephemeral (detached)` with
  `Tasks=16`. Logs show active training/checkpointing rather than only dataset
  prep. Current named checkpoint lines include `puzzle_4x4_play/action_nce`,
  `puzzle_4x4_play/sigreg`, `powderworld_medium_play/action_nce`,
  `powderworld_medium_play/sigreg`, `antmaze_teleport_navigate/action_nce`,
  `antmaze_teleport_navigate/sigreg`, and
  `antmaze_large_stitch/{action_nce,sigreg}`. No eval summaries have landed yet.
- 2026-06-28: Diagnosed Modal dashboard errors for broader OGBench app
  `ap-t8vac1tgm8uhKScE0Olhv3`. Modal preempted several supervisor containers
  and retried them; because those supervisors spawn child training calls, the
  retry path created duplicate child attempts against the same output subdirs.
  Some duplicates failed fast with `Refusing to start fresh in non-empty run
  directory without a checkpoint`, while later duplicates for
  `antmaze_large_stitch` checkpointed to the same paths as the original lanes.
  Treat the current app's `antmaze_large_stitch` artifacts as contaminated
  unless rerun cleanly.
- 2026-06-28: Stopped contaminated broader OGBench app
  `ap-t8vac1tgm8uhKScE0Olhv3` and patched `modal_app.py` so OGB comparison
  supervisors persist child `FunctionCall` IDs and reattach on retry instead of
  spawning duplicate writers. Relaunched the same requested grid under fresh
  root `ogb_broad_clean1`; clean app `ap-Hh88l6rZHlJPTwnMCfcNHF` is
  `ephemeral (detached)` with 17 tasks on the latest check. Logs show active
  training, including `powderworld_medium_play/sigreg_s3072`, and no
  duplicate-writer traceback or Python error in the checked tail.
- 2026-06-28: Follow-up status for clean OGB app
  `ap-Hh88l6rZHlJPTwnMCfcNHF`: app remains `ephemeral (detached)` with
  `Tasks=16`. Modal preempted some supervisors again, but the patched retry
  path reattached to existing child calls (`FunctionCall.from_id`) instead of
  spawning duplicate writers; no duplicate-directory/runtime traceback appeared
  in the checked logs. No eval outputs have landed yet. Checkpoint depth by
  volume listing: puzzle 4x4 both methods epoch 6; puzzle 4x5 both methods
  epoch 2; antmaze teleport Action-NCE epoch 6 and SIGReg epoch 7; powderworld
  Action-NCE epoch 1 and SIGReg epoch 2; antmaze large Action-NCE epoch 6 and
  SIGReg epoch 7. `antsoccer-medium-stitch-v0` has not started training because
  its state-to-pixel conversion is still rendering (`100000/4000000` frames in
  the latest logs), making it the current bottleneck.
- 2026-06-28: Fixed the antsoccer bottleneck for the broad OGB screening run.
  There is no public `visual-antsoccer-*` OGB dataset, and the original
  MuJoCo-rendered conversion would have had to render 4M frames. Added a
  bounded synthetic top-down pixelization for `antsoccer-medium-stitch-v0`:
  first 250k rows / 500 whole episodes, 32x32 RGB, agent and ball drawn from
  qpos, with eval using state observations and the same rasterizer
  (`eval.synthetic_pixels=antsoccer_topdown`). Stopped the two slow rendered
  attempts (`ap-q72vDVXXZjdeQS0RLF985k`, `ap-keEo3MEimX4guvzE3xxVGX`) and
  launched synthetic app `ap-WTlOcVMs0qH2ej1WrympIk` under
  `ogb_broad_antsoccer_topdown1`. The synthetic HDF5 exists in the Modal
  volume as `ogbench/antsoccer_medium_stitch_topdown32_250k.h5`; Action-NCE
  and SIGReg child calls have been spawned but had not yet emitted training
  startup logs at the last check. Treat this antsoccer result as a screening
  comparison, not a direct visual OGB protocol result.
- 2026-06-28: Current broad OGB status: clean app `ap-Hh88l6rZHlJPTwnMCfcNHF`
  is still live with `Tasks=7`. Epoch-10 checkpoints exist for both methods on
  puzzle 4x4, antmaze teleport, antmaze large, and synthetic-topdown
  antsoccer. Puzzle 4x5 SIGReg is done while puzzle 4x5 Action-NCE is still
  training around epoch 9/10; powderworld Action-NCE is around epoch 8/10 and
  SIGReg around epoch 9/10. Completed n50 online eval files have landed for
  puzzle 4x4, antmaze teleport, and antmaze large, but every completed
  method-task result is 0/50 with full-step timeouts. This looks like an
  OGB online protocol/planner mismatch or overly expensive default CEM setting,
  not a training crash. Antsoccer topdown training is complete for both
  methods, but the fixed eval-only apps were stopped because default CEM
  projected multi-day runtime.
- 2026-06-28: Patched the broad OGB eval path. Added `eval.protocol=ogb_trajectory`
  for future-observation goals sampled from converted HDF5 trajectories, with
  task-specific success checks for puzzle button states, antmaze/antsoccer qpos
  goals, and powderworld pixel goals. Fixed vectorized discrete action-space
  handling so powderworld PGD gets the single-env `Discrete(8)` space instead
  of Gymnasium's vector `MultiDiscrete`. Local validation passed:
  `py_compile`, Hydra compose, discrete-solver configuration smoke, and a fake
  antmaze HDF5 `policy=random` trajectory smoke. Modal relaunch is currently
  blocked by `workspace billing cycle spend limit reached`; app
  `ap-Hh88l6rZHlJPTwnMCfcNHF` now shows `Tasks=0`, and all requested broad OGB
  epoch-10 checkpoints are present.
- 2026-06-29: Billing was unblocked and the corrected broad OGB trajectory
  matrix completed. Non-powderworld results came from the corrected matrix app,
  while powderworld was rerun after the final discrete-output/warm-start patch.
  Final single-seed n50 table: Action-NCE vs SIGReg was `34/50` on puzzle 4x4,
  `26/52` on puzzle 4x5, `46/40` on antmaze teleport, `16/6` on powderworld,
  `30/30` on antmaze large, and `88/88` on synthetic top-down antsoccer. All
  relevant Modal apps are stopped with zero active tasks.
- 2026-06-30: Diagnosed the surprising broad-OGB puzzle gap. The same
  start/goal pairs were used by both methods, and the already-solved step-1
  successes were identical. The gap is in nontrivial button-state transitions:
  Action-NCE solved 5/38 vs SIGReg 13/38 on puzzle 4x4, and 3/40 vs 16/40 on
  puzzle 4x5. Linear probes show Action-NCE has near-zero/negative button-state
  R2 on both puzzle tasks, while SIGReg linearly decodes the active button bits
  at about 0.95-0.99. qpos probes confirm Action-NCE still encodes continuous
  pose, so the failure is selective: it learns motion/action geometry but loses
  the discrete combinatorial button-state bits needed by puzzle goals.
- 2026-06-30: Added user-facing broad OGB play support and README media.
  `play.py` now exposes `puzzle4x4`, `puzzle4x5`, `antmaze_teleport`,
  `powderworld`, `antmaze_large`, and `antsoccer` with OGB observation
  adaptation, task-specific keyboard controls, discrete Powderworld handling,
  and native-goal HUD status. Generated stills/GIFs under `assets/datasets/`
  and updated `README.md` to show commands, controls, and the playable-task
  gallery while keeping them separate from the released four-task benchmark.
- 2026-07-02: Replaced the OGBench Scene still used by the README and LaTeX
  paper with a 1024x1024 MuJoCo render of the same representative first-episode
  state (`visual-scene-play-v0`, capped-frame index 83). The tracked files are
  `assets/datasets/scene_still.png` and `paper/latex/images/scene_still.png`.
  Updated `make_task_media.py` so future Scene media reruns keep the compact
  GIF but overwrite the Scene still with a high-resolution render from
  `qpos/qvel/button_states`.
- 2026-07-02: Active paper-polish pass for Jack's comments in
  `paper/latex/iclr_main.tex`. Initial audit found live `\jack{...}` markers in
  the method explanation, MTM-MSE framing, Scene emphasis, TwoRoom-long seed
  count, PushT mechanism probes, AC-CPC positioning, SigReg baseline variance,
  limitations, and conclusion. Existing ledgers already answer the
  TwoRoom-long seed-count comment: AC-MTM and MTM-MSE both have three training
  seeds at n=200. Missing paper-facing data to run before final prose edits:
  PushT linear probes for AC-MTM (`pusht/mtm_action_nce_w030_e10_s{3072,1,2}`)
  and n=200 evals for available extra SigReg seeds on TwoRoom, PushT, and Cube.
- 2026-08-08: Prepared the repository for public release alongside the arXiv
  paper. Renamed the GitHub repo `long-horizon-world-model` →
  `action-contrastive-jepa` (the old name described the abandoned multi-horizon
  direction; the new one matches the method abbreviation readers will search
  for). Rewrote `README.md` around AC-MTM with explicit fork attribution to
  `lucas-maes/le-wm`; kept the MIT licence under the upstream copyright with no
  copyright line of our own, so the release stays as open as possible. Replaced
  the upstream `assets/lewm.gif` hero with `assets/hero.gif`, generated by the
  new `scripts/make_hero_gif.py` from our own task media and result deltas.
  Moved the paper out of this repo into the companion git repo
  `~/workspace/Desktop/action-contrastive-jepa-paper`, which syncs with the
  canonical Overleaf project via `olcli` (`make diff` force-pulls and diffs
  against HEAD) — this keeps Overleaf canonical while making revisions
  diffable. Untracked and gitignored `paper/`, `.yakanban/`, and LaTeX build
  artifacts; the third-party `LeWorldModel.md` scrape is kept locally in the
  paper repo but gitignored there, not redistributed. Scrubbed absolute
  absolute macOS home paths and the Overleaf project id / read-share link from
  `progress/`. Squashed all 128 commits to one fresh commit so none of the
  purged material survives in history; the full pre-release history is
  preserved locally at
  `~/workspace/Desktop/long-horizon-world-model-prerelease-history.bundle`.
  Not yet done: force-push, flip to public, and fill the author list and arXiv
  id placeholders in the README citation block.
