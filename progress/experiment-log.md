# Experiment Log

Chronological, append-only research ledger. See `AGENTS.md` for the entry format.

---

## 2026-05-31 — TwoRoom Multi-Horizon (LeWM-Direct-H) vs original LeWM

### Intent

First head-to-head of **direct multi-horizon prediction** (`LeWM-Direct-H`, `config/train/lewm_mh.yaml`)
against the **original autoregressive LeWM** on TwoRoom. `LeWM-Direct-H` had never been trained or evaluated
on TwoRoom before this (Stage 2 was paused during PushT/Reacher speed profiling).

Tests the core project hypothesis (`paper/MultiFutureLeWorldModel.md`): directly predicting an H-step latent
trajectory should reduce autoregressive rollout drift and improve MPC planning, especially as horizon grows.

Three from-scratch training runs (a horizon sweep), each matched to the AR baseline's protocol
(10 epoch, `batch_size=128`, `early_stopping.enabled=false`, `seed=3072`, TwoRoom dataset). The only changes
vs the AR baseline are the predictor (`FutureQueryPredictor`) / objective (`mh_lejepa_forward`) and the horizon:

- **MH-H5-uniform** — H=5, uniform horizon weights → `tworoom/lewm_direct_h5`
- **MH-H5-discount** — H=5, discount (γ=0.95) horizon weights → `tworoom/lewm_direct_h5_discount`
- **MH-H10-uniform** — H=10, uniform horizon weights → `tworoom/lewm_direct_h10`

**Baseline to beat (reused, not retrained):** `tworoom/lewm_from_scratch` (original AR LeWM),
**86% SR (43/50)** on Protocol A (`goal_offset_steps=25`, `eval_budget=50`), final `validate/loss=0.157995`.

### Commands

#### Train

```bash
# MH-H5-uniform  (Modal app ap-402UJ5Rng4Wtcj1oAY2Egi, A100-40GB)
.venv/bin/modal run --detach modal_app.py::train \
  --config-name lewm_mh --data tworoom --subdir tworoom/lewm_direct_h5 \
  --overrides "early_stopping.enabled=false" --no-wait

# MH-H5-discount  (Modal app ap-RrgEfDL905hYKPKOECz0FF, A100-40GB)
.venv/bin/modal run --detach modal_app.py::train \
  --config-name lewm_mh --data tworoom --subdir tworoom/lewm_direct_h5_discount \
  --overrides "early_stopping.enabled=false loss.horizon_weights.schedule=discount" --no-wait

# MH-H10-uniform  (Modal app ap-Jr9gbTKiWy72shsrIpojEP, A100-80GB)
MODAL_TRAIN_GPU=A100-80GB .venv/bin/modal run --detach modal_app.py::train \
  --config-name lewm_mh --data tworoom --subdir tworoom/lewm_direct_h10 \
  --overrides "early_stopping.enabled=false wm.horizon=10" --no-wait
```

#### Eval

```bash
# Protocol A (25/50), plan horizon 5
.venv/bin/modal run modal_app.py::evaluate --config-name tworoom \
  --policy tworoom/lewm_direct_h5 \
  --overrides "eval.env_batch_size=10 output.save_video=false output.filename=lewm_direct_h5_tworoom_results.txt"

# Long (100/150), plan horizon 5  (repeat for each MH run + the AR baseline tworoom/lewm_from_scratch)
.venv/bin/modal run modal_app.py::evaluate --config-name tworoom_long \
  --policy tworoom/lewm_direct_h5 \
  --overrides "eval.env_batch_size=10 output.save_video=false output.filename=lewm_direct_h5_tworoom_long_results.txt"
```

### Artifacts

* Train metadata / metrics / objects (Modal volume `multi-future-lewm-cache`):
  * `tworoom/lewm_direct_h5/` — `metrics.jsonl`, `lewm_direct_h_epoch_{1..10}_object.ckpt`
  * `tworoom/lewm_direct_h5_discount/`
  * `tworoom/lewm_direct_h10/`
* Eval results: `tworoom/<policy parent>/<output.filename>` on the volume.
* W&B project: `multi-future-lewm` (runs `tworoom_lewm_direct_h5`, `_discount`, `tworoom_lewm_direct_h10`).

### Result

Base (AR LeWM, Protocol A 25/50): **86% SR (43/50)**. All cells `num_eval=50`, `seed=42` (episode sets
aligned across models, so MH-vs-AR can be compared paired). `±` is the binomial standard error of the rate.

| Model | Protocol A (25/50) | Long (100/150) |
|-------|--------------------|----------------|
| AR baseline (`lewm_from_scratch`) | 86.0% (43/50) ±4.9 | 16.0% (8/50) ±5.2 |
| MH-H5-uniform  | 88.0% (44/50) ±4.6 | 12.0% (6/50) ±4.6 |
| MH-H5-discount | 88.0% (44/50) ±4.6 | 10.0% (5/50) ±4.2 |
| MH-H10-uniform | 90.0% (45/50) ±4.2 | 24.0% (12/50) ±6.0 |

**Paired McNemar vs AR baseline** (discordant pairs; `net` = MH-only-wins − AR-only-wins; exact two-sided p):

| Model | Protocol A net (p) | Long net (p) |
|-------|--------------------|--------------|
| MH-H5-uniform  | +1 (p=1.00) | −2 (p=0.75) |
| MH-H5-discount | +1 (p=1.00) | −3 (p=0.51) |
| MH-H10-uniform | +2 (p=0.69) | +4 (p=0.45) |

* Wins / losses: nominally, all 3 MH variants edge out AR on Protocol A; MH-H10 is the only variant nominally
  ahead of AR on the Long protocol (the two H5 variants are nominally *behind* AR there).
* **None of these deltas are statistically significant at n=50** — every paired p > 0.4 and every rate gap is
  ≤ ~1 standard error. The result is suggestive, not decisive.
* Recall collapse? no for all runs. Loss decreased cleanly (no NaN/OOM); episode-success arrays are well-mixed
  (no degenerate all-pass/all-fail); all four Long-protocol eval logs are error-free.

### Decision

**Rerun before claiming a win.** The hypothesis is *directionally supported but not yet established*:
- MH-H10-uniform is the lead candidate — nominally best on **both** protocols (90% vs 86% on A; 24% vs 16% on
  Long) and never behind AR. This is the only variant that improves the Long protocol, consistent with the
  "longer direct horizon reduces compounding drift" story.
- But at n=50 the gaps are within noise (paired p=0.69 / 0.45), so this is not yet evidence the idea *beats* AR —
  only that it is at least on par and trends favorably.
- MH-H5-uniform / MH-H5-discount are on par with AR on Protocol A but trend *worse* on the Long protocol;
  the discount schedule gives no edge over uniform at H=5.

Next: confirm MH-H10-uniform vs AR with higher power — `num_eval=200` and ≥3 training/eval seeds on **both**
protocols — before promoting the multi-horizon variant or moving to Stage 3 fair-baseline plots. Drop the
discount-schedule arm at H=5; carry uniform H5 and H10 forward.

### Notes

* MH eval planning path (`rollout_direct`→`predict_future`) had never run end-to-end on a trained checkpoint;
  a 2-episode smoke-eval of the H5 epoch-1 object confirmed the path works before trusting full runs.
* H=10 needs A100-80GB: 13-step windows allocate ~39 GiB (would OOM on A100-40GB); kept `batch_size=128`
  for schedule parity with the H=5 runs and the AR baseline.
* Train/eval context mismatch (trained C=3, eval world `history_size=1`) is identical to the AR baseline and
  is tolerated by `FutureQueryPredictor` (supports C<num_context).
* Horizon-weighted loss ablation shows no measurable difference between uniform and discount schedules at H=5
  (both 88% on A; 12% vs 10% on Long — within noise).
* The Long protocol (100/150) is hard for every model (≤24%), so it has the most headroom but also the widest
  error bars at n=50; the higher `num_eval` rerun matters most here.
* Statistical caveat: n=50, single seed. Treat the table as a screening pass, not a confirmatory result.

---

## 2026-06-01 — TwoRoom MH-H10 confirmatory run (num_eval=200, 3 training seeds)

### Intent

Higher-power confirmation of the 2026-05-31 screening result, which suggested MH-H10-uniform might beat the AR
baseline but at n=50 had every delta within noise. Goal: decide whether MH-H10's edge is real or a small-sample
artifact before promoting it or starting Stage 3.

Per the prior decision, this run (a) raises `num_eval` 50→**200** (SE ~5%→~2.5%) and (b) adds a **training-seed
distribution** for MH-H10. Only the MH model is retrained at new seeds; the AR baseline reuses its single
paper-validated checkpoint (no value in retraining a reproduced baseline). All 8 cells share eval `seed=42`, so
each model is scored on the *identical* 200-episode set per protocol → fully paired (McNemar valid).

- MH-H10 training seeds: **3072** (existing, from screening), **1**, **2** (new). Identical config otherwise
  (`lewm_mh`, `wm.horizon=10`, uniform weights, 10 epochs, `early_stopping.enabled=false`).
- Compared against AR `tworoom/lewm_from_scratch` (single checkpoint).

### Commands

#### Train

```bash
# Two new MH-H10 seeds (seed=3072 already trained as tworoom/lewm_direct_h10)
MODAL_TRAIN_GPU=A100-80GB .venv/bin/modal run --detach modal_app.py::train \
  --config-name lewm_mh --data tworoom --subdir tworoom/lewm_direct_h10_seed1 \
  --overrides "early_stopping.enabled=false wm.horizon=10 seed=1" --no-wait
MODAL_TRAIN_GPU=A100-80GB .venv/bin/modal run --detach modal_app.py::train \
  --config-name lewm_mh --data tworoom --subdir tworoom/lewm_direct_h10_seed2 \
  --overrides "early_stopping.enabled=false wm.horizon=10 seed=2" --no-wait
```

#### Eval

```bash
# 8 cells = {AR, MH-s3072, MH-s1, MH-s2} x {Protocol A, Long}, all at num_eval=200, eval seed=42.
# IMPORTANT: launch eval with --detach so a local client/network drop cannot orphan the remote job;
# treat the volume result file as the source of truth (see Notes).
.venv/bin/modal run --detach modal_app.py::evaluate --config-name tworoom \
  --policy tworoom/lewm_direct_h10_seed1 \
  --overrides "eval.num_eval=200 eval.env_batch_size=10 output.save_video=false output.filename=lewm_direct_h10_s1_tworoom_n200_results.txt"
# ...repeat for each policy and for --config-name tworoom_long (filename *_long_n200_*).
```

### Artifacts

* New train runs (Modal volume `multi-future-lewm-cache`): `tworoom/lewm_direct_h10_seed1/`,
  `tworoom/lewm_direct_h10_seed2/` — full `lewm_direct_h_epoch_{1..10}_object.ckpt`, `metrics.jsonl`,
  `best_weights`. Both completed 10 epochs / 20,200 steps, no NaN; final train loss ≈0.32, pred_loss ≈0.07–0.09.
* Eval results (volume `tworoom/`): `lewm_{from_scratch,direct_h10_s3072,direct_h10_s1,direct_h10_s2}_tworoom[_long]_n200_results.txt`.
* Local copies + parser: `/tmp/tworoom_volume_n200/tworoom/*`, `/tmp/analyze_confirmatory.py`.

### Result

Base = AR `lewm_from_scratch` (single checkpoint). MH = 3 training seeds {3072, 1, 2}. n=200, eval seed=42.
`±` is binomial SE of the rate. Paired McNemar exact p is each MH seed vs AR on the shared 200-episode set.

**Protocol A (25/50)**

| Model | SR (n=200) | net vs AR | McNemar p |
|-------|------------|-----------|-----------|
| AR `lewm_from_scratch` | 85.0% (170/200) ±2.5 | — | — |
| MH-H10 seed 3072 | 86.0% (172/200) ±2.5 | +2 | 0.864 |
| MH-H10 seed 1 | 91.5% (183/200) ±2.0 | +13 | **0.024** |
| MH-H10 seed 2 | 85.0% (170/200) ±2.5 | +0 | 1.000 |
| **MH-H10 (3-seed mean)** | **87.5%** (sd 3.5, range 85–92) | — | 2/3 seeds > AR, 1 tie |

**Long (100/150)**

| Model | SR (n=200) | net vs AR | McNemar p |
|-------|------------|-----------|-----------|
| AR `lewm_from_scratch` | 18.0% (36/200) ±2.7 | — | — |
| MH-H10 seed 3072 | 21.0% (42/200) ±2.9 | +6 | 0.512 |
| MH-H10 seed 1 | 24.0% (48/200) ±3.0 | +12 | 0.111 |
| MH-H10 seed 2 | 21.5% (43/200) ±2.9 | +7 | 0.419 |
| **MH-H10 (3-seed mean)** | **22.2%** (sd 1.6, range 21–24) | — | **3/3 seeds > AR** |

* Wins / losses: **Long protocol — all 3 MH seeds beat AR** (mean +4.2 pts), with low seed variance (sd 1.6).
  **Protocol A — MH ≈ AR** (mean +2.5 pts) but high seed variance (sd 3.5); only the lucky seed-1 separates.
* vs screening (n=50): the headline screening gaps shrank with power. Protocol A MH-s3072 90%→86% (gap +4→+1);
  Long MH-s3072 24%→21% (gap +8→+3). The screening over-stated the effect; the true effect is smaller.
* Significance: **no single MH seed reaches p<0.05 except Protocol-A seed-1 (p=0.024)**, and that one is the
  high-variance protocol where the other two seeds tie AR — so it does not generalize. Pooling all 3 MH seeds
  (600 trials) vs AR gives ~+2.5 pts on A and ~+4.2 pts on Long, neither significant by a 2-proportion test
  (z≈0.9 and z≈1.3; and pooling over-counts independence, so true p is larger).
* Recall collapse? no. Both new trainings converged cleanly (monotone horizon MSE h1≈0.03 < h10≈0.13, no NaN);
  all 8 eval success-arrays are well-mixed.

### Decision

**Promising on the long-horizon regime, but not a statistically established win — carry MH-H10 forward to
Stage 3 rather than declaring victory.**

- The most credible signal is the **long protocol**: the improvement over AR **replicates across all 3
  independent trainings** (3/3 seeds, tight spread), and the *pattern* matches the hypothesis — MH helps where
  autoregressive rollout drift is worst (long horizon), and not on the short protocol where drift is mild.
- It is **not** a clean significance win: per-seed p-values are 0.11–0.51 on Long, and Protocol A shows no
  reliable advantage. So this confirms "consistent, horizon-specific directional improvement," not "MH beats AR."
- The screening's flashy 90%/+8 numbers were inflated by n=50 noise + a single checkpoint; corrected.

Next: **Stage 3** — test the mechanism directly rather than chasing more planning episodes. Produce the
horizon-wise latent-error curves (does MH's latent MSE diverge slower than AR's rollout under recorded actions?)
and a planning-success-vs-horizon sweep. Those discriminate the drift-reduction claim far more cheaply than
pushing eval to n≥500. Keep MH-H10-uniform as the carried-forward variant; the H=5 discount arm stays dropped.

### Notes

* **Operational (network):** the first eval batch died mid-run to a local client DNS/connection drop
  (`socket.gaierror` → `StreamTerminatedError`). `modal_app.py::evaluate` uses a *blocking* `run_eval.remote()`
  with no `--no-wait`, so losing the client killed the un-detached remote jobs (the `--detach` trainings
  survived). Fix applied: launch evals with `--detach` and treat the **volume result file** as the source of
  truth, not the client connection. Relaunched all 8 cleanly; all exited 0.
* **Operational (tooling):** background `Monitor`/`Bash` scripts run under **zsh**, which does *not* word-split
  unquoted `$VAR` — `for f in $FILES` iterated once over the whole string and a volume-poll monitor falsely
  reported 0/8. Use explicit arrays (`files=(...)`; `for f in "${files[@]}"`). Also `status` is read-only in zsh.
* Wall-clock: each MH-H10 seed trained in ~2.5 h on A100-80GB (20,200 steps). Protocol-A evals ≈20–45 min at
  n=200; Long evals ≈90–120 min (≈6 min/batch for 150-step episodes), 8 cells run concurrently.
* Checkpoint loading is policy-path-driven (`swm.policy.AutoCostModel(cfg.policy)`), so seed1/seed2 dirs resolve
  their epoch-10 object identically to the existing dir — a mis-load would have produced near-random SR, not the
  85–92% observed.

---

## 2026-06-02 — TwoRoom Masked-Transition LeWM (SIGReg-free) — RESULT: WORKS (beats LeWM on Long, p=0.007)

### Intent

First POC of **masked transition modeling** (`paper/MaskedTransitionModel.md`,
`config/train/lewm_masked.yaml`) as a replacement for SIGReg. Mask one element of
`(z_t, a_t, z_{t+1})` and predict it from the other two: `mask z_{t+1}` is the LeWM
forward predictor, `mask a_t` is an inverse-dynamics head. A collapsed encoder makes
inverse dynamics unsolvable, so non-collapse emerges from the task instead of from a
prescribed Gaussian latent marginal. **No SIGReg / variance / covariance term.**

Controlled comparison: everything except the loss is held fixed vs LeWM (encoder,
`ARPredictor`, projectors, latent dim, dataset, planner, compute). The planner uses
only the forward predictor, so eval is the byte-for-byte LeWM path.

Compare against the reproduced AR baseline `tworoom/lewm_from_scratch` (Protocol A
85.0% / Long 18.0% at n=200, see 2026-06-01 entry). Add a **no-reg sanity check**
(plain LeWM, SIGReg off) which should collapse — confirming the inverse term, not
the removal of SIGReg, is what holds it up.

This entry is logged at the **code-complete** milestone (37 tests green, real-ViT
construction + forward + planner-rollout smoke verified on CPU). The TwoRoom HDF5
dataset is not present locally and training needs GPU; Result/Decision below are
TODO until the Modal runs complete.

### Commands

#### Train

```bash
# Masked-transition LeWM on TwoRoom (matched to AR baseline protocol)
.venv/bin/modal run --detach modal_app.py::train \
  --config-name lewm_masked --data tworoom --subdir tworoom/lewm_masked \
  --overrides "early_stopping.enabled=false" --no-wait

# No-reg sanity check: plain LeWM objective with SIGReg off (expected to collapse)
.venv/bin/modal run --detach modal_app.py::train \
  --config-name lewm --data tworoom --subdir tworoom/lewm_noreg \
  --overrides "early_stopping.enabled=false loss.sigreg.weight=0.0" --no-wait

# Optional ablation: literal random masking instead of training both heads each step
.venv/bin/modal run --detach modal_app.py::train \
  --config-name lewm_masked --data tworoom --subdir tworoom/lewm_masked_random \
  --overrides "early_stopping.enabled=false loss.masked.mask_mode=random" --no-wait
```

#### Eval

```bash
# Protocol A (25/50)
.venv/bin/modal run --detach modal_app.py::evaluate --config-name tworoom \
  --policy tworoom/lewm_masked \
  --overrides "eval.num_eval=200 eval.env_batch_size=10 output.save_video=false output.filename=lewm_masked_tworoom_n200_results.txt"

# Long (100/150)
.venv/bin/modal run --detach modal_app.py::evaluate --config-name tworoom_long \
  --policy tworoom/lewm_masked \
  --overrides "eval.num_eval=200 eval.env_batch_size=10 output.save_video=false output.filename=lewm_masked_tworoom_long_n200_results.txt"
```

#### Diagnostics

```bash
# Collapse canary is logged every step during training as fit/emb_std and
# validate/emb_std (metrics.jsonl / W&B). emb_std -> 0 means collapse.
# For the no-reg sanity run, expect emb_std to crater while pred/forward loss -> 0.
```

#### Compare

```bash
# Reuse the AR baseline's n=200 results from the 2026-06-01 entry
# (tworoom/lewm_from_scratch: Protocol A 85.0%, Long 18.0%). Paired McNemar
# vs lewm_masked on the shared eval seed=42 episode set, as in prior entries.
```

### Artifacts

* Train (Modal volume): `tworoom/lewm_masked/`, `tworoom/lewm_noreg/`,
  `tworoom/lewm_masked_random/` — `metrics.jsonl`, `lewm_masked_epoch_{1..10}_object.ckpt`.
* Eval results (volume `tworoom/`): `lewm_masked_tworoom[_long]_n200_results.txt`.
* W&B project: `masked-transition-lewm`.
* Code: `module.InverseDynamics`, `jepa.JEPA.predict_action`,
  `train.masked_transition_forward`, `config/train/lewm_masked.yaml`,
  `tests/test_masked_transition.py`. Branch `worktree-masked-transition`.

### Result

Training: both runs completed 10 epochs on A100-40GB (~4.4 h wall-clock, run
concurrently), no NaN/OOM. Eval: n=200, eval `seed=42` (identical 200-episode set
per protocol → fully paired vs the AR baseline; McNemar valid). `±` is binomial SE.

**Loss trajectories (the mechanism):**

| Run | metric | first → last | reading |
|-----|--------|--------------|---------|
| No-reg (SIGReg off, nothing added) | `pred_loss` | 0.149 → **0.00000** | trivial constant solution `f(o)=c` — collapse |
| | `sigreg_loss` (unweighted monitor) | 46.3 → **51.5 ↑** | latent moving *away* from Gaussian, confirming collapse |
| Masked | `inverse_loss` | 1.001 → **0.739** | latents carry action-distinguishing structure (a collapsed encoder is pinned at the ~1.0 mean-action variance — see `test_collapsed_encoder_cannot_distinguish_actions`) |
| | `forward_loss` | 0.154 → 0.0003 | TwoRoom dynamics are smooth → a compact non-collapsed latent is easy to predict; not collapse (the inverse term proves diversity) |

**Planning success (TwoRoom MPC, CEM):**

| Model | Protocol A (25/50) | Long (100/150) |
|-------|--------------------|----------------|
| AR LeWM baseline (SIGReg), reused | 85.0% (170/200) ±2.5 | 18.0% (36/200) ±2.7 |
| **Masked-transition (no SIGReg)** | **90.5% (181/200) ±2.1** | **29.0% (58/200) ±3.2** |
| No-reg control (SIGReg off) | 29.5% (59/200) ±3.2 | — (not run) |

**Paired McNemar (vs AR baseline; net = candidate-only-wins − baseline-only-wins):**

| Comparison | net | McNemar exact p |
|-----------|-----|-----------------|
| Masked vs baseline — Protocol A | +11 (26 vs 15) | 0.117 |
| Masked vs baseline — Long | **+22 (42 vs 20)** | **0.0071** |
| No-reg vs baseline — Protocol A | −111 (1 vs 112) | ≈0 (collapse) |
| Masked vs no-reg — Protocol A | +122 (123 vs 1) | ≈1e-37 |

Delta:

* Wins / losses: **Masked beats LeWM on the Long protocol (+11 pts, p=0.0071, significant)**
  and is nominally ahead on Protocol A (+5.5 pts, p=0.117 — favorable, not significant
  at single seed). The pattern matches the hypothesis: the SIGReg-free objective helps
  most where the representation must support long-horizon planning.
* Recall collapse? **No for masked; yes for no-reg.** The no-reg control is the smoking
  gun: removing SIGReg with nothing added craters Protocol A 85.0% → 29.5% (net −111,
  p≈0) with `pred_loss`→0. The masked objective *restores and exceeds* LeWM (29.5% →
  90.5%, net +122 vs no-reg) — so the inverse-dynamics task, not a prescribed Gaussian
  marginal, is what prevents collapse. One config, no per-environment tuning.

### Decision

**Keep — the POC works.** With a single task-agnostic config, masked transition modeling
(a) prevents the collapse that removing SIGReg otherwise causes, (b) matches LeWM on the
short protocol, and (c) significantly beats it on the long protocol (p=0.0071). This is
exactly the discriminating result the design targeted: distinct, planning-useful states
emerge from predicting distinguishable action-conditioned transitions rather than from
forcing the latent marginal into an isotropic Gaussian.

Caveats (honest): single training seed per arm; the 2026-06-01 MH work showed Protocol-A
SR has high cross-seed variance (sd ~3.5), so the +5.5 on A could be partly seed luck —
the Long result is the robust signal. Effective rank / `emb_std` were not captured for
these two runs (logging gap, now fixed at `utils.py` to record `*_std`); the functional
planning gap + the `inverse_loss` < mean-baseline evidence stand in for it here.

Next: (1) ≥3 training seeds for masked + AR on **both** protocols to convert the
single-seed result into a confirmatory one (mirror the 2026-06-01 protocol); (2) a
`mask_mode=random` ablation (already launchable, subdir `tworoom/lewm_masked_random`);
(3) extend the controlled comparison to Reacher / PushT / OGB-Cube with the same config;
(4) an effective-rank / linear-probe diagnostic now that `*_std` is logged.

### Notes

* Eval needs no code change: `AutoCostModel` scans the saved object for `get_cost`
  (the `JEPA`), and planning uses only the forward `ARPredictor`. Verified the
  `rollout` path runs on a masked-transition `JEPA` in a CPU smoke test.
* Default `mask_mode=both` trains forward+inverse every step (the `E_m[·]`
  expectation, lower variance); `random` is the literal one-task-per-batch mask.
* `inverse_weight` (default 1.0) is the only new coefficient; the goal is a single
  task-agnostic value, not a per-environment tune.
* Launch trainings/evals with `--detach` (per the 2026-06-01 operational note) and
  treat the volume result file as source of truth.
* Actual runs: train apps `ap-iO5xf9v2qCTjYtnjY0Dj9F` (masked) / `ap-w0OivLEm0tfFuQy5wOLox8`
  (no-reg); eval apps `ap-JGpmNSbICMaQpoJOQ33A3C` (masked A), `ap-58Di8GDdx8b0qq71cTsFGU`
  (masked Long), `ap-z7g7Gjq7YDAE61fPl3x77a` (no-reg A). Eval used L4; train used A100-40GB
  (horizon=1, so 40GB is ample — no need for the 80GB the MH-H10 runs required).
* **Logging gap (fixed):** the masked run's `emb_std` collapse canary was logged to
  Lightning but dropped by `JsonlMetricsCallback`, which only persisted keys containing
  `loss`/`_mse`. Fixed to also keep `*_std`, so future runs capture `emb_std` in
  `metrics.jsonl`. For these runs, collapse is instead read from `pred_loss`→0 (no-reg),
  `inverse_loss` < mean-baseline (masked), and the planning-SR gap.
* Added `--no-wait`/spawn support to `modal_app.py::evaluate` (mirrors `train`) so the
  three evals could run detached and be polled on the volume.
* WANDB_API_KEY was unset locally, so W&B logging was offline/disabled; `metrics.jsonl`
  on the volume is the authoritative training record.

---

## 2026-06-03 — Reacher + OGB-Cube Masked-Transition LeWM — RESULT: ties on Reacher, WINS on Cube (p<1e-4)

### Intent

Extend masked transition modeling (the SIGReg-free objective validated on TwoRoom in the
2026-06-02 entry) to two harder continuous-control environments — **Reacher** (`dmc/reacher_random`)
and **OGB-Cube** (`ogbench/cube_single_expert`) — to test the central POC claim that *one
task-agnostic config* can replace SIGReg across environments.

Controlled, matched-protocol comparison: for each task, train **both** `lewm_masked` (forward +
inverse, SIGReg off) and a from-scratch `lewm` baseline (SIGReg on, weight 0.09) with the identical
protocol (10 epochs, batch 128, seed 3072, `early_stopping.enabled=false`). The only difference is
the objective. Same masked config as TwoRoom — no per-environment tuning (`history_size=3`,
`horizon=1`, `inverse_weight=1.0`, `mask_mode=both`). Fresh baselines rather than reusing the
existing `reacher/lewm` / `cube/lewm` checkpoints, whose training protocol was unknown (the existing
`reacher/lewm` scored only 70% at n=50, below the paper's 86%).

### Commands

#### Train

```bash
# masked + matched baseline, per task (A100-40GB, --detach, --no-wait)
.venv/bin/modal run --detach modal_app.py::train --config-name lewm_masked --data reacher \
  --subdir reacher/lewm_masked --overrides "early_stopping.enabled=false" --no-wait
.venv/bin/modal run --detach modal_app.py::train --config-name lewm        --data reacher \
  --subdir reacher/lewm_base   --overrides "early_stopping.enabled=false" --no-wait
.venv/bin/modal run --detach modal_app.py::train --config-name lewm_masked --data ogb \
  --subdir cube/lewm_masked    --overrides "early_stopping.enabled=false" --no-wait
.venv/bin/modal run --detach modal_app.py::train --config-name lewm        --data ogb \
  --subdir cube/lewm_base      --overrides "early_stopping.enabled=false" --no-wait
```

#### Eval

```bash
# n=200, eval seed 42 (paired masked-vs-baseline per task)
.venv/bin/modal run --detach modal_app.py::evaluate --config-name reacher --policy reacher/lewm_masked \
  --overrides "eval.num_eval=200 eval.env_batch_size=10 output.save_video=false output.filename=lewm_masked_reacher_n200_results.txt"
.venv/bin/modal run --detach modal_app.py::evaluate --config-name reacher --policy reacher/lewm_base \
  --overrides "eval.num_eval=200 eval.env_batch_size=10 output.save_video=false output.filename=lewm_base_reacher_n200_results.txt"
.venv/bin/modal run --detach modal_app.py::evaluate --config-name cube --policy cube/lewm_masked \
  --overrides "eval.num_eval=200 eval.env_batch_size=10 output.save_video=false output.filename=lewm_masked_cube_n200_results.txt"
.venv/bin/modal run --detach modal_app.py::evaluate --config-name cube --policy cube/lewm_base \
  --overrides "eval.num_eval=200 eval.env_batch_size=10 output.save_video=false output.filename=lewm_base_cube_n200_results.txt"
```

### Artifacts

* Train (volume): `reacher/lewm_masked`, `reacher/lewm_base`, `cube/lewm_masked`, `cube/lewm_base`
  — each `lewm{_masked,}_epoch_{1..10}_object.ckpt`, `metrics.jsonl`.
* Eval (volume): `reacher/lewm_{masked,base}_reacher_n200_results.txt`,
  `cube/lewm_{masked,base}_cube_n200_results.txt`.

### Result

n=200, eval `seed=42`, paired (identical episode set per task → McNemar valid). `±` is binomial SE.

| Task | Masked (no SIGReg) | LeWM baseline (SIGReg) | net | McNemar p |
|------|--------------------|------------------------|-----|-----------|
| Reacher | 68.0% (136/200) ±3.3 | 69.0% (138/200) ±3.3 | −2 (36 vs 38) | 0.908 |
| **OGB-Cube** | **81.5% (163/200) ±2.7** | 66.0% (132/200) ±3.3 | **+31 (39 vs 8)** | **<0.0001** |

Delta:

* Wins / losses: **Reacher is a statistical tie** (−1 pt, p=0.91) — masked matches SIGReg.
  **Cube is a decisive masked win** (+15.5 pts, net +31, p<1e-4); masked (81.5%) exceeds even the
  paper's reported LeWM-Cube (74%), while the matched baseline reproduces ~66%.
* Recall collapse? No — both masked runs plan well above random (Cube random ≈48%, Reacher ≈10%).

Combined with TwoRoom (masked +5.5 Protocol A / +11 Long, Long p=0.0071), the full picture:

| Task | Masked | LeWM | verdict |
|------|-------:|-----:|---------|
| TwoRoom (A / Long) | 90.5 / 29.0 | 85.0 / 18.0 | masked ≥; Long win (p=0.007) |
| Reacher | 68.0 | 69.0 | tie |
| OGB-Cube | 81.5 | 66.0 | masked win (p<1e-4) |

### Decision

**Keep — the POC generalizes.** With a single task-agnostic config, masked transition modeling is
**on par or better than SIGReg on every environment tested**, with significant wins on the two
hardest (TwoRoom-Long, Cube) and a clean tie on Reacher. This supports the core thesis: distinct,
planning-useful states emerge from predicting distinguishable action-conditioned transitions, with
no prescribed Gaussian latent marginal — and the largest gains appear where the representation must
do the most work (3D Cube geometry, long-horizon TwoRoom).

Caveats: single training seed per arm (multi-seed confirmation is the next step, esp. for the
within-noise Reacher tie); PushT not yet run (the 4th reported task). Reacher's absolute SR (~68–69%)
is below the paper's 86% for both arms, suggesting the 10-epoch matched protocol under-trains Reacher
relative to the paper — but since both arms share it, the masked-vs-SIGReg comparison is fair.

Next: (1) ≥3 seeds per task to firm up Reacher (tie) and Cube (win); (2) PushT for full 4-task
coverage; (3) effective-rank / linear-probe diagnostics (`*_std` logging now fixed).

### Notes

* **Modal outage mid-run + clean resume.** ~4 h in (epochs 4–6 of 10) Modal's scheduler crashed and
  dropped all four detached apps (state `stopped`, 0 tasks). Recovery was a non-event: each run had
  committed a full Lightning checkpoint (optimizer + `global_step` + RNG) to the volume every ~300s,
  so relaunching the identical command auto-resumed (`resume.mode=auto`) from epoch 4–6, losing only
  ~1k steps. First relaunch was also dropped (Modal still down — containers produced *zero* log
  bytes, the tell that it was infra not code); the second relaunch ~1 h later caught the recovery and
  all four advanced `global_step` past baseline within 4 min. Verified resume via global_step
  advancement, not logs (the log API was flaky during the outage). `prevent_restart_existing=true`
  guarantees a failed resume hard-errors rather than silently restarting from scratch.
* Cube data was added to `LOCAL_TRAIN_DATASETS` (`ogb` → `ogbench/cube_single_expert.h5`) so it
  stages locally and commits periodically like tworoom/reacher — which is exactly what made the
  outage recoverable.
* Same one config across TwoRoom/Reacher/Cube; the inverse head sizes itself from the env's action
  dim (reacher ~2-d, cube larger + merged proprio) with no manual change.

---

## 2026-06-04 — Multi-seed confirmation + PushT + the max_epochs reproduction finding

### Intent

Three things at once: (1) **firm up** the TwoRoom-Long and Cube wins with 3 training seeds each;
(2) add **PushT** (4th reported task) for full coverage; (3) chase the Reacher from-scratch-vs-released
gap raised earlier. All masked-vs-baseline comparisons use the **matched `max_epochs=10` recipe** per
task (masked and baseline share the recipe → fair within-task). Baselines: TwoRoom reuses the AR
`lewm_from_scratch`; Cube/PushT/Reacher use fresh matched-protocol `lewm` baselines.

### Result — multi-seed planning success (n=200, eval seed 42, paired McNemar vs baseline)

| Task | Masked (mean ± sd, seeds {3072,1,2}) | Baseline | Δmean | Per-seed verdict |
|------|--------------------------------------|----------|-------|------------------|
| TwoRoom Protocol A | 90.2% ± 0.5 (90.5/89.5/90.5) | 85.0% (AR) | +5.2 | 3/3 win; p 0.117/0.222/0.117 |
| **TwoRoom Long** | 28.0% ± 0.8 (29.0/27.0/28.0) | 18.0% (AR) | **+10.0** | **3/3 win; p 0.007/0.027/0.015 (all <0.05)** |
| **OGB-Cube** | 79.3% ± 2.4 (81.5/76.0/80.5) | 66.0% | **+13.3** | **3/3 win; p 0.000/0.004/0.000** |
| Reacher | 68.0% (1 seed) | 69.0% | −1.0 | tie; p 0.908 |
| **PushT** | 85.5% ± 0.7 (85.0/85.0/86.5) | 93.5% | **−8.0** | **0/3 — masked LOSES; p 0.002/0.002/0.007** |

**Honest 4-task verdict (at `max_epochs=10`):** masked transition modeling **significantly beats**
SIGReg on **TwoRoom-Long and Cube** (both confirmed across 3 seeds, all p<0.05), modestly wins
**TwoRoom Protocol A**, **ties on Reacher**, and **significantly loses on PushT** (−8pts, 3/3 seeds).
Mixed and task-dependent: masked helps where the inverse-dynamics signal is informative (navigation,
cube manipulation) and hurts on precise contact-based pushing (PushT), where the action's effect is
contact-dependent and hard to invert — so the inverse term adds noise rather than anti-collapse
pressure, and SIGReg's explicit regularization wins.

### The Reacher reproduction finding (the most broadly useful result)

Our from-scratch Reacher (both arms ~68–69%) sits ~13pt below the **released `reacher/lewm` checkpoint
(81.5% @ n=200, cem30)** on the *same* harness. We ruled out, in order:
* **Eval harness** — the released checkpoint reproduces near paper level (81.5% ≈ paper 80–86; issue #41
  reports 80 @ 300/30/30, our exact CEM config). Not eval.
* **CEM iterations (#41)** — cem30 ≥ cem10 for *every* checkpoint (released 81.5 vs 77.5; mine 68 vs 60,
  69 vs 65). The repo's `n_steps=30` is fine; matching the paper's "10 for non-PushT" makes it *worse*.
* **Predictor history (#37)** — introspecting the released checkpoint: `pos_embedding (1,3,192)` →
  history 3, identical to ours. Architecture is **byte-identical** (18,034,478 params).
* **Image preprocessing** — GPU vs CPU path; ruled unlikely (Cube-masked, same GPU path, *beats* the
  released Cube checkpoint). The CPU-preproc retrain was abandoned (dataloader-bound at 2.7 steps/s;
  Modal `cpu=32` didn't yield 32× preprocessing throughput).

**Root cause: `trainer.max_epochs`.** The upstream `lewm.yaml` (raw) ships **`max_epochs: 100`**;
our repo set it to **10** (matching the paper *text*, per issue #52). Every other field matches
(lr 5e-5, wd 1e-3, λ 0.09, knots 17, proj 1024, embed 192, history 3, batch 128, seed 3072). The
catch: the cosine LR schedule is **sized to `max_epochs`** (`T_max = estimated_stepping_batches`), so
our 10-epoch runs **annealed the LR to ~0 by epoch 10** (verified: lr ≈ 7.7e-13 at epoch 9) — the last
several epochs trained at near-zero LR. The released 100-epoch schedule keeps LR near-peak through
epoch 10. So "fixing" `max_epochs` to 10 inadvertently **broke reproduction by collapsing the LR
schedule** — consistent with our `pred_loss` still falling at epoch 10, and the gap hitting *both* arms
equally (it's the schedule, not SIGReg or the masked head).

**Caveat this raises for ALL the above results:** they were trained at `max_epochs=10` (under-trained).
The masked-vs-baseline comparisons are *fair within-task* (matched recipe), but absolute levels are
below the released checkpoints, and conclusions *could* shift at the correct 100-epoch schedule.

### Commands

```bash
# multi-seed (masked seeds {1,2} per task; reuse baselines)
.venv/bin/modal run --detach modal_app.py::train --config-name lewm_masked --data ogb \
  --subdir cube/lewm_masked_s1 --overrides "early_stopping.enabled=false seed=1" --no-wait   # +s2, tworoom s1/s2
# PushT (masked {3072,1,2} + matched baseline), eval uses cem30 (paper-correct for PushT)
.venv/bin/modal run --detach modal_app.py::train --config-name lewm_masked --data pusht --subdir pusht/lewm_masked --overrides "early_stopping.enabled=false" --no-wait
# Reacher max_epochs=100 fix test (IN PROGRESS)
.venv/bin/modal run --detach modal_app.py::train --config-name lewm --data reacher \
  --subdir reacher/lewm_base_e100 --overrides "early_stopping.enabled=false trainer.max_epochs=100" --no-wait
```

### Decision

**Keep the POC with an honest, nuanced claim.** Masked transition modeling is **not a universal
replacement** for SIGReg — it wins on TwoRoom/Cube (confirmed multi-seed), ties Reacher, and loses
PushT. The mechanism story holds (inverse-dynamics anti-collapse helps where actions are
informative; the no-reg control still collapses on TwoRoom). Separately, the **`max_epochs`/LR-schedule
reproduction bug** is a concrete, broadly useful finding for the upstream repo.

Next: confirm the Reacher `max_epochs=100` fix recovers ~80% (test running; eval epoch 10/20/30
checkpoints); if it materially changes verdicts, rerun the 4-task comparison at the 100-epoch schedule.

### Notes

* PushT eval correctly uses CEM `n_steps=30` (the one env where the paper also uses 30).
* All masked seeds tight per task (sd ≤ 2.4), so verdicts are seed-robust, not single-seed artifacts.
* `max_epochs=10` LR-decay bug: the fix is to decouple the LR-schedule horizon from the stop epoch
  (or train with the 100-epoch schedule) — worth upstreaming.

## 2026-06-05 — Reacher max_epochs=100 ep10 probe: baseline recovers, masked degrades

### Intent

Test the Reacher reproduction diagnosis from the 2026-06-04 entry: if the from-scratch gap was caused
by setting `trainer.max_epochs=10`, then training with `trainer.max_epochs=100` and evaluating the
epoch-10 checkpoint should improve the SIGReg baseline because the cosine LR remains near peak instead
of annealing to nearly zero by epoch 10. Also compare the same schedule change on the masked-transition
arm to see whether the masked objective remains stable under the corrected LR schedule.

### Commands

#### Train

```bash
# Exact launched commands were not captured in this local workspace.
# Known intended override: trainer.max_epochs=100, early_stopping.enabled=false.
```

#### Inference

```bash
# Evaluate Reacher epoch-10 object checkpoints with n=200 on the existing CEM-30 harness.
```

#### Diagnostics

```bash
# Pending: inspect metrics.jsonl/W&B for fit/validate forward_loss, inverse_loss, emb_std,
# latent norm/effective rank, optim/lr, and grad norms.
```

#### Eval

```bash
# Pending exact command capture.
```

#### Compare

```bash
# Compare max_epochs=10 epoch-10 runs against max_epochs=100 epoch-10 runs.
```

### Artifacts

* Train metadata: not available locally for this note.
* GPU metrics: not available locally for this note.
* Predictions: not applicable.
* Eval summary: not available locally for this note.
* Per-dialogue eval: not applicable.

### Result

Base:

* `max_epochs=10`: 69.0%
* `max_epochs=100`, evaluated at epoch 10: 76.0%

Candidate:

* Masked `max_epochs=10`: 68.0%
* Masked `max_epochs=100`, evaluated at epoch 10: 39.5%

Delta:

* Wins / losses: baseline gains +7 pts under the corrected schedule, confirming the LR-schedule diagnosis. Masked loses -28.5 pts under the same schedule, which is anomalous and schedule-sensitive.
* Recall collapse? Unknown pending `emb_std`, effective-rank, latent-norm, and loss-trace inspection.

### Decision

Do not over-read the masked ep10 point as a final verdict. Keep the 100-epoch Reacher runs alive and
evaluate ep20/30. In parallel, diagnose the masked arm as a schedule/loss-balance problem: the
SIGReg-free objective may need inverse-loss warmup/annealing, a lower `inverse_weight`, forward-only
warmup, or a weak latent geometry constraint when the LR remains high for many epochs.

### Notes

* Leading hypothesis: at `max_epochs=10`, the collapsing LR may have accidentally made the inverse
  objective gentle late in training; at `max_epochs=100`, `inverse_weight=1.0` keeps a strong
  inverse-dynamics gradient on the encoder during the same early epoch window.
* Planner risk: without SIGReg, masked can learn action-informative latents whose Euclidean geometry,
  scale, or anisotropy is poor for CEM terminal MSE, even if forward/inverse losses improve.
* Reacher-specific risk: many actions may be weakly identifiable from pixel-level frame pairs or
  confounded by near-symmetries/contact-free dynamics; inverse dynamics can therefore inject noisy
  gradients into the representation rather than only preventing collapse.

## 2026-06-05 — Reacher masked e100 inverse-weight / inverse-warmup ablations

### Intent

Follow up the masked `max_epochs=100` ep10 failure by testing whether the inverse-dynamics branch is
too strong under the sustained high-LR schedule. Compare three changes against
`reacher/lewm_masked_e100`: lower inverse weight, inverse warmup, and the combined lower+warmup arm.

Old-vs-current metrics inspection from Modal:

| Run | epoch | mean LR | emb_std | forward_loss | inverse_loss | eval |
|-----|-------|---------|---------|--------------|--------------|------|
| `reacher/lewm_masked` | 9 | ~4.2e-7 mean, final 7.7e-13 | 0.0878 | 0.00017 | 0.832 | 68.0% |
| `reacher/lewm_masked_e100` | 9/10 | ~4.9e-5 | 0.0747 / 0.0710 | 0.00028 / 0.00025 | 0.833 / 0.832 | 39.5% |
| `reacher/lewm_base` | 9 | ~4.2e-7 mean, final 7.7e-13 | n/a | pred 0.0287 | n/a | 69.0% |
| `reacher/lewm_base_e100` | 9/10 | ~4.9e-5 | n/a | pred 0.0279 / 0.0270 | n/a | 76.0% |

The masked loss traces do not show inverse-loss convergence failure; they show a high-LR run with
lower latent spread and similar inverse loss. That points toward loss-balance / latent geometry rather
than a simple underfit inverse head.

### Commands

#### Train

```bash
MODAL_TRAIN_GPU=A100-40GB .venv/bin/modal run --detach modal_app.py::train \
  --config-name lewm_masked --data reacher \
  --subdir reacher/lewm_masked_e100_invw03 \
  --overrides "early_stopping.enabled=false trainer.max_epochs=100 loss.masked.inverse_weight=0.3" \
  --no-wait

MODAL_TRAIN_GPU=A100-40GB .venv/bin/modal run --detach modal_app.py::train \
  --config-name lewm_masked --data reacher \
  --subdir reacher/lewm_masked_e100_warm5 \
  --overrides "early_stopping.enabled=false trainer.max_epochs=100 loss.masked.inverse_warmup_epochs=5" \
  --no-wait

MODAL_TRAIN_GPU=A100-40GB .venv/bin/modal run --detach modal_app.py::train \
  --config-name lewm_masked --data reacher \
  --subdir reacher/lewm_masked_e100_invw03_warm5 \
  --overrides "early_stopping.enabled=false trainer.max_epochs=100 loss.masked.inverse_weight=0.3 loss.masked.inverse_warmup_epochs=5" \
  --no-wait
```

#### Inference

```bash
## Current e100 later-checkpoint checks launched immediately:
.venv/bin/modal run --detach modal_app.py::evaluate --config-name reacher \
  --policy reacher/lewm_masked_e100/lewm_masked_epoch_17 \
  --overrides "eval.num_eval=200 eval.env_batch_size=10 output.save_video=false output.filename=lewm_masked_e100_ep17_reacher_n200.txt" \
  --no-wait

.venv/bin/modal run --detach modal_app.py::evaluate --config-name reacher \
  --policy reacher/lewm_base_e100/lewm_epoch_15 \
  --overrides "eval.num_eval=200 eval.env_batch_size=10 output.save_video=false output.filename=lewm_base_e100_ep15_reacher_n200.txt" \
  --no-wait

# Pending once epoch-10 object checkpoints exist:
.venv/bin/modal run --detach modal_app.py::evaluate --config-name reacher \
  --policy reacher/lewm_masked_e100_invw03/lewm_masked_epoch_10 \
  --overrides "eval.num_eval=200 eval.env_batch_size=10 output.save_video=false output.filename=lewm_masked_e100_invw03_ep10_reacher_n200.txt" \
  --no-wait
```

#### Diagnostics

```bash
# Metrics to inspect: train_step/inverse_weight_effective, train_step/emb_std,
# train_step/forward_loss, train_step/inverse_loss, train_step/loss, optim/lr,
# and optim/grad_norm.
```

#### Eval

```bash
# Pending for all three arms at epoch 10, then optionally epoch 20/30.
```

#### Compare

```bash
# Compare each ablation against reacher/lewm_masked_e100_ep8_reacher_n200.txt
# and against the old reacher/lewm_masked_reacher_n200_results.txt.
```

### Artifacts

* Train metadata: `reacher/lewm_masked_e100_{invw03,warm5,invw03_warm5}/run_metadata.json`
* GPU metrics: `reacher/lewm_masked_e100_{invw03,warm5,invw03_warm5}/metrics.jsonl`
* Predictions: not applicable.
* Eval summary: pending result files in `reacher/`; current-run evals launched for
  `lewm_masked_e100_ep17_reacher_n200.txt` and `lewm_base_e100_ep15_reacher_n200.txt`.
* Per-dialogue eval: not applicable.

### Result

Base:

* Current failed candidate: `reacher/lewm_masked_e100` epoch-10 eval 39.5%.

Candidate:

* `reacher/lewm_masked_e100_invw03`: stopped at epoch 6; no n=200 eval. Last/epoch means show
  inverse loss near 0.83 with loss scaled by 0.3 and `emb_std` around 0.05 by epoch 6.
* `reacher/lewm_masked_e100_warm5`: stopped at epoch 6; no n=200 eval. Warmup-only collapses
  `emb_std` to about 0.004-0.005 and inverse loss stays near 1.0.
* `reacher/lewm_masked_e100_invw03_warm5`: stopped at epoch 6; no n=200 eval. Combined warmup+0.3
  also collapses `emb_std` to about 0.002 and inverse loss stays near 1.0.
* Current-run later evals completed before cleanup: `reacher/lewm_base_e100/lewm_epoch_15` scored
  82.0%; `reacher/lewm_masked_e100/lewm_masked_epoch_17` recovered to 69.5%.

Delta:

* Wins / losses: baseline reaches the released-checkpoint range by epoch 15 (82.0% vs released
  81.5%). Masked recovers from 39.5% at ep10-ish to 69.5% at epoch 17, but still trails baseline by
  12.5 pts and the old 10-epoch masked run by only +1.5 pts.
* Recall collapse? Warmup-only and warmup+0.3 look collapse-adjacent by `emb_std` and inverse loss.
  Lower weight alone does not show the same immediate collapse signal through epoch 6.

### Decision

All active Modal apps were stopped on request to avoid unnecessary 100-epoch compute. Do not run
pure forward warmup as the next default; it appears to allow collapse before the inverse task turns on.
If resuming an ablation, `inverse_weight=0.3` is the only one that looks plausible from the partial
metrics. Baseline e100 has now effectively reproduced the released Reacher level.

### Notes

* Code change: `loss.masked.inverse_warmup_epochs` defaults to 0, so existing masked config behavior
  is unchanged unless overridden.
* Logged metric added: `train_step/inverse_weight_effective`.
* Validation: `pytest -q tests/test_masked_transition.py` passed 11 tests; `py_compile` passed for
  `train.py`, `modal_app.py`, `jepa.py`, and `module.py`; Hydra compose shows the default warmup as 0.
* Modal apps: `ap-PP2TJyWvHW5uzKmlHdkqWC` (`invw03`), `ap-2TxPIgSHX5FDJqaOjoY9dB` (`warm5`),
  `ap-emKWMX0DTyr99XJPtM0wBz` (`invw03_warm5`).
* Current-run eval apps: `ap-oVf28SmEs2tXdBjJyqvR3p` (`lewm_masked_e100` epoch 17),
  `ap-7uwDVMsFaGLLk5evJBFkIp` (`lewm_base_e100` epoch 15).
* Stopped apps on 2026-06-05 at about 21:53 local time: `ap-8C3MaNgaNglBocVS24WU26`,
  `ap-RuaV1zvoPD7RkOn1XfLw5r`, `ap-PP2TJyWvHW5uzKmlHdkqWC`, `ap-2TxPIgSHX5FDJqaOjoY9dB`, and
  `ap-emKWMX0DTyr99XJPtM0wBz`; verified zero tasks remain.
* JSONL logging fix after inspecting partial metrics: `utils.JsonlMetricsCallback` now retains
  `*_weight` / `*_weight_effective` scalars from train-step outputs, so future resumed warmup runs
  will record the actual inverse-weight ramp locally.

## 2026-06-06 — Reacher e100 milestone resumes with automatic evals

### Intent

Continue the schedule-corrected Reacher runs only to useful milestones, not to 100 epochs. Preserve
`trainer.max_epochs=100` so the LR schedule remains the reproduction-correct 100-epoch schedule, but
stop gracefully with `runtime.stop_after_epoch` and automatically evaluate the target epoch checkpoint.

### Commands

#### Train

```bash
# Ablations: resume from epoch 6, stop after epoch 10, then eval epoch 10.
MODAL_TRAIN_GPU=A100-40GB .venv/bin/modal run --detach modal_app.py::train_then_evaluate \
  --config-name lewm_masked --data reacher \
  --subdir reacher/lewm_masked_e100_invw03 \
  --overrides "early_stopping.enabled=false trainer.max_epochs=100 runtime.stop_after_epoch=10 resume.allow_config_mismatch=true loss.masked.inverse_weight=0.3" \
  --eval-config-name reacher \
  --eval-policy reacher/lewm_masked_e100_invw03/lewm_masked_epoch_10 \
  --eval-overrides "eval.num_eval=200 eval.env_batch_size=10 output.save_video=false output.filename=lewm_masked_e100_invw03_ep10_reacher_n200.txt" \
  --no-wait

MODAL_TRAIN_GPU=A100-40GB .venv/bin/modal run --detach modal_app.py::train_then_evaluate \
  --config-name lewm_masked --data reacher \
  --subdir reacher/lewm_masked_e100_warm5 \
  --overrides "early_stopping.enabled=false trainer.max_epochs=100 runtime.stop_after_epoch=10 resume.allow_config_mismatch=true loss.masked.inverse_warmup_epochs=5" \
  --eval-config-name reacher \
  --eval-policy reacher/lewm_masked_e100_warm5/lewm_masked_epoch_10 \
  --eval-overrides "eval.num_eval=200 eval.env_batch_size=10 output.save_video=false output.filename=lewm_masked_e100_warm5_ep10_reacher_n200.txt" \
  --no-wait

MODAL_TRAIN_GPU=A100-40GB .venv/bin/modal run --detach modal_app.py::train_then_evaluate \
  --config-name lewm_masked --data reacher \
  --subdir reacher/lewm_masked_e100_invw03_warm5 \
  --overrides "early_stopping.enabled=false trainer.max_epochs=100 runtime.stop_after_epoch=10 resume.allow_config_mismatch=true loss.masked.inverse_weight=0.3 loss.masked.inverse_warmup_epochs=5" \
  --eval-config-name reacher \
  --eval-policy reacher/lewm_masked_e100_invw03_warm5/lewm_masked_epoch_10 \
  --eval-overrides "eval.num_eval=200 eval.env_batch_size=10 output.save_video=false output.filename=lewm_masked_e100_invw03_warm5_ep10_reacher_n200.txt" \
  --no-wait

# Original e100 runs: resume, stop after epoch 30, then eval epoch 30.
MODAL_TRAIN_GPU=A100-40GB .venv/bin/modal run --detach modal_app.py::train_then_evaluate \
  --config-name lewm_masked --data reacher \
  --subdir reacher/lewm_masked_e100 \
  --overrides "early_stopping.enabled=false trainer.max_epochs=100 runtime.stop_after_epoch=30 resume.allow_config_mismatch=true" \
  --eval-config-name reacher \
  --eval-policy reacher/lewm_masked_e100/lewm_masked_epoch_30 \
  --eval-overrides "eval.num_eval=200 eval.env_batch_size=10 output.save_video=false output.filename=lewm_masked_e100_ep30_reacher_n200.txt" \
  --no-wait

MODAL_TRAIN_GPU=A100-40GB .venv/bin/modal run --detach modal_app.py::train_then_evaluate \
  --config-name lewm --data reacher \
  --subdir reacher/lewm_base_e100 \
  --overrides "early_stopping.enabled=false trainer.max_epochs=100 runtime.stop_after_epoch=30 resume.allow_config_mismatch=true" \
  --eval-config-name reacher \
  --eval-policy reacher/lewm_base_e100/lewm_epoch_30 \
  --eval-overrides "eval.num_eval=200 eval.env_batch_size=10 output.save_video=false output.filename=lewm_base_e100_ep30_reacher_n200.txt" \
  --no-wait
```

#### Inference

```bash
# Inference is run automatically by modal_app.py::train_then_evaluate after training stops.
```

#### Diagnostics

```bash
.venv/bin/modal app list --json
.venv/bin/modal app logs <APP_ID> --since 30m --tail 200
.venv/bin/modal volume ls multi-future-lewm-cache /reacher/<RUN_DIR>
```

#### Eval

```bash
# Expected output files:
# reacher/lewm_masked_e100_invw03/lewm_masked_e100_invw03_ep10_reacher_n200.txt
# reacher/lewm_masked_e100_warm5/lewm_masked_e100_warm5_ep10_reacher_n200.txt
# reacher/lewm_masked_e100_invw03_warm5/lewm_masked_e100_invw03_warm5_ep10_reacher_n200.txt
# reacher/lewm_masked_e100/lewm_masked_e100_ep30_reacher_n200.txt
# reacher/lewm_base_e100/lewm_base_e100_ep30_reacher_n200.txt
```

#### Compare

```bash
# Compare to:
# - masked e100 ep10-ish: 39.5%
# - masked e100 ep17: 69.5%
# - base e100 ep15: 82.0%
# - released reacher/lewm: 81.5%
```

### Artifacts

* Train metadata: each run's `run_metadata.json`.
* GPU metrics: each run's `metrics.jsonl`.
* Predictions: not applicable.
* Eval summary: expected output files listed above.
* Per-dialogue eval: not applicable.

### Result

Base:

* `reacher/lewm_base_e100` epoch 15: 82.0%.
* `reacher/lewm_base_e100` epoch 30: 81.5%.

Candidate:

* `reacher/lewm_masked_e100` epoch 17: 69.5%.
* `reacher/lewm_masked_e100` epoch 30: 75.0%.
* `reacher/lewm_masked_e100_invw03` epoch 10: 60.0%.
* `reacher/lewm_masked_e100_warm5` epoch 10: 10.0%.
* `reacher/lewm_masked_e100_invw03_warm5` epoch 10: 9.5%.

Delta:

* Wins / losses: base e100 reproduces the released checkpoint level by epoch 30 (81.5%). Masked e100
  improves substantially across training (39.5% ep10-ish → 69.5% ep17 → 75.0% ep30) but remains 6.5
  pts below base at epoch 30. Lower inverse weight alone does not rescue epoch-10 masked (60.0%).
  Warmup-only and warmup+0.3 collapse to near-random Reacher performance.
* Recall collapse? Warmup arms are effectively rejected by both early `emb_std`/inverse-loss traces and
  epoch-10 evals. Full-weight masked does not collapse in planning by epoch 30, but still trails SIGReg.

### Decision

Keep the schedule-corrected Reacher diagnosis: baseline recovers to released level under the 100-epoch
LR schedule. For masked, do not use pure forward warmup on Reacher. The result supports continuing to
study full-weight masked over longer training only with milestone stops/evals, but the current Reacher
verdict at the corrected schedule is SIGReg > masked.

### Notes

* Code changes: `runtime.stop_after_epoch` and `StopAfterEpochCallback`; Modal
  `train_then_evaluate` wrapper runs train and then eval sequentially.
* Validation: `py_compile train.py utils.py modal_app.py`, masked tests pass (11), Hydra accepts
  `runtime.stop_after_epoch=10`, and Modal exposes `train_then_evaluate --help`.
* Active apps launched 2026-06-06: `ap-6BvSItA3Jr6C4jwnPY2aYQ` (`invw03`),
  `ap-hVBT4CeDSYZQFkWqeuzaEm` (`warm5`), `ap-2kTU82KAkqYnFozLV2rJmH` (`invw03_warm5`),
  `ap-10eauXpGRa1L4nCTJRnQNT` (`masked_e100` to ep30), `ap-QzdGKaBe2Yix0tAPVTtI22`
  (`base_e100` to ep30).
* Logs confirm checkpoint restore and active training for at least `invw03`, `warm5`, and
  `base_e100`; app list shows all five active with one task each.
* Cleanup/status check: stopped all active Modal apps on 2026-06-06 and verified zero active tasks.
  The active app IDs observed at cleanup were `ap-VE3V5aU7yNVdDwUQPeJOzW`,
  `ap-zkd7TYdKw4bPnBnMLMYkrE`, and `ap-HidYwQBeRvWbVpMTnlzl93`; all transitioned to stopped.

## 2026-06-07 — Reacher e100 epoch-50 continuation

### Intent

Continue the schedule-corrected Reacher baseline and masked runs from epoch 30 to epoch 50 to test
whether either keeps improving. Keep the 100-epoch LR schedule by leaving `trainer.max_epochs=100`,
but stop at `runtime.stop_after_epoch=50` and automatically evaluate the epoch-50 checkpoint.

### Commands

#### Train

```bash
MODAL_TRAIN_GPU=A100-40GB .venv/bin/modal run --detach modal_app.py::train_then_evaluate \
  --config-name lewm_masked --data reacher \
  --subdir reacher/lewm_masked_e100 \
  --overrides "early_stopping.enabled=false trainer.max_epochs=100 runtime.stop_after_epoch=50 resume.allow_config_mismatch=true" \
  --eval-config-name reacher \
  --eval-policy reacher/lewm_masked_e100/lewm_masked_epoch_50 \
  --eval-overrides "eval.num_eval=200 eval.env_batch_size=10 output.save_video=false output.filename=lewm_masked_e100_ep50_reacher_n200.txt" \
  --no-wait

MODAL_TRAIN_GPU=A100-40GB .venv/bin/modal run --detach modal_app.py::train_then_evaluate \
  --config-name lewm --data reacher \
  --subdir reacher/lewm_base_e100 \
  --overrides "early_stopping.enabled=false trainer.max_epochs=100 runtime.stop_after_epoch=50 resume.allow_config_mismatch=true" \
  --eval-config-name reacher \
  --eval-policy reacher/lewm_base_e100/lewm_epoch_50 \
  --eval-overrides "eval.num_eval=200 eval.env_batch_size=10 output.save_video=false output.filename=lewm_base_e100_ep50_reacher_n200.txt" \
  --no-wait
```

#### Inference

```bash
# Inference runs automatically after each train job reaches epoch 50.
```

#### Diagnostics

```bash
.venv/bin/modal app list --json
.venv/bin/modal app logs ap-AJgZPYxZkWbJegmQMs49vL --since 20m --tail 200
.venv/bin/modal app logs ap-lCuLsmXyuvCoY3leXNeSN0 --since 20m --tail 200
```

#### Eval

```bash
# Expected output files:
# reacher/lewm_masked_e100/lewm_masked_e100_ep50_reacher_n200.txt
# reacher/lewm_base_e100/lewm_base_e100_ep50_reacher_n200.txt
```

#### Compare

```bash
# Compare to epoch 30:
# base e100 ep30: 81.5%
# masked e100 ep30: 75.0%
```

### Artifacts

* Train metadata: `reacher/lewm_{masked_,}e100/run_metadata.json`
* GPU metrics: `reacher/lewm_{masked_,}e100/metrics.jsonl`
* Predictions: not applicable.
* Eval summary: epoch-50 result files listed above.
* Per-dialogue eval: not applicable.

### Result

Base:

* `reacher/lewm_base_e100` epoch 30: 81.5%.

Candidate:

* `reacher/lewm_masked_e100` epoch 30: 75.0%.
* Epoch 50 evals: pending.

Delta:

* Wins / losses: pending.
* Recall collapse? pending epoch-50 metrics/eval.

### Decision

Launched both jobs. This is a bounded continuation only; do not let them run all the way to 100 unless
epoch-50 results justify another milestone.

### Notes

* App IDs: masked `ap-AJgZPYxZkWbJegmQMs49vL`, base `ap-lCuLsmXyuvCoY3leXNeSN0`.
* Modal app list shows both active with one task each.
* Masked logs confirm checkpoint restore from epoch 30 and active epoch-31 training.
* Base logs confirm checkpoint restore from epoch 30 and active epoch-31 training.

---

## 2026-06-08 — PushT masked-transition e100 status check

### Intent

Answer why PushT masked-transition is still below the matched LeWM baseline, and whether the next move
should be a better LR/schedule training. This checks the already-running schedule-corrected PushT run
`pusht/lewm_masked_e100` rather than starting a duplicate.

Baseline: matched PushT LeWM `pusht/lewm_base`, n=200 eval seed 42, CEM-30, **93.5%**.
Earlier masked result: `pusht/lewm_masked`, same eval protocol, **85.0%**.

### Commands

#### Diagnostics

```bash
.venv/bin/modal app list --json
.venv/bin/modal app logs ap-QdSga0HL4l3vAyknyuZ5GW --since 2h --tail 240
.venv/bin/modal volume ls multi-future-lewm-cache /pusht/lewm_masked_e100
.venv/bin/modal volume get multi-future-lewm-cache /pusht/lewm_masked_e100/lewm_masked_e100_ep10_pusht_n200.txt /tmp/pusht_masked_e100/
.venv/bin/modal volume get multi-future-lewm-cache /pusht/lewm_masked_e100/lewm_masked_e100_ep15_pusht_n200.txt /tmp/pusht_masked_e100/
.venv/bin/modal volume get multi-future-lewm-cache /pusht/lewm_masked_e100/metrics.jsonl /tmp/pusht_masked_e100/
```

#### Eval

```bash
# Existing completed evals inspected:
# policy=pusht/lewm_masked_e100/lewm_masked_epoch_10
# policy=pusht/lewm_masked_e100/lewm_masked_epoch_15
# Both use config-name=pusht, eval.num_eval=200, eval.env_batch_size=10,
# output.save_video=false, seed=42, CEM n_steps=30.
```

#### Compare

```bash
# Compare against pusht/lewm_base_pusht_n200.txt and pusht/lewm_masked_pusht_n200.txt.
```

### Artifacts

* Train metadata: `pusht/lewm_masked_e100/run_metadata.json`
* GPU metrics: `pusht/lewm_masked_e100/metrics.jsonl`
* Eval summaries:
  * `pusht/lewm_masked_e100/lewm_masked_e100_ep10_pusht_n200.txt`
  * `pusht/lewm_masked_e100/lewm_masked_e100_ep15_pusht_n200.txt`
  * `pusht/lewm_base_pusht_n200.txt`
  * `pusht/lewm_masked_pusht_n200.txt`

### Result

Base:

* `pusht/lewm_base`: **93.5%** (187/200).

Candidate:

* `pusht/lewm_masked` under the old 10-epoch schedule: **85.0%**.
* `pusht/lewm_masked_e100` epoch 10: **78.0%**.
* `pusht/lewm_masked_e100` epoch 15: **81.0%**.
* Active app `ap-QdSga0HL4l3vAyknyuZ5GW` is still training `pusht/lewm_masked_e100`, around epoch 20/100,
  with LR about `4.56e-5`, `emb_std` around `0.15`, forward loss around `0.0013-0.0015`,
  inverse loss around `0.05-0.08`, and roughly 55h ETA for all 100 epochs.

Delta:

* Wins / losses: the corrected 100-epoch LR schedule has not rescued PushT masked so far. It trails the
  old masked checkpoint at both inspected milestones and is far below matched LeWM.
* Recall collapse? no obvious collapse from metrics; the latent spread is stable rather than near zero.

### Decision

Do not launch another plain better-schedule PushT masked run. One is already running. If we keep spending
compute, use bounded milestones (epoch 20/25/30) and stop unless the curve sharply improves. The likely
problem is objective mismatch on contact-heavy PushT: the inverse-dynamics loss can be noisy or
contact-dependent, shaping action-informative latents that are not well aligned with CEM terminal latent MSE.

### Notes

* PushT's matched LeWM baseline is already near the released result scale, unlike the earlier Reacher
  reproduction gap; this leaves much less room for a schedule fix to explain the masked deficit.
* More useful ablations than another plain e100 duplicate: lower `inverse_weight`, weak SIGReg hybrid,
  or inverse-weight annealing after early anti-collapse.

---

## 2026-06-09 — AC-CPC (contrastive anti-collapse) on PushT, matched 10-epoch schedule

### Intent

Test whether a contrastive (InfoNCE) forward objective — a third SIGReg-free anti-collapse mechanism,
distinct from masked transition's inverse-dynamics term — can beat SIGReg on PushT, the one reported
task where masked loses. Hypothesis: InfoNCE grades the predicted future against *other trajectories'*
real futures (not the encoder's own latent, as MSE does), so it should be forced to encode the block
that MSE drops. Compared against the matched 10-epoch base (`pusht/lewm_base`, 93.5%) and masked
(`pusht/lewm_masked`, 85.5%). AC-CPC uses **no inverse head** and **no SIGReg** — InfoNCE only.

### Commands

#### Train

```bash
modal run --detach modal_app.py::train --config-name lewm_accpc --data pusht \
  --subdir pusht/lewm_accpc --overrides "early_stopping.enabled=false" --no-wait
```

#### Diagnostics (linear state probe, identical script across all 3 models)

```bash
modal run --detach modal_app.py::probe --policy pusht/lewm_accpc/lewm_accpc_epoch_10 --dataset pusht_expert_train --n 4000 --no-wait
modal run --detach modal_app.py::probe --policy pusht/lewm_masked/lewm_masked_epoch_10 --dataset pusht_expert_train --n 4000 --no-wait
modal run --detach modal_app.py::probe --policy pusht/lewm_base/lewm_epoch_10 --dataset pusht_expert_train --n 4000 --no-wait
```

#### Eval

```bash
modal run --detach modal_app.py::evaluate --config-name pusht \
  --policy pusht/lewm_accpc/lewm_accpc_epoch_10 \
  --overrides "eval.num_eval=200 eval.env_batch_size=10 output.save_video=false output.filename=lewm_accpc_ep10_pusht_n200.txt" --no-wait
```

### Artifacts

* Checkpoint: `pusht/lewm_accpc/lewm_accpc_epoch_10_object.ckpt`
* Eval summary: `pusht/lewm_accpc/lewm_accpc_ep10_pusht_n200.txt` (success_rate 62.5, 125/200)
* Code: `jepa.JEPA.normalize_latents`, `train.ac_cpc_forward`, `config/train/lewm_accpc.yaml`,
  `tests/test_ac_cpc.py` (9 tests; suite 47 green). Paper: `paper/ActionConditionedCPC.md`.

### Result

Linear-probe R² (n=4000, ridge α=1, 7-dim privileged state) and planning SR (n=200, seed 42, CEM-30):

| Probe R² | base/SIGReg | masked | AC-CPC |
|----------|------------:|-------:|-------:|
| agent x [0] | 0.9472 | 0.9993 | 0.7751 |
| agent y [1] | 0.9429 | 0.9989 | 0.7785 |
| block x [2] | 0.9833 | 0.9262 | 0.7117 |
| block y [3] | 0.9755 | 0.9469 | 0.7679 |
| block orient [4] | 0.7905 | 0.5081 | 0.6546 |
| vel [5] | 0.1248 | 0.1436 | 0.1158 |
| vel [6] | 0.1430 | 0.1945 | 0.1445 |
| **MEAN** | **0.7010** | **0.6739** | **0.5640** |
| **Planning SR** | **93.5%** | **85.5%** | **62.5%** |

Base: 93.5%. Candidate (AC-CPC): **62.5%**. Delta vs base **−31.0**, vs masked **−23.0**.

* Wins / losses: **AC-CPC LOSES** — worst of the three on PushT planning.
* Recall collapse? No (not collapsed: emb_std healthy, cpc started exactly at chance floor
  log(382)=5.945 and trained below it; the latent is non-degenerate, just less physically decodable).

### Decision

**Reject** clean AC-CPC as a PushT win. Mechanism worked as designed (block-orientation R² 0.51→0.65 vs
masked) but it is **not a rebalance** — AC-CPC is uniformly lower-R² (agent 0.78, block-pos 0.74),
yielding the lowest mean decodability and worst planning. Earlier "rebalance" read was corrected by the
full 3-way table.

### Notes

* **Key finding: PushT planning SR is monotonic in mean linear-probe R²** (0.701→93.5, 0.674→85.5,
  0.564→62.5). SIGReg wins by producing the most *balanced, high-decodability* latent (strong on agent
  AND block position AND orientation at once); the SIGReg-free alternatives fail not by dropping one
  variable but by yielding a less physically-organized latent overall — AC-CPC most of all. InfoNCE can
  separate trajectories (low cpc loss) without producing a good metric space for CEM goal-matching.
* Caveat: matched 10-epoch schedule; contrastive methods can converge slower, but −31pp is too large to
  be only under-training. A longer-schedule rerun is possible but low-priority given the gap.
* Backward-compat bug found+fixed mid-run: `self.normalize_latents` raised AttributeError when loading
  any pre-AC-CPC pickled JEPA (base/masked); fixed with `getattr(..., False)` + regression test
  (commit 48cbcea). All re-probes used the fixed code.
* Open follow-up (per `paper/ActionConditionedCPC.md`): `ac_cpc_inverse` hybrid (InfoNCE + inverse
  dynamics) as a separate mode — not yet implemented; the clean ablation (this entry) had to land first.

---

<!-- The two dated entries below were maintained on main as a parallel ledger and
merged here during the 2026-06-10 PR consolidation; kept verbatim for completeness
alongside the branch's structured entries above (which cover the same runs in the
AGENTS.md format). -->

## 2026-06-08 — PushT masked-transition schedule check (existing e100 run)

### Intent

Inspect why PushT masked-transition is below the matched LeWM baseline, and test whether the known
`trainer.max_epochs` / cosine-LR coupling is a plausible explanation. This uses an already-running
schedule-corrected PushT masked run, `pusht/lewm_masked_e100`, rather than launching a fresh run.

Baseline: matched PushT LeWM, `pusht/lewm_base`, n=200 eval seed 42, CEM-30, **93.5%**.
Earlier masked-transition result: `pusht/lewm_masked`, same eval protocol, **85.0%**.

### Commands

#### Diagnostics

```bash
.venv/bin/modal app list --json
.venv/bin/modal app logs ap-QdSga0HL4l3vAyknyuZ5GW --since 2h --tail 240
.venv/bin/modal volume ls multi-future-lewm-cache /pusht/lewm_masked_e100
.venv/bin/modal volume get multi-future-lewm-cache /pusht/lewm_masked_e100/lewm_masked_e100_ep10_pusht_n200.txt /tmp/pusht_masked_e100/
.venv/bin/modal volume get multi-future-lewm-cache /pusht/lewm_masked_e100/lewm_masked_e100_ep15_pusht_n200.txt /tmp/pusht_masked_e100/
.venv/bin/modal volume get multi-future-lewm-cache /pusht/lewm_masked_e100/metrics.jsonl /tmp/pusht_masked_e100/
```

#### Eval

```bash
# Already completed before this inspection:
# policy=pusht/lewm_masked_e100/lewm_masked_epoch_10
# policy=pusht/lewm_masked_e100/lewm_masked_epoch_15
# Both use config-name=pusht, eval.num_eval=200, eval.env_batch_size=10,
# output.save_video=false, seed=42, CEM n_steps=30.
```

#### Compare

```bash
# Compare against:
# pusht/lewm_base_pusht_n200.txt
# pusht/lewm_masked_pusht_n200.txt
```

### Artifacts

* Train metadata: `pusht/lewm_masked_e100/run_metadata.json`
* GPU metrics: `pusht/lewm_masked_e100/metrics.jsonl`
* Eval summaries:
  * `pusht/lewm_masked_e100/lewm_masked_e100_ep10_pusht_n200.txt`
  * `pusht/lewm_masked_e100/lewm_masked_e100_ep15_pusht_n200.txt`
  * `pusht/lewm_base_pusht_n200.txt`
  * `pusht/lewm_masked_pusht_n200.txt`

### Result

Base:

* `pusht/lewm_base`: **93.5%** (187/200).

Candidate:

* `pusht/lewm_masked` under the old 10-epoch schedule: **85.0%**.
* `pusht/lewm_masked_e100` epoch 10: **78.0%**.
* `pusht/lewm_masked_e100` epoch 15: **81.0%**.
* Active app `ap-QdSga0HL4l3vAyknyuZ5GW` is still training `pusht/lewm_masked_e100`, around epoch 20/100,
  with LR still high at about `4.56e-5`, `emb_std` around `0.15`, forward loss around `0.0013-0.0015`,
  inverse loss around `0.05-0.08`, and roughly 55h ETA for a full 100 epochs.

Delta:

* Wins / losses: the corrected 100-epoch LR schedule does **not** explain away the PushT masked loss so far.
  It is worse than the old masked run at epoch 10 and still behind it at epoch 15, while the baseline is already
  near the released PushT level.
* Recall collapse? no obvious collapse from the latest metrics: `emb_std` is stable around `0.15`, not near zero.

### Decision

Do not launch another plain "better schedule" PushT masked run. One is already running, and early/midpoint
checkpoints are below both the matched LeWM baseline and the old 10-epoch masked result. The more likely cause is
objective mismatch on PushT: inverse dynamics is noisy/contact-dependent and can shape a latent that is
action-informative without being well matched to CEM terminal latent MSE. Continue only to bounded milestones
(e.g. epoch 20/25/30) if we want the curve, then stop rather than paying for a full 100 epochs by default.

### Notes

* PushT differs from Reacher: the matched 10-epoch baseline was already strong (93.5%) and close to the released
  checkpoint result scale, so the LR-schedule reproduction fix has less room to rescue the baseline.
* A better next ablation is not a higher-LR/full-schedule duplicate; it is a loss-balance or geometry test:
  lower `inverse_weight`, add weak SIGReg, or anneal inverse weight down after early anti-collapse.

---

## 2026-06-09 — PushT AC-CPC diagnosis after failed clean contrastive run

### Intent

Diagnose why the AC-CPC PushT run in the masked-transition worktree failed. AC-CPC was intended to replace both
SIGReg and inverse dynamics with a contrastive action-conditioned future-identification objective. Compared
against the matched PushT LeWM baseline and masked-transition checkpoint on the same n=200, seed=42 CEM-30 eval.

### Commands

#### Train

```bash
modal run --detach modal_app.py::train --config-name lewm_accpc --data pusht \
  --subdir pusht/lewm_accpc --overrides "early_stopping.enabled=false" --no-wait
```

#### Diagnostics

```bash
modal volume get multi-future-lewm-cache /pusht/lewm_accpc/metrics.jsonl /tmp/pusht_accpc/
modal volume get multi-future-lewm-cache /pusht/lewm_accpc/lewm_accpc_ep10_pusht_n200.txt /tmp/pusht_accpc/
```

#### Eval

```bash
modal run --detach modal_app.py::evaluate --config-name pusht \
  --policy pusht/lewm_accpc/lewm_accpc_epoch_10 \
  --overrides "eval.num_eval=200 eval.env_batch_size=10 output.save_video=false output.filename=lewm_accpc_ep10_pusht_n200.txt" --no-wait
```

#### Compare

```bash
# Compare against pusht/lewm_base_pusht_n200.txt and pusht/lewm_masked_pusht_n200.txt.
```

### Artifacts

* Train metadata: `pusht/lewm_accpc/config.yaml`
* GPU metrics: `pusht/lewm_accpc/metrics.jsonl`
* Predictions: n/a
* Eval summary: `pusht/lewm_accpc/lewm_accpc_ep10_pusht_n200.txt`
* Per-dialogue eval: n/a

### Result

Base:

* LeWM/SIGReg: **93.5%** (187/200)
* Masked transition: **85.0%** (170/200)

Candidate:

* AC-CPC: **62.5%** (125/200)
* Training did not collapse: `cpc_loss` fell from about `6.10` to `0.05`, while `emb_std` stayed around `0.07`.

Delta:

* Wins / losses: AC-CPC is −31 pts vs LeWM and −22.5 pts vs masked transition. Paired with LeWM, AC-CPC solved
  **0** episodes LeWM failed, while LeWM solved **62** episodes AC-CPC failed.
* Recall collapse? no. The contrastive objective optimized successfully; the failure is representational/planning
  geometry, not a degenerate constant latent.

### Decision

Reject clean AC-CPC as a PushT improvement. The best explanation is that InfoNCE learned an easy
trajectory-identification geometry that separates batch futures but is not a balanced, physically decodable,
Euclidean latent space for CEM goal matching. Linear probes from the worktree log agree: mean privileged-state
R² drops from LeWM `0.701` to masked `0.674` to AC-CPC `0.564`, matching the planning-SR ordering.

### Notes

* AC-CPC improved block orientation decodability relative to masked transition, but it damaged agent and block
  position decodability more, so it was not a useful rebalancing.
* A longer schedule is lower priority because the contrastive training loss already saturated by epoch 10.
* Better follow-ups are geometry-constrained hybrids: keep plain AC-CPC as the rejected clean ablation, then test
  `ac_cpc_inverse` or weak-SIGReg/variance-regularized CPC separately.

---

## 2026-06-10 — Reacher masked ep40/ep50 trajectory + VICReg variance floor launch

### Intent

Two parallel workstreams launched from the same analysis session:

1. **Reacher ep40/ep50 trajectory** — the `reacher/lewm_masked_e100` and `reacher/lewm_base_e100`
   runs were previously stopped at ep30. The volume already contained ep40 and ep50 checkpoints and
   eval files from prior work; these were retrieved and the full trajectory compiled to determine
   whether masked is still converging or has hit a representational ceiling.

2. **VICReg variance floor** — a new config `lewm_masked_vicreg` adds a per-dimension variance lower
   bound (`relu(gamma - std(z)).mean()`, weight=0.05, gamma=0.1) to the existing masked-transition
   objective. The hypothesis is that the plain masked/inverse objective develops "lazy" latent
   dimensions for state variables with sparse action-coupling (block orientation on PushT, potentially
   joint velocity dims on Reacher), because inverse dynamics can minimize its loss from position
   information alone without encoding those variables. The variance floor forces every dim to carry
   information without constraining the distribution shape (unlike SIGReg). Launched for all four
   tasks at the standard e10 matched protocol.

Baselines to beat:
* **PushT**: SIGReg 93.5%, masked 85.5% → target: masked_vicreg ≥ 93.5%.
* **Cube**: SIGReg 66.0%, masked 79.3% (already winning) → target: maintain ≥ 79%.
* **TwoRoom-Long**: SIGReg 18.0%, masked 28.0% → target: maintain ≥ 28%.
* **Reacher e10**: SIGReg 69.0%, masked 68.0% → target: beat 69%.

### Commands

#### Diagnostics

```bash
# Fetch completed Reacher ep40/ep50 eval results
mkdir -p /tmp/reacher_results
modal volume get multi-future-lewm-cache /reacher/lewm_masked_e100/lewm_masked_e100_ep25_reacher_n200.txt /tmp/reacher_results/
modal volume get multi-future-lewm-cache /reacher/lewm_masked_e100/lewm_masked_e100_ep40_reacher_n200.txt /tmp/reacher_results/
modal volume get multi-future-lewm-cache /reacher/lewm_masked_e100/lewm_masked_e100_ep50_reacher_n200.txt /tmp/reacher_results/
modal volume get multi-future-lewm-cache /reacher/lewm_base_e100/lewm_base_e100_ep40_reacher_n200.txt /tmp/reacher_results/
modal volume get multi-future-lewm-cache /reacher/lewm_base_e100/lewm_base_e100_ep50_reacher_n200.txt /tmp/reacher_results/
```

#### Train

```bash
# VICReg variance floor — all four tasks, e10 matched protocol (A100-40GB)
MODAL_TRAIN_GPU=A100-40GB modal run --detach modal_app.py::train \
  --config-name lewm_masked_vicreg --data pusht \
  --subdir pusht/lewm_masked_vicreg \
  --overrides "early_stopping.enabled=false" --no-wait

MODAL_TRAIN_GPU=A100-40GB modal run --detach modal_app.py::train \
  --config-name lewm_masked_vicreg --data ogb \
  --subdir cube/lewm_masked_vicreg \
  --overrides "early_stopping.enabled=false" --no-wait

MODAL_TRAIN_GPU=A100-40GB modal run --detach modal_app.py::train \
  --config-name lewm_masked_vicreg --data tworoom \
  --subdir tworoom/lewm_masked_vicreg \
  --overrides "early_stopping.enabled=false" --no-wait

MODAL_TRAIN_GPU=A100-40GB modal run --detach modal_app.py::train \
  --config-name lewm_masked_vicreg --data reacher \
  --subdir reacher/lewm_masked_vicreg \
  --overrides "early_stopping.enabled=false" --no-wait
```

#### Diagnostics

```bash
# Reacher linear probes (masked ep50 vs base ep50)
modal run --detach modal_app.py::probe \
  --policy reacher/lewm_masked_e100/lewm_masked_epoch_50 \
  --dataset dmc/reacher_random \
  --overrides "--state-key observation" --no-wait

modal run --detach modal_app.py::probe \
  --policy reacher/lewm_base_e100/lewm_epoch_50 \
  --dataset dmc/reacher_random \
  --overrides "--state-key observation" --no-wait
```

#### Eval

```bash
# After VICReg training completes — one eval per task, n=200, seed=42, CEM-30
modal run --detach modal_app.py::evaluate --config-name pusht \
  --policy pusht/lewm_masked_vicreg/lewm_masked_vicreg_epoch_10 \
  --overrides "eval.num_eval=200 eval.env_batch_size=10 output.save_video=false output.filename=lewm_masked_vicreg_ep10_pusht_n200.txt" --no-wait

modal run --detach modal_app.py::evaluate --config-name ogb \
  --policy cube/lewm_masked_vicreg/lewm_masked_vicreg_epoch_10 \
  --overrides "eval.num_eval=200 eval.env_batch_size=10 output.save_video=false output.filename=lewm_masked_vicreg_ep10_cube_n200.txt" --no-wait

modal run --detach modal_app.py::evaluate --config-name tworoom \
  --policy tworoom/lewm_masked_vicreg/lewm_masked_vicreg_epoch_10 \
  --overrides "eval.num_eval=200 eval.env_batch_size=10 output.save_video=false output.filename=lewm_masked_vicreg_ep10_tworoom_n200.txt" --no-wait

modal run --detach modal_app.py::evaluate --config-name reacher \
  --policy reacher/lewm_masked_vicreg/lewm_masked_vicreg_epoch_10 \
  --overrides "eval.num_eval=200 eval.env_batch_size=10 output.save_video=false output.filename=lewm_masked_vicreg_ep10_reacher_n200.txt" --no-wait
```

### Artifacts

* VICReg train metadata: `pusht/lewm_masked_vicreg/`, `cube/lewm_masked_vicreg/`,
  `tworoom/lewm_masked_vicreg/`, `reacher/lewm_masked_vicreg/`
* Reacher probe logs: Modal app logs for ap-NUV1wRxfXIBCI8zMcxXWsm (masked) and
  ap-kNIxNsxH7QW9Inag2MhFDr (base)
* VICReg eval summaries (pending): `lewm_masked_vicreg_ep10_*_n200.txt` in each task subdir

### Result

**Reacher ep40/ep50 trajectory (full):**

| epoch | base | masked |
|-------|------|--------|
| ep15 | 82.0% | 69.5% |
| ep25 | — | 67.5% |
| ep30 | 81.5% | 75.0% |
| ep40 | 78.0% | 79.0% |
| ep50 | **87.5%** | **78.0%** |

Masked has plateaued in the 75–79% band (ep30–ep50), not converging to base. Base spiked to 87.5%
at ep50 (above the paper's reported 86 max — within eval noise at n=200, ±3.5pp). **Reacher masked
is a representational ceiling, not a slow-convergence issue.** This updates the prior conclusion that
"masked is still climbing."

**Reacher probe results:** pending (ap-NUV1wRxfXIBCI8zMcxXWsm / ap-kNIxNsxH7QW9Inag2MhFDr).

**VICReg eval results:** pending training completion (~1h per task on A100).

Base:

SIGReg baseline (e10): PushT 93.5%, Cube 66.0%, TwoRoom-Long 18.0%, Reacher 69.0%.

Candidate:

Masked VICReg (e10): pending.

Delta:

* Pending eval results.

### Decision

Pending. Collect VICReg evals and probe results, then update this entry.

### Notes

* **VICReg implementation**: `VarianceReg` added to `module.py`; `masked_transition_forward` in
  `train.py` reads `cfg.loss.variance_reg.{weight,gamma}` and adds the term only when weight > 0,
  so existing masked runs are unaffected. `lewm_masked_vicreg.yaml` inherits from `lewm_masked` with
  `variance_reg.weight=0.05, gamma=0.1`.
* **Reacher ceiling update**: ep40 (79.0%) and ep50 (78.0%) confirm masked has stopped improving
  around 75–79%. Probe results needed to diagnose whether this is a dim-collapse or covariance issue.
* Modal apps: PushT ap-Pst051HUwkP91TiGZIoseP, Cube ap-zhtxnFpdAH5WmXZXKCWv2s,
  TwoRoom ap-IRT31tsNut6IDbEo2wtsWQ, Reacher ap-HX5oZni33bSn5XSUjPkFD1.

---

## 2026-06-11 — BYOL-WM (all 4 tasks)

### Intent

Replace inverse dynamics + SIGReg with a BYOL-style temporal prediction objective to fix the
lazy-dimension collapse on PushT (block orientation) and Reacher (state[1]). The core hypothesis:

1. **EMA target encoder** (τ=0.996): breaks the symmetry that allows trivial constant-output
   collapse. Because the target is a lagged EMA copy, the online encoder must track the moving
   target — a collapsed online network diverges from the EMA path.
2. **Stochastic horizon k ~ Uniform{1,...,5}**: at k=5, predicting z_{t+5} from context requires
   encoding block orientation (changes contact in ~3 steps on PushT) and Reacher state[1] (
   determines where the end-effector ends up at t+5). At k=1, those variables barely matter.
   Averaging over k forces them into the representation.
3. No inverse dynamics, no SIGReg, no VICReg. Anti-collapse is structural.

Compare against: SIGReg baseline (PushT 93.5%, Cube 66.0%, TwoRoom 18.0%, Reacher 69.0%)
and masked transition (PushT 85.5%, Cube 79.0%, TwoRoom 28.0%, Reacher 62.5%).

### Commands

#### Train

```bash
.venv/bin/modal run modal_app.py::train --config-name lewm_byol --data pusht   --subdir pusht/lewm_byol   --overrides "early_stopping.enabled=false" --no-wait
.venv/bin/modal run modal_app.py::train --config-name lewm_byol --data cube    --subdir cube/lewm_byol    --overrides "early_stopping.enabled=false" --no-wait
.venv/bin/modal run modal_app.py::train --config-name lewm_byol --data tworoom --subdir tworoom/lewm_byol --overrides "early_stopping.enabled=false" --no-wait
.venv/bin/modal run modal_app.py::train --config-name lewm_byol --data reacher --subdir reacher/lewm_byol --overrides "early_stopping.enabled=false" --no-wait
```

#### Eval

```bash
.venv/bin/modal run modal_app.py::eval --config-name pusht  --checkpoint pusht/lewm_byol/lewm_byol_ep10.ckpt   --subdir pusht/lewm_byol   --no-wait
.venv/bin/modal run modal_app.py::eval --config-name cube   --checkpoint cube/lewm_byol/lewm_byol_ep10.ckpt    --subdir cube/lewm_byol    --no-wait
.venv/bin/modal run modal_app.py::eval --config-name tworoom --checkpoint tworoom/lewm_byol/lewm_byol_ep10.ckpt --subdir tworoom/lewm_byol --no-wait
.venv/bin/modal run modal_app.py::eval --config-name reacher --checkpoint reacher/lewm_byol/lewm_byol_ep10.ckpt --subdir reacher/lewm_byol --no-wait
```

#### Diagnostics (linear probes)

```bash
.venv/bin/modal run modal_app.py::probe --checkpoint pusht/lewm_byol/lewm_byol_ep10.ckpt --subdir pusht/lewm_byol --no-wait
.venv/bin/modal run modal_app.py::probe --checkpoint reacher/lewm_byol/lewm_byol_ep10.ckpt --subdir reacher/lewm_byol --no-wait
```

### Artifacts

* Checkpoints: `checkpoints/pusht/lewm_byol/`, `cube/lewm_byol/`, etc.
* PushT modal app: ap-s2i2Gl8VLiqisM4EScFaJB
* Cube modal app: ap-1UHR3tdGTb5C0IgSAS3R5J
* TwoRoom modal app: ap-Coz2YLJhkjAdBOzeV9Qava
* Reacher modal app: ap-cwqavSc87cPcN8Ehz7eYAF

### Result

Base (SIGReg, e10): PushT 93.5%, Cube 66.0%, TwoRoom 18.0%, Reacher 69.0%.

Candidate (BYOL-WM, e10): pending.

Delta: pending.

### Decision

Pending eval results.

### Notes

* `byol_wm_forward` in `train.py` (line 522): encodes with online encoder + EMA target,
  samples k ~ Uniform{min_k, max_k}, predicts z_{ctx+k} from context, MSE vs target.
* `ema_update` called in `fit` stage only (not val). Momentum=0.996 → half-life ~340 steps.
* Config `lewm_byol.yaml` inherits `lewm_masked_h` → `lewm` → base LeWM config. The `inverse`
  section is inherited but never read since `byol_wm` mode skips `InverseDynamics` construction.
* `rollout_mode` is set to `direct_horizon` for `byol_wm`, so `rollout_direct` / `FutureQueryPredictor`
  is used for CEM planning at eval time (same as `masked_horizon`).

---

## 2026-06-11 — ms-mtm e10 (all 4 tasks)

### Intent

Multi-scale masked trajectory modeling: H-step forward prediction + endpoint inverse dynamics
at random gap k. Tests whether forcing the encoder to recover a_t from (z_t, z_{t+k}, e_k)
at large k fixes the lazy-dimension collapse on PushT (block orientation) and Reacher (state[1]).

At k=5, recovering a_t from a distant endpoint requires encoding slow state variables that
1-step inverse dynamics ignores (block orientation determines which face is pushed in 5 steps).
No SIGReg, no EMA target — anti-collapse is pure task signal from both objectives.

Merged from .claude/worktrees/ms-mtm-experiment into main.

Compare against: SIGReg (PushT 93.5%, Cube 66.0%, TwoRoom 18.0%, Reacher 69.0%)
and masked_transition (PushT 85.5%, Cube 79.0%, TwoRoom 28.0%, Reacher 62.5%).

### Commands

#### Train

```bash
nohup .venv/bin/modal run modal_app.py::train --config-name lewm_ms_mtm --data pusht   --subdir pusht/lewm_ms_mtm   --overrides "early_stopping.enabled=false" > /tmp/msmtm_pusht.log 2>&1 &
nohup .venv/bin/modal run modal_app.py::train --config-name lewm_ms_mtm --data ogb     --subdir cube/lewm_ms_mtm    --overrides "early_stopping.enabled=false" > /tmp/msmtm_cube.log 2>&1 &
nohup .venv/bin/modal run modal_app.py::train --config-name lewm_ms_mtm --data tworoom --subdir tworoom/lewm_ms_mtm --overrides "early_stopping.enabled=false" > /tmp/msmtm_tworoom.log 2>&1 &
nohup .venv/bin/modal run modal_app.py::train --config-name lewm_ms_mtm --data reacher --subdir reacher/lewm_ms_mtm --overrides "early_stopping.enabled=false" > /tmp/msmtm_reacher.log 2>&1 &
```

#### Eval

```bash
.venv/bin/modal run modal_app.py::eval --config-name pusht   --checkpoint pusht/lewm_ms_mtm/lewm_ms_mtm_ep10.ckpt   --subdir pusht/lewm_ms_mtm   --no-wait
.venv/bin/modal run modal_app.py::eval --config-name cube    --checkpoint cube/lewm_ms_mtm/lewm_ms_mtm_ep10.ckpt    --subdir cube/lewm_ms_mtm    --no-wait
.venv/bin/modal run modal_app.py::eval --config-name tworoom --checkpoint tworoom/lewm_ms_mtm/lewm_ms_mtm_ep10.ckpt --subdir tworoom/lewm_ms_mtm --no-wait
.venv/bin/modal run modal_app.py::eval --config-name reacher --checkpoint reacher/lewm_ms_mtm/lewm_ms_mtm_ep10.ckpt --subdir reacher/lewm_ms_mtm --no-wait
```

#### Diagnostics (linear probes — priority tasks)

```bash
.venv/bin/modal run modal_app.py::probe --checkpoint pusht/lewm_ms_mtm/lewm_ms_mtm_ep10.ckpt   --subdir pusht/lewm_ms_mtm   --no-wait
.venv/bin/modal run modal_app.py::probe --checkpoint reacher/lewm_ms_mtm/lewm_ms_mtm_ep10.ckpt --subdir reacher/lewm_ms_mtm --no-wait
```

### Artifacts

* Logs: /tmp/msmtm_{task}.log

### Result

Base (SIGReg e10): PushT 93.5%, Cube 66.0%, TwoRoom 18.0%, Reacher 69.0%.
Candidate (ms-mtm e10): pending.

### Decision

Pending.

### Notes

* `HorizonInverseDynamics` in module.py: takes (z_t, z_{t+k}, e_k) where e_k is a learned
  nn.Embedding of dimension 32. Input dim = 2*latent_dim + 32.
* Training samples one k per batch (random mode); validation averages over all k (all mode).
* Config inherits from lewm (base) — not lewm_masked_h — so it has no sigreg kwargs conflict.
* Smoke test: forward_loss=0.075, inverse_h1..h5 losses all ~1.1 (untrained), ~3 it/s on A100.

---

## 2026-06-12 — TTA feasibility probe (information wall test)

### Intent

Before building a test-time-adaptation planner (per-goal trainable module that updates the
world model during planning, replacing CEM), test the prerequisite: is block orientation
still recoverable from the FROZEN encoder's features, just not routed to the planning CLS
latent? If yes, a frozen-encoder adapter has a real target. If no, frozen-encoder TTA is
impossible (you cannot read information the encoder discarded).

`probe_features.py` compares orientation R² decodable from: cls (projector output, what the
planner uses) vs patch_mean vs patch_grid (patches adaptive-pooled to GxG then flattened).

### Commands

```bash
nohup .venv/bin/modal run modal_app.py::probe_features --policy pusht/lewm_ms_mtm/lewm_ms_mtm_epoch_10 > /tmp/probe_feat_msmtm.log 2>&1 &
nohup .venv/bin/modal run modal_app.py::probe_features --policy pusht/lewm_ms_mtm/lewm_ms_mtm_epoch_10 --overrides "--grid 7 --alpha-patch 50" > /tmp/probe_feat_g7.log 2>&1 &
```

### Result (ms-mtm PushT ep10; orientation = state[4,5,6])

| dim | cls | patch_mean | patch_grid 4x4 | patch_grid 7x7 |
|---|---|---|---|---|
| state[4] | 0.560 | 0.213 | 0.567 | 0.549 |
| state[5] | 0.315 | 0.044 | 0.137 | 0.138 |
| state[6] | 0.392 | 0.050 | 0.143 | 0.134 |

Patch tokens contain NO MORE orientation than CLS; for the most-collapsed dims (state[5,6])
they are WORSE. Finer grid (7x7) does not help — rules out the pooling confound.

### Decision

INFORMATION WALL CONFIRMED. The encoder discarded block orientation; it is unrecoverable from
any frozen-encoder feature (CLS or patches, any resolution). Therefore:

- Frozen-encoder test-time adaptation CANNOT recover orientation — no adapter on frozen
  features has a target to route. This kills the frozen-encoder TTA variant cheaply (~40 min,
  no planner built).
- TTA could only work by fine-tuning the ENCODER itself with a pixel-grounded loss
  (orientation lives in raw pixels, not in the encoder's collapsed features). But that is
  strictly dominated by adding the same orientation-bearing objective in PRETRAINING: cheaper,
  stable, amortized. And orientation is a UNIFORM need (every PushT goal needs it), not a
  task-specific one — so per-episode adaptation is the wrong tool. Adaptation helps when
  different test instances need different things; here all instances need the same missing info.

### Next

Fix the objective at pretraining so the encoder keeps orientation: add an observation/patch
reconstruction or decode-to-observation auxiliary alongside the masked/forward losses. Re-probe
orientation; only then is any planning-time adaptation/amortization worth building on top.

### Notes

- BYOL PushT e6 run still in flight (background, ~3h left). Now a confirmatory side-run: its
  e2 probe already showed orientation collapsed (state[5,6] = 0.05), consistent with the wall.

---

## 2026-06-13 — Reconstruction auxiliary (PushT viability)

### Intent

The TTA feasibility probe proved orientation is gone from the frozen encoder (info wall), so
the fix must force the encoder to KEEP orientation during pretraining, with a target that
contains it. Reconstruction does: decoding the planning latent back to the frame makes block
orientation un-discardable (you can't reconstruct the block's pixels without its angle).

Add a ConvDecoder (latent -> 64x64 frame) and an MSE reconstruction term to
masked_transition_forward, gated by cfg.loss.reconstruction.weight. Base = masked
(sigreg off, inverse on). Viability question: does orientation R2 (state[5,6]) recover
from masked's ~0.3 toward SIGReg's ~0.79?

### Commands

#### Train
```bash
nohup .venv/bin/modal run modal_app.py::train --config-name lewm_masked_recon --data pusht --subdir pusht/lewm_masked_recon --overrides "early_stopping.enabled=false trainer.max_epochs=5 resume.mode=never resume.prevent_restart_existing=false" > /tmp/recon_pusht.log 2>&1 &
```
#### Probe / Eval
```bash
.venv/bin/modal run modal_app.py::probe --policy pusht/lewm_masked_recon/lewm_masked_recon_epoch_5
.venv/bin/modal run modal_app.py::evaluate --config-name pusht --policy pusht/lewm_masked_recon/lewm_masked_recon_epoch_5
```

### Result

Wiring verified: recon_loss finite + decreasing (3.69 -> 2.77 over first 225 steps), forward
0.024, inverse 1.0, emb_std 0.128. 5.3 it/s, e5 ETA ~3.7h. recon dominates total at weight=1.0.

Orientation probe (state[4,5,6]) vs references — PENDING checkpoints:
| dim | SIGReg | masked | ms-mtm | recon e? |
|---|---|---|---|---|
| state[4] | ~0.79 | 0.51 | 0.560 | pending |
| state[5] | — | — | 0.315 | pending |
| state[6] | — | — | 0.392 | pending |

### Decision

Pending orientation probe on first checkpoints.

### Notes

- ConvDecoder in module.py; JEPA.reconstruct + JEPA.recon_target in jepa.py; decoder built
  in train.py only when reconstruction.weight > 0 (all other modes unchanged).
- BYOL conclusively dead: e6 orientation state[5,6]=0.03/0.07, PushT success 42% (worst of all).
  Third independent mechanism to collapse orientation, consistent with the info wall.

### Result (UPDATE — orientation trajectory)

Orientation probe across epochs (state[4,5,6]):
| dim | e2 | e3 | e4 | e5 | SIGReg target |
|---|---|---|---|---|---|
| state[4] | 0.527 | 0.558 | 0.567 | 0.569 | ~0.79 |
| state[5] | 0.095 | 0.106 | 0.106 | 0.105 | (high) |
| state[6] | 0.205 | 0.197 | 0.199 | 0.198 | (high) |

FLAT. Orientation does not recover even with recon_loss dominating the total. Position dims
(state[0-3]) stay high (0.93-0.99). recon_loss kept decreasing → the decoder learns a blurry
reconstruction that nails blob POSITION but averages over the block's ANGLE. Classic pixel-MSE
high-frequency blur: MSE is minimized by the mean, which discards precise orientation.

### Decision

Pixel-MSE reconstruction (64x64) is NOT viable for restoring orientation. Verdict: reject this
variant. Deeper pattern across all experiments: orientation is dropped by EVERY objective that
has an easy approximate solution (inverse dynamics → agent suffices; forward → position
suffices; MSE recon → blurry blob suffices). It is retained only when the objective forbids
discarding variance outright (SIGReg keeps it at 0.79; that is WHY SIGReg works — not because
orientation helps any task, but because it forbids rank collapse).

### Next (fork)

To give reconstruction a fair last shot, the target must have no blur escape for orientation:
(a) perceptual / frozen-teacher feature reconstruction, or (b) much higher resolution. Else the
pattern says the real mechanism is variance-preservation (proper VICReg covariance, the lighter
cousin of SIGReg), which was only tried weakly before.

---

## 2026-06-15 — Consistent PushT probe sweep (paper foundation)

### Intent

Lock one probe protocol (n=4000, ridge alpha=1, state[4]=block orientation) across ALL
anti-collapse mechanisms for the characterization paper. Corrects earlier confusion where
state[5,6] (velocities, ~0.12 for everyone incl. SIGReg) were misread as orientation.

### Result — block orientation = state[4], and mean R^2 over 7 dims

| mechanism | orientation state[4] | mean R^2 |
|---|---|---|
| SIGReg (lewm_base) | 0.791 | 0.701 |
| AC-CPC | 0.655 | 0.564 |
| BYOL (e6) | 0.638 | 0.647 |
| recon (e5) | 0.569 | 0.679 |
| ms-mtm (e10) | 0.560 | 0.737 |
| MTM/masked | 0.508 | 0.674 |

### Key finding

SIGReg's advantage is SPECIFICALLY block orientation, NOT overall decodability. ms-mtm has the
HIGHEST mean R^2 (0.737 > SIGReg 0.701) yet underencodes orientation (0.560 vs 0.791). So the
dynamics-derived methods do not learn worse representations on average — they selectively
underencode the passively-controlled object orientation, which only a variance/distribution
constraint (SIGReg) forces in. This REFUTES the draft's "planning success monotonic in mean R^2"
claim and replaces it with a sharper, dimension-specific result: the gap is orientation, the
canonical passively-controlled goal variable.

n=200 success-rate evals for all 6 mechanisms in flight to complete the table.

### n=200 SR added — and it REFUTES the orientation-bottleneck thesis

Complete consistent table (PushT, n=200, state[4]=orientation):
| mechanism | planner | orient R^2 | mean R^2 | SR n=200 |
|---|---|---|---|---|
| SIGReg | AR | 0.791 | 0.701 | 93.0 |
| MTM (inverse) | AR | 0.508 | 0.674 | 89.0 |
| reconstruction | AR | 0.569 | 0.679 | 87.5 |
| AC-CPC | AR | 0.655 | 0.564 | 64.5 |
| ms-mtm | direct-H | 0.560 | 0.737 | 62.0 |
| BYOL | direct-H | 0.638 | 0.647 | 42.0 |

TWO findings overturn the planned "passive-state/orientation bottleneck" thesis:

1. ORIENTATION R^2 DOES NOT PREDICT SR. MTM has the WORST orientation (0.508) yet 2nd-best
   SR (89.0). BYOL has good orientation (0.638) yet worst SR (42.0). The within-data evidence
   refutes "orientation underencoding causes planning failure." Probe decodability is decoupled
   from planning utility.

2. PLANNER CONFOUND. The two worst performers (ms-mtm 62, BYOL 42) use the direct_horizon
   planner; the strong ones (SIGReg 93, MTM 89, recon 87.5) use the AR planner. Their low SR
   conflates anti-collapse mechanism with planner architecture. Not apples-to-apples.

Holding planner fixed (AR + MSE latent): SIGReg 93, MTM 89, recon 87.5 are all close; AC-CPC
64.5 is the contrastive outlier. So inverse-dynamics MTM is COMPETITIVE with SIGReg on PushT
DESPITE much worse orientation probing. recon (87.5) also competitive.

IMPLICATION: the planned characterization-by-orientation paper is not supported. Honest
alternatives: (A) "inverse-dynamics anti-collapse is competitive with SIGReg for planning,
including on PushT at n=200" (the original MTM thesis, now better supported); (B) "probe
decodability != planning utility" (MTM plans well despite worse geometry — cautions against
probe-as-proxy). Needs planner held fixed for a clean mechanism comparison.

### COMPLETE n=200 multi-task table (SIGReg vs MTM, locked protocol)

| task | SIGReg | MTM | Delta | note |
|---|---|---|---|---|
| Cube | 66.0 | 79.5 | +13.5 | SIGReg base matches paper (66.0) |
| TwoRoom | 84.0 | 91.5 | +7.5 | SIGReg=lewm_base_e100 (schedule caveat) |
| PushT | 93.0 | 89.0 | -4.0 | AR planner both |
| Reacher | 68.5 | 63.0 | -5.5 | single seed |

MTM net avg Delta = +2.9. Wins (Cube, TwoRoom) exceed losses (PushT, Reacher). Mixed but
net-positive — a competitive distribution-free anti-collapse. Two transient Modal network
blips (Errno 61) on cube_sigreg/tworoom_mtm required one relaunch each; final values clean.

PAPER DIRECTION LOCKED (thesis C): (1) inverse-dynamics anti-collapse is competitive with
SIGReg for planning without distribution matching; (2) probe decodability does NOT predict
planning utility (MTM worst orientation 0.508 yet 2nd-best PushT SR 89; within AR family
orientation R^2 is non-monotonic in SR). All data now at one protocol (n=200, seed 42, CEM 30;
probes n=4000 alpha=1, orientation=state[4]).

---

## 2026-06-18 — Reacher masked-transition seed-collapse investigation

### Intent

Investigate why two newer Reacher seed experiments scored only about 11-14% SR after an initial
masked-transition Reacher result around 63-68% SR. Main question: eval/checkpoint bug, or real
objective instability?

Compare against the matched SIGReg Reacher seeds and the original masked seed-3072 run.

### Commands

#### Train

```bash
# Exact train launches were not captured in local shell history.
# Reconstructed from saved configs:
.venv/bin/modal run modal_app.py::train --config-name lewm_masked --data reacher \
  --subdir reacher/lewm_masked_s1 --overrides "early_stopping.enabled=false seed=1"
.venv/bin/modal run modal_app.py::train --config-name lewm_masked --data reacher \
  --subdir reacher/lewm_masked_s2 --overrides "early_stopping.enabled=false seed=2"
.venv/bin/modal run modal_app.py::train --config-name lewm --data reacher \
  --subdir reacher/lewm_base_s1 --overrides "early_stopping.enabled=false seed=1"
.venv/bin/modal run modal_app.py::train --config-name lewm --data reacher \
  --subdir reacher/lewm_base_s2 --overrides "early_stopping.enabled=false seed=2"
```

#### Inference

```bash
# From Modal app logs:
python eval.py --config-name=reacher policy=reacher/lewm_masked_s1 eval.num_eval=200
python eval.py --config-name=reacher policy=reacher/lewm_masked_s2 eval.num_eval=200
python eval.py --config-name=reacher policy=reacher/lewm_base_s1 eval.num_eval=200
python eval.py --config-name=reacher policy=reacher/lewm_base_s2 eval.num_eval=200
```

#### Diagnostics

```bash
.venv/bin/modal volume get multi-future-lewm-cache /reacher/lewm_masked_s1/{config.yaml,checkpoint_state.json,metrics.jsonl,events.jsonl} /tmp/reacher_investigation/
.venv/bin/modal volume get multi-future-lewm-cache /reacher/lewm_masked_s2/{config.yaml,checkpoint_state.json,metrics.jsonl,events.jsonl} /tmp/reacher_investigation/
.venv/bin/modal app logs ap-Ji5OxTd3Vcb4NKQUihMHqq --tail 300
.venv/bin/modal app logs ap-tsLzzIVH68Q4qH8txgsBHl --tail 300
```

#### Eval

```bash
# No additional successful verification eval was launched during this investigation.
# Attempted nohup verification commands produced empty local logs and no Modal apps, so ignored.
```

#### Compare

```bash
# Parsed metrics/events/result files under /tmp/reacher_investigation.
```

### Artifacts

* Train metadata: `reacher/lewm_masked_s{1,2}/run_metadata.json`
* GPU metrics: `reacher/lewm_masked_s{1,2}/metrics.jsonl`
* Eval logs: Modal apps `ap-Ji5OxTd3Vcb4NKQUihMHqq` (masked s1), `ap-tsLzzIVH68Q4qH8txgsBHl` (masked s2), `ap-xoCeT9RHk7WOhafCQefJyI` (base s1), `ap-PwbkXBwEyL6kj7YXo9boJL` (base s2)
* Eval summary: shared `reacher/dmc_results.txt` contains only `masked_s2` from the concurrent default-output evals; `masked_s1` result is present in app logs only.

### Result

Base:

* SIGReg `reacher/lewm_base`: 69.0% (138/200)
* SIGReg `reacher/lewm_base_s1`: 69.0% (138/200), Modal log
* SIGReg `reacher/lewm_base_s2`: 68.5% (137/200), Modal log

Candidate:

* Masked `reacher/lewm_masked`: 68.0% (136/200) in unique result file; 63.0% (126/200) in earlier default `dmc_results.txt` run.
* Masked `reacher/lewm_masked_s1`: 11.5% (23/200), Modal log.
* Masked `reacher/lewm_masked_s2`: 13.5% (27/200), Modal log and shared `dmc_results.txt`.

Delta:

* Wins / losses: masked seed-3072 ties the base, but seeds 1 and 2 catastrophically lose. Three-seed masked mean is about 31% if using the 68% seed-3072 result, versus about 68.8% for the matched SIGReg seeds.
* Recall collapse? yes for masked s1/s2. Final validation `emb_std`: s1 `0.000119`, s2 `0.000500`; forward loss `~1e-6`; inverse loss `~0.9997`. The inverse head is predicting the normalized action mean and no longer prevents collapse. Seed 3072 did not collapse: final validation `emb_std≈0.0876`, inverse loss `≈0.8329`.

### Decision

Reject the previous "Reacher tie" as a robust masked-transition claim. It was a single lucky/noncollapsed
seed. The low 11-14% results are real collapsed checkpoints, not a Reacher eval bug. Treat plain
`lewm_masked` on Reacher as seed-unstable unless a collapse-prevention change is added.

Next changes should target collapse prevention/early detection: weak variance or weak SIGReg hybrid,
collapse-gated checkpointing, or lower/annealed inverse dynamics that does not let the forward self-MSE
collapse dominate. Future Reacher evals should use exact checkpoint stems plus unique filenames, e.g.
`policy=reacher/lewm_masked_s1/lewm_masked_epoch_10` and
`output.filename=lewm_masked_s1_ep10_reacher_n200.txt`.

### Notes

* The matched SIGReg seeds scoring 69.0/68.5 under the same eval command style rule out a global eval harness issue.
* Config diffs show masked s1/s2 differ from each other only by `seed`, `subdir`, and W&B id.
* Checkpoint states show both bad masked runs reached epoch 10 / global step 127,960; they were not partial runs.
* Tooling issue found: concurrent Reacher evals used the default `output.filename=dmc_results.txt` and `save_video=true`. Modal Volume last-writer behavior caused only one of the concurrent masked seed appends to be retained in `dmc_results.txt`. This is an artifact-recording bug, not the cause of low SR.
* Tooling risk: `AutoCostModel` directory policies pick the newest `*_object.ckpt` by filesystem ctime, not epoch number. Use exact stems for confirmatory evals.

---

## 2026-06-18 — Matched e100 epoch-30 SIGReg vs MTM sweep (in progress)

### Intent

Build a fair longer-budget table for SIGReg vs masked-transition MTM across all four tasks using
the same 100-epoch LR schedule and the same epoch-30 evaluation point. This corrects schedule mixing
between earlier e10 and e100 evidence.

### Commands

#### Train

```bash
.venv/bin/modal run modal_app.py::train --config-name lewm --data pusht \
  --subdir pusht/lewm_base_e100 \
  --overrides "trainer.max_epochs=100 runtime.stop_after_epoch=30 early_stopping.enabled=false"

.venv/bin/modal run modal_app.py::train --config-name lewm --data ogb \
  --subdir cube/lewm_base_e100 \
  --overrides "trainer.max_epochs=100 runtime.stop_after_epoch=30 early_stopping.enabled=false"

.venv/bin/modal run modal_app.py::train --config-name lewm_masked --data ogb \
  --subdir cube/lewm_masked_e100 \
  --overrides "trainer.max_epochs=100 runtime.stop_after_epoch=30 early_stopping.enabled=false"
```

#### Inference

```bash
.venv/bin/modal run modal_app.py::evaluate --config-name pusht \
  --policy pusht/lewm_masked_e100/lewm_masked_epoch_30 --overrides "eval.num_eval=200"

.venv/bin/modal run modal_app.py::evaluate --config-name tworoom \
  --policy tworoom/lewm_base_e100/lewm_epoch_30 --overrides "eval.num_eval=200"

.venv/bin/modal run modal_app.py::evaluate --config-name tworoom \
  --policy tworoom/lewm_masked_e100/lewm_masked_epoch_30 --overrides "eval.num_eval=200"

.venv/bin/modal run modal_app.py::evaluate --config-name reacher \
  --policy reacher/lewm_base_e100/lewm_epoch_30 --overrides "eval.num_eval=200"

.venv/bin/modal run modal_app.py::evaluate --config-name reacher \
  --policy reacher/lewm_masked_e100/lewm_masked_epoch_30 --overrides "eval.num_eval=200"
```

#### Diagnostics

```bash
tail -40 /tmp/e100_*.log
tail -40 /tmp/tr100_*.log
.venv/bin/modal app list | head -60
```

#### Eval

```bash
# Pending after gap-fill training completes:
# cube/lewm_base_e100/lewm_epoch_30
# cube/lewm_masked_e100/lewm_masked_epoch_30
# pusht/lewm_base_e100/lewm_epoch_30
```

#### Compare

```bash
# Assemble final 8-cell table:
# {PushT, Cube, TwoRoom, Reacher} x {SIGReg base, MTM masked}, ep30, n=200.
```

### Artifacts

* Local eval logs: `/tmp/e100_pusht_mtm.log`, `/tmp/e100_tworoom_sig.log`, `/tmp/e100_tworoom_mtm.log`, `/tmp/e100_reacher_sig.log`, `/tmp/e100_reacher_mtm.log`
* Local train logs: `/tmp/tr100_pusht_sig.log`, `/tmp/tr100_cube_sig.log`, `/tmp/tr100_cube_mtm.log`
* PushT SIGReg resumed detached log/app after preemption: `/tmp/tr100_pusht_sig_detach.log`, app `ap-BQI4AkN0lDB1adYe5eveDz` (later stopped after SIGTERM)
* PushT SIGReg tmux resume log/app: `/tmp/tr100_pusht_sig_tmux.log`, tmux session `e100_pusht_sig`, app `ap-fmdghWRPkOM94m11MSsHjd`
* Active/completed Modal apps:
  * PushT MTM eval: `ap-wj6mF8N6cMHJv1EguItf4t`
  * TwoRoom SIGReg eval: `ap-qnQUm8b4VVmOR3PsLM3gGn`
  * TwoRoom MTM eval: `ap-4RS4lu5HhsBPpiincctGdp`
  * Reacher SIGReg eval: `ap-vuzllOpqUa99GGm2E02ABm`
  * Reacher MTM eval: `ap-SPSb4c83EmX9rOpr47VWs4`
  * PushT SIGReg training: `ap-quvhaaPLSHlOm40hVStF4f` (preempted), resumed as `ap-BQI4AkN0lDB1adYe5eveDz` (stopped after SIGTERM), now resumed as `ap-fmdghWRPkOM94m11MSsHjd`
  * Cube SIGReg training: `ap-4DOTZ7f3m4riFkSz4Q3OX3`
  * Cube MTM training: `ap-J1SlwTqAYYCmkHFEICgIQD`

### Result

Base:

* TwoRoom SIGReg epoch 30: **84.0%** (168/200), app `ap-qnQUm8b4VVmOR3PsLM3gGn`.
* Reacher SIGReg epoch 30: **83.5%** (167/200), app `ap-vuzllOpqUa99GGm2E02ABm`.
* PushT SIGReg and Cube SIGReg pending gap-fill training.

Candidate:

* PushT MTM epoch 30: **85.0%** (170/200), app `ap-wj6mF8N6cMHJv1EguItf4t`.
* TwoRoom MTM epoch 30: **87.5%** (175/200), app `ap-4RS4lu5HhsBPpiincctGdp`.
* Reacher MTM epoch 30: **77.0%** (154/200), completed in app `ap-SPSb4c83EmX9rOpr47VWs4`.
* Cube MTM pending gap-fill training.

Delta:

* TwoRoom MTM is +3.5 over SIGReg at epoch 30 under this n=200 eval.
* Reacher MTM is -6.5 below SIGReg at epoch 30 under this n=200 eval.
* PushT and Cube matched deltas pending until gap-fill trainings finish and the missing SIGReg/Cube evals run.
* Recall collapse? no evidence of Reacher MTM collapse at epoch 30 from the completed eval result.

### Decision

Keep monitoring. Do not launch additional e100 seeds until the first matched ep30 table lands; current
gap-fill runs are long and already consume active Modal capacity.

### Notes

* Current eval launches used only `eval.num_eval=200`; they did not override `eval.env_batch_size=10`,
  `output.save_video=false`, or unique filenames. This is inefficient and creates extra video artifacts,
  but the in-progress evals are already far enough along that stopping them would waste more compute.
* At the first status check, the remaining existing-checkpoint evals were about 50% through the 50-step
  eval budget. Gap-fill training status: PushT SIGReg fresh epoch 1, Cube SIGReg resumed epoch 12,
  Cube MTM resumed epoch 17.
* 2026-06-18 10:28 IST status check: Modal app list still shows seven active one-task apps. The four
  unfinished existing-checkpoint eval logs are still progressing around 50% with no errors visible.
  The three gap-fill trainings are writing live training progress: PushT SIGReg epoch 1, Cube SIGReg
  epoch 12, Cube MTM epoch 17. Watcher output files remain empty because they only write after their
  full eval/train groups finish.
* E100 multi-seed recommendation: do a staged expansion, not a full matrix immediately. Finish the
  current single-seed matched table first, then add extra Reacher e100 seeds for both SIGReg and MTM
  because Reacher's e10 result was seed-collapse-sensitive. Add PushT seeds only if the e100 gap is
  small or paper-critical; leave Cube/TwoRoom single-seed unless their matched e100 deltas become
  ambiguous.
* 2026-06-18 14:26 IST status check: all five existing-checkpoint evals finished successfully with
  watcher output:
  PushT MTM 85.0, TwoRoom SIGReg 84.0, TwoRoom MTM 87.5, Reacher SIGReg 83.5,
  Reacher MTM 77.0. PushT SIGReg training app `ap-quvhaaPLSHlOm40hVStF4f` was preempted after saving
  at step 28,000. A short attached resume `ap-JQLK7X2ACg6Rl5SHGlhSMH` verified checkpoint restore, was
  stopped, and the run was relaunched detached as `ap-BQI4AkN0lDB1adYe5eveDz`. Verified with Modal app
  list showing one detached task and logs showing live training at step 28,175. Cube SIGReg is active
  around epoch 16; Cube MTM is active around epoch 22.
* 2026-06-18 14:32 IST correction: the PushT detached resume `ap-BQI4AkN0lDB1adYe5eveDz` later stopped
  after a Modal-client cancellation/SIGTERM, last visible around step 28,725, and no epoch-30 checkpoint
  exists. A canonical `nohup .venv/bin/modal run ... > /tmp/tr100_pusht_sig_resume_canonical.log 2>&1 &`
  relaunch attempt exited silently with an empty log and no new app. PushT SIGReg remains a gap. Decision:
  do not start PushT extra e100 seeds before the matched PushT SIGReg cell is recovered and evaluated.
  Reacher extra e100 seeds are scientifically useful, but should be launched only after deciding whether
  to spend active capacity before the single-seed matched table is complete.
* 2026-06-18 14:49 IST status correction: PushT SIGReg was relaunched successfully in persistent tmux
  session `e100_pusht_sig` with log `/tmp/tr100_pusht_sig_tmux.log`. Modal app `ap-fmdghWRPkOM94m11MSsHjd`
  is active with one task. Verification: log shows checkpoint restore, `Training started`, and live
  progress at epoch 3 / step 28,175. This uses a normal blocking `modal run` inside tmux, not `--no-wait`.
* A stale local Modal client for old `cube/lewm_ms_mtm` exists, but its log reports `APP_STATE_STOPPED`;
  it is not an active Modal task.

---

## 2026-06-18 — LeWM-style n=50 protocol audit and retry evaluation batch

### Intent

Run the paper-facing LeWM Figure 6 style comparison protocol after auditing the known upstream
paper/config mismatches. Current paper decision: use this `n=50` protocol for
all planning tables/figures to avoid mixing episode counts; older matched
`n=200` results remain internal variance-reduction diagnostics.

### Commands

#### Train

```bash
# No new training in this batch. Uses existing checkpoint stems:
# tworoom/lewm_base_e100/lewm_epoch_30
# tworoom/lewm_masked_e100/lewm_masked_epoch_30
# pusht/lewm_base/lewm_epoch_10
# pusht/lewm_masked/lewm_masked_epoch_10
# reacher/lewm_base/lewm_epoch_10
# reacher/lewm_masked/lewm_masked_epoch_10
# cube/lewm_base/lewm_epoch_10
# cube/lewm_masked/lewm_masked_epoch_10
```

#### Inference

```bash
# Initial launch attempt used the canonical nohup form but the local Modal clients exited,
# causing remote cancellation. Those result files are not trusted.
nohup .venv/bin/modal run modal_app.py::evaluate ... > /tmp/n50_<job>.log 2>&1 &

# Retry launch keeps the parent shell alive with wait while still using the required nohup form.
nohup .venv/bin/modal run modal_app.py::evaluate --config-name tworoom \
  --policy tworoom/lewm_base_e100/lewm_epoch_30 \
  --overrides "eval.num_eval=50 eval.env_batch_size=10 seed=42 output.save_video=false output.filename=n50r1_sigreg_tworoom_released_protocol.txt" \
  > /tmp/n50r1_sigreg_tworoom.log 2>&1 &

nohup .venv/bin/modal run modal_app.py::evaluate --config-name tworoom \
  --policy tworoom/lewm_masked_e100/lewm_masked_epoch_30 \
  --overrides "eval.num_eval=50 eval.env_batch_size=10 seed=42 output.save_video=false output.filename=n50r1_mtm_tworoom_released_protocol.txt" \
  > /tmp/n50r1_mtm_tworoom.log 2>&1 &

nohup .venv/bin/modal run modal_app.py::evaluate --config-name pusht \
  --policy pusht/lewm_base/lewm_epoch_10 \
  --overrides "eval.num_eval=50 eval.env_batch_size=10 seed=42 output.save_video=false output.filename=n50r1_sigreg_pusht_released_protocol.txt" \
  > /tmp/n50r1_sigreg_pusht.log 2>&1 &

nohup .venv/bin/modal run modal_app.py::evaluate --config-name pusht \
  --policy pusht/lewm_masked/lewm_masked_epoch_10 \
  --overrides "eval.num_eval=50 eval.env_batch_size=10 seed=42 output.save_video=false output.filename=n50r1_mtm_pusht_released_protocol.txt" \
  > /tmp/n50r1_mtm_pusht.log 2>&1 &

nohup .venv/bin/modal run modal_app.py::evaluate --config-name reacher \
  --policy reacher/lewm_base/lewm_epoch_10 \
  --overrides "eval.num_eval=50 eval.env_batch_size=10 seed=42 output.save_video=false output.filename=n50r1_sigreg_reacher_released_protocol.txt" \
  > /tmp/n50r1_sigreg_reacher.log 2>&1 &

nohup .venv/bin/modal run modal_app.py::evaluate --config-name reacher \
  --policy reacher/lewm_masked/lewm_masked_epoch_10 \
  --overrides "eval.num_eval=50 eval.env_batch_size=10 seed=42 output.save_video=false output.filename=n50r1_mtm_reacher_released_protocol.txt" \
  > /tmp/n50r1_mtm_reacher.log 2>&1 &

nohup .venv/bin/modal run modal_app.py::evaluate --config-name cube \
  --policy cube/lewm_base/lewm_epoch_10 \
  --overrides "eval.num_eval=50 eval.env_batch_size=10 seed=42 output.save_video=false output.filename=n50r1_sigreg_cube_released_protocol.txt" \
  > /tmp/n50r1_sigreg_cube.log 2>&1 &

nohup .venv/bin/modal run modal_app.py::evaluate --config-name cube \
  --policy cube/lewm_masked/lewm_masked_epoch_10 \
  --overrides "eval.num_eval=50 eval.env_batch_size=10 seed=42 output.save_video=false output.filename=n50r1_mtm_cube_released_protocol.txt" \
  > /tmp/n50r1_mtm_cube.log 2>&1 &

nohup .venv/bin/modal run modal_app.py::evaluate --config-name tworoom_long \
  --policy tworoom/lewm_base_e100/lewm_epoch_30 \
  --overrides "eval.num_eval=50 eval.env_batch_size=10 seed=42 output.save_video=false output.filename=n50r1_sigreg_tworoom_long_stress.txt" \
  > /tmp/n50r1_sigreg_tworoom_long.log 2>&1 &

nohup .venv/bin/modal run modal_app.py::evaluate --config-name tworoom_long \
  --policy tworoom/lewm_masked_e100/lewm_masked_epoch_30 \
  --overrides "eval.num_eval=50 eval.env_batch_size=10 seed=42 output.save_video=false output.filename=n50r1_mtm_tworoom_long_stress.txt" \
  > /tmp/n50r1_mtm_tworoom_long.log 2>&1 &
```

#### Diagnostics

```bash
tail -5 /tmp/n50r1_*.log
.venv/bin/modal app list | head -100
.venv/bin/modal app logs <app-id> --tail 120 --timestamps
ps -ef | rg 'modal run modal_app.py::evaluate|n50r1_' | rg -v rg
```

#### Eval

```bash
# Pulled result files from the Modal volume for durable local artifacts.
.venv/bin/modal volume get multi-future-lewm-cache \
  /tworoom/lewm_masked_e100/n50r1_mtm_tworoom_long_stress.txt \
  /tmp/n50r1_results/ --force

mkdir -p progress/evaluations/lewm-style-n50-r1
cp /tmp/n50r1_results/*.txt progress/evaluations/lewm-style-n50-r1/
```

#### Compare

```bash
# Main n=50 LeWM-style protocol:
# TwoRoom: SIGReg 88.0, MTM 96.0
# PushT: SIGReg 96.0, MTM 90.0
# Reacher: SIGReg 74.0, MTM 74.0
# OGBench-Cube: SIGReg 76.0, MTM 86.0
#
# TwoRoom-long stress protocol:
# n=50 paper-facing result: SIGReg 12.0, MTM 28.0
# older n=200 internal diagnostic: SIGReg 18.0, MTM 28.0
```

### Artifacts

* Protocol ledger: `progress/evaluation-protocol-ledger.md`
* Result summary and raw result files: `progress/evaluations/lewm-style-n50-r1/`
* Local retry logs: `/tmp/n50r1_sigreg_tworoom.log`, `/tmp/n50r1_mtm_tworoom.log`,
  `/tmp/n50r1_sigreg_pusht.log`, `/tmp/n50r1_mtm_pusht.log`,
  `/tmp/n50r1_sigreg_reacher.log`, `/tmp/n50r1_mtm_reacher.log`,
  `/tmp/n50r1_sigreg_cube.log`, `/tmp/n50r1_mtm_cube.log`,
  `/tmp/n50r1_sigreg_tworoom_long.log`, `/tmp/n50r1_mtm_tworoom_long.log`
* Retry app ids:
  * TwoRoom SIGReg: `ap-QhLfi7T8QkBnCar5WGf1NZ`
  * TwoRoom MTM: `ap-AsYtUkDmRxMDmnxEsyyiOT`
  * PushT SIGReg: `ap-WKCFeM0Cgr7OKXQCIWNiSC`
  * PushT MTM: `ap-5yMsph3Whcl5rWyYrpEZWC`
  * Reacher SIGReg: `ap-MX1LSlAqWxdtLjwxrvVQpA`
  * Reacher MTM: `ap-dwrxRjbSxL6FaZzlmRX9ck`
  * Cube SIGReg: `ap-KtJBXKyjT9qeNzYAVAFdzC`
  * Cube MTM: `ap-SGGlxQp7BnY0lgzSNwzLxO`
  * TwoRoom-long SIGReg: `ap-XUd5SrzwHQ6GJiTJgG1Vm9`
  * TwoRoom-long MTM: `ap-y3PG5pVHhlN33HeQvva0mW`

### Result

Base:

SIGReg:

| Task/protocol | Success (%) |
|---|---:|
| TwoRoom, released 25/50 | 88.0 |
| PushT, released 25/50 | 96.0 |
| Reacher, released 25/50 | 74.0 |
| OGBench-Cube, released 25/50 | 76.0 |
| TwoRoom-long, paper-text 100/150 | 12.0 |

Candidate:

MTM:

| Task/protocol | Success (%) |
|---|---:|
| TwoRoom, released 25/50 | 96.0 |
| PushT, released 25/50 | 90.0 |
| Reacher, released 25/50 | 74.0 |
| OGBench-Cube, released 25/50 | 86.0 |
| TwoRoom-long, paper-text 100/150 | 28.0 |

Delta:

* Wins / losses: MTM wins TwoRoom (+8), OGBench-Cube (+10), and TwoRoom-long (+16); ties Reacher; loses PushT (-6).
* Recall collapse? no collapse visible from planning output on these exact checkpoints; Reacher remains seed-sensitive from earlier collapsed masked-transition trainings.

### Decision

Use the `n50r1_*` retry outputs for the paper-facing LeWM-style figure and main
planning table. Do not use the first `n50_*` attempt because Modal logs show
cancellation after local client disconnect. Keep older `n=200` runs as internal
variance-reduction diagnostics, but avoid mixing episode counts in the paper.
Keep TwoRoom-long separate from the Figure 6 standard-suite comparison while
reporting it in the main body as the explicit long-horizon stress test.

### Notes

* Issue audit decisions are captured in `progress/evaluation-protocol-ledger.md`.
* The live retry shell keeps the local Modal clients attached with `wait`; this is necessary in this
  execution environment because plain backgrounded clients were killed after the shell returned.
* At launch verification, seven evals had live `python eval.py` commands and `Tasks=1`. The remaining
  three were attached but queued with `Tasks=0`, consistent with hitting active GPU-task capacity while
  three e100 training jobs were also running.
* Final long-horizon MTM app `ap-y3PG5pVHhlN33HeQvva0mW` completed and stopped with `0` tasks after
  writing `n50r1_mtm_tworoom_long_stress.txt`; the result was pulled from the Modal volume.

---

## 2026-06-18 — n50r2 PushT diagnostics and NoReg collapse control

### Intent

Keep the ICLR paper on one planning protocol (`n=50`, seed 42, CEM 300/30/30)
without throwing away useful diagnostic evidence. This reruns the PushT
alternative anti-collapse mechanisms and the TwoRoom NoReg sanity check under the
same paper-facing episode count and unique-output-file discipline.

### Commands

#### Train

```bash
# No new training. Uses existing checkpoint stems:
# pusht/lewm_accpc/lewm_accpc_epoch_10
# pusht/lewm_masked_recon/lewm_masked_recon_epoch_5
# pusht/lewm_ms_mtm/lewm_ms_mtm_epoch_10
# pusht/lewm_byol/lewm_byol_epoch_6
# tworoom/lewm_noreg/lewm_epoch_10
```

#### Inference

```bash
.venv/bin/modal run --detach --name n50r2-accpc-pusht modal_app.py::evaluate \
  --config-name pusht \
  --policy pusht/lewm_accpc/lewm_accpc_epoch_10 \
  --overrides "eval.num_eval=50 eval.env_batch_size=10 seed=42 output.save_video=false output.filename=n50r2_accpc_pusht_probe_protocol.txt"

.venv/bin/modal run --detach --name n50r2-recon-pusht modal_app.py::evaluate \
  --config-name pusht \
  --policy pusht/lewm_masked_recon/lewm_masked_recon_epoch_5 \
  --overrides "eval.num_eval=50 eval.env_batch_size=10 seed=42 output.save_video=false output.filename=n50r2_recon_pusht_probe_protocol.txt"

.venv/bin/modal run --detach --name n50r2-msmtm-pusht modal_app.py::evaluate \
  --config-name pusht \
  --policy pusht/lewm_ms_mtm/lewm_ms_mtm_epoch_10 \
  --overrides "eval.num_eval=50 eval.env_batch_size=10 seed=42 output.save_video=false output.filename=n50r2_msmtm_pusht_probe_protocol.txt"

.venv/bin/modal run --detach --name n50r2-byol-pusht modal_app.py::evaluate \
  --config-name pusht \
  --policy pusht/lewm_byol/lewm_byol_epoch_6 \
  --overrides "eval.num_eval=50 eval.env_batch_size=10 seed=42 output.save_video=false output.filename=n50r2_byol_pusht_probe_protocol.txt"

.venv/bin/modal run --detach --name n50r2-noreg-tworoom modal_app.py::evaluate \
  --config-name tworoom \
  --policy tworoom/lewm_noreg/lewm_epoch_10 \
  --overrides "eval.num_eval=50 eval.env_batch_size=10 seed=42 output.save_video=false output.filename=n50r2_noreg_tworoom_collapse_protocol.txt"
```

#### Diagnostics

```bash
.venv/bin/modal app list | head -45
.venv/bin/modal app logs <app-id> --tail 120 --timestamps
```

#### Eval

```bash
# Completed detached apps:
# ap-sW3W0tIiJehBHR5tjzWhEy n50r2-accpc-pusht -> 62.0
# ap-xUClE3g4RJSHBEFIAFLouG n50r2-recon-pusht -> 92.0
# ap-KsqViPD65ZKvdaVCqYRqX0 n50r2-msmtm-pusht -> 60.0
# ap-r9TeXCKiVpJTMEZRBOHoCG n50r2-byol-pusht -> 38.0
# ap-NocOhedajRDAlIr7ZHcoTw n50r2-noreg-tworoom -> 34.0
```

### Artifacts

* Eval summaries copied to `progress/evaluations/n50r2-diagnostics/`.
* Local paper figure generator: `paper/latex/images/make_results_lewm_style.py`
* Updated 2x2 Figure 3 asset: `paper/latex/images/results_lewm_style.png`

### Result

Base:

LeWM/SIGReg paper-facing comparison points from `n50r1_*`:

| Task/protocol | Success (%) |
|---|---:|
| TwoRoom, released 25/50 | 88.0 |
| PushT, released 25/50 | 96.0 |

Candidate:

Additional diagnostics:

| Method/check | Task/protocol | Success (%) |
|---|---|---:|
| NoReg | TwoRoom, released 25/50 | 34.0 |
| MTM | TwoRoom, released 25/50 | 96.0 |
| Recon | PushT, released 25/50 | 92.0 |
| MTM | PushT, released 25/50 | 90.0 |
| AC-CPC | PushT, released 25/50 | 62.0 |
| MS-MTM | PushT, released 25/50 | 60.0 |
| BYOL-WM | PushT, released 25/50 | 38.0 |

Delta:

* Wins / losses: NoReg confirms that plain next-latent prediction is far below
  either anti-collapse mechanism on TwoRoom (34 vs 88/96). On PushT, Recon is
  close to SIGReg and MTM, while AC-CPC, MS-MTM, and BYOL-WM are clearly weaker
  under the same `n=50` protocol.
* Recall collapse? NoReg planning is poor enough to keep it only as a collapse
  sanity check; the r2 diagnostics do not change the Reacher seed-collapse caveat
  from earlier entries.

### Decision

Use these results in the paper only as diagnostics. The NoReg row strengthens
the anti-collapse sanity check. The PushT rows support a cautious mechanism
story: linear probes are useful for explaining failures, but they do not order
planning performance by themselves.

### Notes

* Initial attempts with `nohup .venv/bin/modal run --detach ... &` produced empty
  local logs and no visible app. Foreground named `modal run --detach --name ...`
  initialized correctly and showed `Tasks=1`; use named launches in this
  environment when confirming a detached app.
* All result files report `eval.num_eval=50`, `seed=42`, released 25/50
  distance/budget, and `output.save_video=false`.

---

## 2026-06-18 — e100/n=200 paper-table gap-fill launches

### Intent

Finish the clean single-seed e100 epoch-30 SIGReg-vs-MTM table before changing
the ICLR paper from completed n=50 results to n=200 controlled results. The
decision is to use n=200 for the final controlled table if all 8 cells land, but
not to edit `paper/latex/iclr_main.tex` while PushT SIGReg, Cube SIGReg, and Cube
MTM are still pending.

### Commands

#### Train

```bash
.venv/bin/modal run --detach --name e100-pusht-sigreg-ep30-train-eval modal_app.py::train_then_evaluate \
  --config-name lewm \
  --data pusht \
  --subdir pusht/lewm_base_e100 \
  --overrides "trainer.max_epochs=100 runtime.stop_after_epoch=30 early_stopping.enabled=false" \
  --eval-config-name pusht \
  --eval-policy pusht/lewm_base_e100/lewm_epoch_30 \
  --eval-overrides "eval.num_eval=200 eval.env_batch_size=10 output.save_video=false output.filename=e100_sigreg_pusht_ep30_n200.txt"
```

#### Inference

```bash
.venv/bin/modal run --detach --name e100-cube-mtm-ep30-n200 modal_app.py::evaluate \
  --config-name cube \
  --policy cube/lewm_masked_e100/lewm_masked_epoch_30 \
  --overrides "eval.num_eval=200 eval.env_batch_size=10 output.save_video=false output.filename=e100_mtm_cube_ep30_n200.txt"
```

#### Diagnostics

```bash
.venv/bin/modal app list | head -30
.venv/bin/modal app logs ap-2fwxdarYZ3E7nVmzK7R2H8 --tail 80
.venv/bin/modal app logs ap-V4d52gU06XW5C3DhzNFrLN --tail 30
.venv/bin/modal app logs ap-4DOTZ7f3m4riFkSz4Q3OX3 --tail 30
.venv/bin/modal volume ls multi-future-lewm-cache /cube/lewm_masked_e100 | rg 'epoch_29|epoch_30|e100_mtm_cube_ep30_n200|n200'
```

#### Eval

```bash
# Pending outputs:
# cube/lewm_masked_e100/e100_mtm_cube_ep30_n200.txt
# pusht/lewm_base_e100/e100_sigreg_pusht_ep30_n200.txt
#
# Still pending after Cube SIGReg reaches epoch 30:
# .venv/bin/modal run --detach --name e100-cube-sigreg-ep30-n200 modal_app.py::evaluate \
#   --config-name cube \
#   --policy cube/lewm_base_e100/lewm_epoch_30 \
#   --overrides "eval.num_eval=200 eval.env_batch_size=10 output.save_video=false output.filename=e100_sigreg_cube_ep30_n200.txt"
```

#### Compare

```bash
# Current completed e100/n=200 cells before these launches:
# PushT MTM 85.0
# TwoRoom SIGReg 84.0
# TwoRoom MTM 87.5
# Reacher SIGReg 83.5
# Reacher MTM 77.0
```

### Artifacts

* Cube MTM eval app: `ap-2fwxdarYZ3E7nVmzK7R2H8`, expected output
  `cube/lewm_masked_e100/e100_mtm_cube_ep30_n200.txt`.
* PushT SIGReg train+eval app: `ap-V4d52gU06XW5C3DhzNFrLN`, expected output
  `pusht/lewm_base_e100/e100_sigreg_pusht_ep30_n200.txt`.
* Cube SIGReg ongoing training app: `ap-4DOTZ7f3m4riFkSz4Q3OX3`.

### Result

Base:

* PushT SIGReg: pending; app `ap-V4d52gU06XW5C3DhzNFrLN` restored from
  `/tmp/stable_worldmodel/pusht/lewm_base_e100/lewm_weights.ckpt` and is live at
  epoch 9 / step 123,375 as of 2026-06-18 21:48 IST.
* Cube SIGReg: pending; app `ap-4DOTZ7f3m4riFkSz4Q3OX3` is live at epoch 22 /
  step 280,250 as of 2026-06-18 21:48 IST.

Candidate:

* Cube MTM: epoch-30 checkpoint exists and n=200 eval is running in app
  `ap-2fwxdarYZ3E7nVmzK7R2H8`. Logs show `episodes=200`, `batches=20`, batch 1
  started, and app list shows `Tasks=1`.

Delta:

* Wins / losses: pending until the three missing cells land.
* Recall collapse? not assessable from these pending jobs.

### Decision

Do not edit `paper/latex/iclr_main.tex` yet. The paper should stay on completed
n=50 results for now, with the n=200 e100 sweep tracked as the intended final
controlled table once all cells are complete. Do not launch broad extra e100
seeds until this first matched table finishes.

### Notes

* The project canonical `nohup .venv/bin/modal run --detach ... &` form produced
  empty local logs/no visible app for these launches in this shell. Named
  foreground `modal run --detach --name ...` created visible detached apps and
  passed the required app-list/log-progress checks.
* App logs include local-client cancellation markers after the local logging
  client was stopped. PushT logs continued advancing after the marker; Cube MTM
  has not yet printed the next batch line, but the eval loop prints per batch and
  Cube batches can be slow. Keep verifying with app list plus result artifacts.

#### 2026-06-18 21:56 IST correction

The first direct detached relaunches did not survive: `ap-2fwxdarYZ3E7nVmzK7R2H8`
and `ap-V4d52gU06XW5C3DhzNFrLN` both stopped with `0` tasks after local-client
interruption. Their output files were not produced and should not be used.

Relaunched the same work inside persistent tmux sessions so the local Modal
client remains attached:

```bash
tmux new-session -d -s e100_cube_mtm_eval 'cd <repo> && .venv/bin/modal run --detach --name e100-cube-mtm-ep30-n200-r2 modal_app.py::evaluate --config-name cube --policy cube/lewm_masked_e100/lewm_masked_epoch_30 --overrides "eval.num_eval=200 eval.env_batch_size=10 output.save_video=false output.filename=e100_mtm_cube_ep30_n200_r2.txt" > /tmp/e100_cube_mtm_ep30_n200_r2.log 2>&1'

tmux new-session -d -s e100_pusht_sig_train_eval 'cd <repo> && .venv/bin/modal run --detach --name e100-pusht-sigreg-ep30-train-eval-r2 modal_app.py::train_then_evaluate --config-name lewm --data pusht --subdir pusht/lewm_base_e100 --overrides "trainer.max_epochs=100 runtime.stop_after_epoch=30 early_stopping.enabled=false" --eval-config-name pusht --eval-policy pusht/lewm_base_e100/lewm_epoch_30 --eval-overrides "eval.num_eval=200 eval.env_batch_size=10 output.save_video=false output.filename=e100_sigreg_pusht_ep30_n200_r2.txt" > /tmp/e100_pusht_sig_ep30_train_eval_r2.log 2>&1'
```

Verified replacement apps:

* Cube MTM eval: `ap-nwk5uMGsuLDtXNdz01grOv`, `Tasks=1`, log shows
  `episodes=200`, `batches=20`, and live CEM/eval progress in batch 1/20.
  Expected output: `cube/lewm_masked_e100/e100_mtm_cube_ep30_n200_r2.txt`.
* PushT SIGReg train+eval: `ap-7URW93LGChFJWh7MTmYYbB`, `Tasks=1`, log shows
  checkpoint restore and live training at epoch 9 / step 124,175. Expected
  output after epoch 30 eval:
  `pusht/lewm_base_e100/e100_sigreg_pusht_ep30_n200_r2.txt`.
* 2026-06-18 21:57 IST latest check: Cube MTM advanced to eval batch 2/20 and
  PushT SIGReg advanced to epoch 9 / step 124,725. App list still shows Cube
  SIGReg, Cube MTM eval, and PushT SIGReg train+eval each with `Tasks=1`.

---

## 2026-06-18 — e100 epoch-30 multi-seed robustness launch

### Intent

Move from a single-seed e100 epoch-30 comparison to paper-grade robustness:
three training seeds per task/method (`3072,1,2`) with a shared n=200 eval
protocol (`seed=42`, `eval.env_batch_size=10`, no videos). Seed 3072 exists or
is already running from the matched table. This launch adds seeds 1 and 2.

Priority rationale: Reacher must be multi-seed because e10 masked-transition
seeds 1/2 collapsed. TwoRoom should also be multi-seed because it supports the
long-horizon figure. PushT/Cube seeds were launched into persistent tmux sessions
but are currently queued at `Tasks=0`, so they are not yet counted as running.

### Commands

#### Train

All commands use `train_then_evaluate` with `trainer.max_epochs=100`,
`runtime.stop_after_epoch=30`, `early_stopping.enabled=false`, exact seed
overrides, and a single automatic n=200 eval after epoch 30.

```bash
tmux new-session -d -s e100_reacher_base_s1 'cd <repo> && .venv/bin/modal run --detach --name e100-reacher-base-s1-ep30 modal_app.py::train_then_evaluate --config-name lewm --data reacher --subdir reacher/lewm_base_e100_s1 --overrides "trainer.max_epochs=100 runtime.stop_after_epoch=30 early_stopping.enabled=false seed=1" --eval-config-name reacher --eval-policy reacher/lewm_base_e100_s1/lewm_epoch_30 --eval-overrides "eval.num_eval=200 eval.env_batch_size=10 seed=42 output.save_video=false output.filename=e100_sigreg_reacher_s1_ep30_n200.txt" > /tmp/e100_reacher_base_s1_ep30.log 2>&1'

tmux new-session -d -s e100_reacher_base_s2 'cd <repo> && .venv/bin/modal run --detach --name e100-reacher-base-s2-ep30 modal_app.py::train_then_evaluate --config-name lewm --data reacher --subdir reacher/lewm_base_e100_s2 --overrides "trainer.max_epochs=100 runtime.stop_after_epoch=30 early_stopping.enabled=false seed=2" --eval-config-name reacher --eval-policy reacher/lewm_base_e100_s2/lewm_epoch_30 --eval-overrides "eval.num_eval=200 eval.env_batch_size=10 seed=42 output.save_video=false output.filename=e100_sigreg_reacher_s2_ep30_n200.txt" > /tmp/e100_reacher_base_s2_ep30.log 2>&1'

tmux new-session -d -s e100_reacher_mtm_s1 'cd <repo> && .venv/bin/modal run --detach --name e100-reacher-mtm-s1-ep30 modal_app.py::train_then_evaluate --config-name lewm_masked --data reacher --subdir reacher/lewm_masked_e100_s1 --overrides "trainer.max_epochs=100 runtime.stop_after_epoch=30 early_stopping.enabled=false seed=1" --eval-config-name reacher --eval-policy reacher/lewm_masked_e100_s1/lewm_masked_epoch_30 --eval-overrides "eval.num_eval=200 eval.env_batch_size=10 seed=42 output.save_video=false output.filename=e100_mtm_reacher_s1_ep30_n200.txt" > /tmp/e100_reacher_mtm_s1_ep30.log 2>&1'

tmux new-session -d -s e100_reacher_mtm_s2 'cd <repo> && .venv/bin/modal run --detach --name e100-reacher-mtm-s2-ep30 modal_app.py::train_then_evaluate --config-name lewm_masked --data reacher --subdir reacher/lewm_masked_e100_s2 --overrides "trainer.max_epochs=100 runtime.stop_after_epoch=30 early_stopping.enabled=false seed=2" --eval-config-name reacher --eval-policy reacher/lewm_masked_e100_s2/lewm_masked_epoch_30 --eval-overrides "eval.num_eval=200 eval.env_batch_size=10 seed=42 output.save_video=false output.filename=e100_mtm_reacher_s2_ep30_n200.txt" > /tmp/e100_reacher_mtm_s2_ep30.log 2>&1'

tmux new-session -d -s e100_tworoom_base_s1 'cd <repo> && .venv/bin/modal run --detach --name e100-tworoom-base-s1-ep30 modal_app.py::train_then_evaluate --config-name lewm --data tworoom --subdir tworoom/lewm_base_e100_s1 --overrides "trainer.max_epochs=100 runtime.stop_after_epoch=30 early_stopping.enabled=false seed=1" --eval-config-name tworoom --eval-policy tworoom/lewm_base_e100_s1/lewm_epoch_30 --eval-overrides "eval.num_eval=200 eval.env_batch_size=10 seed=42 output.save_video=false output.filename=e100_sigreg_tworoom_s1_ep30_n200.txt" > /tmp/e100_tworoom_base_s1_ep30.log 2>&1'

tmux new-session -d -s e100_tworoom_base_s2 'cd <repo> && .venv/bin/modal run --detach --name e100-tworoom-base-s2-ep30 modal_app.py::train_then_evaluate --config-name lewm --data tworoom --subdir tworoom/lewm_base_e100_s2 --overrides "trainer.max_epochs=100 runtime.stop_after_epoch=30 early_stopping.enabled=false seed=2" --eval-config-name tworoom --eval-policy tworoom/lewm_base_e100_s2/lewm_epoch_30 --eval-overrides "eval.num_eval=200 eval.env_batch_size=10 seed=42 output.save_video=false output.filename=e100_sigreg_tworoom_s2_ep30_n200.txt" > /tmp/e100_tworoom_base_s2_ep30.log 2>&1'

tmux new-session -d -s e100_tworoom_mtm_s1 'cd <repo> && .venv/bin/modal run --detach --name e100-tworoom-mtm-s1-ep30 modal_app.py::train_then_evaluate --config-name lewm_masked --data tworoom --subdir tworoom/lewm_masked_e100_s1 --overrides "trainer.max_epochs=100 runtime.stop_after_epoch=30 early_stopping.enabled=false seed=1" --eval-config-name tworoom --eval-policy tworoom/lewm_masked_e100_s1/lewm_masked_epoch_30 --eval-overrides "eval.num_eval=200 eval.env_batch_size=10 seed=42 output.save_video=false output.filename=e100_mtm_tworoom_s1_ep30_n200.txt" > /tmp/e100_tworoom_mtm_s1_ep30.log 2>&1'

tmux new-session -d -s e100_tworoom_mtm_s2 'cd <repo> && .venv/bin/modal run --detach --name e100-tworoom-mtm-s2-ep30 modal_app.py::train_then_evaluate --config-name lewm_masked --data tworoom --subdir tworoom/lewm_masked_e100_s2 --overrides "trainer.max_epochs=100 runtime.stop_after_epoch=30 early_stopping.enabled=false seed=2" --eval-config-name tworoom --eval-policy tworoom/lewm_masked_e100_s2/lewm_masked_epoch_30 --eval-overrides "eval.num_eval=200 eval.env_batch_size=10 seed=42 output.save_video=false output.filename=e100_mtm_tworoom_s2_ep30_n200.txt" > /tmp/e100_tworoom_mtm_s2_ep30.log 2>&1'

tmux new-session -d -s e100_pusht_base_s1 'cd <repo> && .venv/bin/modal run --detach --name e100-pusht-base-s1-ep30 modal_app.py::train_then_evaluate --config-name lewm --data pusht --subdir pusht/lewm_base_e100_s1 --overrides "trainer.max_epochs=100 runtime.stop_after_epoch=30 early_stopping.enabled=false seed=1" --eval-config-name pusht --eval-policy pusht/lewm_base_e100_s1/lewm_epoch_30 --eval-overrides "eval.num_eval=200 eval.env_batch_size=10 seed=42 output.save_video=false output.filename=e100_sigreg_pusht_s1_ep30_n200.txt" > /tmp/e100_pusht_base_s1_ep30.log 2>&1'

tmux new-session -d -s e100_pusht_base_s2 'cd <repo> && .venv/bin/modal run --detach --name e100-pusht-base-s2-ep30 modal_app.py::train_then_evaluate --config-name lewm --data pusht --subdir pusht/lewm_base_e100_s2 --overrides "trainer.max_epochs=100 runtime.stop_after_epoch=30 early_stopping.enabled=false seed=2" --eval-config-name pusht --eval-policy pusht/lewm_base_e100_s2/lewm_epoch_30 --eval-overrides "eval.num_eval=200 eval.env_batch_size=10 seed=42 output.save_video=false output.filename=e100_sigreg_pusht_s2_ep30_n200.txt" > /tmp/e100_pusht_base_s2_ep30.log 2>&1'

tmux new-session -d -s e100_pusht_mtm_s1 'cd <repo> && .venv/bin/modal run --detach --name e100-pusht-mtm-s1-ep30 modal_app.py::train_then_evaluate --config-name lewm_masked --data pusht --subdir pusht/lewm_masked_e100_s1 --overrides "trainer.max_epochs=100 runtime.stop_after_epoch=30 early_stopping.enabled=false seed=1" --eval-config-name pusht --eval-policy pusht/lewm_masked_e100_s1/lewm_masked_epoch_30 --eval-overrides "eval.num_eval=200 eval.env_batch_size=10 seed=42 output.save_video=false output.filename=e100_mtm_pusht_s1_ep30_n200.txt" > /tmp/e100_pusht_mtm_s1_ep30.log 2>&1'

tmux new-session -d -s e100_pusht_mtm_s2 'cd <repo> && .venv/bin/modal run --detach --name e100-pusht-mtm-s2-ep30 modal_app.py::train_then_evaluate --config-name lewm_masked --data pusht --subdir pusht/lewm_masked_e100_s2 --overrides "trainer.max_epochs=100 runtime.stop_after_epoch=30 early_stopping.enabled=false seed=2" --eval-config-name pusht --eval-policy pusht/lewm_masked_e100_s2/lewm_masked_epoch_30 --eval-overrides "eval.num_eval=200 eval.env_batch_size=10 seed=42 output.save_video=false output.filename=e100_mtm_pusht_s2_ep30_n200.txt" > /tmp/e100_pusht_mtm_s2_ep30.log 2>&1'

tmux new-session -d -s e100_cube_base_s1 'cd <repo> && .venv/bin/modal run --detach --name e100-cube-base-s1-ep30 modal_app.py::train_then_evaluate --config-name lewm --data ogb --subdir cube/lewm_base_e100_s1 --overrides "trainer.max_epochs=100 runtime.stop_after_epoch=30 early_stopping.enabled=false seed=1" --eval-config-name cube --eval-policy cube/lewm_base_e100_s1/lewm_epoch_30 --eval-overrides "eval.num_eval=200 eval.env_batch_size=10 seed=42 output.save_video=false output.filename=e100_sigreg_cube_s1_ep30_n200.txt" > /tmp/e100_cube_base_s1_ep30.log 2>&1'

tmux new-session -d -s e100_cube_base_s2 'cd <repo> && .venv/bin/modal run --detach --name e100-cube-base-s2-ep30 modal_app.py::train_then_evaluate --config-name lewm --data ogb --subdir cube/lewm_base_e100_s2 --overrides "trainer.max_epochs=100 runtime.stop_after_epoch=30 early_stopping.enabled=false seed=2" --eval-config-name cube --eval-policy cube/lewm_base_e100_s2/lewm_epoch_30 --eval-overrides "eval.num_eval=200 eval.env_batch_size=10 seed=42 output.save_video=false output.filename=e100_sigreg_cube_s2_ep30_n200.txt" > /tmp/e100_cube_base_s2_ep30.log 2>&1'

tmux new-session -d -s e100_cube_mtm_s1 'cd <repo> && .venv/bin/modal run --detach --name e100-cube-mtm-s1-ep30 modal_app.py::train_then_evaluate --config-name lewm_masked --data ogb --subdir cube/lewm_masked_e100_s1 --overrides "trainer.max_epochs=100 runtime.stop_after_epoch=30 early_stopping.enabled=false seed=1" --eval-config-name cube --eval-policy cube/lewm_masked_e100_s1/lewm_masked_epoch_30 --eval-overrides "eval.num_eval=200 eval.env_batch_size=10 seed=42 output.save_video=false output.filename=e100_mtm_cube_s1_ep30_n200.txt" > /tmp/e100_cube_mtm_s1_ep30.log 2>&1'

tmux new-session -d -s e100_cube_mtm_s2 'cd <repo> && .venv/bin/modal run --detach --name e100-cube-mtm-s2-ep30 modal_app.py::train_then_evaluate --config-name lewm_masked --data ogb --subdir cube/lewm_masked_e100_s2 --overrides "trainer.max_epochs=100 runtime.stop_after_epoch=30 early_stopping.enabled=false seed=2" --eval-config-name cube --eval-policy cube/lewm_masked_e100_s2/lewm_masked_epoch_30 --eval-overrides "eval.num_eval=200 eval.env_batch_size=10 seed=42 output.save_video=false output.filename=e100_mtm_cube_s2_ep30_n200.txt" > /tmp/e100_cube_mtm_s2_ep30.log 2>&1'
```

#### Inference

Automatic after each checkpoint reaches epoch 30 via `train_then_evaluate`.
TwoRoom-long 100/150 evals still need to be launched manually for the new
TwoRoom seed checkpoints after training completes.

#### Diagnostics

```bash
.venv/bin/modal app list | head -120
tmux ls
tail -30 /tmp/e100_reacher_*_ep30.log
tail -30 /tmp/e100_tworoom_*_ep30.log
tail -12 /tmp/e100_pusht_*_ep30.log
tail -12 /tmp/e100_cube_*_ep30.log
```

#### Eval

```bash
# Completed just before launch:
# cube/lewm_masked_e100/e100_mtm_cube_ep30_n200_r2.txt -> 77.0
#
# Pending automatic outputs for seeds 1/2:
# reacher/lewm_base_e100_s{1,2}/e100_sigreg_reacher_s{1,2}_ep30_n200.txt
# reacher/lewm_masked_e100_s{1,2}/e100_mtm_reacher_s{1,2}_ep30_n200.txt
# tworoom/lewm_base_e100_s{1,2}/e100_sigreg_tworoom_s{1,2}_ep30_n200.txt
# tworoom/lewm_masked_e100_s{1,2}/e100_mtm_tworoom_s{1,2}_ep30_n200.txt
# pusht/lewm_base_e100_s{1,2}/e100_sigreg_pusht_s{1,2}_ep30_n200.txt
# pusht/lewm_masked_e100_s{1,2}/e100_mtm_pusht_s{1,2}_ep30_n200.txt
# cube/lewm_base_e100_s{1,2}/e100_sigreg_cube_s{1,2}_ep30_n200.txt
# cube/lewm_masked_e100_s{1,2}/e100_mtm_cube_s{1,2}_ep30_n200.txt
```

#### Compare

```bash
# Final target:
# mean +/- sd over train seeds {3072,1,2}, eval seed 42, n=200,
# for {TwoRoom, PushT, Reacher, Cube} x {SIGReg, MTM}.
```

### Artifacts

* Running Reacher apps/logs:
  * SIGReg s1: `ap-YjUwVFV7AFrwRwI55YXjsk`, `/tmp/e100_reacher_base_s1_ep30.log`
  * SIGReg s2: `ap-2tsAmuzc4QiemOuZllWNgZ`, `/tmp/e100_reacher_base_s2_ep30.log`
  * MTM s1: `ap-Myke76CHpd10OmIkIg4qE2`, `/tmp/e100_reacher_mtm_s1_ep30.log`
  * MTM s2: `ap-sSi7JbJub9XST1tkBedavr`, `/tmp/e100_reacher_mtm_s2_ep30.log`
* Running TwoRoom apps/logs:
  * SIGReg s1: `ap-jh0UGytKoAyrDJGf3YlZnu`, `/tmp/e100_tworoom_base_s1_ep30.log`
  * SIGReg s2: `ap-nJx06nDyjkM1aLIcR4Wuaq`, `/tmp/e100_tworoom_base_s2_ep30.log`
  * MTM s1: `ap-JaSUDq483sn5lMx2pJ9EJo`, `/tmp/e100_tworoom_mtm_s1_ep30.log`
  * MTM s2: `ap-L8Uc5CwEF0kQwefc9IWIBb`, `/tmp/e100_tworoom_mtm_s2_ep30.log`
* Queued PushT apps/logs (`Tasks=0` as of 23:46 IST):
  * SIGReg s1/s2: `ap-CaTDh0n84uwBfA8xpjuLb2`,
    `ap-LAeeH6Q8AFOm7mgJU4ZDZr`
  * MTM s1/s2: `ap-JAJPdjaycTT7GFpbzwG7jB`,
    `ap-dmEXS6IyjH5VduF8hSmItJ`
* Queued Cube apps/logs (`Tasks=0` as of 23:46 IST):
  * SIGReg s1/s2: `ap-6rPc7wQSH4rVOIcmhD6u2k`,
    `ap-BWb9B1eZl1jNmbmhPCrv2X`
  * MTM s1/s2: `ap-e9alMNWPtxiXTOmRiwU1zk`,
    `ap-8AUDCpDMPnJbMurxNpnj10`

### Result

Base:

* Seed-3072 completed results currently include TwoRoom SIGReg 84.0, Reacher
  SIGReg 83.5. PushT SIGReg and Cube SIGReg seed-3072 are still training.
* Reacher SIGReg s1/s2 and TwoRoom SIGReg s1/s2 are verified running with
  step-level logs.
* PushT/Cube SIGReg s1/s2 are created but queued at `Tasks=0`.

Candidate:

* Seed-3072 completed results currently include PushT MTM 85.0, TwoRoom MTM
  87.5, Reacher MTM 77.0, Cube MTM 77.0.
* Reacher MTM s1/s2 and TwoRoom MTM s1/s2 are verified running with step-level
  logs.
* PushT/Cube MTM s1/s2 are created but queued at `Tasks=0`.

Delta:

* Wins / losses: pending until seed 1/2 runs finish.
* Recall collapse? Reacher seed 1/2 e100 MTM will directly test whether the
  e100 schedule removes or reduces the e10 masked-collapse failure.

### Decision

Use the final paper table only after the seed-3072 gaps plus extra seed runs
land. Current paper should still avoid claiming multi-seed e100 results. Track
PushT/Cube queued apps and only mark them running after both app list shows
`Tasks=1` and logs show real training progress.

### Notes

* The full extra-seed matrix is launched/queued, but only Reacher and TwoRoom
  were verified as running at launch time. This is likely a Modal concurrency
  limit rather than a job failure.
* The launch uses one persistent tmux session per job to avoid local-client
  cancellation.

#### 2026-06-19 08:38 IST status

* Cube MTM seed-3072 eval completed: `cube/lewm_masked_e100/e100_mtm_cube_ep30_n200_r2.txt`
  scored **77.0%**.
* Cube SIGReg seed-3072 training completed and wrote
  `cube/lewm_base_e100/lewm_epoch_30_object.ckpt`. Launched its n=200 eval:
  tmux `e100_cube_sig_eval`, app `ap-mgs5w2gOhPZ4iegixD3doZ`, output
  `e100_sigreg_cube_ep30_n200_r2.txt`. It is queued at `Tasks=0`, not yet
  running.
* PushT SIGReg seed-3072 is still running in app `ap-7URW93LGChFJWh7MTmYYbB`,
  around epoch 26 / step 350,550.
* Reacher seed 1/2 SIGReg+MTM runs remain active with `Tasks=1`, around
  epoch 11-12.
* TwoRoom seed 1/2 SIGReg+MTM runs remain active with `Tasks=1`; one MTM seed
  is around epoch 30, the others around epochs 23-24.
* Cube SIGReg seed 2 (`ap-BWb9B1eZl1jNmbmhPCrv2X`) has started and is around
  epoch 3. PushT seed 1/2, Cube SIGReg seed 1, and Cube MTM seed 1/2 apps are
  still queued at `Tasks=0`.

#### 2026-06-19 16:14 IST status

Seed-3072 e100/n=200 table is now complete:

| Task | SIGReg | MTM |
|---|---:|---:|
| PushT | 84.5 | 85.0 |
| TwoRoom | 84.0 | 87.5 |
| Reacher | 83.5 | 77.0 |
| OGBench-Cube | 66.5 | 77.0 |

New completed seed-1/2 TwoRoom standard n=200 evals:

| Method | Seed 1 | Seed 2 |
|---|---:|---:|
| SIGReg | 87.0 | 85.0 |
| MTM | 91.0 | 45.0 |

Status of active/queued extra-seed jobs:

* Reacher SIGReg/MTM seeds 1/2 remain active with `Tasks=1`, around epochs
  20-21.
* PushT SIGReg seed-3072 completed at 84.5. PushT base seed 2 and MTM seed 1
  are active around epoch 7; PushT base seed 1 and MTM seed 2 remain queued at
  `Tasks=0`.
* Cube SIGReg seed-3072 completed at 66.5. Cube base seeds 1/2 and MTM seeds
  1/2 are active, roughly epochs 4-14 depending on seed/method.
* TwoRoom-long evals still need to be launched for the new TwoRoom seed
  checkpoints if the long-horizon figure will report multi-seed e100 results.

---

## 2026-06-19 — TwoRoom MTM e100 seed-2 epoch-30 failure investigation

### Intent

Investigate why `tworoom/lewm_masked_e100_s2/lewm_masked_epoch_30` scored only
45.0% on the standard n=200 TwoRoom protocol while e100 MTM seed 1 scored 91.0%,
e100 SIGReg seeds scored 87.0/85.0, and e10 MTM seed 2 previously scored 90.5%.

Main question: bad eval/provenance bug, intrinsically bad seed, or late
instability caused by evaluating epoch 30 on the 100-epoch LR schedule?

### Commands

#### Train

No new training launched for this investigation. Existing run:

```bash
tmux new-session -d -s e100_tworoom_mtm_s2 'cd <repo> && .venv/bin/modal run --detach --name e100-tworoom-mtm-s2-ep30 modal_app.py::train_then_evaluate --config-name lewm_masked --data tworoom --subdir tworoom/lewm_masked_e100_s2 --overrides "trainer.max_epochs=100 runtime.stop_after_epoch=30 early_stopping.enabled=false seed=2" --eval-config-name tworoom --eval-policy tworoom/lewm_masked_e100_s2/lewm_masked_epoch_30 --eval-overrides "eval.num_eval=200 eval.env_batch_size=10 seed=42 output.save_video=false output.filename=e100_mtm_tworoom_s2_ep30_n200.txt" > /tmp/e100_tworoom_mtm_s2_ep30.log 2>&1'
```

#### Inference

No separate inference.

#### Diagnostics

```bash
mkdir -p /tmp/tworoom_mtm_investigate/tworoom
.venv/bin/modal volume get multi-future-lewm-cache /tworoom/lewm_masked_e100_s2/config.yaml /tmp/tworoom_mtm_investigate/tworoom/lewm_masked_e100_s2/config.yaml --force
.venv/bin/modal volume get multi-future-lewm-cache /tworoom/lewm_masked_e100_s2/metrics.jsonl /tmp/tworoom_mtm_investigate/tworoom/lewm_masked_e100_s2/metrics.jsonl --force
.venv/bin/modal volume get multi-future-lewm-cache /tworoom/lewm_masked_e100_s2/events.jsonl /tmp/tworoom_mtm_investigate/tworoom/lewm_masked_e100_s2/events.jsonl --force
.venv/bin/modal volume get multi-future-lewm-cache /tworoom/lewm_masked_e100_s2/checkpoint_state.json /tmp/tworoom_mtm_investigate/tworoom/lewm_masked_e100_s2/checkpoint_state.json --force
.venv/bin/modal volume get multi-future-lewm-cache /tworoom/lewm_masked_s2/metrics.jsonl /tmp/tworoom_e10_compare/tworoom/lewm_masked_s2/metrics.jsonl --force
.venv/bin/modal volume get multi-future-lewm-cache /tworoom/lewm_masked_s2/events.jsonl /tmp/tworoom_e10_compare/tworoom/lewm_masked_s2/events.jsonl --force
.venv/bin/modal volume get multi-future-lewm-cache /tworoom/lewm_masked_s2_tworoom_n200.txt /tmp/tworoom_e10_compare/tworoom/lewm_masked_s2_tworoom_n200.txt --force
python3 - <<'PY'
import json, os
runs = {
    "e10 mtm s2": "/tmp/tworoom_e10_compare/tworoom/lewm_masked_s2",
    "e100 mtm s1": "/tmp/tworoom_mtm_investigate/tworoom/lewm_masked_e100_s1",
    "e100 mtm s2": "/tmp/tworoom_mtm_investigate/tworoom/lewm_masked_e100_s2",
}
for name, path in runs.items():
    vals = []
    with open(os.path.join(path, "events.jsonl")) as f:
        for line in f:
            row = json.loads(line)
            if row.get("event") == "validation_epoch_end" and row.get("global_step", 0) > 0:
                vals.append((row["epoch"], row["metrics"]))
    best = min(vals, key=lambda item: item[1]["validate/loss"])
    final = vals[-1]
    print(name, "best", best[0], best[1]["validate/loss"], "final", final[0], final[1])
PY
```

#### Eval

Queued targeted n=200 re-evals for the bad seed's earlier/recovered checkpoints
and an epoch-30 rerun:

```bash
for ep in 10 15 16 29; do
  sess="investigate_tworoom_mtm_s2_ep${ep}"
  out="investigate_mtm_tworoom_s2_ep${ep}_n200.txt"
  tmux has-session -t "$sess" 2>/dev/null && continue
  tmux new-session -d -s "$sess" "cd <repo> && .venv/bin/modal run --detach --name investigate-tworoom-mtm-s2-ep${ep} modal_app.py::evaluate --config-name tworoom --policy tworoom/lewm_masked_e100_s2/lewm_masked_epoch_${ep} --overrides \"eval.num_eval=200 eval.env_batch_size=10 seed=42 output.save_video=false output.filename=${out}\" > /tmp/${sess}.log 2>&1"
done
sess="investigate_tworoom_mtm_s2_ep30_rerun"
out="investigate_mtm_tworoom_s2_ep30_rerun_n200.txt"
if ! tmux has-session -t "$sess" 2>/dev/null; then
  tmux new-session -d -s "$sess" "cd <repo> && .venv/bin/modal run --detach --name investigate-tworoom-mtm-s2-ep30-rerun modal_app.py::evaluate --config-name tworoom --policy tworoom/lewm_masked_e100_s2/lewm_masked_epoch_30 --overrides \"eval.num_eval=200 eval.env_batch_size=10 seed=42 output.save_video=false output.filename=${out}\" > /tmp/${sess}.log 2>&1"
fi
.venv/bin/modal app list | head -220
```

As of launch, all five investigation eval apps were created but queued at
`Tasks=0`, so none are counted as running yet.

#### Compare

```bash
python3 - <<'PY'
from pathlib import Path
import re
files = {
    "e10_mtm_s2": Path("/tmp/tworoom_e10_compare/tworoom/lewm_masked_s2_tworoom_n200.txt"),
    "e100_mtm_s1": Path("/tmp/tworoom_mtm_results/e100_mtm_tworoom_s1_ep30_n200.txt"),
    "e100_mtm_s2": Path("/tmp/tworoom_mtm_results/e100_mtm_tworoom_s2_ep30_n200.txt"),
    "e100_sig_s2": Path("/tmp/tworoom_mtm_results/e100_sigreg_tworoom_s2_ep30_n200.txt"),
}
def parse(path):
    text = path.read_text()
    match = re.search(r"episode_successes': array\(\[(.*?)\]\)", text, re.S)
    return [token == "True" for token in re.findall(r"True|False", match.group(1))]
arr = {key: parse(path) for key, path in files.items()}
for key, successes in arr.items():
    print(key, 100 * sum(successes) / len(successes))
for first, second in [("e100_mtm_s2", "e100_sig_s2"), ("e100_mtm_s2", "e10_mtm_s2"), ("e100_mtm_s2", "e100_mtm_s1")]:
    a, b = arr[first], arr[second]
    print(first, second, {
        "both": sum(x and y for x, y in zip(a, b)),
        "first_only": sum(x and not y for x, y in zip(a, b)),
        "second_only": sum((not x) and y for x, y in zip(a, b)),
        "neither": sum((not x) and (not y) for x, y in zip(a, b)),
    })
PY
```

### Artifacts

* Train metadata: `tworoom/lewm_masked_e100_s2/run_metadata.json`
* Train config: `tworoom/lewm_masked_e100_s2/config.yaml`
* Train metrics: `tworoom/lewm_masked_e100_s2/metrics.jsonl`
* Train events: `tworoom/lewm_masked_e100_s2/events.jsonl`
* Checkpoint state: `tworoom/lewm_masked_e100_s2/checkpoint_state.json`
* Bad eval summary: `tworoom/lewm_masked_e100_s2/e100_mtm_tworoom_s2_ep30_n200.txt`
* Good e10 comparison: `tworoom/lewm_masked_s2_tworoom_n200.txt`
* Queued investigation eval outputs:
  * `tworoom/lewm_masked_e100_s2/investigate_mtm_tworoom_s2_ep10_n200.txt`
  * `tworoom/lewm_masked_e100_s2/investigate_mtm_tworoom_s2_ep15_n200.txt`
  * `tworoom/lewm_masked_e100_s2/investigate_mtm_tworoom_s2_ep16_n200.txt`
  * `tworoom/lewm_masked_e100_s2/investigate_mtm_tworoom_s2_ep29_n200.txt`
  * `tworoom/lewm_masked_e100_s2/investigate_mtm_tworoom_s2_ep30_rerun_n200.txt`

### Result

Base:

SIGReg e100 seed 2 scored 85.0% at epoch 30. e10 SIGReg seed 3072 scored 85.0%.

Candidate:

MTM e100 seed 2 scored 45.0% at epoch 30, while the same training seed under
the original 10-epoch LR schedule scored 90.5%. MTM e100 seed 1 scored 91.0%.

Delta:

* Wins / losses: e100 MTM seed 2 loses 91 paired episodes to e100 SIGReg seed 2
  and gains only 11. It loses 93 paired episodes to e10 MTM seed 2 and gains
  only 2.
* Recall collapse? Not a full zero-variance collapse, but collapse-adjacent late
  representation degradation. E100 seed 2 final validation has
  `validate/emb_std=0.065`, `validate/inverse_loss=0.959`, and
  `validate/loss=0.965`, versus seed 1 final `emb_std=0.103`,
  `inverse_loss=0.746`, and `loss=0.747`.
* LR schedule: e10 final LR is effectively zero by epoch 10
  (`~1.2e-12`), while e100 epoch 30 is still high (`~4.0e-5`). The bad seed's
  gradients spike late under this high-LR regime, including epoch-level grad-norm
  maxima above 2,700 by epoch 20 and above 38,000 by epoch 27.

### Decision

Do not present e100 epoch-30 TwoRoom MTM as a robust win yet. The e10 TwoRoom
MTM result is still robust across seeds (`90.5/89.5/90.5`), but e100 epoch-30
seed 2 is a real instability unless the queued epoch-30 rerun contradicts it.
Use the queued checkpoint-trajectory evals to decide whether TwoRoom MTM should
use an earlier e100 checkpoint, the original e10 schedule, or be described as
schedule-sensitive.

### Notes

* Provenance check passed: the bad result file points to
  `policy: tworoom/lewm_masked_e100_s2/lewm_masked_epoch_30`, `seed: 42`,
  `eval.num_eval: 200`, `goal_offset_steps: 25`, and `eval_budget: 50`.
* Config check passed: seed 2 used `seed: 2`, `trainer.max_epochs: 100`,
  `runtime.stop_after_epoch: 30`, `wm.type: lewm_masked`, and SIGReg off.
* `checkpoint_state.json` reports the best validation at zero-indexed epoch 15
  / step 82,208, corresponding to the nearby one-indexed object checkpoint stem
  `lewm_masked_epoch_16`.
* The original eval log completed all 20 batches and wrote the 45.0% result; no
  exception or NaN path was found.

#### 2026-06-19 19:17 IST status

* TwoRoom MTM seed-2 investigation evals for checkpoint stems `epoch_10`,
  `epoch_15`, `epoch_16`, `epoch_29`, and rerun `epoch_30` are still created
  but not executing: all five app entries show `Tasks=0`, logs contain only
  Modal initialization/object creation, and the Modal volume contains no
  `investigate_mtm_tworoom_s2_*` result files yet.
* Reacher extra-seed jobs are all live with `Tasks=1`: SIGReg s1 around epoch
  27, SIGReg s2 around epoch 26, MTM s1 around epoch 25, and MTM s2 around
  epoch 25.
* PushT extra-seed jobs: SIGReg s2 and MTM s1 are live around epoch 12
  validation; SIGReg s1 and MTM s2 remain created but not executing
  (`Tasks=0`).
* Cube extra-seed jobs are live with `Tasks=1`: SIGReg s1 is in validation,
  SIGReg s2 around epoch 20, MTM s1 around epoch 8, and MTM s2 around epoch 13.
* No new remaining seed-1/2 n=200 result files or epoch-30 object checkpoints
  were visible on the Modal volume at this check.

#### 2026-06-20 11:25 IST status

Modal appears budget/workspace-blocked. The app list shows only four old
detached apps, all with `Tasks=0`, and Cube MTM logs ended with:
`ConflictError: workspace ac-nyKhb80scQoYoNq4AUcvIi is disabled`. Several
long-running train/eval jobs also hit Modal's 86400s input timeout.

New completed results since the previous check:

| Result | Success |
|---|---:|
| TwoRoom MTM seed 2 e100 `epoch_10` | 60.5 |
| TwoRoom MTM seed 2 e100 `epoch_15` | 88.0 |
| TwoRoom MTM seed 2 e100 `epoch_16` | 90.5 |
| TwoRoom MTM seed 2 e100 `epoch_29` | 90.0 |
| TwoRoom MTM seed 2 e100 `epoch_30` rerun | 46.5 |
| Cube SIGReg seed 2 e100 `epoch_30` | 62.0 |

Interpretation:

* The original TwoRoom MTM seed-2 epoch-30 failure is confirmed. The rerun
  scored 46.5 vs the original 45.0.
* The same training seed is good at nearby checkpoints: epoch 15/16/29 score
  88.0/90.5/90.0. So the bad result is not an intrinsically bad seed or eval
  protocol; it is a bad final checkpoint / late schedule instability.
* Reacher SIGReg seed 1 finished training and wrote
  `reacher/lewm_base_e100_s1/lewm_epoch_30_object.ckpt`, but its automatic
  n=200 eval timed out at batch 11/20, so no final result file landed.
* Reacher SIGReg seed 2 stopped around epoch 29, Reacher MTM seeds 1/2 stopped
  around epoch 28, before epoch-30 object/eval outputs landed.
* PushT stopped before epoch 30 for all extra seeds: SIGReg s1 around epoch 8,
  SIGReg s2 around epoch 26, MTM s1 around epoch 25, MTM s2 around epoch 0.
* Cube SIGReg seed 2 completed at epoch 30 and evaluated to 62.0. Cube SIGReg
  seed 1 stopped around epoch 20. Cube MTM seeds 1/2 stopped around epochs
  15/24.

Recommendation:

Do not bump the budget to continue the entire broad matrix blindly. The highest
value follow-up is small and targeted:

1. Evaluate already-finished Reacher SIGReg seed 1 epoch-30 object
   (`reacher/lewm_base_e100_s1/lewm_epoch_30`) because training is complete and
   only the eval timed out.
2. Resume/finish Reacher SIGReg seed 2 and MTM seeds 1/2 to epoch 30, then run
   n=200 evals, because Reacher is the main schedule-sensitive task.
3. Skip or defer PushT/Cube extra e100 seeds unless the paper specifically needs
   a full e100 robustness table. Existing evidence already argues PushT should
   use e10, and Cube e10 already has the clean multi-seed win.

#### 2026-06-20 08:25 IST status

Budget/workspace access is restored. I stopped the stale broad-matrix apps that
were still listed with `Tasks=0`:

* PushT SIGReg seed 1: `ap-CaTDh0n84uwBfA8xpjuLb2`
* PushT SIGReg seed 2: `ap-LAeeH6Q8AFOm7mgJU4ZDZr`
* Cube SIGReg seed 1: `ap-6rPc7wQSH4rVOIcmhD6u2k`
* PushT MTM seed 2: `ap-dmEXS6IyjH5VduF8hSmItJ`

Then I relaunched only the targeted Reacher gap-fill jobs. The first direct
background launches were canceled after local wrapper interruption, so those
stopped apps should not be used:

* Reacher SIGReg seed 1 eval: `ap-sB0sQ1AVpCbTqoaweWmIXJ`
* Reacher SIGReg seed 2 resume: `ap-iKkSqQr8i46Yv0zutRFCqB`
* Reacher MTM seed 1 resume: `ap-fEa0oe0RAIOLisFyAkhxls`
* Reacher MTM seed 2 resume: `ap-7Pgzd5NjTgnsw0ea3EOCz7`

The trusted relaunches are running from persistent tmux sessions with detached
Modal apps and verified task/log progress:

| Job | App | Local log | Expected output |
|---|---|---|---|
| Reacher SIGReg seed 1 epoch-30 eval only | `ap-H6CjUdUHDxVY8t0SSkE7sL` | `/tmp/e100_reacher_base_s1_ep30_eval_r2.log` | `reacher/lewm_base_e100_s1/e100_sigreg_reacher_s1_ep30_n200_r2.txt` |
| Reacher SIGReg seed 2 resume to epoch 30 + eval | `ap-oYk2Lz7uQVPzsTFwZOC1pb` | `/tmp/e100_reacher_base_s2_ep30_resume_r2.log` | `reacher/lewm_base_e100_s2/e100_sigreg_reacher_s2_ep30_n200_r2.txt` |
| Reacher MTM seed 1 resume to epoch 30 + eval | `ap-XZBL6jLWNi7aHKUA4mWMRT` | `/tmp/e100_reacher_mtm_s1_ep30_resume_r2.log` | `reacher/lewm_masked_e100_s1/e100_mtm_reacher_s1_ep30_n200_r2.txt` |
| Reacher MTM seed 2 resume to epoch 30 + eval | `ap-n2WXDZilTfXsidjYJ9tP3s` | `/tmp/e100_reacher_mtm_s2_ep30_resume_r2.log` | `reacher/lewm_masked_e100_s2/e100_mtm_reacher_s2_ep30_n200_r2.txt` |

Current verified progress:

* SIGReg seed 1 eval is running `eval.num_eval=200`, `eval.env_batch_size=10`,
  `seed=42`; logs show it evaluating 20 batches.
* SIGReg seed 2 restored from
  `/tmp/stable_worldmodel/reacher/lewm_base_e100_s2/lewm_weights.ckpt` at step
  376,000 / epoch 29 and is training.
* MTM seed 1 restored from
  `/tmp/stable_worldmodel/reacher/lewm_masked_e100_s1/lewm_masked_weights.ckpt`
  at step 360,000 / epoch 29 and is training.
* MTM seed 2 restored from
  `/tmp/stable_worldmodel/reacher/lewm_masked_e100_s2/lewm_masked_weights.ckpt`
  at step 361,000 / epoch 29 and is training.

#### 2026-06-20 08:45 IST status

Reacher SIGReg seed 1 epoch-30 eval completed:

* App: `ap-H6CjUdUHDxVY8t0SSkE7sL`
* Policy: `reacher/lewm_base_e100_s1/lewm_epoch_30`
* Eval: `eval.num_eval=200`, `eval.env_batch_size=10`, `seed=42`,
  `goal_offset_steps=25`, `eval_budget=50`
* Result: 79.5% success
* Raw artifact: `progress/evaluations/e100-reacher-r2/e100_sigreg_reacher_s1_ep30_n200_r2.txt`
* Modal volume artifact:
  `reacher/lewm_base_e100_s1/e100_sigreg_reacher_s1_ep30_n200_r2.txt`

Remaining trusted Reacher r2 jobs are still running with `Tasks=1`:

* SIGReg seed 2 resume/eval `ap-oYk2Lz7uQVPzsTFwZOC1pb`: epoch 30 around
  step 380,225.
* MTM seed 1 resume/eval `ap-XZBL6jLWNi7aHKUA4mWMRT`: epoch 29 around
  step 366,075.
* MTM seed 2 resume/eval `ap-n2WXDZilTfXsidjYJ9tP3s`: epoch 29 around
  step 366,875.

#### 2026-06-20 13:25 IST status

The targeted Reacher r2 gap-fill is complete. Modal app list is empty and there
are no active tmux sessions. All three remaining jobs finished cleanly and wrote
their epoch-30 objects and n=200 result files:

| Approach | Train seed | App | Policy | Success |
|---|---:|---|---|---:|
| SIGReg | 1 | `ap-H6CjUdUHDxVY8t0SSkE7sL` | `reacher/lewm_base_e100_s1/lewm_epoch_30` | 79.5 |
| SIGReg | 2 | `ap-oYk2Lz7uQVPzsTFwZOC1pb` | `reacher/lewm_base_e100_s2/lewm_epoch_30` | 80.5 |
| MTM | 1 | `ap-XZBL6jLWNi7aHKUA4mWMRT` | `reacher/lewm_masked_e100_s1/lewm_masked_epoch_30` | 10.5 |
| MTM | 2 | `ap-n2WXDZilTfXsidjYJ9tP3s` | `reacher/lewm_masked_e100_s2/lewm_masked_epoch_30` | 12.0 |

All four r2 result files are archived under
`progress/evaluations/e100-reacher-r2/`. Protocol was identical across cells:
`eval.num_eval=200`, `eval.env_batch_size=10`, `seed=42`,
`goal_offset_steps=25`, and `eval_budget=50`.

Interpretation:

* The Reacher e100 SIGReg result is stable across extra seeds: 79.5/80.5,
  compared with seed-3072's 81.5 at epoch 30 and 87.5 at epoch 50.
* Reacher e100 epoch-30 MTM is not robust. The extra seed results collapse to
  10.5/12.0, despite the seed-3072 MTM epoch-30 result of 75.0.
* Do not report Reacher e100 epoch-30 as an MTM success. If Reacher appears in
  the paper's main controlled table, it should be framed as a failure/instability
  case for MTM under the 100-epoch schedule, or kept in an appendix while the
  main claim uses tasks where MTM is robust.

## 2026-06-20 — Reacher MTM collapse rescue screen

### Intent

Investigate why Reacher MTM collapses for seeds 1/2 and test targeted fixes on
the failing seed before committing to full e10+n=200 reruns. Collapse is visible
within the first epoch, so this first wave uses `runtime.stop_after_epoch=1`
with the e10 LR schedule (`trainer.max_epochs=10`) as a cheap screen.

### Diagnostics

Existing Reacher MTM metrics show a real encoder-collapse failure, not an eval
bug:

| Run | Seed | Schedule | Final `emb_std` | Final inverse loss | n=200 SR |
|---|---:|---|---:|---:|---:|
| `reacher/lewm_masked` | 3072 | e10 | 0.0879 | 0.845 | 68.0 |
| `reacher/lewm_masked_s1` | 1 | e10 | 0.00012 | 0.978 | 11.5 |
| `reacher/lewm_masked_s2` | 2 | e10 | 0.00050 | 0.981 | 13.5 |
| `reacher/lewm_masked_e100` | 3072 | e100 epoch 30 | ~0.044 | ~0.831 | 75.0 |
| `reacher/lewm_masked_e100_s1` | 1 | e100 epoch 30 | 0.00005 | 1.019 | 10.5 |
| `reacher/lewm_masked_e100_s2` | 2 | e100 epoch 30 | 0.00005 | 0.984 | 12.0 |
| `reacher/lewm_masked_vicreg` | 3072 | e10 | 0.1069 | 0.845 | 66.0 |

The bad seeds collapse during epoch 0/1. For example, `lewm_masked_s1` ends
epoch 0 at `emb_std=0.0183` and epoch 1 at `0.00427`, while the good seed
remains around `0.14`.

### Commands

All screens use:

```bash
modal_app.py::train --config-name lewm_masked --data reacher \
  --overrides "trainer.max_epochs=10 runtime.stop_after_epoch=1 early_stopping.enabled=false seed=1 ..."
```

The launched variants are:

| Variant | Subdir | Override delta | App | Local log |
|---|---|---|---|---|
| variance floor soft | `reacher/mtm_screen_var005_g005_s1` | `+loss.variance_reg.weight=0.005 +loss.variance_reg.gamma=0.05` | `ap-CR4zXwOIk31bJ4RfWdy42O` | `/tmp/r_mtm_scr_var005_g005_s1.log` |
| variance floor soft 2 | `reacher/mtm_screen_var01_g005_s1` | `+loss.variance_reg.weight=0.01 +loss.variance_reg.gamma=0.05` | `ap-2Nz5i7lHNOh5lJfuEcMnds` | `/tmp/r_mtm_scr_var01_g005_s1.log` |
| variance floor medium | `reacher/mtm_screen_var01_g01_s1` | `+loss.variance_reg.weight=0.01 +loss.variance_reg.gamma=0.1` | `ap-tb9ezkMXIhnoZ41mt2wCs7` | `/tmp/r_mtm_scr_var01_g01_s1.log` |
| variance floor existing strength | `reacher/mtm_screen_var05_g01_s1` | `+loss.variance_reg.weight=0.05 +loss.variance_reg.gamma=0.1` | `ap-szJ99f6nvXwQtscW5PM0Zx` | `/tmp/r_mtm_scr_var05_g01_s1.log` |
| stronger inverse | `reacher/mtm_screen_inv3_s1` | `loss.masked.inverse_weight=3.0` | `ap-rqYhFIm5lYWuWnw8IqgUA3` | `/tmp/r_mtm_scr_inv3_s1.log` |
| much stronger inverse | `reacher/mtm_screen_inv10_s1` | `loss.masked.inverse_weight=10.0` | `ap-GYcsVYcFEFrOjWohfPGD8D` | `/tmp/r_mtm_scr_inv10_s1.log` |
| weak hybrid SIGReg | `reacher/mtm_screen_hybrid01_s1` | `loss.sigreg.weight=0.01` | `ap-eA2n3bQL2o2lfvXeffyKAC` | `/tmp/r_mtm_scr_hybrid01_s1.log` |
| stronger hybrid SIGReg | `reacher/mtm_screen_hybrid03_s1` | `loss.sigreg.weight=0.03` | `ap-r5fytdFTf5e84BnfqcAbfR` | `/tmp/r_mtm_scr_hybrid03_s1.log` |
| low LR | `reacher/mtm_screen_lr1e5_s1` | `optimizer.lr=1e-5` | `ap-JTLHdZ5enUyCjuJKFGcniv` | `/tmp/r_mtm_scr_lr1e5_s1.log` |
| lower LR | `reacher/mtm_screen_lr2e5_s1` | `optimizer.lr=2e-5` | `ap-Y93YokVx0QjXj1Uk3S3eih` | `/tmp/r_mtm_scr_lr2e5_s1.log` |

The `loss.sigreg.weight=0.005` screen app `ap-HIEwaBXeb2b4oVHzR1snWC` never
acquired a task while the other ten apps were running, so it was stopped.

### Code

Patched `train.py` so `masked_transition_forward` can optionally include
`loss.sigreg.weight` inside the masked objective. Existing pure MTM configs are
unchanged because `config/train/lewm_masked.yaml` sets this weight to `0.0`.
Added regression coverage in `tests/test_masked_transition.py`.

Validation:

```bash
.venv/bin/python -m pytest -q tests/test_masked_transition.py
.venv/bin/python -m py_compile train.py modal_app.py
```

### Artifacts

Expected screen metrics:

* `reacher/mtm_screen_var005_g005_s1/metrics.jsonl`
* `reacher/mtm_screen_var01_g005_s1/metrics.jsonl`
* `reacher/mtm_screen_var01_g01_s1/metrics.jsonl`
* `reacher/mtm_screen_var05_g01_s1/metrics.jsonl`
* `reacher/mtm_screen_inv3_s1/metrics.jsonl`
* `reacher/mtm_screen_inv10_s1/metrics.jsonl`
* `reacher/mtm_screen_hybrid01_s1/metrics.jsonl`
* `reacher/mtm_screen_hybrid03_s1/metrics.jsonl`
* `reacher/mtm_screen_lr1e5_s1/metrics.jsonl`
* `reacher/mtm_screen_lr2e5_s1/metrics.jsonl`

### Result

Pending. As of launch verification, the ten screen apps above all had
`Tasks=1` and step-level training progress in logs. The next decision is to
promote variants that keep epoch-1 `emb_std` meaningfully above the collapse
basin, roughly `>0.03`, to full e10 training and n=200 evaluation on seeds 1/2.

#### Early live metric snapshot

Pulled live `metrics.jsonl` snapshots while the screens were still in epoch 0:

| Screen | Step | `emb_std` | Inverse loss | Note |
|---|---:|---:|---:|---|
| `mtm_screen_hybrid01_s1` | 925 | 0.8047 | 0.908 | Strongly noncollapsed, but SIGReg term is large |
| `mtm_screen_hybrid03_s1` | 900 | 0.8711 | 0.925 | Strongly noncollapsed, stronger Gaussian pressure |
| `mtm_screen_inv10_s1` | 900 | 0.3574 | 0.926 | Noncollapsed; loss scale large |
| `mtm_screen_inv3_s1` | 1750 | 0.1914 | 0.922 | Noncollapsed; plausible pure-MTM rescue |
| `mtm_screen_lr1e5_s1` | 2300 | 0.0938 | 0.988 | Alive but inverse still near baseline |
| `mtm_screen_lr2e5_s1` | 1275 | 0.0972 | 0.993 | Alive but inverse still near baseline |
| `mtm_screen_var005_g005_s1` | 1350 | 0.0806 | 0.993 | Alive, soft variance penalty inactive at this point |
| `mtm_screen_var01_g005_s1` | 2150 | 0.0703 | 0.987 | Alive, soft variance penalty inactive at this point |
| `mtm_screen_var01_g01_s1` | 1200 | 0.0879 | 1.009 | Alive, variance penalty active |
| `mtm_screen_var05_g01_s1` | 900 | 0.0952 | 0.984 | Alive, matches existing VICReg direction |

All ten are far above the seed-1 plain-MTM collapse trajectory at comparable
stage (`lewm_masked_s1` ended epoch 0 at `emb_std=0.0183`), but this is not yet
enough to select a winner. Need epoch-1 final metrics, then promote the best
variant(s) to full e10+n=200 confirmation.

### Decision

Keep running. Prioritize full e10+n=200 confirmation for:

1. the best soft variance-floor screen,
2. the best weak hybrid SIGReg screen,
3. inverse-weight-only only if it keeps `emb_std` alive without pathological
   loss scale.

#### 2026-06-20 14:55 IST promotion

Stopped four low-priority screens to free Modal capacity:

| Screen | App | Reason |
|---|---|---|
| `mtm_screen_lr1e5_s1` | `ap-JTLHdZ5enUyCjuJKFGcniv` | variance alive but inverse loss still near baseline |
| `mtm_screen_lr2e5_s1` | `ap-Y93YokVx0QjXj1Uk3S3eih` | variance alive but inverse loss still near baseline |
| `mtm_screen_var005_g005_s1` | `ap-CR4zXwOIk31bJ4RfWdy42O` | tracking too close to bad seed trajectory |
| `mtm_screen_var01_g005_s1` | `ap-2Nz5i7lHNOh5lJfuEcMnds` | tracking too close to bad seed trajectory |

Latest live snapshot before promotion:

| Screen | Step | `emb_std` | Inverse loss | Forward loss | Interpretation |
|---|---:|---:|---:|---:|---|
| `mtm_screen_hybrid01_s1` | 1675 | 0.836 | 0.929 | 0.0209 | noncollapsed, but aided by SIGReg |
| `mtm_screen_hybrid03_s1` | 1650 | 0.926 | 0.841 | 0.0518 | noncollapsed, stronger SIGReg pressure |
| `mtm_screen_inv10_s1` | 1675 | 0.356 | 0.855 | 0.0215 | noncollapsed, large inverse scale |
| `mtm_screen_inv3_s1` | 2825 | 0.260 | 0.828 | 0.0063 | best pure-MTM rescue signal |
| `mtm_screen_lr1e5_s1` | 3475 | 0.084 | 1.003 | 0.0082 | low priority |
| `mtm_screen_lr2e5_s1` | 2150 | 0.084 | 0.989 | 0.0080 | low priority |
| `mtm_screen_var005_g005_s1` | 2400 | 0.068 | 0.984 | 0.0052 | low priority |
| `mtm_screen_var01_g005_s1` | 3475 | 0.059 | 1.002 | 0.0039 | low priority |
| `mtm_screen_var01_g01_s1` | 2000 | 0.071 | 1.000 | 0.0058 | still running as fallback |
| `mtm_screen_var05_g01_s1` | 1750 | 0.074 | 1.020 | 0.0062 | still running as fallback |

Promoted `loss.masked.inverse_weight=3.0` to full e10 training plus n=200
evaluation on all three Reacher training seeds (`3072,1,2`). These are pure MTM
ablations: SIGReg remains off.

```bash
tmux new-session -d -s r_mtm_full_inv3_s3072 'cd <repo> && nohup .venv/bin/modal run --detach --name r-mtm-full-inv3-s3072 modal_app.py::train_then_evaluate --config-name lewm_masked --data reacher --subdir reacher/mtm_inv3_e10_s3072 --overrides "trainer.max_epochs=10 early_stopping.enabled=false seed=3072 loss.masked.inverse_weight=3.0" --eval-config-name reacher --eval-policy reacher/mtm_inv3_e10_s3072/lewm_masked_epoch_10 --eval-overrides "eval.num_eval=200 eval.env_batch_size=10 seed=42 output.save_video=false output.filename=mtm_inv3_e10_s3072_ep10_n200.txt" > /tmp/r_mtm_full_inv3_s3072.log 2>&1'
tmux new-session -d -s r_mtm_full_inv3_s1 'cd <repo> && nohup .venv/bin/modal run --detach --name r-mtm-full-inv3-s1 modal_app.py::train_then_evaluate --config-name lewm_masked --data reacher --subdir reacher/mtm_inv3_e10_s1 --overrides "trainer.max_epochs=10 early_stopping.enabled=false seed=1 loss.masked.inverse_weight=3.0" --eval-config-name reacher --eval-policy reacher/mtm_inv3_e10_s1/lewm_masked_epoch_10 --eval-overrides "eval.num_eval=200 eval.env_batch_size=10 seed=42 output.save_video=false output.filename=mtm_inv3_e10_s1_ep10_n200.txt" > /tmp/r_mtm_full_inv3_s1.log 2>&1'
tmux new-session -d -s r_mtm_full_inv3_s2 'cd <repo> && nohup .venv/bin/modal run --detach --name r-mtm-full-inv3-s2 modal_app.py::train_then_evaluate --config-name lewm_masked --data reacher --subdir reacher/mtm_inv3_e10_s2 --overrides "trainer.max_epochs=10 early_stopping.enabled=false seed=2 loss.masked.inverse_weight=3.0" --eval-config-name reacher --eval-policy reacher/mtm_inv3_e10_s2/lewm_masked_epoch_10 --eval-overrides "eval.num_eval=200 eval.env_batch_size=10 seed=42 output.save_video=false output.filename=mtm_inv3_e10_s2_ep10_n200.txt" > /tmp/r_mtm_full_inv3_s2.log 2>&1'
```

Verified running:

| Train seed | App | Log | Status at verification |
|---:|---|---|---|
| 3072 | `ap-RCksn1PAtRcqGexnedR2ha` | `/tmp/r_mtm_full_inv3_s3072.log` | `Tasks=1`; training at step 200 |
| 1 | `ap-VqE5DexBFKRHmSftzizfJg` | `/tmp/r_mtm_full_inv3_s1.log` | `Tasks=1`; training at step 575 |
| 2 | `ap-vCBxz6cvXn6WzGXtC99z7T` | `/tmp/r_mtm_full_inv3_s2.log` | `Tasks=1`; training at step 600 |

#### 2026-06-20 15:02 IST backup sentinel promotion

Pulled one last partial metrics snapshot, then stopped all remaining screen apps
to make the long-running jobs the source of truth. Final pre-stop screen
snapshot:

| Screen | Step | `emb_std` | Inverse loss | Forward loss | Interpretation |
|---|---:|---:|---:|---:|---|
| `mtm_screen_hybrid01_s1` | 3625 | 0.883 | 0.855 | 0.0287 | strong fallback, but includes weak SIGReg |
| `mtm_screen_hybrid03_s1` | 3725 | 0.949 | 0.831 | 0.0359 | strong fallback, more SIGReg-like |
| `mtm_screen_inv10_s1` | 3650 | 0.414 | 0.844 | 0.0099 | pure-MTM backup, high inverse scale |
| `mtm_screen_inv3_s1` | 5425 | 0.235 | 0.847 | 0.0041 | pure-MTM primary rescue |
| `mtm_screen_var01_g01_s1` | 4025 | 0.053 | 1.020 | 0.0032 | rejected, inverse near baseline |
| `mtm_screen_var05_g01_s1` | 3725 | 0.065 | 1.005 | 0.0048 | rejected, inverse near baseline |

Launched two seed-1 full sentinels as backups:

```bash
tmux new-session -d -s r_mtm_full_inv10_s1 'cd <repo> && nohup .venv/bin/modal run --detach --name r-mtm-full-inv10-s1 modal_app.py::train_then_evaluate --config-name lewm_masked --data reacher --subdir reacher/mtm_inv10_e10_s1 --overrides "trainer.max_epochs=10 early_stopping.enabled=false seed=1 loss.masked.inverse_weight=10.0" --eval-config-name reacher --eval-policy reacher/mtm_inv10_e10_s1/lewm_masked_epoch_10 --eval-overrides "eval.num_eval=200 eval.env_batch_size=10 seed=42 output.save_video=false output.filename=mtm_inv10_e10_s1_ep10_n200.txt" > /tmp/r_mtm_full_inv10_s1.log 2>&1'
tmux new-session -d -s r_mtm_full_hybrid01_s1 'cd <repo> && nohup .venv/bin/modal run --detach --name r-mtm-full-hybrid01-s1 modal_app.py::train_then_evaluate --config-name lewm_masked --data reacher --subdir reacher/mtm_hybrid01_e10_s1 --overrides "trainer.max_epochs=10 early_stopping.enabled=false seed=1 loss.sigreg.weight=0.01" --eval-config-name reacher --eval-policy reacher/mtm_hybrid01_e10_s1/lewm_masked_epoch_10 --eval-overrides "eval.num_eval=200 eval.env_batch_size=10 seed=42 output.save_video=false output.filename=mtm_hybrid01_e10_s1_ep10_n200.txt" > /tmp/r_mtm_full_hybrid01_s1.log 2>&1'
```

Verified running:

| Variant | Train seed | App | Log | Status at verification |
|---|---:|---|---|---|
| MTM inverse weight 10 | 1 | `ap-D0eLDTracUpnlcjktkcGAo` | `/tmp/r_mtm_full_inv10_s1.log` | `Tasks=1`; training at step 475 |
| MTM + weak SIGReg 0.01 | 1 | `ap-kwvnHENDeVLc0hg9qYM2PW` | `/tmp/r_mtm_full_hybrid01_s1.log` | `Tasks=1`; training at step 600 |

All one-epoch screen apps are now intentionally stopped. Current active Reacher
rescue jobs are only the five full train-then-eval apps:

* `inv3` seeds `3072,1,2`
* `inv10` seed `1`
* `hybrid01` seed `1`

#### 2026-06-20 15:12 IST early full-run metrics

Pulled committed `metrics.jsonl` snapshots from the five full runs. The promoted
`inv3` runs reproduce the noncollapse signal across all three seeds:

| Run | Step | `emb_std` | Inverse loss | Forward loss | SIGReg loss |
|---|---:|---:|---:|---:|---:|
| `mtm_inv3_e10_s3072` | 2775 | 0.247 | 0.826 | 0.0059 | - |
| `mtm_inv3_e10_s1` | 3750 | 0.242 | 0.828 | 0.0047 | - |
| `mtm_inv3_e10_s2` | 3725 | 0.244 | 0.835 | 0.0045 | - |
| `mtm_inv10_e10_s1` | 800 | 0.344 | 0.942 | 0.0201 | - |
| `mtm_hybrid01_e10_s1` | 950 | 0.816 | 0.947 | 0.0385 | 13.75 |

This is the first strong evidence that the Reacher MTM badness is not intrinsic
to the e10 schedule: increasing the inverse-dynamics weight keeps the failing
seeds out of the collapse basin early in training. Final answer still depends
on n=200 eval success at epoch 10.

#### 2026-06-20 15:15 IST pure-MTM lower-LR controls

The user raised the cleaner hypothesis that Reacher MTM may simply use too high
an LR. This is worth testing because it keeps the method as plain MTM: no new
module, no SIGReg, no variance floor, and no extra regularizer. Launched three
seed-1 full train-then-eval LR-only controls with the default MTM objective
(`loss.masked.inverse_weight=1.0`, `loss.sigreg.weight=0.0`):

```bash
tmux new-session -d -s r_mtm_full_lr1e5_s1 'cd <repo> && nohup .venv/bin/modal run --detach --name r-mtm-full-lr1e5-s1 modal_app.py::train_then_evaluate --config-name lewm_masked --data reacher --subdir reacher/mtm_lr1e5_e10_s1 --overrides "trainer.max_epochs=10 early_stopping.enabled=false seed=1 optimizer.lr=1e-5" --eval-config-name reacher --eval-policy reacher/mtm_lr1e5_e10_s1/lewm_masked_epoch_10 --eval-overrides "eval.num_eval=200 eval.env_batch_size=10 seed=42 output.save_video=false output.filename=mtm_lr1e5_e10_s1_ep10_n200.txt" > /tmp/r_mtm_full_lr1e5_s1.log 2>&1'
tmux new-session -d -s r_mtm_full_lr2e5_s1 'cd <repo> && nohup .venv/bin/modal run --detach --name r-mtm-full-lr2e5-s1 modal_app.py::train_then_evaluate --config-name lewm_masked --data reacher --subdir reacher/mtm_lr2e5_e10_s1 --overrides "trainer.max_epochs=10 early_stopping.enabled=false seed=1 optimizer.lr=2e-5" --eval-config-name reacher --eval-policy reacher/mtm_lr2e5_e10_s1/lewm_masked_epoch_10 --eval-overrides "eval.num_eval=200 eval.env_batch_size=10 seed=42 output.save_video=false output.filename=mtm_lr2e5_e10_s1_ep10_n200.txt" > /tmp/r_mtm_full_lr2e5_s1.log 2>&1'
tmux new-session -d -s r_mtm_full_lr3e5_s1 'cd <repo> && nohup .venv/bin/modal run --detach --name r-mtm-full-lr3e5-s1 modal_app.py::train_then_evaluate --config-name lewm_masked --data reacher --subdir reacher/mtm_lr3e5_e10_s1 --overrides "trainer.max_epochs=10 early_stopping.enabled=false seed=1 optimizer.lr=3e-5" --eval-config-name reacher --eval-policy reacher/mtm_lr3e5_e10_s1/lewm_masked_epoch_10 --eval-overrides "eval.num_eval=200 eval.env_batch_size=10 seed=42 output.save_video=false output.filename=mtm_lr3e5_e10_s1_ep10_n200.txt" > /tmp/r_mtm_full_lr3e5_s1.log 2>&1'
```

Verified running:

| Variant | Train seed | App | Log | Status at verification |
|---|---:|---|---|---|
| MTM LR 1e-5 | 1 | `ap-qPcUCjldA8MlxWeB9xxaAt` | `/tmp/r_mtm_full_lr1e5_s1.log` | `Tasks=1`; training at step 625 |
| MTM LR 2e-5 | 1 | `ap-lmBm268itga3ovNGlANt1x` | `/tmp/r_mtm_full_lr2e5_s1.log` | `Tasks=1`; training at step 600 |
| MTM LR 3e-5 | 1 | `ap-8eT4X70lKBonvnpzZFCJav` | `/tmp/r_mtm_full_lr3e5_s1.log` | `Tasks=1`; training at step 575 |

Early committed metric snapshot:

| Run | Step | Peak LR so far | `emb_std` | Inverse loss | Forward loss |
|---|---:|---:|---:|---:|---:|
| `mtm_lr1e5_e10_s1` | 675 | 5.28e-6 | 0.127 | 0.993 | 0.0203 |
| `mtm_lr2e5_e10_s1` | 675 | 1.06e-5 | 0.120 | 0.991 | 0.0167 |
| `mtm_lr3e5_e10_s1` | 650 | 1.53e-5 | 0.115 | 0.993 | 0.0150 |

Interpretation: lower LR remains a plausible pure-MTM rescue, but the early
signal is weaker than `inverse_weight=3.0`. The decisive criterion is whether
inverse loss starts dropping below the constant-action baseline before
`emb_std` collapses; final answer requires the epoch-10 n=200 evaluations.

#### 2026-06-21 00:55 IST completed rescue results and next LR3 seeds

All first-wave full Reacher rescue runs completed. Raw n=200 result files were
archived under `progress/evaluations/reacher-mtm-rescue/`.

| Variant | Train seed(s) | n=200 success | Final `emb_std` | Final inverse loss | Interpretation |
|---|---:|---:|---:|---:|---|
| MTM `inverse_weight=3.0` | 3072 | 60.5 | 0.130 | 0.845 | noncollapsed, but below seed's default MTM 68.0 |
| MTM `inverse_weight=3.0` | 1 | 81.5 | 0.134 | 0.833 | strong rescue of collapsed seed 1 |
| MTM `inverse_weight=3.0` | 2 | 67.5 | 0.139 | 0.807 | strong rescue of collapsed seed 2, but not top-tier |
| MTM `inverse_weight=10.0` | 1 | 57.5 | 0.211 | 0.833 | stable but over-weighted / planner-poor |
| MTM + weak SIGReg 0.01 | 1 | 60.0 | 0.965 | 0.834 | noncollapsed but not a useful paper direction |
| MTM `lr=1e-5` | 1 | 21.5 | 0.089 | 0.949 | underfits / ineffective |
| MTM `lr=2e-5` | 1 | 54.0 | 0.071 | 0.835 | learns inverse but not enough control quality |
| MTM `lr=3e-5` | 1 | 73.0 | 0.066 | 0.834 | best pure-objective/LR-only result so far |

Takeaways:

* Reacher default MTM failure is not inevitable. Both inverse-weight balancing
  and lower LR can rescue seed 1 from the 11.5% collapse.
* `inverse_weight=3.0` is the strongest single rescue result (seed 1 at 81.5)
  and rescues seed 2 to 67.5, but it hurts seed 3072 and is not cleanly robust.
* If we want no added regularizer/module and no objective-shape change, the
  best current direction is `optimizer.lr=3e-5`. It scored 73.0 on collapsed
  seed 1, while `1e-5` and `2e-5` were worse.
* Do not make a paper-facing Reacher MTM claim from these yet. The next necessary
  check is whether `lr=3e-5` holds on seeds 3072 and 2.

Launched the remaining `lr=3e-5` full e10+n=200 train-then-eval jobs:

```bash
tmux new-session -d -s r_mtm_full_lr3e5_s3072 'cd <repo> && nohup .venv/bin/modal run --detach --name r-mtm-full-lr3e5-s3072 modal_app.py::train_then_evaluate --config-name lewm_masked --data reacher --subdir reacher/mtm_lr3e5_e10_s3072 --overrides "trainer.max_epochs=10 early_stopping.enabled=false seed=3072 optimizer.lr=3e-5" --eval-config-name reacher --eval-policy reacher/mtm_lr3e5_e10_s3072/lewm_masked_epoch_10 --eval-overrides "eval.num_eval=200 eval.env_batch_size=10 seed=42 output.save_video=false output.filename=mtm_lr3e5_e10_s3072_ep10_n200.txt" > /tmp/r_mtm_full_lr3e5_s3072.log 2>&1'
tmux new-session -d -s r_mtm_full_lr3e5_s2 'cd <repo> && nohup .venv/bin/modal run --detach --name r-mtm-full-lr3e5-s2 modal_app.py::train_then_evaluate --config-name lewm_masked --data reacher --subdir reacher/mtm_lr3e5_e10_s2 --overrides "trainer.max_epochs=10 early_stopping.enabled=false seed=2 optimizer.lr=3e-5" --eval-config-name reacher --eval-policy reacher/mtm_lr3e5_e10_s2/lewm_masked_epoch_10 --eval-overrides "eval.num_eval=200 eval.env_batch_size=10 seed=42 output.save_video=false output.filename=mtm_lr3e5_e10_s2_ep10_n200.txt" > /tmp/r_mtm_full_lr3e5_s2.log 2>&1'
```

Initial verification:

| Train seed | App | Log | Status |
|---:|---|---|---|
| 3072 | `ap-h5WVAU61PmzK0Vqrbpf2VM` | `/tmp/r_mtm_full_lr3e5_s3072.log` | `Tasks=1`; training at step 250 |
| 2 | `ap-5QrZRkPy7bWlGxHAIYKJ2q` | `/tmp/r_mtm_full_lr3e5_s2.log` | `Tasks=1`; training at step 275 |

#### 2026-06-21 01:25 IST LR3 robustness completed

The `optimizer.lr=3e-5` robustness check completed and outputs were archived
under `progress/evaluations/reacher-mtm-rescue/`.

| Variant | Train seed | n=200 success | Final `emb_std` | Final inverse loss |
|---|---:|---:|---:|---:|
| MTM `lr=3e-5` | 3072 | 60.0 | 0.074 | 0.846 |
| MTM `lr=3e-5` | 1 | 73.0 | 0.066 | 0.834 |
| MTM `lr=3e-5` | 2 | 61.0 | 0.081 | 0.807 |

Decision:

* Lowering LR to `3e-5` helps enough to show the default `5e-5` LR contributes
  to the collapse, but it is not a robust Reacher rescue.
* Final Reacher rescue candidates tested so far are all insufficient for a clean
  paper-facing MTM win:
  * default e10 MTM: 68.0/11.5/13.5
  * `inverse_weight=3.0`: 60.5/81.5/67.5
  * `lr=3e-5`: 60.0/73.0/61.0
* Paper recommendation remains: do not present Reacher as an MTM win. Use it as
  an instability/failure case or omit from the main MTM-win narrative unless a
  stronger pure-MTM schedule lands later.

Modal app list is empty and no tmux sessions remain active.

## 2026-06-21 — Reacher checkpoint-selection audit

### Intent

Test the hypothesis that epoch 10 is not the best Reacher MTM checkpoint. This
first audit evaluates intermediate checkpoints for the two rescue families that
matter most:

* MTM `inverse_weight=3.0`
* MTM `optimizer.lr=3e-5`

To keep the first wave bounded, launch epochs 3/5/7/9 for the weaker final
seeds (`3072` and `2`) at the same n=200 protocol (`eval.num_eval=200`,
`eval.env_batch_size=10`, `seed=42`, `goal_offset_steps=25`, `eval_budget=50`).

### Commands

Pattern:

```bash
tmux new-session -d -s r_ckpt_<variant>_s<seed>_e<epoch> \
  'cd <repo> && nohup .venv/bin/modal run --detach --name r-ckpt-<variant>-s<seed>-e<epoch> modal_app.py::evaluate --config-name reacher --policy reacher/<subdir>/lewm_masked_epoch_<epoch> --overrides "eval.num_eval=200 eval.env_batch_size=10 seed=42 output.save_video=false output.filename=ckpt_<variant>_s<seed>_e<epoch>_n200.txt" > /tmp/r_ckpt_<variant>_s<seed>_e<epoch>.log 2>&1'
```

Launched first wave:

| Variant | Seed | Epochs |
|---|---:|---|
| `inv3` | 3072 | 3, 5, 7, 9 |
| `inv3` | 2 | 3, 5, 7, 9 |
| `lr3` | 3072 | 3, 5, 7, 9 |
| `lr3` | 2 | 3, 5, 7, 9 |

### Status

Modal only admitted 10 concurrent eval tasks. Verified running with `Tasks=1`
and eval logs:

* `inv3` seed 3072 epochs 3, 5
* `inv3` seed 2 epochs 3, 5, 7, 9
* `lr3` seed 3072 epochs 5, 9
* `lr3` seed 2 epochs 3, 9

Stopped six zero-task apps to avoid ambiguous queued state. Relaunch these after
the first wave finishes:

* `inv3` seed 3072 epochs 7, 9
* `lr3` seed 3072 epochs 3, 7
* `lr3` seed 2 epochs 5, 7

### 2026-06-21 23:03 IST Update

First-wave completed checkpoint results so far:

| Variant | Train seed | Epoch | n=200 success |
|---|---:|---:|---:|
| `inv3` | 3072 | 3 | 41.0 |
| `inv3` | 3072 | 5 | 54.5 |
| `inv3` | 2 | 3 | 55.0 |
| `inv3` | 2 | 5 | 63.5 |
| `inv3` | 2 | 7 | 53.0 |
| `lr3` | 3072 | 5 | 55.5 |
| `lr3` | 3072 | 9 | 68.5 |
| `lr3` | 2 | 3 | 1.5 |
| `lr3` | 2 | 9 | 65.5 |

Still running from first wave:

* `inv3` seed 2 epoch 9

Relaunched the six deferred zero-task checkpoint evals with distinct app/log
names and verified all are live (`Tasks=1`) with real eval progress:

| Variant | Train seed | Epoch | Log |
|---|---:|---:|---|
| `inv3` | 3072 | 7 | `/tmp/r_ckpt2_inv3_s3072_e7.log` |
| `inv3` | 3072 | 9 | `/tmp/r_ckpt2_inv3_s3072_e9.log` |
| `lr3` | 3072 | 3 | `/tmp/r_ckpt2_lr3_s3072_e3.log` |
| `lr3` | 3072 | 7 | `/tmp/r_ckpt2_lr3_s3072_e7.log` |
| `lr3` | 2 | 5 | `/tmp/r_ckpt2_lr3_s2_e5.log` |
| `lr3` | 2 | 7 | `/tmp/r_ckpt2_lr3_s2_e7.log` |

Early read: epoch 10 is not obviously dominant for Reacher. For the lower-LR
family, epoch 9 is already stronger than several earlier checkpoints on the
same seeds, but epoch 10 still needs to be compared against the full grid before
making a paper-facing checkpoint-selection decision.

### 2026-06-21 23:32 IST Default-MTM checkpoint audit launch

Checked default Reacher MTM checkpoint inventories:

* `reacher/lewm_masked`
* `reacher/lewm_masked_s1`
* `reacher/lewm_masked_s2`

All have epoch 1-10 object checkpoints. Pulled default metrics locally to
choose audit points. The collapsed seeds already have very low epoch-end
embedding std by epoch 1 (`s1`: 0.00427, `s2`: 0.00449), while seed 3072 stays
noncollapsed through epoch 10. Therefore launched a narrow n=200 default audit:

| Subdir | Train seed | Epochs | Rationale |
|---|---:|---|---|
| `reacher/lewm_masked` | 3072 | 9 | check whether healthy default seed improves before e10 |
| `reacher/lewm_masked_s1` | 1 | 1, 5, 9 | test earliest saved checkpoint plus late collapsed controls |
| `reacher/lewm_masked_s2` | 2 | 1, 5, 9 | test earliest saved checkpoint plus late collapsed controls |

Launch pattern:

```bash
tmux new-session -d -s r_ckpt_def_<seed>_e<epoch> \
  'cd <repo> && nohup .venv/bin/modal run --detach --name r-ckpt-def-<seed>-e<epoch> modal_app.py::evaluate --config-name reacher --policy reacher/<subdir>/lewm_masked_epoch_<epoch> --overrides "eval.num_eval=200 eval.env_batch_size=10 seed=42 output.save_video=false output.filename=ckpt_default_<seed>_e<epoch>_n200.txt" > /tmp/r_ckpt_def_<seed>_e<epoch>.log 2>&1'
```

Initial verification: all seven default-audit apps were admitted with
`Tasks=1`; logs reached the `eval.py` command. Awaiting `Evaluating batch`
lines before marking the default audit fully underway.

### 2026-06-22 01:15 IST Final checkpoint-audit result

Archived raw outputs and logs under:

```text
progress/evaluations/reacher-checkpoint-audit/
```

Default MTM checkpoint audit:

| Train seed | Epoch 1 | Epoch 5 | Epoch 9 | Epoch 10 baseline |
|---:|---:|---:|---:|---:|
| 3072 | - | - | 64.5 | 68.0 |
| 1 | 12.0 | 12.0 | 10.0 | 11.5 |
| 2 | 10.0 | 11.5 | 10.5 | 13.5 |

`loss.masked.inverse_weight=3.0` checkpoint audit:

| Train seed | Epoch 3 | Epoch 5 | Epoch 7 | Epoch 9 | Epoch 10 baseline |
|---:|---:|---:|---:|---:|---:|
| 3072 | 41.0 | 54.5 | 54.0 | 68.0 | 60.5 |
| 1 | - | - | - | 76.0 | 81.5 |
| 2 | 55.0 | 63.5 | 53.0 | 67.0 | 67.5 |

`optimizer.lr=3e-5` checkpoint audit:

| Train seed | Epoch 3 | Epoch 5 | Epoch 7 | Epoch 9 | Epoch 10 baseline |
|---:|---:|---:|---:|---:|---:|
| 3072 | 37.5 | 55.5 | 60.0 | 68.5 | 60.0 |
| 1 | - | - | - | 74.0 | 73.0 |
| 2 | 1.5 | 42.5 | 54.0 | 65.5 | 61.0 |

Decision:

* Default MTM e10 is not hiding a better checkpoint. The collapsed seeds are
  already poor at epoch 1, and the healthy seed is better at epoch 10 than
  epoch 9.
* For `inverse_weight=3.0`, epoch 9 and epoch 10 are effectively tied by
  three-seed mean: epoch 9 = 70.3, epoch 10 = 69.8.
* For pure MTM with only lower LR (`optimizer.lr=3e-5`), epoch 9 is clearly
  better than epoch 10 across all three seeds: 68.5 / 74.0 / 65.5, mean 69.3,
  versus epoch 10 mean 64.7.
* Paper recommendation for Reacher MTM, if included without adding modules or
  regularizers: report `optimizer.lr=3e-5`, epoch 9, n=200, and frame it as
  competitive/schedule-sensitive rather than a strong Reacher win.

Final Modal state: no active/running apps; no tmux sessions.

## 2026-06-22 — Reacher representation-objective experiments

### Intent

Test objective-derived representation improvements for Reacher without task
labels, SIGReg, variance-floor regularizers, LR changes, or inverse-weight
tweaks. Start with fragile seed 1 only and monitor early collapse signals before
spending on a full multi-seed matrix.

Variants:

* Multi-step latent overshooting: `lewm_masked_h`, H=5, standard inverse weight.
* Multi-step action discrimination: new `lewm_masked_ms_inverse`, standard
  one-step forward loss plus horizon-conditioned endpoint inverse
  `(z_t, z_{t+k}, e_k) -> a_t`, k in [1,5].

### Commands

#### Diagnostics

```bash
.venv/bin/python -m pytest -q tests/test_masked_transition.py tests/test_config_sanity.py
.venv/bin/python -m py_compile train.py module.py jepa.py
```

Result: targeted tests passed (`19 passed`), py_compile passed.

#### Train + Eval

Launching seed-1 sentinels with the standard e10 schedule and n=200 eval:

```bash
tmux new-session -d -s r_obj_overshoot_s1 'cd <repo> && nohup .venv/bin/modal run --detach --name r-obj-overshoot-s1 modal_app.py::train_then_evaluate --config-name lewm_masked_h --data reacher --subdir reacher/mtm_overshoot_e10_s1 --overrides "trainer.max_epochs=10 early_stopping.enabled=false seed=1" --eval-config-name reacher --eval-policy reacher/mtm_overshoot_e10_s1 --eval-overrides "eval.num_eval=200 eval.env_batch_size=10 seed=42 output.save_video=false output.filename=mtm_overshoot_e10_s1_n200.txt" > /tmp/r_obj_overshoot_s1.log 2>&1'
tmux new-session -d -s r_obj_msinv_s1 'cd <repo> && nohup .venv/bin/modal run --detach --name r-obj-msinv-s1 modal_app.py::train_then_evaluate --config-name lewm_masked_ms_inverse --data reacher --subdir reacher/mtm_ms_inverse_e10_s1 --overrides "trainer.max_epochs=10 early_stopping.enabled=false seed=1" --eval-config-name reacher --eval-policy reacher/mtm_ms_inverse_e10_s1 --eval-overrides "eval.num_eval=200 eval.env_batch_size=10 seed=42 output.save_video=false output.filename=mtm_ms_inverse_e10_s1_n200.txt" > /tmp/r_obj_msinv_s1.log 2>&1'
```

### Monitoring criteria

Stop early if the known collapse signature appears: `train_step/emb_std` falls
near zero during epoch 0/1 while inverse/action loss stays near chance (~1.0).
Promote to seeds 3072 and 2 only if seed 1 stays noncollapsed.

### Artifacts

* Overshooting app: `ap-Sglp40SzY4Co2Z0Jrwch0i`
* Multi-step inverse app: `ap-DCoc1Ot1BhhYs1HRu1L0YR`
* Overshooting subdir: `reacher/mtm_overshoot_e10_s1`
* Multi-step inverse subdir: `reacher/mtm_ms_inverse_e10_s1`
* Local logs: `/tmp/r_obj_overshoot_s1.log`,
  `/tmp/r_obj_msinv_s1.log`, `/tmp/r_obj_collapse_monitor.log`

### Result

Initial monitoring, still epoch 0:

* At about 50 logged train-step rows, overshooting had recent
  `train_step/emb_std` min/mean/max = 0.0762 / 0.0825 / 0.0869.
* At about 50 logged train-step rows, multi-step inverse had recent
  `train_step/emb_std` min/mean/max = 0.0781 / 0.0830 / 0.0898.
* The local collapse monitor's first pull saw overshooting row 59
  `last=0.07617`, `mean10=0.07544`, and multi-step inverse row 62
  `last=0.07715`, `mean10=0.07656`.
* The second monitor pull saw overshooting row 96 `last=0.06079`,
  `mean10=0.06177`, and multi-step inverse row 108 `last=0.06641`,
  `mean10=0.06455`.
* Final overshooting metrics before stop: 423 logged train-step rows, roughly
  step 10,575 / epoch 0. `train_step/emb_std` fell from 0.2871 at startup and
  0.0776 near row 50 to `last=0.01624`, `mean10=0.01700`. Forward loss fell to
  about `3e-4`, while inverse loss stayed near chance (`~1.0`).
* Final multi-step inverse metrics before stop: 472 logged train-step rows,
  roughly step 11,800 / epoch 0. `train_step/emb_std` fell from 0.2949 at
  startup and 0.0781 near row 50 to `last=0.01562`, `mean10=0.01718`. Forward
  loss fell to about `3e-4`, while inverse/action loss stayed near chance
  (`~1.0`).
* Both runs were stopped before completing training, so no n=200 eval ran.

### Decision

Reject both Reacher objective sentinels under the standard e10 schedule. The
multi-step overshooting loss and the separate horizon-conditioned inverse/action
discrimination loss both reproduced the same collapse mode as fragile default
MTM seeds. Do not launch seeds 3072/2 for these arms without a more fundamental
change to the objective or collapse gate.

Final Modal/local state: both apps stopped with zero tasks; local
`r_obj_collapse_monitor` tmux session stopped.

## 2026-06-22 — Reacher collapse mechanism sentinel: AC-CPC short run

### Intent

Test the leading collapse hypothesis with the smallest useful experiment.

Hypothesis: the Reacher MTM failures are not fixed by adding more horizon or a
larger inverse endpoint task because the core failure is the same-encoder latent
MSE fixed point. Forward MSE can reduce loss by shrinking the online encoder's
latent scale; continuous inverse-action MSE is too weak/noisy to oppose this
before collapse, so inverse loss stays near chance while forward loss goes to
zero. A contrastive future-identification objective should remove that collapse
direction: a constant latent cannot identify the true future among negatives.

Minimum falsification experiment: run a short Reacher seed-1 AC-CPC sentinel,
same e10 optimizer/LR schedule and same AR predictor family, but stop at 4000
steps with no checkpoint/eval. Success criterion is not planning SR yet; it is
whether `emb_std` stays alive while the contrastive loss falls. Failure would
mean collapse is broader than the latent-MSE self-target mechanism.

### Commands

#### Diagnostics

```bash
.venv/bin/python -m pytest -q tests/test_ac_cpc.py tests/test_config_sanity.py
.venv/bin/python -m py_compile train.py module.py jepa.py
```

Result: targeted AC-CPC/config tests passed (`15 passed`, one existing warning),
py_compile passed.

#### Train

```bash
tmux new-session -d -s r_accpc_sentinel_s1 'cd <repo> && nohup .venv/bin/modal run --detach --name r-accpc-sentinel-s1 modal_app.py::train --config-name lewm_accpc --data reacher --subdir reacher/accpc_collapse_sentinel_s1_4k --overrides "seed=1 +trainer.max_steps=4000 trainer.max_epochs=10 checkpoint.enabled=false dump_object=false wandb.enabled=false early_stopping.enabled=false logging.log_every_n_steps=25" > /tmp/r_accpc_sentinel_s1.log 2>&1'
```

### Artifacts

* App: `ap-eVLarM0CAl64w8R3YLH6fI`
* Subdir: `reacher/accpc_collapse_sentinel_s1_4k`
* Local log: `/tmp/r_accpc_sentinel_s1.log`
* Local pulled metrics: `/tmp/reacher_accpc_sentinel_metrics/metrics.jsonl`

### Result

Initial committed metrics through about step 500:

* `train_step/emb_std` increased from `0.06284` to `0.07123` instead of
  shrinking toward zero.
* `train_step/cpc_loss` fell from `5.48` to `1.78`.
* This is the opposite early signature from the MSE/inverse sentinel failures,
  which drifted downward early and later crossed `mean10 emb_std < 0.02`.

Mid-run metrics at step 2250:

* `train_step/emb_std` was `0.07142`, recent mean10 `0.07147`.
* `train_step/cpc_loss` was `1.05`, recent mean10 `1.08`.

Final metrics at step 4000:

* `train_step/emb_std` was `0.07149`, recent mean10 `0.07154`.
* `train_step/cpc_loss` was `1.02`, recent mean10 `1.00`.
* The app stopped normally at `max_steps=4000`; no checkpoints or eval were
  requested.

### Decision

This validates the narrow collapse hypothesis: removing same-encoder latent MSE
and using future identification prevents the early Reacher collapse under the
same seed/LR schedule. It does not yet validate planning quality; PushT evidence
already warns that clean AC-CPC can produce noncollapsed but CEM-poor geometry.
The next minimum experiment should therefore test a geometry-preserving hybrid
or MSE-lite objective, not another stronger inverse-MSE arm.

## 2026-06-22 — Reacher targeted fix sentinel: MTM + true action NCE

### Intent

Test the targeted fix implied by the AC-CPC mechanism run. The earlier
"multi-step inverse/action discrimination" run was still continuous action MSE
regression. This sentinel keeps standard MTM's forward latent MSE and planner
geometry, but replaces inverse-action MSE with true in-batch action
discrimination:

```text
score((z_t,z_{t+1}), a_j) = -||g(z_t,z_{t+1}) - a_j||^2 / temperature
```

Each transition must classify its own action among all other transition actions
in the batch/window. A collapsed encoder gives one constant inverse prediction,
so it cannot satisfy many different action labels.

### Commands

#### Diagnostics

```bash
.venv/bin/python -m pytest -q tests/test_masked_transition.py tests/test_config_sanity.py
.venv/bin/python -m py_compile train.py module.py jepa.py
```

Result: targeted tests passed (`22 passed`), py_compile passed, and Hydra
composition for `lewm_masked_action_nce` on Reacher with `+trainer.max_steps=4000`
succeeded.

#### Train

```bash
tmux new-session -d -s r_mtm_action_nce_s1 'cd <repo> && nohup .venv/bin/modal run --detach --name r-mtm-action-nce-s1 modal_app.py::train --config-name lewm_masked_action_nce --data reacher --subdir reacher/mtm_action_nce_sentinel_s1_4k --overrides "seed=1 +trainer.max_steps=4000 trainer.max_epochs=10 checkpoint.enabled=false dump_object=false wandb.enabled=false early_stopping.enabled=false logging.log_every_n_steps=25" > /tmp/r_mtm_action_nce_s1.log 2>&1'
```

### Artifacts

* App: `ap-7WCuUOLOVBjdKDI0SQAgel`
* Subdir: `reacher/mtm_action_nce_sentinel_s1_4k`
* Local log: `/tmp/r_mtm_action_nce_s1.log`
* Local pulled metrics: `/tmp/reacher_action_nce_sentinel_metrics/metrics.jsonl`

### Result

Initial metrics through about step 575:

* `train_step/emb_std` increased from `0.26367` to `0.47461`, recent mean10
  `0.47695`.
* `train_step/forward_loss` fell from `0.1199` to `0.0198`.
* `train_step/inverse_loss` fell from `9.15` to `8.17`.

Mid-run metrics at step 2150:

* `train_step/emb_std` was `0.37500`, recent mean10 `0.37930`.
* `train_step/forward_loss` was `0.01266`, recent mean10 `0.01411`.
* `train_step/inverse_loss` was `7.56`, recent mean10 `7.30`.

Final metrics at step 4000:

* `train_step/emb_std` was `0.34570`, recent mean10 `0.34648`.
* `train_step/forward_loss` was `0.00710`, recent mean10 `0.00738`.
* `train_step/inverse_loss` was `7.24`, recent mean10 `7.25`.
* The app stopped normally at `max_steps=4000`; no checkpoints or eval were
  requested.

### Decision

Promote MTM + true action NCE to the next Reacher fix candidate. It preserves
MTM's forward latent MSE path while preventing the early collapse that killed
default MTM, multi-step overshooting, and endpoint inverse MSE. The next minimum
promotion should be a bounded train-then-eval seed-1 run with checkpoint/eval
enabled, not immediate multi-seed fanout.

## 2026-06-22 — Reacher MTM + action NCE full seed-1 candidate

### Intent

Promote the successful 4000-step MTM + action-NCE sentinel to one full seed-1
train-then-evaluate run, still under the standard e10 LR schedule. This is the
minimum performance-bearing run for the candidate fix: one seed only, n=200 eval
if training survives, with an early collapse monitor to avoid wasting compute.

### Commands

#### Train + Eval

```bash
tmux new-session -d -s r_action_nce_full_s1 'cd <repo> && nohup .venv/bin/modal run --detach --name r-action-nce-full-s1 modal_app.py::train_then_evaluate --config-name lewm_masked_action_nce --data reacher --subdir reacher/mtm_action_nce_e10_s1 --overrides "trainer.max_epochs=10 early_stopping.enabled=false seed=1" --eval-config-name reacher --eval-policy reacher/mtm_action_nce_e10_s1 --eval-overrides "eval.num_eval=200 eval.env_batch_size=10 seed=42 output.save_video=false output.filename=mtm_action_nce_e10_s1_n200.txt" > /tmp/r_action_nce_full_s1.log 2>&1'
```

#### Monitor

Local tmux monitor `r_action_nce_full_monitor` pulls
`reacher/mtm_action_nce_e10_s1/metrics.jsonl` every 5 minutes and stops app
`ap-3z4LVRSm4r0c8JR9hRs2t1` if recent mean10 `train_step/emb_std < 0.05` after
at least 80 logged train-step rows.

### Artifacts

* App: `ap-3z4LVRSm4r0c8JR9hRs2t1`
* Subdir: `reacher/mtm_action_nce_e10_s1`
* Local train log: `/tmp/r_action_nce_full_s1.log`
* Local monitor log: `/tmp/r_action_nce_full_monitor.log`
* Eval output: `reacher/mtm_action_nce_e10_s1_n200.txt`

### Result

Initial monitor pulls:

* rows=24, global step 600, `last emb_std=0.35352`,
  `mean10 emb_std=0.28535`.
* rows=79, global step 1975, `last emb_std=0.39258`,
  `mean10 emb_std=0.37676`.
* rows=136, global step 3400, `last emb_std=0.47656`,
  `mean10 emb_std=0.47383`.
* rows=193, global step 4825, `last emb_std=0.47852`,
  `mean10 emb_std=0.48242`.

Final training metrics:

* 5118 logged train-step rows through global step 127,950.
* `train_step/emb_std`: first `0.30664`, final `0.28906`, final mean10
  `0.28887`.
* `train_step/forward_loss`: first `0.2471`, final `0.00162`, final mean10
  `0.00154`.
* `train_step/inverse_loss`: first `9.19`, final `7.26`, final mean10 `7.12`.

Eval result:

* Protocol: Reacher n=200, eval seed 42, CEM n_steps=30, video off.
* Success rate: **57.0%** (`113/200`).
* Evaluation time: 1836.2 seconds.

### Decision

Action-NCE fixes the Reacher collapse mechanism, but the seed-1 e10 planning
result is not strong enough to promote to multi-seed paper reporting. It is a
useful mechanistic result: contrastive inverse pressure preserves latent
variance while MTM forward MSE trains, but the resulting geometry/control signal
is still below the better Reacher MTM lower-LR checkpoint (~74% for seed 1) and
below matched SIGReg seed-1 (~69% in the e10 comparison / stronger under e100).
Next candidates should focus on geometry/performance, not just anti-collapse.

Final Modal/local state: no active Modal apps and no `r_action_nce_full_*` tmux
sessions.

## 2026-06-23 — Reacher action-NCE inverse-weight sentinel sweep

### Intent

Find the smallest action-NCE inverse weight that prevents Reacher collapse while
allowing MTM's forward geometry to dominate more than the failed weight-1.0 full
candidate. The weight-1.0 run fixed collapse but scored only 57.0% n=200,
suggesting the contrastive inverse objective may have over-shaped the latent
space.

Weights: `0.02`, `0.05`, `0.10`. Each run is Reacher seed 1, standard e10 LR
schedule, `max_steps=4000`, no checkpoints, no eval.

### Commands

#### Train

```bash
tmux new-session -d -s r_nce_w002_s1 'cd <repo> && nohup .venv/bin/modal run --detach --name r-nce-w002-s1 modal_app.py::train --config-name lewm_masked_action_nce --data reacher --subdir reacher/mtm_action_nce_w002_sentinel_s1_4k --overrides "seed=1 +trainer.max_steps=4000 trainer.max_epochs=10 checkpoint.enabled=false dump_object=false wandb.enabled=false early_stopping.enabled=false logging.log_every_n_steps=25 loss.masked.inverse_weight=0.02" > /tmp/r_nce_w002_s1.log 2>&1'
tmux new-session -d -s r_nce_w005_s1 'cd <repo> && nohup .venv/bin/modal run --detach --name r-nce-w005-s1 modal_app.py::train --config-name lewm_masked_action_nce --data reacher --subdir reacher/mtm_action_nce_w005_sentinel_s1_4k --overrides "seed=1 +trainer.max_steps=4000 trainer.max_epochs=10 checkpoint.enabled=false dump_object=false wandb.enabled=false early_stopping.enabled=false logging.log_every_n_steps=25 loss.masked.inverse_weight=0.05" > /tmp/r_nce_w005_s1.log 2>&1'
tmux new-session -d -s r_nce_w010_s1 'cd <repo> && nohup .venv/bin/modal run --detach --name r-nce-w010-s1 modal_app.py::train --config-name lewm_masked_action_nce --data reacher --subdir reacher/mtm_action_nce_w010_sentinel_s1_4k --overrides "seed=1 +trainer.max_steps=4000 trainer.max_epochs=10 checkpoint.enabled=false dump_object=false wandb.enabled=false early_stopping.enabled=false logging.log_every_n_steps=25 loss.masked.inverse_weight=0.10" > /tmp/r_nce_w010_s1.log 2>&1'
```

### Artifacts

* `0.02` app: `ap-By6HNt0fNbvkYyrL60YuD9`, subdir
  `reacher/mtm_action_nce_w002_sentinel_s1_4k`
* `0.05` app: `ap-D2S6BKC6LDtUmdXspBwsSc`, subdir
  `reacher/mtm_action_nce_w005_sentinel_s1_4k`
* `0.10` app: `ap-wQdmwcRv5XSydUaPYNPDxK`, subdir
  `reacher/mtm_action_nce_w010_sentinel_s1_4k`
* Local logs: `/tmp/r_nce_w002_s1.log`, `/tmp/r_nce_w005_s1.log`,
  `/tmp/r_nce_w010_s1.log`

### Result

Initial committed metrics:

| Weight | Step | `emb_std` last | `emb_std` mean10 | Forward loss mean10 | NCE loss mean10 |
|---:|---:|---:|---:|---:|---:|
| 0.02 | 375 | 0.0815 | 0.0842 | 0.00847 | 9.096 |
| 0.05 | 375 | 0.0859 | 0.0901 | 0.00948 | 9.102 |
| 0.10 | 550 | 0.0879 | 0.0901 | 0.00936 | 9.074 |

### Decision

Continue all three to 4000 steps. Early metrics are not collapsed, but they are
much closer to the collapse-risk regime than weight 1.0, so final short-run
metrics are required before promoting any low-weight candidate.

Final 4000-step metrics:

| Weight | Final `emb_std` | Mean10 `emb_std` | Mean10 forward loss | Mean10 NCE loss | Decision |
|---:|---:|---:|---:|---:|---|
| 0.02 | 0.0471 | 0.0470 | 0.00238 | 9.095 | Reject: effectively at collapse threshold |
| 0.05 | 0.0569 | 0.0567 | 0.00340 | 9.095 | Reject/borderline: too little margin for full e10 |
| 0.10 | 0.0674 | 0.0665 | 0.00463 | 9.094 | Not enough margin; do not promote directly |

Updated decision: low weights reduce contrastive domination but allow the latent
scale to keep shrinking. The next minimum bracket is `0.20`, `0.30`, `0.50`,
still as 4000-step sentinels, looking for a final `emb_std` comfortably above
0.1 without the weight-1.0 over-shaping risk.

Second bracket, final 4000-step metrics:

| Weight | Final `emb_std` | Mean10 `emb_std` | Mean10 forward loss | Mean10 NCE loss | Decision |
|---:|---:|---:|---:|---:|---|
| 0.20 | 0.1157 | 0.1170 | 0.00198 | 8.222 | Borderline; may decay too close to collapse in full e10 |
| 0.30 | 0.1650 | 0.1656 | 0.00564 | 7.363 | Promote as balanced candidate |
| 0.50 | 0.2148 | 0.2150 | 0.00467 | 7.276 | Stable but closer to weight-1.0 over-shaping |

Decision: promote `loss.masked.inverse_weight=0.30` to one full seed-1
train-then-evaluate run. Do not fan out seeds yet.

## 2026-06-23 — Reacher action-NCE weight-0.30 full seed-1 candidate

### Intent

Run the balanced action-NCE candidate selected by the 4000-step weight sweep:
`loss.masked.inverse_weight=0.30`. This is intended to keep enough contrastive
inverse pressure to prevent collapse while reducing the over-shaping seen at
weight 1.0.

### Commands

#### Train + Eval

```bash
tmux new-session -d -s r_nce_w030_full_s1 'cd <repo> && nohup .venv/bin/modal run --detach --name r-nce-w030-full-s1 modal_app.py::train_then_evaluate --config-name lewm_masked_action_nce --data reacher --subdir reacher/mtm_action_nce_w030_e10_s1 --overrides "trainer.max_epochs=10 early_stopping.enabled=false seed=1 loss.masked.inverse_weight=0.30" --eval-config-name reacher --eval-policy reacher/mtm_action_nce_w030_e10_s1 --eval-overrides "eval.num_eval=200 eval.env_batch_size=10 seed=42 output.save_video=false output.filename=mtm_action_nce_w030_e10_s1_n200.txt" > /tmp/r_nce_w030_full_s1.log 2>&1'
```

#### Monitor

Local tmux monitor `r_nce_w030_full_monitor` pulls
`reacher/mtm_action_nce_w030_e10_s1/metrics.jsonl` every 5 minutes and stops
app `ap-3oddALYeDGAaHJFTbqLrg5` if recent mean10 `train_step/emb_std < 0.08`
after at least 80 logged train-step rows.

### Artifacts

* App: `ap-3oddALYeDGAaHJFTbqLrg5`
* Subdir: `reacher/mtm_action_nce_w030_e10_s1`
* Local train log: `/tmp/r_nce_w030_full_s1.log`
* Local monitor log: `/tmp/r_nce_w030_full_monitor.log`
* Eval output target: `reacher/mtm_action_nce_w030_e10_s1_n200.txt`

### Result

Initial monitor pull: rows=18, global step 450, `last emb_std=0.13672`,
`mean10 emb_std=0.14785`.

Live status at 2026-06-23 13:06 IST: app `ap-3oddALYeDGAaHJFTbqLrg5`
still has one running task. Training is around epoch 4/10, global step
44,350/127,960. The monitor has reached rows=1,745, step=43,625 with
`last emb_std=0.27344` and `mean10 emb_std=0.27461`, comfortably above the
0.08 kill threshold.

Final train state: completed 10 epochs / 127,960 steps and wrote
`reacher/mtm_action_nce_w030_e10_s1/lewm_masked_action_nce_epoch_10_object.ckpt`.
The final logged train-step `emb_std` was `0.18262`, with final monitor
`mean10 emb_std=0.18252`, so the run remained noncollapsed through the end.

Eval: `n=200`, `eval.env_batch_size=10`, `seed=42`, CEM `300/30/30`, output
`reacher/mtm_action_nce_w030_e10_s1_n200.txt`.

Result: `70.5%` success rate (`141/200`), evaluation time 1656.6s.

### Decision

Keep as an improved Reacher action-NCE candidate. Weight 0.30 is much better
than weight 1.0 (`57.0%`) and clearly prevents collapse, but it is still only
one training seed and remains below the best SIGReg/Reacher e100 controls. The
next decision should be whether to run seeds `3072` and `2` for this exact
setting or audit earlier checkpoints for this weight before seed fanout.

---

## 2026-06-23 — Reacher e100 MTM seed-1/2 earlier-checkpoint sweep

### Intent

Check whether the e100 Reacher MTM seed-1/2 epoch-30 failures are late-training
instabilities or whether those seeds are already bad earlier in the 100-epoch LR
schedule. Compare against the existing seed-3072 e100 MTM milestone trajectory
and the existing seed-1/2 epoch-30 results.

Evaluate seeds `1` and `2` at epochs `10`, `17`, `25`, and `29`. Epoch 30 is
already complete: seed 1 scored 10.5 and seed 2 scored 12.0.

### Commands

#### Diagnostics

```bash
.venv/bin/modal volume ls multi-future-lewm-cache /reacher/lewm_masked_e100_s1
.venv/bin/modal volume ls multi-future-lewm-cache /reacher/lewm_masked_e100_s2
.venv/bin/modal app list | head -30
```

Both seed directories contain `lewm_masked_epoch_1_object.ckpt` through
`lewm_masked_epoch_30_object.ckpt`.

#### Eval

```bash
nohup .venv/bin/modal run --detach modal_app.py::evaluate --config-name reacher --policy reacher/lewm_masked_e100_s1/lewm_masked_epoch_10 --overrides "eval.num_eval=200 eval.env_batch_size=10 seed=42 output.save_video=false output.filename=e100_mtm_reacher_s1_ep10_n200_early.txt" > /tmp/e100_mtm_reacher_s1_ep10_early.log 2>&1 &
nohup .venv/bin/modal run --detach modal_app.py::evaluate --config-name reacher --policy reacher/lewm_masked_e100_s1/lewm_masked_epoch_17 --overrides "eval.num_eval=200 eval.env_batch_size=10 seed=42 output.save_video=false output.filename=e100_mtm_reacher_s1_ep17_n200_early.txt" > /tmp/e100_mtm_reacher_s1_ep17_early.log 2>&1 &
nohup .venv/bin/modal run --detach modal_app.py::evaluate --config-name reacher --policy reacher/lewm_masked_e100_s1/lewm_masked_epoch_25 --overrides "eval.num_eval=200 eval.env_batch_size=10 seed=42 output.save_video=false output.filename=e100_mtm_reacher_s1_ep25_n200_early.txt" > /tmp/e100_mtm_reacher_s1_ep25_early.log 2>&1 &
nohup .venv/bin/modal run --detach modal_app.py::evaluate --config-name reacher --policy reacher/lewm_masked_e100_s1/lewm_masked_epoch_29 --overrides "eval.num_eval=200 eval.env_batch_size=10 seed=42 output.save_video=false output.filename=e100_mtm_reacher_s1_ep29_n200_early.txt" > /tmp/e100_mtm_reacher_s1_ep29_early.log 2>&1 &
nohup .venv/bin/modal run --detach modal_app.py::evaluate --config-name reacher --policy reacher/lewm_masked_e100_s2/lewm_masked_epoch_10 --overrides "eval.num_eval=200 eval.env_batch_size=10 seed=42 output.save_video=false output.filename=e100_mtm_reacher_s2_ep10_n200_early.txt" > /tmp/e100_mtm_reacher_s2_ep10_early.log 2>&1 &
nohup .venv/bin/modal run --detach modal_app.py::evaluate --config-name reacher --policy reacher/lewm_masked_e100_s2/lewm_masked_epoch_17 --overrides "eval.num_eval=200 eval.env_batch_size=10 seed=42 output.save_video=false output.filename=e100_mtm_reacher_s2_ep17_n200_early.txt" > /tmp/e100_mtm_reacher_s2_ep17_early.log 2>&1 &
nohup .venv/bin/modal run --detach modal_app.py::evaluate --config-name reacher --policy reacher/lewm_masked_e100_s2/lewm_masked_epoch_25 --overrides "eval.num_eval=200 eval.env_batch_size=10 seed=42 output.save_video=false output.filename=e100_mtm_reacher_s2_ep25_n200_early.txt" > /tmp/e100_mtm_reacher_s2_ep25_early.log 2>&1 &
nohup .venv/bin/modal run --detach modal_app.py::evaluate --config-name reacher --policy reacher/lewm_masked_e100_s2/lewm_masked_epoch_29 --overrides "eval.num_eval=200 eval.env_batch_size=10 seed=42 output.save_video=false output.filename=e100_mtm_reacher_s2_ep29_n200_early.txt" > /tmp/e100_mtm_reacher_s2_ep29_early.log 2>&1 &
```

#### Compare

```bash
# Existing seed-3072 e100 MTM trajectory:
# epoch 10/17/25/30/40/50 = 39.5 / 69.5 / 67.5 / 75.0 / 79.0 / 78.0
# Existing seed-1/2 epoch-30 results:
# seed 1 / seed 2 = 10.5 / 12.0
```

### Artifacts

* Eval outputs:
  * `reacher/lewm_masked_e100_s1/e100_mtm_reacher_s1_ep10_n200_early_r2.txt`
  * `reacher/lewm_masked_e100_s1/e100_mtm_reacher_s1_ep{17,25,29}_n200_early.txt`
  * `reacher/lewm_masked_e100_s2/e100_mtm_reacher_s2_ep{10,17,25,29}_n200_early.txt`
* Local archived copies: `progress/evaluations/e100-reacher-r2/e100_mtm_reacher_s{1,2}_ep*_n200_early*.txt`
* Local launch logs: `/tmp/e100_mtm_reacher_s{1,2}_ep{10,17,25,29}_early.log`

### Result

All cells use n=200, eval seed 42, CEM 300/30/30, Reacher 25/50.

| Train seed | Epoch 10 | Epoch 17 | Epoch 25 | Epoch 29 | Epoch 30 existing |
|---:|---:|---:|---:|---:|---:|
| 1 | 11.5 | 14.0 | 13.5 | 13.5 | 10.5 |
| 2 | 12.0 | 11.5 | 15.0 | 10.0 | 12.0 |

For reference, seed 3072 on the same e100 schedule scored
39.5/69.5/67.5/75.0 at epochs 10/17/25/30 and 79.0/78.0 at epochs 40/50.

* Wins / losses: seeds 1/2 are poor at every audited checkpoint; there is no
  earlier strong checkpoint comparable to TwoRoom seed 2's epoch-15/16/29
  recovery.
* Recall collapse? yes / collapse-like for seeds 1/2. This matches the prior
  metric audit: final e100 epoch-30 `emb_std` was about `0.00005` for both
  seeds and inverse loss remained near the constant-action baseline.

### Decision

Reject e100 Reacher MTM as a checkpoint-selection rescue. The failure is already
present by epoch 10 for seeds 1/2 and persists through epochs 17/25/29/30.
Unlike TwoRoom seed 2, this is not a late bad-final-checkpoint issue. Treat it
as a seed-level collapse/failure case under the 100-epoch LR schedule.

### Launch verification

Initial direct background wrappers exited with zero-byte logs and created no
Modal apps, so the sweep was relaunched with persistent local wrappers. Seed-1
epoch-10 was launched in a foreground local wrapper and reached batch 3, but
the app stopped after the local client was killed. That partial run is not
trusted and was rerun via tmux as `e100_mtm_reacher_s1_ep10_n200_early_r2.txt`.
The remaining seven local wrappers ran inside detached tmux sessions.

Verified state after trusted launch: all eight trusted evals reached
`Evaluating batch 1`, and Modal app list showed each as
`State=ephemeral (detached)` with `Tasks=1`. All eight later completed and wrote
raw result files copied into `progress/evaluations/e100-reacher-r2/`.

| Eval | App | Local log / wrapper |
|---|---|---|
| seed 1 epoch 10 | `ap-dNNPChWVnvwHDiVNI8BrEI` | Abandoned partial run: app stopped after local client kill, no final result expected |
| seed 1 epoch 10 rerun | `ap-5rpdqif0MiYKH56M6Ybi8D` | `/tmp/e100_mtm_reacher_s1_ep10_early_r2.log`; tmux `e100_r_mtm_s1_e10_r2`; output `e100_mtm_reacher_s1_ep10_n200_early_r2.txt` |
| seed 1 epoch 17 | `ap-4CjzAYVlwZravHxxMzAQsx` | `/tmp/e100_mtm_reacher_s1_ep17_early.log`; tmux `e100_r_mtm_s1_e17` |
| seed 1 epoch 25 | `ap-BmHPSmcJKADRBoW8tAfReD` | `/tmp/e100_mtm_reacher_s1_ep25_early.log`; tmux `e100_r_mtm_s1_e25` |
| seed 1 epoch 29 | `ap-1xoQzANRzNblifzFUTiX5g` | `/tmp/e100_mtm_reacher_s1_ep29_early.log`; tmux `e100_r_mtm_s1_e29` |
| seed 2 epoch 10 | `ap-EEPA3FS8ARstrOCx2Nndry` | `/tmp/e100_mtm_reacher_s2_ep10_early.log`; tmux `e100_r_mtm_s2_e10` |
| seed 2 epoch 17 | `ap-qM5E0OZkQZLYfFx3Rxg9yd` | `/tmp/e100_mtm_reacher_s2_ep17_early.log`; tmux `e100_r_mtm_s2_e17` |
| seed 2 epoch 25 | `ap-ZgICyBYduZTMSTS4UE4M74` | `/tmp/e100_mtm_reacher_s2_ep25_early.log`; tmux `e100_r_mtm_s2_e25` |
| seed 2 epoch 29 | `ap-PcArtr1jw81N2q3NESmwqr` | `/tmp/e100_mtm_reacher_s2_ep29_early.log`; tmux `e100_r_mtm_s2_e29` |

---

## 2026-06-24 — Reacher action-NCE weight-0.30 checkpoint audit and seed fanout

### Intent

Follow up the first promising Reacher action-NCE result:
`lewm_masked_action_nce`, `loss.masked.inverse_weight=0.30`, seed 1, epoch 10
scored `70.5%` n=200 and stayed noncollapsed. This run tests whether that result
is checkpoint luck and whether the same setting rescues the historically
collapsed seeds `3072` and `2`.

Compare against:

* Same-schedule SIGReg e10 three-seed mean: `69.0 / 69.0 / 68.5`.
* Default MTM e10 seeds: `68.0 / 11.5 / 13.5`.
* Lower-LR MTM epoch-9 audit: `68.5 / 74.0 / 65.5`.
* MTM inverse-weight-3 epoch-9/10 rescue: around `70%` mean but objective-weight
  tuning and weaker mechanism story.

### Commands

#### Eval

```bash
tmux new-session -d -s r_w030_ep7_s1_eval 'cd <repo> && nohup .venv/bin/modal run --detach --name r-w030-ep7-s1 modal_app.py::evaluate --config-name reacher --policy reacher/mtm_action_nce_w030_e10_s1/lewm_masked_action_nce_epoch_7 --overrides "eval.num_eval=200 eval.env_batch_size=10 seed=42 output.save_video=false output.filename=mtm_action_nce_w030_e10_s1_ep7_n200.txt" > /tmp/r_w030_ep7_s1_eval.log 2>&1'
tmux new-session -d -s r_w030_ep8_s1_eval 'cd <repo> && nohup .venv/bin/modal run --detach --name r-w030-ep8-s1 modal_app.py::evaluate --config-name reacher --policy reacher/mtm_action_nce_w030_e10_s1/lewm_masked_action_nce_epoch_8 --overrides "eval.num_eval=200 eval.env_batch_size=10 seed=42 output.save_video=false output.filename=mtm_action_nce_w030_e10_s1_ep8_n200.txt" > /tmp/r_w030_ep8_s1_eval.log 2>&1'
tmux new-session -d -s r_w030_ep9_s1_eval 'cd <repo> && nohup .venv/bin/modal run --detach --name r-w030-ep9-s1 modal_app.py::evaluate --config-name reacher --policy reacher/mtm_action_nce_w030_e10_s1/lewm_masked_action_nce_epoch_9 --overrides "eval.num_eval=200 eval.env_batch_size=10 seed=42 output.save_video=false output.filename=mtm_action_nce_w030_e10_s1_ep9_n200.txt" > /tmp/r_w030_ep9_s1_eval.log 2>&1'
```

#### Train + Eval

```bash
tmux new-session -d -s r_nce_w030_full_s3072 'cd <repo> && nohup .venv/bin/modal run --detach --name r-nce-w030-full-s3072 modal_app.py::train_then_evaluate --config-name lewm_masked_action_nce --data reacher --subdir reacher/mtm_action_nce_w030_e10_s3072 --overrides "trainer.max_epochs=10 early_stopping.enabled=false seed=3072 loss.masked.inverse_weight=0.30" --eval-config-name reacher --eval-policy reacher/mtm_action_nce_w030_e10_s3072 --eval-overrides "eval.num_eval=200 eval.env_batch_size=10 seed=42 output.save_video=false output.filename=mtm_action_nce_w030_e10_s3072_n200.txt" > /tmp/r_nce_w030_full_s3072.log 2>&1'
tmux new-session -d -s r_nce_w030_full_s2 'cd <repo> && nohup .venv/bin/modal run --detach --name r-nce-w030-full-s2 modal_app.py::train_then_evaluate --config-name lewm_masked_action_nce --data reacher --subdir reacher/mtm_action_nce_w030_e10_s2 --overrides "trainer.max_epochs=10 early_stopping.enabled=false seed=2 loss.masked.inverse_weight=0.30" --eval-config-name reacher --eval-policy reacher/mtm_action_nce_w030_e10_s2 --eval-overrides "eval.num_eval=200 eval.env_batch_size=10 seed=42 output.save_video=false output.filename=mtm_action_nce_w030_e10_s2_n200.txt" > /tmp/r_nce_w030_full_s2.log 2>&1'
```

#### Diagnostics

```bash
.venv/bin/modal app list | head -40
tail -80 /tmp/r_w030_ep7_s1_eval.log
tail -80 /tmp/r_w030_ep8_s1_eval.log
tail -80 /tmp/r_w030_ep9_s1_eval.log
tail -80 /tmp/r_nce_w030_full_s3072.log
tail -80 /tmp/r_nce_w030_full_s2.log
```

Two monitor loops are running from tmux parent process `51591`, pulling
`metrics.jsonl` for seeds `3072` and `2` every five minutes and stopping the
corresponding app if recent mean10 `train_step/emb_std < 0.08` after at least
80 logged train-step rows.

### Artifacts

* Seed-1 checkpoint audit apps:
  * epoch 7: `ap-IjSQPih2XQCuIswhyaly2O`, log `/tmp/r_w030_ep7_s1_eval.log`
  * epoch 8: `ap-1zWogr8NyS0pH1LeWYrJRP`, log `/tmp/r_w030_ep8_s1_eval.log`
  * epoch 9: `ap-USO0WwaZWft3FxcraLwvxC`, log `/tmp/r_w030_ep9_s1_eval.log`
* Full fanout apps:
  * seed 3072: `ap-bav4j9lbktRcgGWjY4t486`, log `/tmp/r_nce_w030_full_s3072.log`
  * seed 2: `ap-YkUFkYrajX6znb54NrUQdg`, log `/tmp/r_nce_w030_full_s2.log`
* Monitor logs:
  * `/tmp/r_nce_w030_full_s3072_monitor.log`
  * `/tmp/r_nce_w030_full_s2_monitor.log`
* Expected eval outputs:
  * `reacher/mtm_action_nce_w030_e10_s1_ep{7,8,9}_n200.txt`
  * `reacher/mtm_action_nce_w030_e10_s3072_n200.txt`
  * `reacher/mtm_action_nce_w030_e10_s2_n200.txt`

### Result

Initial launch verification:

* Modal app list shows all five apps as `State=ephemeral (detached)` with
  `Tasks=1`.
* Epoch 7/8/9 eval logs show the exact `python eval.py` command, cached Reacher
  actions, `1760000 valid starting points`, and active rollout batches. At
  launch verification, epoch 7/8 were around batch 3 and epoch 9 was around
  batch 3, with early batch success rates in the 70-90% range.
* Full seed `3072` and `2` train-then-eval logs show dataset staging,
  `python train.py` launch, validation, first backward pass, and train progress
  bars. At launch verification seed `3072` was around step 225/127,960 and seed
  `2` was around step 475/127,960. The first monitor pull found no metrics yet,
  so collapse status is pending until train rows are committed to the Volume.
* Early committed metrics are healthy. Seed `3072` reached step 2250 with
  last / mean10 `emb_std` 0.260 / 0.222, forward loss 0.015, action-NCE loss
  7.89. Seed `2` reached step 2525 with last / mean10 `emb_std` 0.271 / 0.262,
  forward loss 0.0086, action-NCE loss 7.15. Both are far above the 0.08 kill
  threshold and unlike default collapsed seeds, which were already around
  `emb_std=0.066` by step 2000.

Completed seed-1 checkpoint audit:

| Checkpoint | App | Output artifact | n=200 SR | Eval time |
|---|---|---|---:|---:|
| epoch 7 | `ap-IjSQPih2XQCuIswhyaly2O` | `reacher/mtm_action_nce_w030_e10_s1/mtm_action_nce_w030_e10_s1_ep7_n200.txt` | 64.5 | 1187.9s |
| epoch 8 | `ap-1zWogr8NyS0pH1LeWYrJRP` | `reacher/mtm_action_nce_w030_e10_s1/mtm_action_nce_w030_e10_s1_ep8_n200.txt` | 73.5 | 1174.2s |
| epoch 9 | `ap-USO0WwaZWft3FxcraLwvxC` | `reacher/mtm_action_nce_w030_e10_s1/mtm_action_nce_w030_e10_s1_ep9_n200.txt` | 68.5 | 1313.8s |
| epoch 10 | `ap-3oddALYeDGAaHJFTbqLrg5` | `reacher/mtm_action_nce_w030_e10_s1_n200.txt` | 70.5 | 1656.6s |

The checkpoint-audit apps are now stopped with `0` tasks. The epoch-8 checkpoint
is the current best seed-1 Reacher action-NCE result and is above both the
same-schedule SIGReg seed-1 score (`69.0`) and the three-seed SIGReg mean
(`68.8`). This is a promising checkpoint-selection result, not yet a robust
method claim.

Current full fanout status after the audit completed:

* Seed `3072` app `ap-bav4j9lbktRcgGWjY4t486` remains live. Local monitor pull
  at 206 train rows / step 5150: mean10 `emb_std=0.2889`, mean10 forward loss
  `0.00543`, mean10 action-NCE loss `7.196`.
* Seed `2` app `ap-YkUFkYrajX6znb54NrUQdg` remains live. Local monitor pull at
  220 train rows / step 5500: mean10 `emb_std=0.3043`, mean10 forward loss
  `0.00487`, mean10 action-NCE loss `7.179`.

Existing seed-1 weight-0.30 metrics for context:

| Run | Final / mean10 `emb_std` | Mean10 forward loss | Mean10 action-NCE loss | n=200 SR |
|---|---:|---:|---:|---:|
| weight 1.0 seed 1 | 0.2891 / 0.2889 | 0.00154 | 7.115 | 57.0 |
| weight 0.30 seed 1 | 0.1826 / 0.1825 | 0.00059 | 7.117 | 70.5 |

### Decision

Keep action-NCE weight 0.30 as the active Reacher candidate. The epoch-8 seed-1
checkpoint is the first MTM-family Reacher result in this branch that clearly
exceeds the same-schedule SIGReg seed result without reintroducing SIGReg or
using the inverse-weight-3 rescue. Do not promote it to the paper-level method
yet: robustness depends on the running seeds `3072` and `2`, and a Reacher-only
fix is not enough for a claim that replaces SIGReg. The cross-task generality
jobs below are therefore mandatory before any stronger claim.

---

## 2026-06-24 — Action-NCE weight-0.30 cross-task generality check

### Intent

Test whether the Reacher action-NCE fix is a general masked-transition variant
or a Reacher-specific patch. Use the same fixed `loss.masked.inverse_weight=0.30`
and seed `3072` on TwoRoom, PushT, and OGBench-Cube, then evaluate with the
paper-facing n=50 protocol. This is an exploratory generality check; do not
replace the paper-facing MTM rows unless the results are clearly better and
repeatable.

Baselines to compare against the n=50 table:

* TwoRoom SIGReg / MTM: `88.0 / 96.0`.
* PushT SIGReg / MTM: `96.0 / 90.0`.
* OGBench-Cube SIGReg / MTM: `76.0 / 86.0`.

### Commands

#### Train + Eval

The first direct background launches produced zero-byte logs and no Modal apps,
so they are treated as failed wrappers. Trusted relaunches used persistent tmux
wrappers:

```bash
tmux new-session -d -s tr_nce_w030_s3072 'cd <repo> && nohup .venv/bin/modal run --detach --name tr-nce-w030-s3072 modal_app.py::train_then_evaluate --config-name lewm_masked_action_nce --data tworoom --subdir tworoom/mtm_action_nce_w030_e10_s3072 --overrides "trainer.max_epochs=10 early_stopping.enabled=false seed=3072 loss.masked.inverse_weight=0.30" --eval-config-name tworoom --eval-policy tworoom/mtm_action_nce_w030_e10_s3072 --eval-overrides "eval.num_eval=50 eval.env_batch_size=10 seed=42 output.save_video=false output.filename=mtm_action_nce_w030_e10_s3072_n50.txt" > /tmp/tr_nce_w030_s3072.log 2>&1'
tmux new-session -d -s pt_nce_w030_s3072 'cd <repo> && nohup .venv/bin/modal run --detach --name pt-nce-w030-s3072 modal_app.py::train_then_evaluate --config-name lewm_masked_action_nce --data pusht --subdir pusht/mtm_action_nce_w030_e10_s3072 --overrides "trainer.max_epochs=10 early_stopping.enabled=false seed=3072 loss.masked.inverse_weight=0.30" --eval-config-name pusht --eval-policy pusht/mtm_action_nce_w030_e10_s3072 --eval-overrides "eval.num_eval=50 eval.env_batch_size=10 seed=42 output.save_video=false output.filename=mtm_action_nce_w030_e10_s3072_n50.txt" > /tmp/pt_nce_w030_s3072.log 2>&1'
tmux new-session -d -s cb_nce_w030_s3072 'cd <repo> && nohup .venv/bin/modal run --detach --name cb-nce-w030-s3072 modal_app.py::train_then_evaluate --config-name lewm_masked_action_nce --data ogb --subdir cube/mtm_action_nce_w030_e10_s3072 --overrides "trainer.max_epochs=10 early_stopping.enabled=false seed=3072 loss.masked.inverse_weight=0.30" --eval-config-name cube --eval-policy cube/mtm_action_nce_w030_e10_s3072 --eval-overrides "eval.num_eval=50 eval.env_batch_size=10 seed=42 output.save_video=false output.filename=mtm_action_nce_w030_e10_s3072_n50.txt" > /tmp/cb_nce_w030_s3072.log 2>&1'
```

#### Diagnostics

```bash
tail -80 /tmp/tr_nce_w030_s3072.log
tail -80 /tmp/pt_nce_w030_s3072.log
tail -80 /tmp/cb_nce_w030_s3072.log
.venv/bin/modal app list | head -80
```

### Artifacts

* TwoRoom app: `ap-8SglVluM3bxoLSCJ8bFqaP`, log `/tmp/tr_nce_w030_s3072.log`,
  expected output `tworoom/mtm_action_nce_w030_e10_s3072_n50.txt`.
* PushT app: `ap-ywPTjA7FCyE2HqE9IasNno`, log `/tmp/pt_nce_w030_s3072.log`,
  expected output `pusht/mtm_action_nce_w030_e10_s3072_n50.txt`.
* Cube app: `ap-n2YqOQcdbNIf77oCZbGB8R`, log `/tmp/cb_nce_w030_s3072.log`,
  expected output `cube/mtm_action_nce_w030_e10_s3072_n50.txt`.

### Result

Initial launch verification:

* The first direct background wrappers failed silently: `/tmp/tr_*`,
  `/tmp/pt_*`, and `/tmp/cb_*` were zero-byte logs and no apps appeared.
* The tmux relaunches are live. Modal app list shows all three as
  `State=ephemeral (detached)` with `Tasks=1`.
* Logs show dataset staging for `tworoom.h5`, `pusht_expert_train.h5`, and
  `ogbench/cube_single_expert.h5`.
* Follow-up log check confirms all three are past first backward pass with train
  progress bars: TwoRoom around step 450/51,380, PushT around step 325/139,330,
  and Cube around step 400/127,960.

### Decision

Pending. A generalizable candidate needs to preserve TwoRoom/Cube wins and avoid
the PushT degradation. If it only helps Reacher, keep it as a diagnostic branch,
not as the new main method.

## 2026-06-24 — PREEMPTION + RESUME (nce w030 s3072 sweep)

### Intent

Operational entry, not a new experiment. Recover the two w030/s3072 jobs that
Modal killed via **worker preemption** (NOT a budget/spend limit — no spend error
in any log; the other 3 apps kept running, which a workspace cap would not allow).
Root cause: `run_train` / `run_train_then_evaluate` / `run_eval` have no `retries=`
set, so a preempted task dies with no auto-restart.

### What was preempted

* **Cube** (`ap-n2YqOQcdbNIf77oCZbGB8R`) — died mid-train at e6/10 (~58%, step
  ~74,825/127,960). Latest volume ckpt = `epoch_5_object.ckpt` + rolling
  `weights.ckpt`.
* **TwoRoom** (`ap-8SglVluM3bxoLSCJ8bFqaP`) — training had COMPLETED
  (`epoch_10_object.ckpt` saved); only the eval phase was preempted.
* Survived (untouched): PushT `ap-ywPTjA7FCyE2HqE9IasNno`, Reacher
  `ap-bav4j9lbktRcgGWjY4t486` and `ap-YkUFkYrajX6znb54NrUQdg`.

### Commands

#### Cube — resume (auto-resume via `resume.mode: auto` in lewm.yaml)

```bash
tmux new-session -d -s cb_nce_w030_s3072_resume 'nohup .venv/bin/modal run --detach --name cb-nce-w030-s3072 modal_app.py::train_then_evaluate --config-name lewm_masked_action_nce --data ogb --subdir cube/mtm_action_nce_w030_e10_s3072 --overrides "trainer.max_epochs=10 early_stopping.enabled=false seed=3072 loss.masked.inverse_weight=0.30" --eval-config-name cube --eval-policy cube/mtm_action_nce_w030_e10_s3072 --eval-overrides "eval.num_eval=50 eval.env_batch_size=10 seed=42 output.save_video=false output.filename=mtm_action_nce_w030_e10_s3072_n50.txt" > /tmp/cb_nce_w030_s3072_resume.log 2>&1'
```

#### TwoRoom — eval only (`::evaluate`, ckpt already trained)

```bash
tmux new-session -d -s tr_eval_w030_s3072 'cd <repo> && nohup .venv/bin/modal run --detach --name tr-eval-w030-s3072 modal_app.py::evaluate --config-name tworoom --policy tworoom/mtm_action_nce_w030_e10_s3072 --overrides "eval.num_eval=50 eval.env_batch_size=10 seed=42 output.save_video=false output.filename=mtm_action_nce_w030_e10_s3072_n50.txt" > /tmp/tr_eval_w030_s3072.log 2>&1'
```

### Artifacts

* Cube resume app: `ap-NvZGbe7qrAntEdc6spnxNy`, log `/tmp/cb_nce_w030_s3072_resume.log`.
* TwoRoom eval app: `ap-Zl72Qtee37RdORIECqUWDr`, log `/tmp/tr_eval_w030_s3072.log`.

### Result

* Both relaunches verified `Tasks=1` (ephemeral/detached) with real progress.
* Cube confirmed RESUMING (not from scratch): `Resuming training from checkpoint:
  .../lewm_masked_action_nce_weights.ckpt`. Caveat: auto picked the rolling
  mid-epoch-6 `weights.ckpt`, so Lightning warned the dataloader is not resumable
  (minor data-ordering imperfection in epoch 6). For a pristine number, resume
  instead from `epoch_5_object.ckpt` via `resume.ckpt_path=`.
* All five w030/s3072 jobs ran to `✓ App completed`. Final success rates (from the
  canonical `..._n50.txt` / `..._n200.txt` result files):

  | Task | Seed | Episodes | success_rate | eval_time |
  |---|---|---|---|---|
  | TwoRoom | s3072 | 50 | **92.0%** | 258s |
  | PushT | s3072 | 50 | **92.0%** | 375s |
  | Cube (resumed) | s3072 | 50 | **82.0%** | 673s |
  | Reacher | s2 | 200 | **64.0%** | 1562s |
  | Reacher | s3072 | 200 | **70.5%** | 1536s |

* Cube resume succeeded end-to-end: training reached e10/10 100% and eval wrote
  `cube/...n50.txt` (82.0%). The mid-epoch-6 dataloader caveat did not block
  completion. No further preemption occurred, so retries were not exercised.
* Cube eval log shows harmless `Renderer.__del__` EGL teardown tracebacks
  (`EGL_NOT_INITIALIZED`) + missing `libGLU.so.0` — cosmetic, post-success.

### Decision

Recovery complete; whole sweep finished. Hardening landed for future runs:
`retries=RETRIES` (default 3, env `MODAL_RETRIES`) now on all 8 `@app.function`s in
`modal_app.py`, so a preempted task auto-restarts and resumes from the latest
checkpoint. Scientific keep/reject of the w030 method still pending (see prior
entry: must preserve TwoRoom/Cube wins without the PushT degradation).

## 2026-06-25 — ACTION-NCE w030 CROSS-TASK VERDICT + s1/s2 CONFIRMATORY FANOUT

### Intent

(1) Record the completed cross-task generality comparison for the MTM +
action-NCE weight-0.30 ("w030", `lewm_masked_action_nce`) candidate against MTM
(`lewm_masked`) and SIGReg/LeWM (`lewm_base`). (2) Launch the s1/s2 confirmatory
seeds for TwoRoom, PushT, and Cube so cross-task preservation becomes 3-seed
robust like Reacher already is. Baseline source: curated numbers in
`progress/experiment-tables.md` (updated 2026-06-23).

### Result — candidate vs baselines

Reacher (e10, n=200, 3 seeds `3072/1/2`, fully matched):

* SIGReg    69.0 / 69.0 / 68.5  -> mean 68.8
* MTM       68.0 / 11.5 / 13.5  -> mean 31.0  (seeds 1,2 collapse)
* NCE-w030  70.5 / 70.5 / 64.0  -> mean 68.3  (no collapse)

Cross-task (e10, n=50, single seed `3072`):

| Task | SIGReg | MTM | NCE-w030 | vs MTM |
|---|---:|---:|---:|---|
| PushT   | 96.0  | 90.0  | 92.0 | +2.0 (clean: same e10/n50/s3072) |
| Cube    | 76.0  | 86.0  | 82.0 | -4.0 (clean: same e10/n50/s3072) |
| TwoRoom | 88.0* | 96.0* | 92.0 | *baseline e100-ep30; matched e10 MTM 3-seed n200 = 90.2 |

Verdict:

* Reacher collapse FIXED — NCE-w030 reaches SIGReg parity (68.3 vs 68.8), stable
  across all 3 seeds, vs MTM 31.0.
* TwoRoom/Cube wins preserved (Cube small -4 vs MTM, still > SIGReg).
* PushT not degraded (slightly above MTM; residual gap to SIGReg is pre-existing
  MTM behavior, not introduced by action-NCE).
* GATING CAVEAT: cross-task evals are single-seed (s3072). Only Reacher is 3-seed
  robust; TwoRoom/PushT/Cube need s1/s2 before a robust generality claim.

### Commands — s1/s2 confirmatory fanout (this entry's launches)

Six `train_then_evaluate` jobs (e10, n50, `inverse_weight=0.30`), via tmux. Pattern
(task->data/evalcfg: tworoom->tworoom/tworoom, pusht->pusht/pusht, cube->ogb/cube;
seed in {1,2}):

```bash
tmux new-session -d -s <task>_nce_w030_s<seed> 'nohup .venv/bin/modal run --detach \
  --name <prefix>-nce-w030-s<seed> modal_app.py::train_then_evaluate \
  --config-name lewm_masked_action_nce --data <data> \
  --subdir <task>/mtm_action_nce_w030_e10_s<seed> \
  --overrides "trainer.max_epochs=10 early_stopping.enabled=false seed=<seed> loss.masked.inverse_weight=0.30" \
  --eval-config-name <evalcfg> --eval-policy <task>/mtm_action_nce_w030_e10_s<seed> \
  --eval-overrides "eval.num_eval=50 eval.env_batch_size=10 seed=42 output.save_video=false output.filename=mtm_action_nce_w030_e10_s<seed>_n50.txt" \
  > /tmp/<task>_nce_w030_s<seed>.log 2>&1'
```

### Artifacts (expected)

* `{tworoom,pusht,cube}/mtm_action_nce_w030_e10_s{1,2}_n50.txt`
* Logs: `/tmp/{tr,pt,cb}_nce_w030_s{1,2}.log`
* Modal apps now carry `retries=3`, so preemption auto-recovers from checkpoint.

### Decision

Keep/promote action-NCE w030 as the MTM-family candidate of record, CONTINGENT on
the s1/s2 fanout confirming TwoRoom/PushT/Cube preservation. If those hold, w030 is
the cleanest "MTM that doesn't collapse on Reacher" story for the paper.

### Notes

* Launch verification in-session: tmux sessions created; app-list `Tasks>=1` +
  first `[train]` backward pass confirmation pending via watcher.

### Update — all 6 confirmatory runs completed (2026-06-25 ~18:00 IST)

All six `✓ App completed`, no preemption, no errors. Final e10 n=50 success rates
(from `{task}/mtm_action_nce_w030_e10_s{1,2}_n50.txt`):

| Task | s3072 | s1 | s2 | Mean | MTM 3-seed | vs MTM |
|---|---:|---:|---:|---:|---:|:--:|
| TwoRoom | 92.0 | 92.0 | 92.0 | 92.0 | 90.2 | +1.8 |
| PushT | 92.0 | 88.0 | 86.0 | 88.7 | 85.5 | +3.2 |
| Cube | 82.0 | 78.0 | 80.0 | 80.0 | 79.3 | +0.7 |
| Reacher (n200) | 70.5 | 70.5 | 64.0 | 68.3 | 31.0 | +37.3 |

VERDICT UPGRADED to robust 3-seed: NCE-w030 >= MTM 3-seed mean on every task, with
no collapse anywhere (TwoRoom zero variance; PushT/Cube/Reacher tight spreads).
The gating single-seed caveat is resolved. Only PushT trails SIGReg (93.5), which
is the pre-existing MTM gap, not introduced by action-NCE. Decision: promote
action-NCE w030 as the MTM-family candidate of record. `experiment-tables.md`
"Completed Cross-Task Results" section updated with full 3-seed numbers.

Cleanup: stopped the two stale 2026-06-24 origin apps (`ap-n2YqOQ…` cube,
`ap-8SglVl…` tworoom) and cleared all dead tmux streamers.

### Update — matched n=200 evals complete (2026-06-25)

Re-evaluated the candidate TwoRoom/PushT/Cube checkpoints at n=200 (eval-only, L4)
to remove the n50-vs-n200 episode-count asymmetry against the MTM 3-seed n=200
baselines. 9 evals (`{task}/mtm_action_nce_w030_e10_s{seed}_n200.txt`), all
completed, no preemption.

| Task | NCE-w030 n200 (s3072/s1/s2) | Mean | MTM n200 | Δ |
|---|---|---:|---:|:--:|
| TwoRoom | 90.5 / 90.0 / 91.5 | 90.7 | 90.2 | +0.5 |
| PushT | 87.5 / 88.0 / 84.5 | 86.7 | 85.5 | +1.2 |
| Cube | 80.5 / 76.5 / 79.5 | 78.8 | 79.3 | -0.5 |
| Reacher | 70.5 / 70.5 / 64.0 | 68.3 | 31.0 | +37.3 |

Refined verdict: at fully matched n=200, NCE-w030 is within ±1.2 of MTM on all
three non-Reacher tasks (statistical tie) and +37 on Reacher. Pure Reacher rescue,
zero cross-task cost. `experiment-tables.md` updated to the matched n=200 table.

Still running: 3 TwoRoom-long (100/150) n=200 evals (slow, ~2h/seed) — apps
`tr-long-w030-s{3072,1,2}-n200`, files `..._tworoom_long_n200.txt`. Compare vs MTM
28.0 ±0.8 / AR ~18.0. Will append the long-horizon row when they land.

### Update — TwoRoom-long (100/150) n=200 complete (2026-06-25)

| Protocol | NCE-w030 (s3072/s1/s2) | Mean | MTM | AR/SIGReg | Direct-H10 |
|---|---|---:|---:|---:|---:|
| TwoRoom-long 100/150 | 25.0 / 23.5 / 24.0 | 24.2 | 28.0 ±0.8 | 18.0 | 22.2 ±1.6 |

FINDING (the user-prompted long-horizon check): this is the one place action-NCE
costs something. NCE-w030 24.2 trails MTM 28.0 by ~3.8; seed ranges (23.5–25.0 vs
27–29) do not overlap, so the gap is likely real, not noise. It still beats
AR/SIGReg (+6.2) and Direct-H10 (+2.0), so most of the masked-transition
long-horizon advantage is retained, just attenuated.

FINAL VERDICT (action-NCE w030, all evals complete):
* Reacher collapse fixed: 68.3 vs MTM 31.0 (+37), SIGReg parity, 3-seed stable.
* Standard protocol (25/50) matched n=200: ties MTM on TwoRoom/PushT/Cube (±1.2),
  no collapse — zero standard-horizon cost.
* Long-horizon (100/150): small but apparently-real ~4pt regression vs MTM (24.2
  vs 28.0); still >> AR/SIGReg.
* Net: robust Reacher-collapse rescue with no standard-horizon cost and a modest
  long-horizon trade-off vs pure MTM. Promote as the Reacher-robust MTM-family
  variant; if the paper headlines TwoRoom-long, note pure MTM keeps a ~4pt edge.

All eval apps complete; no preemption. Candidate result files archived in volume
under `{task}/mtm_action_nce_w030_e10_s{seed}_{n200,tworoom_long_n200}.txt`.

## 2026-06-25 — play-reacher-success-audit

### Intent

Check whether the Reacher human-play demo uses an accidentally strict or
incorrect success condition when the rendered arm appears close to the goal.

### Commands

#### Diagnostics

```bash
MUJOCO_GL=glfw .venv/bin/python play.py --task reacher --selftest --seed 0
```

```bash
.venv/bin/python - <<'PY'
from project_paths import configure_stablewm_home
configure_stablewm_home()
import gymnasium as gym, stable_worldmodel

env = gym.make(
    "swm/ReacherDMControl-v0", task="qpos_match", max_episode_steps=200
)
env.reset(seed=0)
u = env.unwrapped
print(u.env.task.qpos_threshold)
env.close()
PY
```

### Artifacts

* Self-test preview: `/tmp/play_selftest_reacher.png`

### Result

Base:

The installed `ReacherQPosMatchTask` uses a per-joint strict threshold of
`0.05` radians and terminates only when every joint satisfies it.

Candidate:

`play.py` computes `max(abs(qpos - target_qpos)) < 0.05`, which is equivalent to
the native task check. The seed-0 proportional-control self-test solved the
rollout goal in 3 steps at a displayed distance of `0.017` radians.

Delta:

* Wins / losses: no success-logic discrepancy found.
* Recall collapse? no

### Decision

Keep the success criterion. Treat difficulty under keyboard control as a
usability issue: `0.126` radians is about 2.5 times the allowed error, and
full-scale torque is easy to overshoot. Use `[` to reduce action magnitude near
the goal; consider displaying the `0.05 rad` threshold explicitly in a future
UI refinement.

### Notes

* The criterion matches evaluation semantics and is intentionally based on both
  joint coordinates, not visual/end-effector proximity.
* No Modal/vLLM issue.

## 2026-06-25 — play-task-switch-repeat-fix

### Intent

Diagnose and fix the interactive demo repeatedly cycling through tasks after a
single held `N` press, especially while the slower Cube environment is loading.

### Commands

#### Diagnostics

```bash
.venv/bin/python - <<'PY'
import pygame
event = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_n, repeat=True)
print(pygame.version.ver, event.repeat)
PY
```

#### Eval

```bash
.venv/bin/pytest -q tests/test_config_sanity.py tests/test_play.py
```

```bash
MUJOCO_GL=glfw .venv/bin/python play.py --task reacher --selftest --seed 0
```

### Artifacts

* Regression tests: `tests/test_play.py`
* Interactive input fix: `play.py`

### Result

Base:

`pygame.key.set_repeat(220, 55)` generated repeated `KEYDOWN` events for all
keys. While an environment was synchronously constructing, repeated `N` events
accumulated and triggered successive task loads when the event loop resumed.

Candidate:

Ignore `repeat=True` keydown events for one-shot controls. Preserve repeat
handling for task action keys and Space so turn-based held-key control is
unchanged.

Delta:

* Wins / losses: one held `N` now causes one task transition; movement repeat is retained.
* Recall collapse? no

### Decision

Keep the filtered repeat behavior and regression tests.

### Notes

* Targeted tests: 9 passed.
* Reacher self-test still solved in 3 steps at `0.017` rad.
* No Modal/vLLM issue.

## 2026-06-26 — play-readme-demo-docs

### Intent

Document the human-play demo for repository users, add a README GIF, and fix
control conflicts found while writing the task-specific controls.

### Commands

#### Diagnostics

```bash
MUJOCO_GL=glfw .venv/bin/python make_play_demo_gif.py
```

```bash
.venv/bin/python -B -m py_compile play.py make_play_demo_gif.py
```

#### Eval

```bash
.venv/bin/pytest -q tests/test_play.py
```

```bash
.venv/bin/pytest -q tests/test_config_sanity.py tests/test_play.py
```

```bash
MUJOCO_GL=glfw .venv/bin/python play.py --task reacher --selftest --seed 0
```

### Artifacts

* README demo GIF: `assets/play_demo.gif`
* GIF generator: `make_play_demo_gif.py`
* User-facing docs: `README.md`
* Interactive input/control fix: `play.py`
* Regression tests: `tests/test_play.py`

### Result

Base:

The README did not explain how to launch or control `play.py`. Cube also had
two control conflicts: `Q` was both quit and z-up, and Space was both no-op and
gripper close.

Candidate:

Added a Human play demo README section with launch commands, global controls,
task-specific controls, screenshots, self-test commands, and a generated
Reacher play GIF. Updated the demo so task-specific `Q` actions are not stolen
by quit, and changed Cube gripper close from Space to `X`.

Delta:

* Wins / losses: README now documents the demo; Cube z and gripper controls are usable.
* Recall collapse? no

### Decision

Keep the docs, GIF, and Cube control fixes.

### Notes

* `tests/test_play.py`: 5 passed.
* `tests/test_config_sanity.py tests/test_play.py`: 11 passed.
* Reacher self-test still solved in 3 steps at `0.017` rad.
* No Modal/vLLM issue.

## 2026-06-27 — ogbench-scene-setup-smoke-and-e10-s3072

### Intent

Add OGBench visual Scene as a more complex task than the existing paper table,
verify that the data and evaluation path work end to end, then run a matched
seed-3072 e10 comparison of SIGReg vs Action-NCE. The target evidence is whether
Action-NCE beats SIGReg on `visual-scene-play-v0` under the same `n=50`,
seed-42 evaluation style used for the LeWM-style paper-facing rows.

### Commands

#### Train

```bash
tmux new-session -d -s scene_sigreg "cd <repo> && nohup .venv/bin/modal run --detach modal_app.py::train_then_evaluate --config-name lewm --data scene --subdir scene/lewm_sigreg_e10_s3072 --overrides 'seed=3072 wandb.config.name=scene_lewm_sigreg_e10_s3072 early_stopping.enabled=false' --eval-config-name scene --eval-overrides 'eval.num_eval=50 eval.env_batch_size=10 output.filename=scene_sigreg_e10_s3072_n50.txt output.save_video=false' > /tmp/scene_sigreg_train_eval_tmux.log 2>&1"
```

```bash
tmux new-session -d -s scene_action_nce "cd <repo> && nohup .venv/bin/modal run --detach modal_app.py::train_then_evaluate --config-name lewm_masked_action_nce --data scene --subdir scene/lewm_masked_action_nce_e10_s3072 --overrides 'seed=3072 wandb.config.name=scene_lewm_masked_action_nce_e10_s3072 early_stopping.enabled=false' --eval-config-name scene --eval-overrides 'eval.num_eval=50 eval.env_batch_size=10 output.filename=scene_action_nce_e10_s3072_n50.txt output.save_video=false' > /tmp/scene_action_nce_train_eval_tmux.log 2>&1"
```

#### Inference

```bash
# train_then_evaluate ran evaluation automatically after training completed.
```

#### Diagnostics

```bash
.venv/bin/modal run --detach scene_prep.py::prepare
```

```bash
.venv/bin/pytest -q tests/test_config_sanity.py
```

```bash
.venv/bin/python -B -m py_compile scene_prep.py modal_app.py train.py eval.py
```

```bash
.venv/bin/python train.py --config-name=lewm data=scene wandb.enabled=false --cfg job
```

```bash
.venv/bin/python eval.py --config-name=scene policy=random --cfg job
```

```bash
tail -80 /tmp/scene_sigreg_train_eval_tmux.log
tail -80 /tmp/scene_action_nce_train_eval_tmux.log
.venv/bin/modal app list | head -20
```

#### Eval

```bash
.venv/bin/modal run --detach modal_app.py::evaluate --config-name scene --policy random --overrides "eval.num_eval=2 eval.env_batch_size=2 output.filename=scene_random_smoke.txt output.save_video=false"
```

#### Compare

```bash
.venv/bin/modal volume get multi-future-lewm-cache /scene/scene_sigreg_e10_s3072_n50.txt /tmp/scene_results/
.venv/bin/modal volume get multi-future-lewm-cache /scene/scene_action_nce_e10_s3072_n50.txt /tmp/scene_results/
rg -n "metrics:|evaluation_time" /tmp/scene_results/*.txt
```

### Artifacts

* Train metadata: `finetuning/results/<RUN_ID>/train_result.json` (not used for this Hydra/Modal path)
* GPU metrics: `finetuning/results/<RUN_ID>/gpu_metrics.csv` (not used for this Hydra/Modal path)
* Converted dataset: `.stable_worldmodel/ogbench/visual_scene_play.h5`
* Train config: `config/train/data/scene.yaml`
* Eval config: `config/eval/scene.yaml`
* SIGReg run dir: `.stable_worldmodel/scene/lewm_sigreg_e10_s3072`
* Action-NCE run dir: `.stable_worldmodel/scene/lewm_masked_action_nce_e10_s3072`
* SIGReg app/log: `ap-IexNJ27FhZGX5Yuy9pUiiu`, `/tmp/scene_sigreg_train_eval_tmux.log`
* Action-NCE app/log: `ap-YeuKRYAXXTvkcCnDGqbHCb`, `/tmp/scene_action_nce_train_eval_tmux.log`
* Predictions: not applicable
* Eval summary: `.stable_worldmodel/scene/scene_sigreg_e10_s3072_n50.txt` and `.stable_worldmodel/scene/scene_action_nce_e10_s3072_n50.txt`
* Per-dialogue eval: not applicable

### Result

Base:

SIGReg completed. Initial direct client app `ap-sZLYdUPv3B2FORkepwHeHA` saved
`lewm_weights.ckpt` at step 1000 but then stopped after a local-client
cancellation. Relaunched under tmux as `ap-IexNJ27FhZGX5Yuy9pUiiu`; verified
`Tasks=1`, checkpoint restore from step 1000, and progress line at step 1,025.
Final n50/eval-seed-42 Scene result: **56.0% SR** (`28/50`) in 736.8s eval
time.

Candidate:

Action-NCE completed. Initial direct client app `ap-il19EIHKXOjvkJJcliH1PT`
saved `lewm_masked_action_nce_weights.ckpt` at step 1000 but then stopped after
a local-client cancellation. Relaunched under tmux as
`ap-YeuKRYAXXTvkcCnDGqbHCb`; verified `Tasks=1`, checkpoint restore from step
1000, and progress line at step 1,025. Final n50/eval-seed-42 Scene result:
**80.0% SR** (`40/50`) in 855.2s eval time.

Delta:

* Wins / losses: Action-NCE +24 points over SIGReg. Paired discordants:
  14 Action-NCE wins / 2 SIGReg wins, 26 both-success, 8 both-fail. Exact
  two-sided binomial/McNemar p ~= 0.0042 for the discordant split.
* Recall collapse? no. Action-NCE validation `emb_std` stayed around 0.31-0.37
  during the observed epoch summaries, `effective_rank` stayed around 6-7.5,
  and inverse margin was large (~0.54-0.75), not near chance.

### Decision

Keep the Scene setup and treat this as a completed complex-task add-on result:
Action-NCE clearly beats SIGReg under matched seed-3072 e10 training and
n50/eval-seed-42 evaluation. Use the interrupted direct-client apps only as
provenance for the step-1000 resume checkpoints, not as final runs.

### Notes

* Scene conversion created 1,000,000 transition rows from raw
  `visual-scene-play-v0` after dropping terminal observations. Important fields
  include `pixels`, `action`, `qpos`, `qvel`, `button_state_0/1`,
  `privileged_block_0_pos`, `privileged_block_0_quat`,
  `privileged_drawer_pos`, and `privileged_window_pos`.
* First Scene random smoke failed because eval hardcoded simulator render size
  to 224x224 while the Scene dataset/world buffers were 64x64. Fixed
  `eval.py` to honor eval-config render `width`/`height`; Scene eval now
  renders 64x64 and still feeds 224x224 policy inputs after transforms.
* Modal random-policy smoke completed with 2 episodes and expected 0% SR.
* Tests: `tests/test_config_sanity.py` passed with 8 tests; py_compile passed
  for `scene_prep.py`, `modal_app.py`, `train.py`, and `eval.py`; Hydra train
  and eval config composition passed.
* Direct `nohup ... &` local launches can initialize Modal apps with zero tasks
  for this local-entrypoint path. Persistent tmux sessions are currently the
  reliable way to keep the blocking Modal client alive.

## 2026-06-27 — ogbench-scene-robustness-fanout-s1-s2

### Intent

Make the OGBench visual Scene result robust across training seeds. The completed
seed-3072 study showed Action-NCE beating SIGReg 80.0% vs 56.0% at n=50 with a
strong paired split. This fanout adds train seeds `1` and `2` for both methods
under the same e10 training budget and n50/eval-seed-42 Scene evaluation.

### Commands

#### Train

```bash
tmux new-session -d -s scene_sigreg_s1 "cd <repo> && nohup .venv/bin/modal run --detach modal_app.py::train_then_evaluate --config-name lewm --data scene --subdir scene/lewm_sigreg_e10_s1 --overrides 'seed=1 wandb.config.name=scene_lewm_sigreg_e10_s1 early_stopping.enabled=false' --eval-config-name scene --eval-overrides 'eval.num_eval=50 eval.env_batch_size=10 output.filename=scene_sigreg_e10_s1_n50.txt output.save_video=false' > /tmp/scene_sigreg_e10_s1_n50.log 2>&1"
```

```bash
tmux new-session -d -s scene_action_nce_s1 "cd <repo> && nohup .venv/bin/modal run --detach modal_app.py::train_then_evaluate --config-name lewm_masked_action_nce --data scene --subdir scene/lewm_masked_action_nce_e10_s1 --overrides 'seed=1 wandb.config.name=scene_lewm_masked_action_nce_e10_s1 early_stopping.enabled=false' --eval-config-name scene --eval-overrides 'eval.num_eval=50 eval.env_batch_size=10 output.filename=scene_action_nce_e10_s1_n50.txt output.save_video=false' > /tmp/scene_action_nce_e10_s1_n50.log 2>&1"
```

```bash
tmux new-session -d -s scene_sigreg_s2 "cd <repo> && nohup .venv/bin/modal run --detach modal_app.py::train_then_evaluate --config-name lewm --data scene --subdir scene/lewm_sigreg_e10_s2 --overrides 'seed=2 wandb.config.name=scene_lewm_sigreg_e10_s2 early_stopping.enabled=false' --eval-config-name scene --eval-overrides 'eval.num_eval=50 eval.env_batch_size=10 output.filename=scene_sigreg_e10_s2_n50.txt output.save_video=false' > /tmp/scene_sigreg_e10_s2_n50.log 2>&1"
```

```bash
tmux new-session -d -s scene_action_nce_s2 "cd <repo> && nohup .venv/bin/modal run --detach modal_app.py::train_then_evaluate --config-name lewm_masked_action_nce --data scene --subdir scene/lewm_masked_action_nce_e10_s2 --overrides 'seed=2 wandb.config.name=scene_lewm_masked_action_nce_e10_s2 early_stopping.enabled=false' --eval-config-name scene --eval-overrides 'eval.num_eval=50 eval.env_batch_size=10 output.filename=scene_action_nce_e10_s2_n50.txt output.save_video=false' > /tmp/scene_action_nce_e10_s2_n50.log 2>&1"
```

#### Inference

```bash
# train_then_evaluate will run evaluation automatically after training completes.
```

#### Diagnostics

```bash
.venv/bin/modal app list | head -20
tail -80 /tmp/scene_sigreg_e10_s1_n50.log
tail -80 /tmp/scene_action_nce_e10_s1_n50.log
tail -80 /tmp/scene_sigreg_e10_s2_n50.log
tail -80 /tmp/scene_action_nce_e10_s2_n50.log
```

#### Eval

```bash
# Pending automatic eval after training.
```

#### Compare

```bash
# Pending output files:
# .stable_worldmodel/scene/scene_sigreg_e10_s1_n50.txt
# .stable_worldmodel/scene/scene_action_nce_e10_s1_n50.txt
# .stable_worldmodel/scene/scene_sigreg_e10_s2_n50.txt
# .stable_worldmodel/scene/scene_action_nce_e10_s2_n50.txt
```

### Artifacts

* SIGReg s1 app/log: `ap-C9fUecyMne2StJJ8AtpaWO`, `/tmp/scene_sigreg_e10_s1_n50.log`
* Action-NCE s1 app/log: `ap-GvU5gGnNVCv4gludCVc6ld`, `/tmp/scene_action_nce_e10_s1_n50.log`
* SIGReg s2 app/log: `ap-akHZWstHjZLN7jCpuILd3k`, `/tmp/scene_sigreg_e10_s2_n50.log`
* Action-NCE s2 app/log: `ap-e7C8jJYpLV85QQb9ZM4K4n`, `/tmp/scene_action_nce_e10_s2_n50.log`
* Run dirs: `.stable_worldmodel/scene/lewm_sigreg_e10_s1`, `.stable_worldmodel/scene/lewm_masked_action_nce_e10_s1`, `.stable_worldmodel/scene/lewm_sigreg_e10_s2`, `.stable_worldmodel/scene/lewm_masked_action_nce_e10_s2`
* Eval summary: pending four `scene_*_e10_s{1,2}_n50.txt` files in `.stable_worldmodel/scene/`

### Result

Base:

SIGReg seeds `1` and `2` are running. Both apps were verified with `Tasks=1`
and first-epoch training progress. Initial validation has low `emb_std` because
this is before training has spread representations; the first seed-3072 run
showed SIGReg spreading by later epochs.

Candidate:

Action-NCE seeds `1` and `2` are running. Both apps were verified with
`Tasks=1`, initial validation, first backward, and first-epoch training progress.

Delta:

* Wins / losses: pending automatic n50 eval outputs.
* Recall collapse? pending; use epoch summaries for `emb_std`, `effective_rank`,
  and `inverse_margin` as the first collapse sentinels.

### Decision

Let all four runs complete. If Action-NCE wins on both new seeds, report Scene
as a three-training-seed complex-task result. If one seed is ambiguous, run
n=200 eval on all three completed checkpoints before drawing a stronger claim.

### Notes

* Persistent tmux wrappers were used because direct local-client interruption
  previously canceled Modal inputs.
* Running four jobs concurrently increases the initial HDF5 staging time, but
  all four jobs reached training.

## 2026-06-27 — ogbench-scene-official-eval-plumbing

### Intent

Add the official OGBench Visual Scene fixed-goal evaluation path so our Scene
checkpoint can be compared to public `visual-scene-play-v0` task1..task5
numbers. This is different from the repo's current trajectory-goal Scene eval:
the official envs reset to five fixed goal tasks and allow up to 750 env steps.

### Commands

#### Train

```bash
# No new training; evaluate existing scene/lewm_masked_action_nce_e10_s3072.
```

#### Inference

```bash
# N/A
```

#### Diagnostics

```bash
.venv/bin/python -m py_compile eval.py modal_app.py train.py
.venv/bin/pytest -q tests/test_config_sanity.py
.venv/bin/python eval.py --config-name=scene_official policy=random eval.num_eval=1 eval.task_ids='[1]' eval.max_episode_steps=25 eval.eval_budget=25 eval.progress_bar=false output.filename=scene_official_random_smoke_local.txt
```

#### Eval

```bash
tmux new-session -d -s scene_official_an_smoke2 "cd <repo> && nohup .venv/bin/modal run --detach modal_app.py::evaluate --config-name scene_official --policy scene/lewm_masked_action_nce_e10_s3072 --overrides 'eval.num_eval=1 eval.task_ids=[1] eval.max_episode_steps=75 eval.eval_budget=75 eval.env_batch_size=1 output.filename=scene_official_action_nce_s3072_task1_n1_h75_smoke_v2.txt output.save_video=false' > /tmp/scene_official_an_smoke2.log 2>&1"
tmux new-session -d -s scene_official_an_task1_full "cd <repo> && nohup .venv/bin/modal run --detach modal_app.py::evaluate --config-name scene_official --policy scene/lewm_masked_action_nce_e10_s3072 --overrides 'eval.num_eval=1 eval.task_ids=[1] eval.max_episode_steps=750 eval.eval_budget=750 eval.env_batch_size=1 output.filename=scene_official_action_nce_s3072_task1_n1_h750.txt output.save_video=false' > /tmp/scene_official_an_task1_full.log 2>&1"
```

#### Compare

```bash
tail -120 /tmp/scene_official_an_smoke2.log
tail -120 /tmp/scene_official_an_task1_full.log
```

### Artifacts

* Official eval config: `config/eval/scene_official.yaml`
* Official eval implementation: `eval.py`
* Tests: `tests/test_config_sanity.py`
* Short smoke app/log: `ap-tAJM7U1zL9dsm7SzR1cEVI`, `/tmp/scene_official_an_smoke2.log`
* Full task1 one-episode app/log: `ap-BHlNBSYqJxAjgwxNXQf6mL`, `/tmp/scene_official_an_task1_full.log`
* Eval summaries on Modal Volume:
  `.stable_worldmodel/scene/scene_official_action_nce_s3072_task1_n1_h75_smoke_v2.txt`
  and `.stable_worldmodel/scene/scene_official_action_nce_s3072_task1_n1_h750.txt`

### Result

Base:

Not run yet under official fixed-goal protocol.

Candidate:

Action-NCE seed-3072 official task1:

* 75-step smoke: 0/1 success, completed cleanly in about 25s.
* 750-step official cap: 0/1 success, completed cleanly in about 167s.

Delta:

* Wins / losses: no SIGReg official comparison yet.
* Recall collapse? not applicable; this was eval-only.

### Decision

Keep the official evaluator. Do not claim OGBench public SOTA from the current
trajectory-goal 80% result. The first official task1 smoke suggests the current
final-latent MPC objective is not enough for the fixed-goal OGBench task, even
with the full 750-step cap. Next step, if pursuing official comparison, is to
run tiny paired SIGReg/Action-NCE task1 videos or planning artifacts to diagnose
whether the issue is long-horizon subgoal planning, goal-image encoding, or
action scaling.

### Notes

* Official Visual Scene envs are `visual-scene-singletask-task{1..5}-v0`; the
  dataset names include `play`, but gym env ids do not.
* All official Scene tasks have `max_episode_steps=750`.
* Task primitive estimates from the official task definitions:
  task1 open drawer/window = about 2 object-level subtasks; task2
  unlock/close/lock drawer+window = about 6; task3 rearrange-medium = about 4;
  task4 put-in-drawer = about 4; task5 rearrange-hard = about 8. Average is
  roughly 4.8 atomic subtasks, but the official score is over environment
  steps, not this primitive count.
* The first Modal model smoke exposed a narrow contract issue: `JEPA.get_cost`
  expected `goal_action` because dataset-goal eval provides it and then drops it
  before encoding the goal. Official reset goals are image-only, so
  `eval.py::_policy_step_info` now provides a dummy `goal_action`.

## 2026-06-27 — ogbench-scene-official-full-s3072-n50

### Intent

Run the full official OGBench Visual Scene fixed-goal benchmark for the matched
seed-3072 Scene checkpoints. This compares Action-NCE against SIGReg under the
public-style five-task protocol rather than the repo's trajectory-goal protocol.

### Commands

#### Train

```bash
# No new training; evaluate existing seed-3072 Scene checkpoints.
```

#### Inference

```bash
# N/A
```

#### Diagnostics

```bash
tail -80 /tmp/scene_official_action_nce_s3072_alltasks_n50.log
tail -80 /tmp/scene_official_sigreg_s3072_alltasks_n50.log
.venv/bin/modal app list | head -30
```

#### Eval

```bash
tmux new-session -d -s scene_official_an_s3072_n50 "cd <repo> && nohup .venv/bin/modal run --detach modal_app.py::evaluate --config-name scene_official --policy scene/lewm_masked_action_nce_e10_s3072 --overrides 'eval.num_eval=50 eval.env_batch_size=1 output.filename=scene_official_action_nce_s3072_alltasks_n50.txt output.save_video=false' > /tmp/scene_official_action_nce_s3072_alltasks_n50.log 2>&1"
tmux new-session -d -s scene_official_sigreg_s3072_n50 "cd <repo> && nohup .venv/bin/modal run --detach modal_app.py::evaluate --config-name scene_official --policy scene/lewm_sigreg_e10_s3072 --overrides 'eval.num_eval=50 eval.env_batch_size=1 output.filename=scene_official_sigreg_s3072_alltasks_n50.txt output.save_video=false' > /tmp/scene_official_sigreg_s3072_alltasks_n50.log 2>&1"
```

#### Compare

```bash
# Pending: compare the two Modal Volume result files after both evals finish.
```

### Artifacts

* Action-NCE app/log: `ap-wl9bagnRqgjDtXR1T6EryU`,
  `/tmp/scene_official_action_nce_s3072_alltasks_n50.log`
* SIGReg app/log: `ap-BGB8FeZXYJfeyV3w64VDge`,
  `/tmp/scene_official_sigreg_s3072_alltasks_n50.log`
* Expected Action-NCE result:
  `.stable_worldmodel/scene/scene_official_action_nce_s3072_alltasks_n50.txt`
* Expected SIGReg result:
  `.stable_worldmodel/scene/scene_official_sigreg_s3072_alltasks_n50.txt`

### Result

Base:

SIGReg seed-3072 official Visual Scene fixed-goal five-task result pending.

Candidate:

Action-NCE seed-3072 official Visual Scene fixed-goal five-task result pending.

Delta:

* Wins / losses: pending.
* Recall collapse? not applicable; eval-only. Use the already-completed training
  diagnostics for collapse interpretation.

### Decision

Let both detached Modal evals continue. As of launch verification, both jobs
loaded checkpoints, created the official task1 env, printed `official-scene`
progress, and Modal showed `ephemeral (detached)` with `Tasks=1`.

### Notes

* This is the full official sweep: tasks `1..5`, `50` episodes per task,
  `750` env-step cap, `187500` maximum env steps per model.
* With `env_batch_size=1` and CEM replanning every `action_block=5` steps, this
  is expected to take hours.
* The earlier one-episode task1 Action-NCE smoke scored 0/1, so the official
  result may be much lower than the repo's trajectory-goal Scene result.

## 2026-06-27 — ogbench-scene-robustness-three-seed-results

### Intent

Finalize the matched OGBench Scene trajectory-goal robustness fanout. Compare
Action-NCE against SIGReg over seeds `3072,1,2` with the same n50 eval seed and
protocol.

### Commands

#### Train

```bash
# Training commands were launched in the earlier ogbench-scene-robustness-fanout-s1-s2 entry.
```

#### Inference

```bash
# N/A
```

#### Diagnostics

```bash
.venv/bin/modal app list | head -30
tmux list-sessions
```

#### Eval

```bash
.venv/bin/modal volume get multi-future-lewm-cache /scene/scene_sigreg_e10_s1_n50.txt /tmp/scene_results_robust/
.venv/bin/modal volume get multi-future-lewm-cache /scene/scene_action_nce_e10_s1_n50.txt /tmp/scene_results_robust/
.venv/bin/modal volume get multi-future-lewm-cache /scene/scene_sigreg_e10_s2_n50.txt /tmp/scene_results_robust/
.venv/bin/modal volume get multi-future-lewm-cache /scene/scene_action_nce_e10_s2_n50.txt /tmp/scene_results_robust/
```

#### Compare

```bash
rg -n "metrics:|evaluation_time" /tmp/scene_results_robust/*.txt /tmp/scene_results/*.txt
.venv/bin/python - <<'PY'
from pathlib import Path
import re, math
pairs = [
    ("s3072", Path("/tmp/scene_results/scene_sigreg_e10_s3072_n50.txt"), Path("/tmp/scene_results/scene_action_nce_e10_s3072_n50.txt")),
    ("s1", Path("/tmp/scene_results_robust/scene_sigreg_e10_s1_n50.txt"), Path("/tmp/scene_results_robust/scene_action_nce_e10_s1_n50.txt")),
    ("s2", Path("/tmp/scene_results_robust/scene_sigreg_e10_s2_n50.txt"), Path("/tmp/scene_results_robust/scene_action_nce_e10_s2_n50.txt")),
]
# Parse episode_successes arrays and count paired discordants.
PY
```

### Artifacts

* SIGReg seed-3072 output: `.stable_worldmodel/scene/scene_sigreg_e10_s3072_n50.txt`
* Action-NCE seed-3072 output: `.stable_worldmodel/scene/scene_action_nce_e10_s3072_n50.txt`
* SIGReg seed-1 output: `.stable_worldmodel/scene/scene_sigreg_e10_s1_n50.txt`
* Action-NCE seed-1 output: `.stable_worldmodel/scene/scene_action_nce_e10_s1_n50.txt`
* SIGReg seed-2 output: `.stable_worldmodel/scene/scene_sigreg_e10_s2_n50.txt`
* Action-NCE seed-2 output: `.stable_worldmodel/scene/scene_action_nce_e10_s2_n50.txt`

### Result

Base:

SIGReg Scene n50 trajectory-goal success rates:

* seed `3072`: 56.0% (28/50)
* seed `1`: 58.0% (29/50)
* seed `2`: 60.0% (30/50)
* mean: 58.0%

Candidate:

Action-NCE Scene n50 trajectory-goal success rates:

* seed `3072`: 80.0% (40/50)
* seed `1`: 78.0% (39/50)
* seed `2`: 82.0% (41/50)
* mean: 80.0%

Delta:

* Mean delta: +22 pts for Action-NCE.
* Wins / losses: combined paired discordants across 150 matched episodes were
  40 Action-NCE-only successes vs 7 SIGReg-only successes; exact two-sided
  binomial/McNemar p ~= 1.1e-6.
* Recall collapse? no eval-side sign of collapse; success rates are stable
  across seeds for both methods. Use train-side `emb_std`, `effective_rank`,
  and inverse-margin logs if a collapse-specific figure is needed.

### Decision

Keep. This is the strongest complex-task trajectory-goal Scene result: a clean
three-training-seed Action-NCE win over SIGReg. Report it separately from public
OGBench Scene SOTA because this protocol samples trajectory goals from
`visual-scene-play-v0` and uses a 50-step budget, while public OGBench Visual
Scene scores use five fixed tasks with a 750-step cap.

### Notes

* All Modal apps from this robustness fanout are stopped with zero tasks.
* The OpenGL/EGL teardown tracebacks in some logs occurred after eval summaries
  were written and did not prevent result files from landing.

## 2026-06-27 — ogbench-scene-official-full-s3072-n50-interrupted

### Intent

Record the outcome of the monolithic full official OGBench Visual Scene
fixed-goal benchmark attempt for seed-3072 Action-NCE and SIGReg.

### Commands

#### Train

```bash
# No new training; evaluated existing seed-3072 Scene checkpoints.
```

#### Inference

```bash
# N/A
```

#### Diagnostics

```bash
tail -80 /tmp/scene_official_action_nce_s3072_alltasks_n50.log
tail -80 /tmp/scene_official_sigreg_s3072_alltasks_n50.log
.venv/bin/modal app list | head -30
.venv/bin/modal volume get multi-future-lewm-cache /scene/scene_official_action_nce_s3072_alltasks_n50.txt /tmp/scene_official_results/
.venv/bin/modal volume get multi-future-lewm-cache /scene/scene_official_sigreg_s3072_alltasks_n50.txt /tmp/scene_official_results/
```

#### Eval

```bash
tmux new-session -d -s scene_official_an_s3072_n50 "cd <repo> && nohup .venv/bin/modal run --detach modal_app.py::evaluate --config-name scene_official --policy scene/lewm_masked_action_nce_e10_s3072 --overrides 'eval.num_eval=50 eval.env_batch_size=1 output.filename=scene_official_action_nce_s3072_alltasks_n50.txt output.save_video=false' > /tmp/scene_official_action_nce_s3072_alltasks_n50.log 2>&1"
tmux new-session -d -s scene_official_sigreg_s3072_n50 "cd <repo> && nohup .venv/bin/modal run --detach modal_app.py::evaluate --config-name scene_official --policy scene/lewm_sigreg_e10_s3072 --overrides 'eval.num_eval=50 eval.env_batch_size=1 output.filename=scene_official_sigreg_s3072_alltasks_n50.txt output.save_video=false' > /tmp/scene_official_sigreg_s3072_alltasks_n50.log 2>&1"
```

#### Compare

```bash
# No comparison possible; neither result file exists.
```

### Artifacts

* Action-NCE app/log: `ap-wl9bagnRqgjDtXR1T6EryU`,
  `/tmp/scene_official_action_nce_s3072_alltasks_n50.log`
* SIGReg app/log: `ap-BGB8FeZXYJfeyV3w64VDge`,
  `/tmp/scene_official_sigreg_s3072_alltasks_n50.log`
* Missing Action-NCE result:
  `.stable_worldmodel/scene/scene_official_action_nce_s3072_alltasks_n50.txt`
* Missing SIGReg result:
  `.stable_worldmodel/scene/scene_official_sigreg_s3072_alltasks_n50.txt`

### Result

Base:

SIGReg did not finish. The log reached about `26,945 / 187,500` env steps
(14.4%) before the Modal client printed `[Errno 8] nodename nor servname
provided, or not known`; no result file was written.

Candidate:

Action-NCE did not finish. The log reached about `25,648 / 187,500` env steps
(13.7%) before the same Modal client DNS/network error; no result file was
written.

Delta:

* Wins / losses: unavailable.
* Recall collapse? not applicable; eval-only and no completed official result.

### Decision

Reject the monolithic launch pattern for official Scene n50. Relaunch as
task/episode chunks with separate output files, or add incremental metric
flushing to `evaluate_official_scene`, before making any public OGBench
fixed-goal comparison.

### Notes

* These failed jobs do not affect the completed trajectory-goal Scene result.
* The official evaluator itself is still useful: local random smoke and
  one-episode Action-NCE task1 runs completed correctly. The failure here was
  job durability/output granularity, not an evaluator API incompatibility.

## 2026-06-27 — ogbench-scene-official-chunked-rerun-s3072

### Intent

Relaunch the official OGBench Visual Scene fixed-goal benchmark in recoverable
chunks after the monolithic Action-NCE and SIGReg n50 jobs stopped around 14%
with a Modal client DNS/network error and no result files.

### Commands

#### Train

```bash
# No new training; evaluate existing seed-3072 Scene checkpoints.
```

#### Inference

```bash
# N/A
```

#### Diagnostics

```bash
.venv/bin/python -m py_compile eval.py modal_app.py train.py
.venv/bin/pytest -q tests/test_config_sanity.py
.venv/bin/python eval.py --config-name=scene_official policy=random eval.num_eval=1 eval.episode_start=10 eval.task_ids='[1]' eval.max_episode_steps=25 eval.eval_budget=25 eval.progress_bar=false output.filename=scene_official_random_chunk_smoke_local.txt
bash -n scripts/run_scene_official_chunk_lane.sh
cd paper/latex && pdflatex -interaction=nonstopmode -halt-on-error iclr_main.tex
```

#### Eval

```bash
for method in action_nce sigreg; do
  for task in 1 2 3 4 5; do
    session="scene_official_${method}_t${task}"
    log="/tmp/${session}_lane.log"
    tmux new-session -d -s "${session}" "cd <repo> && ./scripts/run_scene_official_chunk_lane.sh ${method} ${task} > ${log} 2>&1"
  done
done
```

Each lane runs chunks with:

```bash
nohup .venv/bin/modal run --detach modal_app.py::evaluate \
  --config-name scene_official \
  --policy <scene/lewm_masked_action_nce_e10_s3072-or-scene/lewm_sigreg_e10_s3072> \
  --overrides 'eval.num_eval=10 eval.episode_start=<0|10|20|30|40> eval.task_ids=[<task>] eval.env_batch_size=1 output.filename=<unique-chunk-file> output.save_video=false'
```

#### Compare

```bash
# Pending: aggregate 50 chunk files after all lanes finish.
```

### Artifacts

* Official eval config with chunk offset: `config/eval/scene_official.yaml`
* Official evaluator: `eval.py`
* Modal eval durability change: `modal_app.py`
* Chunk lane launcher: `scripts/run_scene_official_chunk_lane.sh`
* Lane logs: `/tmp/scene_official_{action_nce,sigreg}_t{1..5}_lane.log`
* Per-chunk logs:
  `/tmp/scene_official_{action_nce,sigreg}_s3072_task{1..5}_eps{0_9,10_19,20_29,30_39,40_49}_attempt*.log`
* Expected chunk outputs:
  `.stable_worldmodel/scene/scene_official_{action_nce,sigreg}_s3072_task{1..5}_eps{0_9,10_19,20_29,30_39,40_49}_n10.txt`

### Result

Base:

SIGReg official fixed-goal result pending. Five SIGReg task lanes launched and
the first chunks show real `official-scene` progress.

Candidate:

Action-NCE official fixed-goal result pending. Five Action-NCE task lanes
launched and the first chunks show real `official-scene` progress.

Delta:

* Wins / losses: pending aggregation after all chunk files land.
* Recall collapse? not applicable; eval-only.

### Decision

Let the chunk lanes continue. This launch pattern should survive intermittent
client/network failures better than the monolithic jobs because each chunk is
short, writes a unique output file, and is verified from the Modal Volume before
the lane advances to the next chunk.

### Notes

* Verified after launch: ten tmux sessions exist, ten detached Modal apps show
  `Tasks=1`, and the first per-chunk logs contain `official-scene` progress
  lines rather than just initialization.
* The paper-facing trajectory-goal Scene result is already complete and has
  been promoted into `paper/latex/iclr_main.tex`; this official fixed-goal
  rerun is only for public OGBench comparison.

## 2026-06-27 — ogbench-scene-official-remote-supervisor-s3072

### Intent

Move official OGBench Visual Scene chunk orchestration fully into Modal so the
benchmark does not depend on local tmux or the laptop staying online.

### Commands

#### Train

```bash
# No new training; evaluate existing seed-3072 Scene checkpoints.
```

#### Inference

```bash
# N/A
```

#### Diagnostics

```bash
.venv/bin/python -m py_compile modal_app.py eval.py train.py
.venv/bin/pytest -q tests/test_config_sanity.py
.venv/bin/modal run modal_app.py::scene_official_chunks --help
git diff --check
tmux list-sessions
.venv/bin/modal app list | head -80
```

#### Eval

```bash
# Stop the local tmux-supervised attempt after repeated Modal RemoteError /
# Deadline exceeded failures, preserving landed chunk files.
for s in $(tmux list-sessions | awk -F: '/scene_official_/ {print $1}'); do tmux kill-session -t "$s"; done
for app in ap-teS3MeMyqHJyfYxtUyNwtR ap-b8SUl1GsydOuasEXXh7tTE ap-WWriHrMbNbLNM1eYorPfWN ap-5bMsmNcTQveJXvvhCeLIoI ap-2HlmPJAh4wrsmC83AQbGdW ap-UB6e1PhA0zoYTnQm8VJBXg ap-QBQcXomjeTQdgKnIiyRK1b ap-OvgPriQdKWgOYyG1QO35m7 ap-IJHKMof2BzVo42uVao5Ng7; do .venv/bin/modal app stop --yes "$app" || true; done

# Launch the Modal-resident supervisor.
.venv/bin/modal run --detach modal_app.py::scene_official_chunks --methods action_nce,sigreg --tasks 1,2,3,4,5 --starts 0,10,20,30,40 --chunk-size 10 --max-attempts 3
```

#### Compare

```bash
# Pending: aggregate all 50 chunk outputs after supervisor app finishes.
```

### Artifacts

* Supervisor implementation: `modal_app.py::run_scene_official_chunk_supervisor`
* Lane implementation: `modal_app.py::run_scene_official_chunk_lane`
* Active remote supervisor app: `ap-lP9ONtY1E5FEgKbaGHOKCV`
* Completed local chunk outputs before supervisor switch:
  `.stable_worldmodel/scene/scene_official_action_nce_s3072_task1_eps0_9_n10.txt`,
  `.stable_worldmodel/scene/scene_official_action_nce_s3072_task2_eps0_9_n10.txt`,
  `.stable_worldmodel/scene/scene_official_action_nce_s3072_task5_eps0_9_n10.txt`,
  `.stable_worldmodel/scene/scene_official_sigreg_s3072_task2_eps0_9_n10.txt`,
  `.stable_worldmodel/scene/scene_official_sigreg_s3072_task3_eps0_9_n10.txt`,
  `.stable_worldmodel/scene/scene_official_sigreg_s3072_task4_eps0_9_n10.txt`

### Result

Base:

SIGReg official fixed-goal result still pending. Three SIGReg first chunks have
completed and all are 0/10 so far.

Candidate:

Action-NCE official fixed-goal result still pending. Three Action-NCE first
chunks have completed and all are 0/10 so far.

Delta:

* Wins / losses: pending full chunk aggregation.
* Recall collapse? not applicable; eval-only.

### Decision

Keep the remote-supervisor approach. The local tmux-supervised approach was not
robust enough: it preserved some outputs but still depended on local lane
processes and had many Modal input-plane failures. The new supervisor is the
last triggered Modal function, runs inside Modal, spawns ten GPU lane functions,
and survived killing the local Modal CLI: app `ap-lP9ONtY1E5FEgKbaGHOKCV`
remained live with `Tasks=11`.

### Notes

* The six completed chunks are skipped by the remote lanes because the result
  files already exist on the Modal Volume.
* No tmux sessions remain. The laptop can go offline without controlling the
  remaining chunk sequence.

## 2026-06-27 — ogbench-scene-trajectory-goal-paper-audit

### Intent

Interrogate the completed OGBench Visual Scene trajectory-goal result because
the Action-NCE margin over SIGReg is large enough to look suspicious. Check raw
episode arrays, configs, checkpoints, train metrics, and eval logs before
strengthening or qualifying the paper interpretation.

### Commands

#### Train

```bash
# No new training; audited existing Scene e10 runs.
```

#### Inference

```bash
# No new inference; retrieved completed result and run metadata files.
```

#### Diagnostics

```bash
.venv/bin/modal volume ls multi-future-lewm-cache /scene
.venv/bin/modal volume get multi-future-lewm-cache /scene/scene_sigreg_e10_s3072_n50.txt /tmp/scene_audit/
.venv/bin/modal volume get multi-future-lewm-cache /scene/scene_action_nce_e10_s3072_n50.txt /tmp/scene_audit/
.venv/bin/modal volume get multi-future-lewm-cache /scene/scene_sigreg_e10_s1_n50.txt /tmp/scene_audit/
.venv/bin/modal volume get multi-future-lewm-cache /scene/scene_action_nce_e10_s1_n50.txt /tmp/scene_audit/
.venv/bin/modal volume get multi-future-lewm-cache /scene/scene_sigreg_e10_s2_n50.txt /tmp/scene_audit/
.venv/bin/modal volume get multi-future-lewm-cache /scene/scene_action_nce_e10_s2_n50.txt /tmp/scene_audit/
.venv/bin/modal volume get --force multi-future-lewm-cache /scene/<run>/config.yaml /tmp/scene_audit/runs/<run>/
.venv/bin/modal volume get --force multi-future-lewm-cache /scene/<run>/run_metadata.json /tmp/scene_audit/runs/<run>/
.venv/bin/modal volume get --force multi-future-lewm-cache /scene/<run>/checkpoint_state.json /tmp/scene_audit/runs/<run>/
.venv/bin/modal volume get --force multi-future-lewm-cache /scene/<run>/metrics.jsonl /tmp/scene_audit/runs/<run>/
rg -n "Loading model from checkpoint|scene/lewm_(sigreg|masked_action).*epoch|success_rate" /tmp progress outputs -g '*scene*' -g '*.log' -g '*.txt'
cd paper/latex && pdflatex -interaction=nonstopmode -halt-on-error iclr_main.tex
```

#### Eval

```bash
# No new eval launched.
```

#### Compare

```bash
.venv/bin/python - <<'PY'
# Parsed /tmp/scene_audit result files, counted paired discordants, and
# summarized last logged train metrics.
PY
```

### Artifacts

* Retrieved result files: `/tmp/scene_audit/scene_{sigreg,action_nce}_e10_s{3072,1,2}_n50.txt`
* Retrieved metadata: `/tmp/scene_audit/runs/{sigreg,action_nce}_s{3072,1,2}/{config.yaml,run_metadata.json,checkpoint_state.json,metrics.jsonl}`
* Paper edit: `paper/latex/iclr_main.tex`
* Rebuilt PDF: `paper/latex/iclr_main.pdf`

### Result

Base:

SIGReg Scene trajectory-goal n50 scores remain:

* seed `3072`: 56.0% (28/50)
* seed `1`: 58.0% (29/50)
* seed `2`: 60.0% (30/50)
* mean: 58.0%

Candidate:

Action-NCE Scene trajectory-goal n50 scores remain:

* seed `3072`: 80.0% (40/50)
* seed `1`: 78.0% (39/50)
* seed `2`: 82.0% (41/50)
* mean: 80.0%

Delta:

* Wins / losses: seed-level discordants were `14/2`, `13/3`, and `13/2`;
  combined paired discordants were 40 Action-NCE-only successes versus 7
  SIGReg-only successes across 150 matched episodes. Exact two-sided
  binomial/McNemar p ~= `1.07e-6`.
* Recall collapse? no obvious collapse. Final train `emb_std` is nonzero for
  both methods: about `0.99` for SIGReg and `0.24` for Action-NCE over the last
  logged batches. This points to planner-usable geometry or dynamics quality
  rather than trivial constant collapse.

Audit checks:

* Raw result configs match on eval seed 42, `num_eval=50`,
  `goal_offset_steps=25`, `eval_budget=50`, CEM `300/30/30`, horizon 5, action
  block 5, and dataset `ogbench/visual_scene_play`.
* Training configs match on dataset, batch size, LR, 10 epochs, encoder/action
  encoder/forward predictor, and differ in the intended objective.
* Eval logs confirm epoch-10 object checkpoints were loaded for both methods.
* The planner cost path encodes goal pixels and drops goal action; privileged
  state configures simulator targets/success under the trajectory-goal protocol
  rather than entering the learned cost.

### Decision

Keep the Scene trajectory-goal result, but present it with stronger caveats.
The paper now says the result is a robust matched trajectory-goal planning
comparison, not a public OGBench leaderboard claim. The appendix lists plausible
mechanisms and remaining checks: eval-seed/larger-n confirmation, per-factor
Scene probes/decomposition, checkpoint sweep, and separate official fixed-goal
OGBench evaluation.

### Notes

* The result may combine better action-aligned latent geometry with better
  one-step latent dynamics; final logged forward MSE is about `0.002` for
  Action-NCE versus `0.004` for SIGReg, so do not frame it as a pure proof that
  latent marginal normality alone is harmful.
* The official fixed-goal chunked rerun is separate and still should not be
  mixed with this trajectory-goal table.

## 2026-06-28 — ogbench-scene-fixed-goal-singletask-s3072-complete

### Intent

Aggregate the fully remote OGBench Visual Scene fixed-goal singletask chunk run
for matched seed-3072 Action-NCE and SIGReg checkpoints, and record whether the
current world-model/CEM policy transfers to the fixed-goal task setting.

### Commands

#### Train

```bash
# No new training; evaluated existing seed-3072 Scene checkpoints.
```

#### Inference

```bash
# N/A
```

#### Diagnostics

```bash
.venv/bin/modal app list | head -80
.venv/bin/modal app logs ap-lP9ONtY1E5FEgKbaGHOKCV --tail 200
mkdir -p /tmp/scene_official_chunks_pull
for method in action_nce sigreg; do for task in 1 2 3 4 5; do for start in 0 10 20 30 40; do end=$((start+9)); .venv/bin/modal volume get --force multi-future-lewm-cache /scene/scene_official_${method}_s3072_task${task}_eps${start}_${end}_n10.txt /tmp/scene_official_chunks_pull/ >/dev/null; done; done; done
find /tmp/scene_official_chunks_pull -maxdepth 1 -type f -name 'scene_official_*_n10.txt' | sort | wc -l
```

#### Eval

```bash
# Completed by the already-launched Modal-resident supervisor:
.venv/bin/modal run --detach modal_app.py::scene_official_chunks --methods action_nce,sigreg --tasks 1,2,3,4,5 --starts 0,10,20,30,40 --chunk-size 10 --max-attempts 3
```

#### Compare

```bash
.venv/bin/python - <<'PY'
# Parsed all 50 /tmp/scene_official_chunks_pull/scene_official_*_n10.txt files,
# reading the final metrics dict from each chunk and aggregating per task/method.
PY
```

### Artifacts

* Summary: `progress/evaluations/scene-official-s3072/summary.json`
* Modal Volume chunk outputs: `.stable_worldmodel/scene/scene_official_{action_nce,sigreg}_s3072_task{1..5}_eps{0_9,10_19,20_29,30_39,40_49}_n10.txt`
* Local pulled chunk mirror: `/tmp/scene_official_chunks_pull/`
* Remote supervisor app: `ap-lP9ONtY1E5FEgKbaGHOKCV`

### Result

Base:

SIGReg fixed-goal singletask result: 0/250 successes, 0.0% overall. Each of the
five official tasks scored 0/50, and every episode hit the 750-step cap.

Candidate:

Action-NCE fixed-goal singletask result: 0/250 successes, 0.0% overall. Each of
the five official tasks scored 0/50, and every episode hit the 750-step cap.

Delta:

* Wins / losses: 0 Action-NCE wins, 0 SIGReg wins, 250 ties at failure.
* Recall collapse? not applicable; eval-only. This does not diagnose latent
  collapse and should not be mixed with train-time collapse metrics.

### Decision

Keep the fixed-goal evaluator and remote-supervisor orchestration, but do not
use this result as a public OGBench performance claim until the base-env
`visual-scene-v0`/`task_id` path is also smoke-tested. The result says the
current final-latent CEM planner is compatible with the fixed-goal Visual Scene
singletask environments but does not solve them out of the box. The paper-facing
complex-task result remains the repo trajectory-goal Scene comparison:
Action-NCE 80.0% mean versus SIGReg 58.0% mean over three training seeds.

### Notes

* The remote-supervisor design fixed the local-client dependency: all 50 chunk
  files landed even after the local Modal CLI process was killed during the run.
* Fixed-goal Visual Scene is a different protocol from the repo trajectory-goal
  Scene protocol. It uses five fixed OGBench goal tasks and a 750-step cap; the
  existing paper result samples dataset future goals with `goal_offset_steps=25`
  and `eval_budget=50`.
* A local registry sanity check on 2026-06-28 showed `visual-scene-v0` and
  `visual-scene-singletask-taskN-v0` have matching `task_infos`, but their
  rendered goal images are not byte-identical on reset. This is likely render or
  reset-context detail, not a wrong-task bug, but it is enough reason to run one
  cheap base-env `task_id` smoke before calling the singletask run exact public
  protocol.

## 2026-06-28 — broader-ogbench-single-seed-launch

### Intent

Run single-seed (`3072`) SIGReg versus Action-NCE comparisons on broader OGBench
tasks requested by Jack:

* Combinatorial: `puzzle-4x4-play-v0`, `puzzle-4x5-play-v0`
* Stochastic / uncertainty: `antmaze-teleport-navigate-v0`, `powderworld-medium-play-v0`
* Stitching / long-horizon: `antmaze-large-stitch-v0`, `antsoccer-medium-stitch-v0`

Because this repo's world model is RGB-pixel based, use OGB visual dataset/env
aliases where available:

* `puzzle-4x4-play-v0` -> `visual-puzzle-4x4-play-v0`
* `puzzle-4x5-play-v0` -> `visual-puzzle-4x5-play-v0`
* `antmaze-teleport-navigate-v0` -> `visual-antmaze-teleport-navigate-v0`
* `antmaze-large-stitch-v0` -> `visual-antmaze-large-stitch-v0`
* `powderworld-medium-play-v0` -> first three RGB channels of its native 6-channel observation
* `antsoccer-medium-stitch-v0` -> render pixels from qpos/qvel if present

### Commands

#### Train

```bash
# Launched as one Modal-resident supervisor. It prepares each OGB dataset once,
# then spawns 12 train_then_evaluate GPU lanes. The successful submission was:
.venv/bin/modal run --detach modal_app.py::ogb_comparison \
  --tasks puzzle-4x4-play-v0,puzzle-4x5-play-v0,antmaze-teleport-navigate-v0,powderworld-medium-play-v0,antmaze-large-stitch-v0,antsoccer-medium-stitch-v0 \
  --methods action_nce,sigreg \
  --seed 3072 \
  --eval-seed 42 \
  --eval-num 50 \
  --eval-env-batch-size 5
```

#### Inference

```bash
# Included in train_then_evaluate after epoch-10 training.
```

#### Diagnostics

```bash
.venv/bin/python -m py_compile eval.py modal_app.py ogb_prep.py
.venv/bin/python eval.py --config-name=ogb_generic policy=random eval.num_eval=1 eval.env_batch_size=1 eval.max_episode_steps=2 eval.dataset_name=dummy eval.ogb_dataset_name=visual-puzzle-4x4-play-v0 output.filename=/tmp/ogb_online_random_smoke.txt
.venv/bin/python eval.py --config-name=ogb_generic solver=pgd policy=random eval.num_eval=1 eval.env_batch_size=1 eval.max_episode_steps=2 eval.dataset_name=dummy eval.ogb_dataset_name=powderworld-medium-play-v0 output.filename=/tmp/ogb_online_powder_random_smoke.txt
.venv/bin/modal app list | head -20
tail -5 /tmp/<JOB>.log
```

#### Eval

```bash
# Generic online OGB eval path:
python eval.py --config-name=ogb_generic policy=<POLICY_DIR> \
  eval.dataset_name=<CONVERTED_H5_STEM> \
  eval.ogb_dataset_name=<OGB_ENV_DATASET_NAME> \
  eval.num_eval=50 eval.env_batch_size=5 \
  output.save_video=false output.filename=<RESULT>.txt
```

#### Compare

```bash
# Pending until result files land.
```

### Artifacts

* Converted datasets: `.stable_worldmodel/ogbench/{visual_puzzle_4x4_play,visual_puzzle_4x5_play,visual_antmaze_teleport_navigate,powderworld_medium_play_rgb,visual_antmaze_large_stitch,antsoccer_medium_stitch_rendered}.h5`
* SIGReg policies: `.stable_worldmodel/ogb_broad/<task_slug>/sigreg_s3072/`
* Action-NCE policies: `.stable_worldmodel/ogb_broad/<task_slug>/action_nce_s3072/`
* Eval outputs: `.stable_worldmodel/ogb_broad/<task_slug>/<task_slug>_{sigreg,action_nce}_s3072_n50.txt`
* Local Modal wrapper log: none useful; the first `nohup` wrapper exited with
  an empty `/tmp/ogb_broad_supervisor.log`, then the successful foreground
  `modal run --detach` submission was interrupted after verification.
  Use `modal app logs ap-t8vac1tgm8uhKScE0Olhv3`.
* Modal supervisor app: `ap-t8vac1tgm8uhKScE0Olhv3`

### Result

Base:

Partial. Completed SIGReg online n50 evals so far are 0/50 on
`puzzle-4x4-play-v0`, 0/50 on `antmaze-teleport-navigate-v0`, and 0/50 on
`antmaze-large-stitch-v0`; each episode timed out at the env step cap.
`puzzle-4x5-play-v0`, `powderworld-medium-play-v0`, and
`antsoccer-medium-stitch-v0` evals are not complete.

Candidate:

Partial. Completed Action-NCE online n50 evals so far are also 0/50 on
`puzzle-4x4-play-v0`, 0/50 on `antmaze-teleport-navigate-v0`, and 0/50 on
`antmaze-large-stitch-v0`; each episode timed out at the env step cap.
`puzzle-4x5-play-v0`, `powderworld-medium-play-v0`, and
`antsoccer-medium-stitch-v0` evals are not complete.

Delta:

* Wins / losses: no completed-task separation yet; all completed online evals tie at 0/50.
* Recall collapse? pending train-metric review; checkpoints exist for most runs.

### Decision

Do not treat the online OGB results as paper-facing claims yet. The all-zero
finished evals and frequent full-step timeouts point to a protocol/planner
mismatch or overly hard default CEM settings on these new OGB envs, not a
training crash. Next step is to audit/reset-goal handling and run bounded,
cheaper evals before spending more hours on full 50 x 1000-step CEM.

### Notes

* Added `ogb_prep.py`, `config/train/data/ogb_generic.yaml`,
  `config/eval/ogb_generic.yaml`, and `config/eval/solver/pgd.yaml`.
* Added `eval.protocol=ogb_online` for reset-goal OGB evaluation.
* Local random-policy smoke passed for visual puzzle and powderworld.
* Launch verification: `modal app list` showed `ap-t8vac1tgm8uhKScE0Olhv3`
  as `ephemeral (detached)` with `Tasks=8`; `modal app logs` showed active
  download progress for all six requested OGB datasets.
* 2026-06-28 status check: app `ap-t8vac1tgm8uhKScE0Olhv3` remains
  `ephemeral (detached)` with `Tasks=16`. Logs show live training and checkpoint
  saves, including named lanes for puzzle 4x4, powderworld, antmaze teleport,
  and antmaze large stitch across both methods. No `Completed OGB`/eval summary
  lines are present yet.
* 2026-06-28 error diagnosis: Modal dashboard errors are mainly worker
  preemptions on supervisor containers. Because `run_ogb_comparison_task` has
  `retries=RETRIES` and spawns child `run_train_then_evaluate` calls, retried
  supervisors spawned duplicate child calls with the same `subdir`. Early
  duplicates failed with `Refusing to start fresh in non-empty run directory
  without a checkpoint`; later `antmaze_large_stitch` duplicates checkpointed to
  the same paths as the original lanes. Do not use this app's
  `antmaze_large_stitch` outputs as clean results without a rerun under a
  non-duplicating supervisor or a fresh subdir.
* 2026-06-28 remediation: stopped contaminated app
  `ap-t8vac1tgm8uhKScE0Olhv3`, patched `modal_app.py` so OGB comparison
  supervisors persist child `FunctionCall` IDs and reattach on retry instead of
  spawning duplicate writers, and relaunched the same requested grid under fresh
  root `ogb_broad_clean1`. Clean app `ap-Hh88l6rZHlJPTwnMCfcNHF` is
  `ephemeral (detached)` with 17 tasks on the latest check. Logs show active
  training, including `powderworld_medium_play/sigreg_s3072`, and no
  duplicate-writer traceback or Python error in the checked tail.
* 2026-06-28 follow-up status: clean app `ap-Hh88l6rZHlJPTwnMCfcNHF` remains
  `ephemeral (detached)` with `Tasks=16`. Modal preempted some supervisors
  again, but the patched retry path reattached to existing child calls
  (`FunctionCall.from_id`) instead of spawning duplicate writers; no
  duplicate-directory/runtime traceback appeared in the checked logs. No eval
  outputs have landed yet. Checkpoint depth by volume listing: puzzle 4x4 both
  methods epoch 6; puzzle 4x5 both methods epoch 2; antmaze teleport Action-NCE
  epoch 6 and SIGReg epoch 7; powderworld Action-NCE epoch 1 and SIGReg epoch
  2; antmaze large Action-NCE epoch 6 and SIGReg epoch 7.
  `antsoccer-medium-stitch-v0` has not started training because its
  state-to-pixel conversion is still rendering (`100000/4000000` frames in the
  latest logs), making it the current bottleneck.
* 2026-06-28 antsoccer fix: there is no public `visual-antsoccer-*` OGB
  dataset, and the original MuJoCo-rendered conversion would have had to render
  4M frames. Added a bounded synthetic top-down pixelization for
  `antsoccer-medium-stitch-v0`: first 250k rows / 500 whole episodes, 32x32
  RGB, agent and ball drawn from qpos, with eval using state observations and
  the same rasterizer (`eval.synthetic_pixels=antsoccer_topdown`). Stopped the
  two slow rendered attempts (`ap-q72vDVXXZjdeQS0RLF985k`,
  `ap-keEo3MEimX4guvzE3xxVGX`) and launched synthetic app
  `ap-WTlOcVMs0qH2ej1WrympIk` under `ogb_broad_antsoccer_topdown1`. The
  synthetic HDF5 exists in the Modal volume as
  `ogbench/antsoccer_medium_stitch_topdown32_250k.h5`; Action-NCE and SIGReg
  child calls have been spawned but had not yet emitted training startup logs
  at the last check. Treat this antsoccer result as a screening comparison, not
  a direct visual OGB protocol result.
* 2026-06-28 current status: clean broad app `ap-Hh88l6rZHlJPTwnMCfcNHF`
  remains live with `Tasks=7`. Volume artifacts show completed epoch-10
  checkpoints for both methods on puzzle 4x4, antmaze teleport, antmaze large,
  and synthetic-topdown antsoccer; puzzle 4x5 SIGReg is done while puzzle 4x5
  Action-NCE is still training around epoch 9/10; powderworld Action-NCE is
  around epoch 8/10 and SIGReg around epoch 9/10. Completed n50 eval outputs
  have landed for puzzle 4x4, antmaze teleport, and antmaze large; all six
  method-task results are 0/50 with full-step timeouts. The active app is also
  running a slow default online CEM eval at roughly 3% of 50,000 total steps
  with an ETA of several hours. Separate antsoccer eval-only apps loaded the
  fixed synthetic checkpoint path but were stopped because default CEM projected
  multi-day runtime.
* 2026-06-28 eval fix: patched `eval.py` with an `ogb_trajectory` protocol for
  future-observation goals sampled from the converted OGB HDF5, plus
  task-specific success checks for puzzle button states, antmaze/antsoccer
  qpos goals, and powderworld pixel goals. Also fixed the eval policy's solver
  setup for vectorized discrete envs, where powderworld exposes a
  `MultiDiscrete` vector action space but PGD expects the single-env
  `Discrete(8)` space. Added defaults to `config/eval/ogb_generic.yaml` for
  `goal_offset_steps`, distance tolerance, and pixel tolerance. Validation:
  `py_compile` passed; Hydra compose for `eval.protocol=ogb_trajectory`
  passed; a local fake-HDF5 antmaze trajectory smoke with `policy=random`
  completed and wrote results.
* 2026-06-28 Modal blocker: all requested broad OGB training checkpoints are
  now present, including puzzle 4x5 Action-NCE and both powderworld methods,
  but launching the corrected Modal eval smoke failed with
  `workspace billing cycle spend limit reached`. Current `modal app list` shows
  `ap-Hh88l6rZHlJPTwnMCfcNHF` with `Tasks=0`. When Modal spend is unblocked,
  first run a tiny corrected smoke:

  ```bash
  .venv/bin/modal run --detach modal_app.py::evaluate \
    --config-name ogb_generic \
    --policy ogb_broad_clean1/puzzle_4x4_play/action_nce_s3072 \
    --overrides "eval.protocol=ogb_trajectory eval.dataset_name=ogbench/visual_puzzle_4x4_play eval.ogb_dataset_name=visual-puzzle-4x4-play-v0 eval.num_eval=2 eval.env_batch_size=2 eval.goal_offset_steps=25 eval.eval_budget=25 solver.num_samples=16 solver.n_steps=3 solver.topk=4 output.save_video=false output.filename=puzzle_4x4_action_nce_traj_smoke_n2.txt"
  ```

  Then run the trajectory matrix from existing checkpoints with
  `eval.num_eval=50`, `eval.env_batch_size=5`, `eval.goal_offset_steps=25`,
  `eval.eval_budget=50`, and a bounded solver such as
  `solver.num_samples=64 solver.n_steps=10 solver.topk=8` for CEM tasks.

---

## 2026-06-29 — broader-ogbench-corrected-trajectory-eval-s3072

### Intent

After billing was unblocked, run the corrected trajectory-goal evaluation from
the existing broad OGB checkpoints for the requested single training seed
(`3072`). This replaces the rejected online reset-goal sweep, whose completed
cells were all 0/50 timeouts, with `eval.protocol=ogb_trajectory`: sample
start/goal pairs from the converted OGB HDF5 trajectories and evaluate whether
MPC reaches the future trajectory state/image.

Baseline: SIGReg (`lewm`) epoch-10 checkpoints. Candidate: Action-NCE
(`lewm_masked_action_nce`, default inverse weight 0.30) epoch-10 checkpoints.
All evals use eval seed 42, `eval.num_eval=50`, `eval.env_batch_size=5`,
`goal_offset_steps=25`, and `eval_budget=50`. Continuous tasks use bounded CEM
(`num_samples=64`, `n_steps=10`, `topk=8`); powderworld uses PGD for its
discrete `Discrete(8)` action space.

### Commands

#### Train

```bash
# Training was completed by the prior clean broad OGB apps after the duplicate-
# writer fix. Reused epoch-10 checkpoints from these roots:
#   ogb_broad_clean1/{puzzle_4x4_play,puzzle_4x5_play,antmaze_teleport_navigate,powderworld_medium_play,antmaze_large_stitch}/{action_nce_s3072,sigreg_s3072}
#   ogb_broad_antsoccer_topdown1/antsoccer_medium_stitch/{action_nce_s3072,sigreg_s3072}
```

#### Inference

```bash
# Not applicable; MPC evaluation calls the world-model policy online.
```

#### Diagnostics

```bash
.venv/bin/python -m py_compile eval.py modal_app.py ogb_prep.py

# Local shape smoke: discrete policy outputs are int64 (num_envs,), continuous
# outputs still pass through scaler inverse, and PGD receives from_scalar=True
# for discrete warm starts/replanning.

# Modal verification after each launch:
tail -5 /tmp/<job>.log
.venv/bin/modal app list | head -20
```

#### Eval

```bash
# Tiny corrected smoke after billing unblocked.
nohup .venv/bin/modal run --detach modal_app.py::evaluate \
  --config-name ogb_generic \
  --policy ogb_broad_clean1/puzzle_4x4_play/action_nce_s3072 \
  --overrides "eval.protocol=ogb_trajectory eval.dataset_name=ogbench/visual_puzzle_4x4_play eval.ogb_dataset_name=visual-puzzle-4x4-play-v0 eval.num_eval=2 eval.env_batch_size=2 eval.goal_offset_steps=25 eval.eval_budget=25 solver.num_samples=16 solver.n_steps=3 solver.topk=4 output.save_video=false output.filename=puzzle_4x4_action_nce_traj_smoke_n2.txt" \
  > /tmp/ogb_puzzle_traj_smoke.log 2>&1 &

# Corrected trajectory matrix from existing checkpoints.
nohup .venv/bin/modal run --detach modal_app.py::ogb_eval_matrix \
  --tasks puzzle-4x4-play-v0,puzzle-4x5-play-v0,antmaze-teleport-navigate-v0,powderworld-medium-play-v0,antmaze-large-stitch-v0,antsoccer-medium-stitch-v0 \
  --methods action_nce,sigreg \
  --seed 3072 \
  --eval-seed 42 \
  --eval-num 50 \
  --eval-env-batch-size 5 \
  --goal-offset-steps 25 \
  --eval-budget 50 \
  --subdir-root ogb_broad_clean1 \
  --cem-num-samples 64 \
  --cem-n-steps 10 \
  --cem-topk 8 \
  --pgd-num-samples 64 \
  --pgd-n-steps 10 \
  > /tmp/ogb_broad_traj_matrix.log 2>&1 &

# Powderworld replacement after the final discrete-output/warm-start patch.
nohup .venv/bin/modal run --detach modal_app.py::evaluate \
  --config-name ogb_generic \
  --policy ogb_broad_clean1/powderworld_medium_play/action_nce_s3072 \
  --overrides "solver=pgd eval.protocol=ogb_trajectory eval.dataset_name=ogbench/powderworld_medium_play_rgb eval.ogb_dataset_name=powderworld-medium-play-v0 eval.num_eval=50 eval.env_batch_size=5 eval.goal_offset_steps=25 eval.eval_budget=50 solver.num_samples=64 solver.n_steps=10 output.save_video=false output.filename=powderworld_medium_play_action_nce_s3072_traj_g25_b50_n50_fixed_discrete2.txt" \
  > /tmp/ogb_powder_action_nce_fixed2.log 2>&1 &

nohup .venv/bin/modal run --detach modal_app.py::evaluate \
  --config-name ogb_generic \
  --policy ogb_broad_clean1/powderworld_medium_play/sigreg_s3072 \
  --overrides "solver=pgd eval.protocol=ogb_trajectory eval.dataset_name=ogbench/powderworld_medium_play_rgb eval.ogb_dataset_name=powderworld-medium-play-v0 eval.num_eval=50 eval.env_batch_size=5 eval.goal_offset_steps=25 eval.eval_budget=50 solver.num_samples=64 solver.n_steps=10 output.save_video=false output.filename=powderworld_medium_play_sigreg_s3072_traj_g25_b50_n50_fixed_discrete2.txt" \
  > /tmp/ogb_powder_sigreg_fixed2.log 2>&1 &
```

#### Compare

```bash
.venv/bin/modal volume get multi-future-lewm-cache \
  ogb_broad_clean1/puzzle_4x4_play/puzzle_4x4_play_action_nce_s3072_traj_g25_b50_n50.txt \
  /tmp/ogb_results_pull/

# Repeated for all 12 result files, then parsed locally into
# progress/evaluations/ogb-broad-corrected-s3072/summary.json.
```

### Artifacts

* Train metadata: `finetuning/results/<RUN_ID>/train_result.json` not used for
  these LeWM Modal jobs; checkpoint metadata lives in each policy directory.
* GPU metrics: Modal/W&B training metrics under the policy directories.
* Predictions: not applicable; MPC policy evaluated online.
* Eval summary:
  `progress/evaluations/ogb-broad-corrected-s3072/summary.json`
* Per-dialogue eval: not applicable.
* Modal result files pulled to `/tmp/ogb_results_pull/`:
  * `puzzle_4x4_play_action_nce_s3072_traj_g25_b50_n50.txt`
  * `puzzle_4x4_play_sigreg_s3072_traj_g25_b50_n50.txt`
  * `puzzle_4x5_play_action_nce_s3072_traj_g25_b50_n50.txt`
  * `puzzle_4x5_play_sigreg_s3072_traj_g25_b50_n50.txt`
  * `antmaze_teleport_navigate_action_nce_s3072_traj_g25_b50_n50.txt`
  * `antmaze_teleport_navigate_sigreg_s3072_traj_g25_b50_n50.txt`
  * `powderworld_medium_play_action_nce_s3072_traj_g25_b50_n50_fixed_discrete2.txt`
  * `powderworld_medium_play_sigreg_s3072_traj_g25_b50_n50_fixed_discrete2.txt`
  * `antmaze_large_stitch_action_nce_s3072_traj_g25_b50_n50.txt`
  * `antmaze_large_stitch_sigreg_s3072_traj_g25_b50_n50.txt`
  * `antsoccer_medium_stitch_action_nce_s3072_traj_g25_b50_n50.txt`
  * `antsoccer_medium_stitch_sigreg_s3072_traj_g25_b50_n50.txt`

### Result

Base:

SIGReg results under corrected trajectory-goal eval:

| Task | SIGReg success | Mean steps |
|---|---:|---:|
| `visual-puzzle-4x4-play-v0` | 50.0% (25/50) | 30.08 |
| `visual-puzzle-4x5-play-v0` | 52.0% (26/50) | 31.84 |
| `visual-antmaze-teleport-navigate-v0` | 40.0% (20/50) | 30.58 |
| `powderworld-medium-play-v0` | 6.0% (3/50) | 47.10 |
| `visual-antmaze-large-stitch-v0` | 30.0% (15/50) | 36.14 |
| `antsoccer-medium-stitch-v0` | 88.0% (44/50) | 6.94 |

Candidate:

Action-NCE results under corrected trajectory-goal eval:

| Task | Action-NCE success | Mean steps |
|---|---:|---:|
| `visual-puzzle-4x4-play-v0` | 34.0% (17/50) | 36.30 |
| `visual-puzzle-4x5-play-v0` | 26.0% (13/50) | 38.84 |
| `visual-antmaze-teleport-navigate-v0` | 46.0% (23/50) | 29.08 |
| `powderworld-medium-play-v0` | 16.0% (8/50) | 43.32 |
| `visual-antmaze-large-stitch-v0` | 30.0% (15/50) | 37.26 |
| `antsoccer-medium-stitch-v0` | 88.0% (44/50) | 6.94 |

Delta:

| Task | Delta (Action-NCE - SIGReg) | Paired wins / losses / both / neither |
|---|---:|---|
| `visual-puzzle-4x4-play-v0` | -16 pts | 4 / 12 / 13 / 21 |
| `visual-puzzle-4x5-play-v0` | -26 pts | 3 / 16 / 10 / 21 |
| `visual-antmaze-teleport-navigate-v0` | +6 pts | 3 / 0 / 20 / 27 |
| `powderworld-medium-play-v0` | +10 pts | 6 / 1 / 2 / 41 |
| `visual-antmaze-large-stitch-v0` | 0 pts | 2 / 2 / 13 / 33 |
| `antsoccer-medium-stitch-v0` | 0 pts | 0 / 0 / 44 / 6 |

* Wins / losses: SIGReg clearly wins both combinatorial puzzle tasks; Action-NCE
  modestly wins antmaze teleport and powderworld; antmaze-large and synthetic
  antsoccer are ties at this single seed.
* Recall collapse? no broad collapse signal from the completed run. The main
  failure mode was evaluation plumbing: the first online protocol was rejected,
  then powderworld needed discrete-action output and PGD warm-start fixes.

### Decision

Keep as a useful screening result, but do not treat it as a public OGBench
leaderboard claim or paper-facing proof. The corrected trajectory-goal protocol
is now working across all requested task families. It says Action-NCE is not a
uniform broad-OGB win: it helps on stochastic/discrete powderworld and teleport
navigation, ties on the long-horizon stitching tasks at this seed, and loses
both combinatorial puzzles to SIGReg.

Next decision point: either run more seeds under this repo trajectory protocol
for the broad-task story, or implement task-specific/public OGB protocols before
making external-comparison claims.

### Notes

* Fixed Modal supervisor retry duplication earlier by persisting child
  `FunctionCall` ids and reattaching after supervisor preemptions.
* Fixed powderworld discrete planning after the first trajectory matrix:
  vectorized env action spaces are converted to the single-env `Discrete(8)`;
  policy actions return clipped integer ids; scaler inverse is skipped for
  discrete outgoing env actions; PGD warm starts/replans pass `from_scalar=True`.
* Tiny powderworld fixed-path smoke completed after the final patch:
  Action-NCE, `eval.num_eval=2`, `eval_budget=30`, PGD `16 x 3`, result 50%
  (1/2), confirming the discrete output path.
* Corrected matrix app `ap-oywkV63S5vv5hCSYVIxHCb` wrote the non-powderworld
  result files before failing in the old powderworld lane. Final powderworld
  replacement apps were `ap-zBUfufFovBiJ6CQMmqD99p` (Action-NCE) and
  `ap-cg7MtSHKITOGInYDUtjsbG` (SIGReg).
* Antsoccer uses the bounded synthetic top-down state-to-pixel conversion
  (`antsoccer_medium_stitch_topdown32_250k.h5`) because no public visual
  antsoccer OGB dataset exists and MuJoCo-rendering 4M frames was too slow.
* Final `modal app list` check showed no active tasks for the relevant broad
  OGB apps.

---

## 2026-06-30 — puzzle-gap-diagnostics-action-nce-vs-sigreg

### Intent

Diagnose why the corrected broad OGB trajectory screen showed a large SIGReg
advantage on the two combinatorial puzzle tasks:
`visual-puzzle-4x4-play-v0` scored Action-NCE 34% vs SIGReg 50%, and
`visual-puzzle-4x5-play-v0` scored Action-NCE 26% vs SIGReg 52%.

### Commands

#### Train

```bash
# No new training.
```

#### Inference

```bash
# No standalone inference; diagnostics use existing eval outputs and encoder probes.
```

#### Diagnostics

```bash
# Parse paired eval result files from /tmp/ogb_results_pull.
.venv/bin/python - <<'PY'
# Extract episode_successes, episode_steps, start_episode_ids, and start_steps
# from the four puzzle result files and compare immediate vs nontrivial wins.
PY

# Pull training metadata/metrics.
.venv/bin/modal volume get --force multi-future-lewm-cache \
  /ogb_broad_clean1/puzzle_4x4_play/action_nce_s3072/metrics.jsonl \
  /tmp/ogb_puzzle_diag/puzzle_4x4_play/action_nce_s3072/

# Button-state linear probes, n=1000.
.venv/bin/modal run --detach modal_app.py::probe \
  --policy ogb_broad_clean1/puzzle_4x4_play/action_nce_s3072 \
  --dataset ogbench/visual_puzzle_4x4_play \
  --n 1000 \
  --overrides "--state-key button_states"

.venv/bin/modal run --detach modal_app.py::probe \
  --policy ogb_broad_clean1/puzzle_4x4_play/sigreg_s3072 \
  --dataset ogbench/visual_puzzle_4x4_play \
  --n 1000 \
  --overrides "--state-key button_states"

.venv/bin/modal run --detach modal_app.py::probe \
  --policy ogb_broad_clean1/puzzle_4x5_play/action_nce_s3072 \
  --dataset ogbench/visual_puzzle_4x5_play \
  --n 1000 \
  --overrides "--state-key button_states"

.venv/bin/modal run --detach modal_app.py::probe \
  --policy ogb_broad_clean1/puzzle_4x5_play/sigreg_s3072 \
  --dataset ogbench/visual_puzzle_4x5_play \
  --n 1000 \
  --overrides "--state-key button_states"

# qpos probes, n=1000, to check whether Action-NCE still encodes continuous pose.
.venv/bin/modal run --detach modal_app.py::probe \
  --policy ogb_broad_clean1/puzzle_4x4_play/action_nce_s3072 \
  --dataset ogbench/visual_puzzle_4x4_play \
  --n 1000 \
  --overrides "--state-key qpos"
```

#### Eval

```bash
# Reused corrected trajectory eval outputs from the previous entry.
```

#### Compare

```bash
# Pairwise compare Action-NCE and SIGReg episode_successes on the same
# start_episode_ids/start_steps arrays.
```

### Artifacts

* Eval outputs: `/tmp/ogb_results_pull/puzzle_4x4_play_{action_nce,sigreg}_s3072_traj_g25_b50_n50.txt`
* Eval outputs: `/tmp/ogb_results_pull/puzzle_4x5_play_{action_nce,sigreg}_s3072_traj_g25_b50_n50.txt`
* Pulled metrics/configs: `/tmp/ogb_puzzle_diag/puzzle_{4x4,4x5}_play/{action_nce_s3072,sigreg_s3072}/`
* Probe apps:
  * Button states: `ap-eZ5OeccsneXhIwwmT9g1Y7`, `ap-gKSDsm7t0b6q5QDuV3hRNo`,
    `ap-QeUq1XmROewncDaSVARlTS`, `ap-sY3KSBocBQ7lCIgbNMgzBc`
  * qpos: `ap-WYf8pU3ughiB65UQ8kTEt2`, `ap-GSzzgkS3GreqmpIOPeFwaP`,
    `ap-DWBr7JLJM7dbUS8JE7nsYP`, `ap-iLVdZ1PIH5cPEeBRCs98Xj`

### Result

Base:

SIGReg uses the same start/goal pairs as Action-NCE. The step-1 already-solved
successes are identical: 12/50 on 4x4 and 10/50 on 4x5. On nontrivial puzzle
transitions, SIGReg solves 13/38 on 4x4 and 16/40 on 4x5.

Button-state probes show SIGReg linearly exposes the variable button bits:
4x4 mean R2 is 0.1969 with bits 11-15 at about 0.98; 4x5 mean R2 is 0.1914
with bits 0 and 14-19 at about 0.95-0.97.

Candidate:

Action-NCE has the same immediate successes, but only solves 5/38 nontrivial
4x4 transitions and 3/40 nontrivial 4x5 transitions.

Button-state probes are negative/near-zero for every bit: 4x4 mean R2 -0.0620;
4x5 mean R2 -0.1002. qpos probes show Action-NCE is not globally broken: the
main continuous pose coordinates are still highly decodable (first four qpos
dims around 0.99 on both puzzle tasks), often comparable to SIGReg.

Training metrics also show a scale/geometry difference. Final train-step
latent `emb_std` is about 0.27-0.29 for Action-NCE versus about 0.99 for
SIGReg. Although Action-NCE has lower raw forward MSE, normalized by latent
variance its forward error is roughly 5-6x larger than SIGReg on these puzzle
runs. Most Action-NCE total loss is the inverse-NCE term, not forward geometry.

Delta:

* Wins / losses: the puzzle gap comes from nontrivial button-state transitions,
  not from already-solved cases or mismatched episode sampling.
* Recall collapse? no full collapse; Action-NCE retains continuous pose
  information. The failure is selective: poor encoding of discrete
  combinatorial button state and compressed latent geometry for final-latent
  CEM.

### Decision

Treat the puzzle gap as a real objective/representation mismatch, not a Modal
or eval pairing bug. The likely mechanism is that Action-NCE on puzzle learns
the controllable motion/action geometry but drops the discrete button-state
bits that define the combinatorial goal. SIGReg's Gaussian marginal pressure
keeps enough latent spread to preserve those bits, making the final-latent CEM
cost more useful for puzzle goals.

Next fixes to test if we want Action-NCE to cover puzzles:

1. Add a weak SIGReg or variance-floor hybrid to Action-NCE on puzzle.
2. Reduce Action-NCE inverse weight on low-dimensional deterministic puzzle
   tasks so forward/button-state geometry is not dominated by inverse NCE.
3. Add a button-state/goal-state probe gate before spending full puzzle evals:
   fail the run if button bits are not linearly decodable.

### Notes

* All diagnostic Modal apps completed and later `modal app list` showed zero
  active tasks.

## 2026-06-30 — ogb-play-readme-smoke

### Intent

Verify the user-facing README/media/play.py pass for the six broad OGB tasks.
This is a playability and documentation smoke test, not a model-training or
policy-evaluation result.

### Commands

#### Train

```bash
# not applicable
```

#### Inference

```bash
# not applicable
```

#### Diagnostics

```bash
.venv/bin/python -m py_compile play.py make_task_media.py
.venv/bin/python -m pytest -q tests/test_play.py
PYGAME_HIDE_SUPPORT_PROMPT=1 .venv/bin/python play.py --task puzzle4x4 --selftest
PYGAME_HIDE_SUPPORT_PROMPT=1 .venv/bin/python play.py --task puzzle4x5 --selftest
PYGAME_HIDE_SUPPORT_PROMPT=1 .venv/bin/python play.py --task antmaze_teleport --selftest
PYGAME_HIDE_SUPPORT_PROMPT=1 .venv/bin/python play.py --task powderworld --selftest
PYGAME_HIDE_SUPPORT_PROMPT=1 .venv/bin/python play.py --task antmaze_large --selftest
PYGAME_HIDE_SUPPORT_PROMPT=1 .venv/bin/python play.py --task antsoccer --selftest
PYGAME_HIDE_SUPPORT_PROMPT=1 .venv/bin/python play.py --list
git diff --check -- README.md play.py make_task_media.py progress/README.md
```

#### Eval

```bash
# not applicable
```

#### Compare

```bash
# not applicable
```

### Artifacts

* README media: `assets/datasets/{puzzle4x4,puzzle4x5,antmaze_teleport,powderworld,antmaze_large,antsoccer}.{gif,png}`
* Play previews: `/tmp/play_selftest_{puzzle4x4,puzzle4x5,antmaze_teleport,powderworld,antmaze_large,antsoccer}.png`

### Result

Base:

Existing play tasks and `tests/test_play.py` still pass.

Candidate:

All six broad OGB tasks instantiate through `play.py`, produce 224x224 RGB
current/goal panels, step for 40 random/selftest actions, and report native HUD
status without crashing.

Delta:

* Wins / losses: no policy comparison.
* Recall collapse? not applicable.

### Decision

Keep the README media and broad OGB play support. These tasks are documented as
playable/internal screening tasks, distinct from the released four-task
benchmark.

### Notes

* Powderworld uses `env.render()` for the human-facing frame instead of raw
  six-channel observations.
* AntSoccer has no public visual OGB dataset; `play.py` renders the native
  state environment for human play.

## 2026-06-30 — ogb_state_nce_puzzle_4x4_play_action_state_nce_s3072_e3

### Intent

Test the simplest principled fix for the OGB puzzle gap: keep the current
Action-NCE anti-collapse objective, and add a small in-batch contrastive loss on
predicted future-state deltas. This directly targets the observed failure mode
where Action-NCE preserves continuous pose but does not encode puzzle button
state. Compare first against the existing puzzle-4x4 Action-NCE and SIGReg
button-state probes before spending on full n50 trajectory eval.

### Commands

#### Train

```bash
.venv/bin/modal run --detach modal_app.py::train \
  --config-name lewm_masked_action_state_nce \
  --data ogb_generic \
  --subdir ogb_state_nce/puzzle_4x4_play/action_state_nce_s3072_e3 \
  --ogb-task puzzle-4x4-play-v0 \
  --overrides "seed=3072 runtime.stop_after_epoch=3 early_stopping.enabled=false"
```

#### Inference

```bash
# not applicable
```

#### Diagnostics

```bash
.venv/bin/python -m py_compile train.py modal_app.py eval.py
PYTHONPATH=. .venv/bin/pytest -q tests/test_masked_transition.py
git diff --check
.venv/bin/modal app list | head -20
```

#### Eval

```bash
# pending; run button-state probe before any full trajectory eval
```

#### Compare

```bash
# pending
```

### Artifacts

* Train metadata: `finetuning/results/ogb_state_nce/puzzle_4x4_play/action_state_nce_s3072_e3/train_result.json`
* GPU metrics: `finetuning/results/ogb_state_nce/puzzle_4x4_play/action_state_nce_s3072_e3/gpu_metrics.csv`
* Checkpoints: `ogb_state_nce/puzzle_4x4_play/action_state_nce_s3072_e3/*_object.ckpt`
* W&B: `https://wandb.ai/jack-b/masked-transition-lewm/runs/ogb_state_nce_puzzle_4x4_play_action_state_nce_s3072_e3`
* Modal app: `ap-rLiKMWb4Po6Vw597CO9tCh`

### Result

Base:

Existing Action-NCE puzzle probes showed near-zero/negative button-state
decodability despite preserving main continuous pose coordinates. SIGReg
retained strong active-button decodability and outperformed Action-NCE on
nontrivial puzzle episodes.

Candidate:

Running. Validation before training logged `state_nce_loss`, `state_nce_acc`,
and `state_nce_weight=0.03`, and first backward passed with all tracked
parameters receiving gradients.

Delta:

* Wins / losses: pending button-state probe.
* Recall collapse? no evidence yet; early training loss is decreasing normally.

### Decision

Pending. If the epoch-3 button-state probe improves materially, scale to a full
e10 additive run and then n50 trajectory eval. If it remains near zero, scale
the loss weight or try the replacement `lewm_masked_state_nce` config as the
next controlled step.

### Notes

* This is loss-only: no new module, head, privileged state target, or
  task-specific puzzle hack.
* The initial weight is deliberately small (`0.03`) because the state NCE term
  starts near `log(B * H)` and otherwise could dominate the existing
  Action-NCE/MSE balance.
* The replacement config exists but should remain second-stage until the
  additive sentinel is diagnosed.

## 2026-06-30 — ogb_state_nce_puzzle_4x4_play_action_state_nce_s3072_epoch1_button_probe

### Intent

Run an early button-state probe on the epoch-1 additive state-NCE checkpoint
while the epoch-3 sentinel continues. This is an early trend check only; the
planned decision gate remains epoch 3.

### Commands

#### Train

```bash
# continued from ogb_state_nce_puzzle_4x4_play_action_state_nce_s3072_e3
```

#### Inference

```bash
# not applicable
```

#### Diagnostics

```bash
.venv/bin/modal run --detach modal_app.py::probe \
  --policy ogb_state_nce/puzzle_4x4_play/action_state_nce_s3072_e3/lewm_masked_action_state_nce_epoch_1 \
  --dataset ogbench/visual_puzzle_4x4_play \
  --n 1000 \
  --overrides "--state-key button_states"
```

#### Eval

```bash
# not applicable
```

#### Compare

```bash
# compare mean R2 against prior Action-NCE/SIGReg button-state probes
```

### Artifacts

* Probe Modal app: `ap-09OvjyNBLhEptXp0XLavmx`
* Policy object: `ogb_state_nce/puzzle_4x4_play/action_state_nce_s3072_e3/lewm_masked_action_state_nce_epoch_1_object.ckpt`

### Result

Base:

Existing Action-NCE puzzle-4x4 button-state mean R2 was near zero/negative
(`-0.0620`), while SIGReg was positive (`0.1969`) with active button bits near
`0.95-0.99`.

Candidate:

Epoch-1 additive state-NCE button-state probe:

```text
PROBE_R2 MEAN = -0.0059
```

Delta:

* Wins / losses: no clear win at epoch 1.
* Recall collapse? no; training metrics at epoch 1 were healthy
  (`fit/emb_std=0.2886`, `fit/state_nce_acc=0.9896`,
  `validate/state_nce_acc=0.1509`).

### Decision

Continue to the planned epoch-3 gate. If epoch 3 remains near zero on button
R2, treat `state_nce.weight=0.03` as too weak or misaligned and escalate to a
larger additive weight or the replacement config.

### Notes

* The training objective is learning transition discrimination, but epoch-1
  linear decodability has not yet shown the missing puzzle-button variables.

## 2026-06-30 — ogb_state_nce_puzzle_4x4_play_action_state_nce_s3072_epoch3_button_probe

### Intent

Run the planned decision-gate button-state probe on the epoch-3 additive
Action-NCE + delta-state-NCE checkpoint.

### Commands

#### Train

```bash
# completed by ogb_state_nce_puzzle_4x4_play_action_state_nce_s3072_e3
```

#### Inference

```bash
# not applicable
```

#### Diagnostics

```bash
.venv/bin/modal run --detach modal_app.py::probe \
  --policy ogb_state_nce/puzzle_4x4_play/action_state_nce_s3072_e3/lewm_masked_action_state_nce_epoch_3 \
  --dataset ogbench/visual_puzzle_4x4_play \
  --n 1000 \
  --overrides "--state-key button_states"
```

#### Eval

```bash
# not applicable; no full trajectory eval until button-state decodability recovers
```

#### Compare

```bash
# compare mean R2 against prior Action-NCE/SIGReg button-state probes
```

### Artifacts

* Train Modal app: `ap-rLiKMWb4Po6Vw597CO9tCh`
* Probe Modal app: `ap-eh5WKx04H1DAxmeG5sRbTQ`
* Policy object: `ogb_state_nce/puzzle_4x4_play/action_state_nce_s3072_e3/lewm_masked_action_state_nce_epoch_3_object.ckpt`
* W&B: `https://wandb.ai/jack-b/masked-transition-lewm/runs/ogb_state_nce_puzzle_4x4_play_action_state_nce_s3072_e3`

### Result

Base:

Existing Action-NCE puzzle-4x4 button-state mean R2 was near zero/negative
(`-0.0620`), while SIGReg was positive (`0.1969`) with active button bits near
`0.95-0.99`.

Candidate:

Epoch-3 additive delta-state-NCE button-state probe:

```text
PROBE_R2 MEAN = -0.0854
```

Training itself was healthy:

```text
validate/emb_std = 0.2918
validate/effective_rank = 7.1640
validate/state_nce_acc = 0.9135
validate/state_nce_loss = 1.1981
```

Delta:

* Wins / losses: loss vs SIGReg, and no improvement over Action-NCE on the
  button-state probe.
* Recall collapse? no. The failure is representation content/alignment, not
  collapse or optimization.

### Decision

Reject small additive delta-state-NCE as the fix for puzzle button-state
decodability. Next controlled step: test `lewm_masked_state_nce`, replacing
Action-NCE with delta-state-NCE, to isolate whether Action-NCE pressure is
actively suppressing the button bits. If replacement also fails while
state-NCE metrics are healthy, pivot from delta targets to an absolute or
hard-negative state-identity objective.

### Notes

* The result is diagnostic: state transition discrimination can become easy
  without recovering persistent combinatorial state variables.
* Do not run full n50 trajectory eval for this checkpoint.

## 2026-06-30 — ogb_state_nce_puzzle_4x4_play_state_nce_replace_s3072_e3

### Intent

Second controlled scale-up after additive delta-state-NCE failed the button
probe. Replace Action-NCE by setting `inverse_weight=0` and training with
delta-state-NCE (`weight=0.10`) plus the standard forward latent loss. This
tests whether Action-NCE pressure is actively suppressing puzzle button-state
information.

### Commands

#### Train

```bash
.venv/bin/modal run --detach modal_app.py::train \
  --config-name lewm_masked_state_nce \
  --data ogb_generic \
  --subdir ogb_state_nce/puzzle_4x4_play/state_nce_replace_s3072_e3 \
  --ogb-task puzzle-4x4-play-v0 \
  --overrides "seed=3072 runtime.stop_after_epoch=3 early_stopping.enabled=false"
```

#### Inference

```bash
# pending
```

#### Diagnostics

```bash
.venv/bin/modal app list | head -20
git diff --check
```

#### Eval

```bash
.venv/bin/modal run --detach modal_app.py::probe \
  --policy ogb_state_nce/puzzle_4x4_play/state_nce_replace_s3072_e3/lewm_masked_state_nce_epoch_3 \
  --dataset ogbench/visual_puzzle_4x4_play \
  --n 1000 \
  --overrides "--state-key button_states"
```

#### Compare

```bash
# pending
```

### Artifacts

* Modal app: `ap-1KHVGY0UjPWJsIrw1fX76Y`
* Train metadata: `finetuning/results/ogb_state_nce/puzzle_4x4_play/state_nce_replace_s3072_e3/train_result.json`
* GPU metrics: `finetuning/results/ogb_state_nce/puzzle_4x4_play/state_nce_replace_s3072_e3/gpu_metrics.csv`
* Checkpoints: `ogb_state_nce/puzzle_4x4_play/state_nce_replace_s3072_e3/*_object.ckpt`

### Result

Base:

Small additive delta-state-NCE reached healthy transition-discrimination metrics
but failed the epoch-3 button probe (`PROBE_R2 MEAN = -0.0854`).

Candidate:

Replacement delta-state-NCE completed 3 epochs. Final metrics showed the
objective fit the training batch but generalized poorly:

```text
fit/emb_std = 0.0575
fit/effective_rank = 37.10
fit/state_nce_acc = 1.0000
fit/state_nce_loss = 0.2583
validate/emb_std = 0.0748
validate/effective_rank = 23.09
validate/state_nce_acc = 0.0370
validate/state_nce_loss = 7.0115
```

Epoch-3 button-state probe:

```text
PROBE_R2 MEAN = -0.0563
```

Delta:

* Wins / losses: no meaningful button-state recovery; still below SIGReg and
  not an improvement over the existing Action-NCE failure.
* Recall collapse? borderline low latent scale, but the decisive issue is that
  removing Action-NCE did not restore puzzle button-state information.

### Decision

Reject replacement delta-state-NCE. Do not spend on full trajectory eval for
this checkpoint. Since removing Action-NCE did not help, the next controlled
step is to change the state-NCE target from pure deltas to absolute future-state
identity while restoring the additive Action-NCE setup.

### Notes

* This still uses no new module/head and no privileged button labels.
* This is evidence against "Action-NCE suppresses buttons" and in favor of
  "delta targets discard persistent latch state."

## 2026-06-30 — ogb_state_nce_puzzle_4x4_play_state_nce_replace_s3072_epoch1_button_probe

### Intent

Run an early button-state probe on the epoch-1 replacement state-NCE checkpoint.
This checks whether removing Action-NCE immediately restores puzzle button
state in the latent.

### Commands

#### Train

```bash
# continued from ogb_state_nce_puzzle_4x4_play_state_nce_replace_s3072_e3
```

#### Inference

```bash
# not applicable
```

#### Diagnostics

```bash
.venv/bin/modal run --detach modal_app.py::probe \
  --policy ogb_state_nce/puzzle_4x4_play/state_nce_replace_s3072_e3/lewm_masked_state_nce_epoch_1 \
  --dataset ogbench/visual_puzzle_4x4_play \
  --n 1000 \
  --overrides "--state-key button_states"
```

#### Eval

```bash
# not applicable
```

#### Compare

```bash
# compare mean R2 against prior Action-NCE/SIGReg and additive state-NCE probes
```

### Artifacts

* Probe Modal app: `ap-FIeZhOlip7IYAUEFbJZToO`
* Policy object: `ogb_state_nce/puzzle_4x4_play/state_nce_replace_s3072_e3/lewm_masked_state_nce_epoch_1_object.ckpt`

### Result

Base:

Additive delta-state-NCE epoch-3 button-state probe scored
`PROBE_R2 MEAN = -0.0854`.

Candidate:

Replacement delta-state-NCE epoch-1 button-state probe:

```text
PROBE_R2 MEAN = -0.0110
```

Epoch-1 replacement training was noncollapsed but did not yet generalize the
state-NCE objective:

```text
fit/emb_std = 0.0887
fit/effective_rank = 83.27
validate/emb_std = 0.1514
validate/effective_rank = 16.73
validate/state_nce_acc = 0.0081
```

Delta:

* Wins / losses: no meaningful button-state recovery at epoch 1.
* Recall collapse? no hard collapse, but low train `emb_std` and poor
  validation state-NCE generalization make this arm borderline.

### Decision

Continue to epoch 3 for a clean replacement sentinel. If epoch 3 remains flat,
do not spend on trajectory eval; change the target to absolute or hard-negative
state identity.

### Notes

* Removing Action-NCE alone is not an immediate fix for button-state
  decodability.

## 2026-06-30 — ogb_state_nce_puzzle_4x4_play_state_nce_replace_s3072_epoch3_button_probe

### Intent

Run the planned epoch-3 button-state probe on the replacement delta-state-NCE
checkpoint. This closes the control for whether Action-NCE is suppressing the
puzzle button variables.

### Commands

#### Train

```bash
# completed by ogb_state_nce_puzzle_4x4_play_state_nce_replace_s3072_e3
```

#### Inference

```bash
# not applicable
```

#### Diagnostics

```bash
.venv/bin/modal run --detach modal_app.py::probe \
  --policy ogb_state_nce/puzzle_4x4_play/state_nce_replace_s3072_e3/lewm_masked_state_nce_epoch_3 \
  --dataset ogbench/visual_puzzle_4x4_play \
  --n 1000 \
  --overrides "--state-key button_states"
```

#### Eval

```bash
# not applicable; no full trajectory eval until button-state decodability recovers
```

#### Compare

```bash
# compare mean R2 against prior Action-NCE/SIGReg and additive state-NCE probes
```

### Artifacts

* Train Modal app: `ap-1KHVGY0UjPWJsIrw1fX76Y`
* Probe Modal app: `ap-pErZmsb0xm2gj6DUD8i8a3`
* Policy object: `ogb_state_nce/puzzle_4x4_play/state_nce_replace_s3072_e3/lewm_masked_state_nce_epoch_3_object.ckpt`
* W&B: `https://wandb.ai/jack-b/masked-transition-lewm/runs/ogb_state_nce_puzzle_4x4_play_state_nce_replace_s3072_e3`

### Result

Base:

Existing Action-NCE puzzle-4x4 button-state mean R2 was near zero/negative
(`-0.0620`), while SIGReg was positive (`0.1969`) with active button bits near
`0.95-0.99`. Additive delta-state-NCE epoch 3 scored `-0.0854`.

Candidate:

Replacement delta-state-NCE epoch-3 button-state probe:

```text
PROBE_R2 MEAN = -0.0563
```

Delta:

* Wins / losses: no meaningful gain over Action-NCE; clear loss vs SIGReg.
* Recall collapse? borderline low scale at epoch 3 (`validate/emb_std=0.0748`),
  but the main conclusion is that dropping Action-NCE did not recover buttons.

### Decision

Reject replacement delta-state-NCE. The next rung is additive absolute
state-NCE: keep Action-NCE and weight `0.03`, but set
`loss.masked.state_nce.mode=absolute` so the contrastive target includes
persistent state identity instead of only transition deltas.

### Notes

* This falsifies the simple suppressor story. Delta-state contrast is not
  enough because the missing puzzle variable is mostly a persistent latch state.

## 2026-06-30 — ogb_state_nce_puzzle_4x4_play_action_state_nce_abs_s3072_e3

### Intent

Third controlled rung after both delta-state variants failed the button probe.
Keep the current Action-NCE method and the small additive state-NCE weight
(`0.03`), but switch `state_nce` from `delta` to `absolute` so the contrastive
target includes persistent state identity rather than only the latent change.

### Commands

#### Train

```bash
.venv/bin/modal run --detach modal_app.py::train \
  --config-name lewm_masked_action_state_nce \
  --data ogb_generic \
  --subdir ogb_state_nce/puzzle_4x4_play/action_state_nce_abs_s3072_e3 \
  --ogb-task puzzle-4x4-play-v0 \
  --overrides "seed=3072 runtime.stop_after_epoch=3 early_stopping.enabled=false loss.masked.state_nce.mode=absolute"
```

#### Inference

```bash
# pending
```

#### Diagnostics

```bash
.venv/bin/modal app list | head -20
```

#### Eval

```bash
.venv/bin/modal run --detach modal_app.py::probe \
  --policy ogb_state_nce/puzzle_4x4_play/action_state_hard_nce_s3072_e3/lewm_masked_action_state_hard_nce_epoch_1 \
  --dataset ogbench/visual_puzzle_4x4_play \
  --n 1000 \
  --overrides "--state-key button_states"

# pending epoch-3 button-state probe
```

#### Compare

```bash
# compare mean R2 against prior Action-NCE, SIGReg, and delta-state-NCE probes
```

### Artifacts

* Modal app: `ap-vUb3SBagjFOwwavTj38rQ2`
* Train metadata: `finetuning/results/ogb_state_nce/puzzle_4x4_play/action_state_nce_abs_s3072_e3/train_result.json`
* GPU metrics: `finetuning/results/ogb_state_nce/puzzle_4x4_play/action_state_nce_abs_s3072_e3/gpu_metrics.csv`
* Checkpoints: `ogb_state_nce/puzzle_4x4_play/action_state_nce_abs_s3072_e3/*_object.ckpt`
* W&B: `https://wandb.ai/jack-b/masked-transition-lewm/runs/ogb_state_nce_puzzle_4x4_play_action_state_nce_abs_s3072_e3`

### Result

Base:

Action-NCE and both delta-state-NCE variants fail the puzzle-4x4 button-state
probe. Replacement delta-state-NCE scored `PROBE_R2 MEAN = -0.0563`; additive
delta-state-NCE scored `-0.0854`; prior SIGReg scored `0.1969`.

Candidate:

Running. Initial validation before training:

```text
validate/emb_std = 0.00495
validate/effective_rank = 3.66
validate/state_nce_acc = 0.0052
validate/state_nce_loss = 5.9510
```

First backward passed with all tracked parameters receiving gradients.

Epoch-1 training/validation stayed healthy:

```text
fit/emb_std = 0.3107
fit/effective_rank = 12.13
fit/state_nce_acc = 0.8542
fit/state_nce_loss = 1.4136
validate/emb_std = 0.3412
validate/effective_rank = 6.01
validate/state_nce_acc = 0.1504
validate/state_nce_loss = 3.6399
```

Epoch-1 button-state probe:

```text
PROBE_R2 MEAN = 0.0304
```

Epoch-2 validation improved without collapse:

```text
fit/emb_std = 0.2705
fit/effective_rank = 16.58
fit/state_nce_acc = 0.9323
fit/state_nce_loss = 0.9675
validate/emb_std = 0.2959
validate/effective_rank = 8.36
validate/state_nce_acc = 0.3664
validate/state_nce_loss = 3.1002
```

Epoch-2 button-state probe:

```text
PROBE_R2 MEAN = -0.0394
```

Epoch-3 validation was healthy on the training objective:

```text
fit/emb_std = 0.2621
fit/effective_rank = 18.57
fit/state_nce_acc = 0.9688
fit/state_nce_loss = 0.7686
validate/emb_std = 0.2659
validate/effective_rank = 11.35
validate/state_nce_acc = 0.7651
validate/state_nce_loss = 1.6658
```

But the epoch-3 button-state probe failed:

```text
PROBE_R2 MEAN = -0.0722
```

Delta:

* Wins / losses: transient epoch-1 win over delta-state-NCE probes, but no
  durable button-state recovery; epoch 3 is below the prior Action-NCE probe and
  far below SIGReg (`0.1969`).
* Recall collapse? no. The loss objective and latent scale are healthy, but the
  representation still discards the persistent latch variables.

### Decision

Reject easy in-batch absolute state-NCE as a fix. Next step: keep the same
loss family but focus the contrast on hard negatives, because easy in-batch
absolute negatives can be solved by non-button visual/pose variables.

### Notes

* Still loss-only: no new module/head and no privileged button labels.
* This run changes only one axis from the rejected additive sentinel:
  `loss.masked.state_nce.mode=absolute`.

## 2026-06-30 — ogb_state_nce_puzzle_4x4_play_action_state_nce_abs_s3072_epoch1_button_probe

### Intent

Run an early button-state probe on the epoch-1 additive absolute-state-NCE
checkpoint. This checks whether absolute state identity starts recovering
persistent puzzle latch variables earlier than the rejected delta-state targets.

### Commands

#### Train

```bash
# continued from ogb_state_nce_puzzle_4x4_play_action_state_nce_abs_s3072_e3
```

#### Inference

```bash
# not applicable
```

#### Diagnostics

```bash
.venv/bin/modal run --detach modal_app.py::probe \
  --policy ogb_state_nce/puzzle_4x4_play/action_state_nce_abs_s3072_e3/lewm_masked_action_state_nce_epoch_1 \
  --dataset ogbench/visual_puzzle_4x4_play \
  --n 1000 \
  --overrides "--state-key button_states"
```

#### Eval

```bash
# not applicable
```

#### Compare

```bash
# compare mean R2 against prior Action-NCE/SIGReg and delta-state-NCE probes
```

### Artifacts

* Train Modal app: `ap-vUb3SBagjFOwwavTj38rQ2`
* Probe Modal app: `ap-I3QfFmBIO6yWxeX4zIqKMy`
* Policy object: `ogb_state_nce/puzzle_4x4_play/action_state_nce_abs_s3072_e3/lewm_masked_action_state_nce_epoch_1_object.ckpt`

### Result

Base:

Epoch-1 additive delta-state-NCE scored `PROBE_R2 MEAN = -0.0059`;
epoch-1 replacement delta-state-NCE scored `-0.0110`; prior SIGReg scored
`0.1969`.

Candidate:

Epoch-1 additive absolute-state-NCE button-state probe:

```text
PROBE_R2 MEAN = 0.0304
```

Delta:

* Wins / losses: directional win over both epoch-1 delta probes and over the
  rejected epoch-3 delta probes; still far below SIGReg.
* Recall collapse? no; epoch-1 validation `emb_std=0.3412`.

### Decision

Continue to epoch 3 for the real gate. This result supports the hypothesis that
persistent state identity, not transition delta, is the right loss target, but it
is not yet strong enough to scale to full trajectory eval.

### Notes

* Active dimensions are uneven: some button bits are positive (`state[1]=0.1644`,
  `state[12]=0.1301`), while others remain negative. Need epoch-3 probe before
  deciding whether easy in-batch absolute negatives are enough.

## 2026-06-30 — ogb_state_nce_puzzle_4x4_play_action_state_nce_abs_s3072_epoch2_button_probe

### Intent

Check whether the epoch-1 positive button-state signal from additive
absolute-state-NCE persists at epoch 2 or was only an early-training blip.

### Commands

#### Train

```bash
# continued from ogb_state_nce_puzzle_4x4_play_action_state_nce_abs_s3072_e3
```

#### Inference

```bash
# not applicable
```

#### Diagnostics

```bash
.venv/bin/modal run --detach modal_app.py::probe \
  --policy ogb_state_nce/puzzle_4x4_play/action_state_nce_abs_s3072_e3/lewm_masked_action_state_nce_epoch_2 \
  --dataset ogbench/visual_puzzle_4x4_play \
  --n 1000 \
  --overrides "--state-key button_states"
```

#### Eval

```bash
# not applicable
```

#### Compare

```bash
# compare mean R2 against epoch-1 absolute probe and prior delta probes
```

### Artifacts

* Train Modal app: `ap-vUb3SBagjFOwwavTj38rQ2`
* Probe Modal app: `ap-ES2n2QGW69oY5M1X9pR29F`
* Policy object: `ogb_state_nce/puzzle_4x4_play/action_state_nce_abs_s3072_e3/lewm_masked_action_state_nce_epoch_2_object.ckpt`

### Result

Base:

Epoch-1 additive absolute-state-NCE scored `PROBE_R2 MEAN = 0.0304`.

Candidate:

Epoch-2 additive absolute-state-NCE button-state probe:

```text
PROBE_R2 MEAN = -0.0394
```

Delta:

* Wins / losses: lost the small epoch-1 gain.
* Recall collapse? no; epoch-2 validation `emb_std=0.2959`.

### Decision

Continue to the epoch-3 gate, but treat the epoch-1 positive result as
transient unless epoch 3 recovers.

### Notes

* State-NCE validation improved at epoch 2, so the button regression failure is
  not explained by objective underfitting.

## 2026-06-30 — ogb_state_nce_puzzle_4x4_play_action_state_nce_abs_s3072_epoch3_button_probe

### Intent

Run the decision-gate button-state probe on the epoch-3 additive
absolute-state-NCE checkpoint.

### Commands

#### Train

```bash
# completed by ogb_state_nce_puzzle_4x4_play_action_state_nce_abs_s3072_e3
```

#### Inference

```bash
# not applicable
```

#### Diagnostics

```bash
.venv/bin/modal run --detach modal_app.py::probe \
  --policy ogb_state_nce/puzzle_4x4_play/action_state_nce_abs_s3072_e3/lewm_masked_action_state_nce_epoch_3 \
  --dataset ogbench/visual_puzzle_4x4_play \
  --n 1000 \
  --overrides "--state-key button_states"
```

#### Eval

```bash
# not applicable; no full trajectory eval because button-state decodability failed
```

#### Compare

```bash
# compare mean R2 against prior Action-NCE/SIGReg and state-NCE probes
```

### Artifacts

* Train Modal app: `ap-vUb3SBagjFOwwavTj38rQ2`
* Probe Modal app: `ap-sv9mA3QXYUM2bwCVhSp8Yn`
* Policy object: `ogb_state_nce/puzzle_4x4_play/action_state_nce_abs_s3072_e3/lewm_masked_action_state_nce_epoch_3_object.ckpt`
* W&B: `https://wandb.ai/jack-b/masked-transition-lewm/runs/ogb_state_nce_puzzle_4x4_play_action_state_nce_abs_s3072_e3`

### Result

Base:

Prior SIGReg button-state probe scored `0.1969`; Action-NCE scored `-0.0620`;
additive delta-state-NCE scored `-0.0854`; replacement delta-state-NCE scored
`-0.0563`.

Candidate:

Epoch-3 additive absolute-state-NCE button-state probe:

```text
PROBE_R2 MEAN = -0.0722
```

Delta:

* Wins / losses: loss vs SIGReg and no durable improvement over Action-NCE.
* Recall collapse? no; epoch-3 validation `emb_std=0.2659` and
  `state_nce_acc=0.7651`.

### Decision

Reject easy in-batch absolute-state-NCE. Move to hard-negative state identity:
same loss family and no new module/labels, but train against the highest-scoring
wrong target states so easy pose/background negatives cannot dominate the
objective.

### Notes

* The objective learned and generalized, but on variables other than the puzzle
  latches. This is exactly the case for hard negatives rather than more epochs.

## 2026-06-30 — ogb_state_nce_hard_negative_implementation

### Intent

Implement the next principled rung after easy absolute state-NCE failed: keep
the same state-NCE loss family, but restrict each contrastive row to the positive
target plus the top-k highest-scoring wrong targets. This preserves the no-new
module/no-label constraint while focusing gradient on states the representation
currently confuses.

### Commands

#### Train

```bash
# not applicable
```

#### Inference

```bash
# not applicable
```

#### Diagnostics

```bash
.venv/bin/python -m py_compile train.py modal_app.py eval.py
PYTHONPATH=. .venv/bin/pytest -q tests/test_masked_transition.py
```

#### Eval

```bash
# not applicable
```

#### Compare

```bash
# compare future hard-negative sentinel against rejected easy absolute-state-NCE
```

### Artifacts

* Code: `train.py`
* Tests: `tests/test_masked_transition.py`
* Config: `config/train/lewm_masked_action_state_hard_nce.yaml`

### Result

Base:

Easy absolute-state-NCE trained well but failed button decodability at epoch 3
(`PROBE_R2 MEAN = -0.0722`), indicating that easy negatives let the objective be
solved using non-button variables.

Candidate:

Added `hard_negatives` to `state_transition_discrimination_loss`. When enabled,
the loss uses each diagonal positive and the top-k wrong targets by current
similarity. Added `lewm_masked_action_state_hard_nce` with
`mode=absolute`, `hard_negatives=64`, and the same additive weight `0.03`.

Validation:

```text
25 passed in 8.47s
```

Delta:

* Wins / losses: implementation only; training pending.
* Recall collapse? not applicable.

### Decision

Launch the same seed-3072, puzzle-4x4, 3-epoch sentinel. If the epoch-3
button-state probe improves, scale to e10 and then trajectory eval; if not,
the next rung is the existing multi-step inverse path or a more explicit
hard-negative sampler.

### Notes

* The default remains full in-batch NCE (`hard_negatives=0`), so existing configs
  are unchanged.

## 2026-06-30 — ogb_state_nce_puzzle_4x4_play_action_state_hard_nce_s3072_e3

### Intent

Run the hard-negative absolute state-NCE sentinel on the same puzzle-4x4 seed-3072
3-epoch gate. This tests whether focusing state contrast on the currently most
confusable target states can preserve puzzle button variables where easy
in-batch absolute state-NCE failed.

### Commands

#### Train

```bash
.venv/bin/modal run --detach modal_app.py::train \
  --config-name lewm_masked_action_state_hard_nce \
  --data ogb_generic \
  --subdir ogb_state_nce/puzzle_4x4_play/action_state_hard_nce_s3072_e3 \
  --ogb-task puzzle-4x4-play-v0 \
  --overrides "seed=3072 runtime.stop_after_epoch=3 early_stopping.enabled=false"
```

#### Inference

```bash
# pending
```

#### Diagnostics

```bash
.venv/bin/modal app list | head -20
```

#### Eval

```bash
.venv/bin/modal run --detach modal_app.py::probe \
  --policy ogb_state_nce/puzzle_4x4_play/action_state_hard_nce_s3072_e3/lewm_masked_action_state_hard_nce_epoch_1 \
  --dataset ogbench/visual_puzzle_4x4_play \
  --n 1000 \
  --overrides "--state-key button_states"

.venv/bin/modal run --detach modal_app.py::probe \
  --policy ogb_state_nce/puzzle_4x4_play/action_state_hard_nce_s3072_e3/lewm_masked_action_state_hard_nce_epoch_3 \
  --dataset ogbench/visual_puzzle_4x4_play \
  --n 1000 \
  --overrides "--state-key button_states"
```

#### Compare

```bash
# compare mean R2 against easy absolute-state-NCE and SIGReg button probes
```

### Artifacts

* Modal app: `ap-qcDEiPUUKehwCZDn2mzQrU`
* Train metadata: `finetuning/results/ogb_state_nce/puzzle_4x4_play/action_state_hard_nce_s3072_e3/train_result.json`
* GPU metrics: `finetuning/results/ogb_state_nce/puzzle_4x4_play/action_state_hard_nce_s3072_e3/gpu_metrics.csv`
* Checkpoints: `ogb_state_nce/puzzle_4x4_play/action_state_hard_nce_s3072_e3/*_object.ckpt`
* Epoch-1 probe app: `ap-3qcUEIEydUHGqrZ6LUU1oc`
* Epoch-3 probe app: `ap-QjpslS4ZFfEGPepTv9FACb`
* W&B: `https://wandb.ai/jack-b/masked-transition-lewm/runs/ogb_state_nce_puzzle_4x4_play_action_state_hard_nce_s3072_e3`

### Result

Base:

Easy absolute-state-NCE trained to `validate/state_nce_acc=0.7651` but scored
`PROBE_R2 MEAN = -0.0722` at epoch 3.

Candidate:

Initial validation before training:

```text
validate/emb_std = 0.00495
validate/effective_rank = 3.66
validate/state_nce_acc = 0.0052
validate/state_nce_loss = 4.1992
```

First backward passed with all tracked parameters receiving gradients.

Epoch 1 validation:

```text
fit/emb_std = 0.3128
fit/effective_rank = 11.9512
fit/state_nce_acc = 0.8724
fit/state_nce_loss = 1.4069
validate/emb_std = 0.3354
validate/effective_rank = 7.7583
validate/state_nce_acc = 0.4630
validate/state_nce_loss = 2.4285
validate/loss = 1.0760
```

Epoch-1 button-state probe:

```text
PROBE_R2 MEAN = -0.0039
```

Epoch 2 validation:

```text
fit/emb_std = 0.2682
fit/effective_rank = 16.2348
fit/state_nce_acc = 0.9401
fit/state_nce_loss = 0.9531
validate/emb_std = 0.3108
validate/effective_rank = 7.4121
validate/state_nce_acc = 0.3480
validate/state_nce_loss = 3.1366
validate/loss = 1.5731
```

Epoch 3 validation:

```text
fit/emb_std = 0.2619
fit/effective_rank = 18.3691
fit/state_nce_acc = 0.9609
fit/state_nce_loss = 0.7571
validate/emb_std = 0.2683
validate/effective_rank = 12.1603
validate/state_nce_acc = 0.7216
validate/state_nce_loss = 1.7947
validate/loss = 0.7690
```

Epoch-3 button-state probe:

```text
PROBE_R2 MEAN = -0.0681
```

Delta:

* Wins / losses: loss vs SIGReg and no improvement over prior rejected
  state-NCE variants. Epoch-3 R2 is worse than the epoch-1 hard-negative probe
  and roughly in the same failed range as Action-NCE/easy absolute.
* Recall collapse? no; this is an objective-alignment failure, not latent
  collapse.

### Decision

Reject additive hard-negative state-NCE. Do not launch trajectory evals. The
state contrast can become numerically healthy while still missing button latch
variables, so the next principled rung should force long-horizon controllable
state into the representation, e.g. endpoint Action-NCE rather than another
same-step state-identity contrast.

### Notes

* The initial state-NCE loss is on the expected `log(65)` scale for one positive
  plus `hard_negatives=64`.
* The failed `nohup` probe wrapper produced a zero-byte log and no Modal app; the
  foreground `modal run --detach` launch created app `ap-3qcUEIEydUHGqrZ6LUU1oc`
  and completed normally.
* Epoch 3 proves that good validation state-NCE accuracy is not sufficient for
  puzzle button decodability on this task.

## 2026-06-30 — ogb_endpoint_action_nce_implementation

### Intent

Implement the next rung after same-step state identity contrast failed:
endpoint Action-NCE. This keeps the planner-facing one-step AR forward loss and
the existing `HorizonInverseDynamics` module, but makes the auxiliary inverse
target contrastive over `(z_t, z_{t+k}, k) -> a_t` endpoint pairs. The hypothesis
is that longer endpoint discrimination should make persistent controllable
state useful where same-step state identity contrast selected easier visual
factors.

### Commands

#### Train

```bash
# not applicable
```

#### Inference

```bash
# not applicable
```

#### Diagnostics

```bash
.venv/bin/python -m py_compile train.py modal_app.py eval.py
PYTHONPATH=. .venv/bin/pytest -q tests/test_masked_transition.py
```

#### Eval

```bash
# not applicable
```

#### Compare

```bash
# compare future sentinel against rejected state-NCE variants and SIGReg button probes
```

### Artifacts

* Code: `train.py`
* Tests: `tests/test_masked_transition.py`
* Config: `config/train/lewm_masked_endpoint_action_nce.yaml`

### Result

Base:

Additive hard-negative state-NCE reached healthy validation contrast metrics but
failed button-state decodability at epoch 3 (`PROBE_R2 MEAN = -0.0681`).

Candidate:

Added `inverse_loss_type: action_nce` support to endpoint inverse losses and a
new config, `lewm_masked_endpoint_action_nce`, with `inverse_weight=0.30`,
`inverse_min_horizon=1`, and `inverse_max_horizon=${wm.horizon}`. A first launch
(`ap-c5iKZB27TjDC60Ns41Vivm`) was stopped at step 486 because the config still
loaded `wm.history_size + wm.num_preds = 4` frames, limiting endpoint gaps to
1-3. The config was corrected to set `data.dataset.num_steps =
wm.history_size + wm.horizon` while keeping `wm.num_preds=1`.

Validation:

```text
27 passed in 8.29s
```

Delta:

* Wins / losses: implementation only; training pending.
* Recall collapse? not applicable.

### Decision

Launch the same seed-3072, puzzle-4x4, 3-epoch button-probe gate. Do not scale
to other tasks unless the button probe improves.

### Notes

* This avoids adding a new module or privileged labels; it reuses the existing
  horizon-conditioned inverse head and changes the endpoint auxiliary loss from
  MSE to contrastive action discrimination.

## 2026-06-30 — ogb_endpoint_action_nce_puzzle_4x4_play_h5_s3072_e3

### Intent

Run the corrected endpoint Action-NCE sentinel on puzzle-4x4, seed 3072. This
uses 8-frame windows (`history_size=3`, `horizon=5`) while preserving the
one-step AR forward target (`num_preds=1`), so endpoint inverse losses cover
gaps 1-5. Compare button-state probe results against the rejected state-NCE
variants and SIGReg.

### Commands

#### Train

```bash
.venv/bin/modal run --detach modal_app.py::train \
  --config-name lewm_masked_endpoint_action_nce \
  --data ogb_generic \
  --subdir ogb_endpoint_action_nce/puzzle_4x4_play/endpoint_action_nce_h5_s3072_e3 \
  --ogb-task puzzle-4x4-play-v0 \
  --overrides "seed=3072 runtime.stop_after_epoch=3 early_stopping.enabled=false"
```

#### Inference

```bash
# pending
```

#### Diagnostics

```bash
.venv/bin/modal app list | head -16
```

#### Eval

```bash
.venv/bin/modal run --detach modal_app.py::probe \
  --policy ogb_endpoint_action_nce/puzzle_4x4_play/endpoint_action_nce_h5_s3072_e3/lewm_masked_endpoint_action_nce_epoch_1 \
  --dataset ogbench/visual_puzzle_4x4_play \
  --n 1000 \
  --overrides "--state-key button_states"

.venv/bin/modal run --detach modal_app.py::probe \
  --policy ogb_endpoint_action_nce/puzzle_4x4_play/endpoint_action_nce_h5_s3072_e3/lewm_masked_endpoint_action_nce_epoch_3 \
  --dataset ogbench/visual_puzzle_4x4_play \
  --n 1000 \
  --overrides "--state-key button_states"
```

#### Compare

```bash
# compare mean R2 against SIGReg, Action-NCE, and rejected state-NCE probes
```

### Artifacts

* Modal app: `ap-n4GhUjHZitzjIEvyysgdR7`
* Train metadata: `finetuning/results/ogb_endpoint_action_nce/puzzle_4x4_play/endpoint_action_nce_h5_s3072_e3/train_result.json`
* GPU metrics: `finetuning/results/ogb_endpoint_action_nce/puzzle_4x4_play/endpoint_action_nce_h5_s3072_e3/gpu_metrics.csv`
* Checkpoints: `ogb_endpoint_action_nce/puzzle_4x4_play/endpoint_action_nce_h5_s3072_e3/*_object.ckpt`
* Epoch-1 probe app: `ap-TrN27YokYxEbYX2eCAX0IJ`
* Epoch-3 probe app: `ap-CBan5HaaLkneHd4IyspUc8`
* W&B: `https://wandb.ai/jack-b/masked-transition-lewm/runs/ogb_endpoint_action_nce_puzzle_4x4_play_endpoint_action_nce_h5_s3072_e3`

### Result

Base:

Hard-negative state-NCE failed at epoch 3 (`PROBE_R2 MEAN = -0.0681`) despite
healthy validation contrast metrics.

Candidate:

Initial validation confirms endpoint gaps h1-h5 are active:

```text
validate/emb_std = 0.0047
validate/effective_rank = 3.9934
validate/inverse_acc = 0.0017
validate/inverse_baseline = 0.0017
validate/inverse_h1_loss = 10.9032
validate/inverse_h2_loss = 10.6105
validate/inverse_h3_loss = 10.2707
validate/inverse_h4_loss = 9.9722
validate/inverse_h5_loss = 9.8410
validate/inverse_loss = 10.3195
validate/loss = 3.1777
```

Epoch 1 validation:

```text
fit/emb_std = 0.3541
fit/effective_rank = 6.7734
fit/inverse_acc = 0.2207
fit/inverse_loss = 3.6231
validate/emb_std = 0.3790
validate/effective_rank = 5.9187
validate/inverse_acc = 0.0268
validate/inverse_loss = 9.7968
validate/loss = 3.1963
```

Epoch-1 button-state probe:

```text
PROBE_R2 MEAN = -0.0021
```

Epoch 3 validation:

```text
fit/emb_std = 0.3559
fit/effective_rank = 7.9985
fit/inverse_acc = 0.3578
fit/inverse_loss = 2.4442
validate/emb_std = 0.3753
validate/effective_rank = 7.1220
validate/inverse_acc = 0.1325
validate/inverse_loss = 5.5684
validate/loss = 1.7171
```

Epoch-3 button-state probe:

```text
PROBE_R2 MEAN = -0.0988
```

Delta:

* Wins / losses: clear loss vs SIGReg (`0.1969`) and no improvement over
  Action-NCE/state-NCE failures. Epoch 3 is worse than epoch 1 despite better
  endpoint-action validation metrics.
* Recall collapse? no; representation scale and endpoint validation were
  healthy. This is another objective-alignment failure.

### Decision

Reject endpoint Action-NCE for puzzle. Do not launch trajectory evals. The
auxiliary can learn h1-h5 endpoint action discrimination without making the
button latch variables linearly decodable, so another generic contrastive
auxiliary is unlikely to close the puzzle gap by itself.

### Notes

* A previous launch in `endpoint_action_nce_s3072_e3` was stopped at step 486
  because it only loaded 4-frame windows and therefore did not test h5 endpoints.
* The decisive negative result is epoch 3: validation endpoint-action accuracy
  improved to `0.1325` while button-state R2 fell to `-0.0988`.

## 2026-07-02 - ogb-scene-highres-readme-paper-image

### Intent

Generate a higher-resolution OGBench Scene still for the README and paper from
the original `visual-scene-play-v0` simulator state, rather than upscaling the
existing 160x160 tracked thumbnail. Compare against the current representative
Scene still generated by `make_task_media.py`.

### Commands

#### Train

```bash
# not applicable
```

#### Inference

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
import numpy as np
from PIL import Image
import gymnasium as gym
import ogbench.manipspace

idx = 83
npz = Path.home() / ".ogbench" / "data" / "visual-scene-play-v0.npz"
with np.load(npz) as data:
    qpos = data["qpos"][idx].astype(np.float64)
    qvel = data["qvel"][idx].astype(np.float64)
    buttons = data["button_states"][idx].astype(np.int64)

env = gym.make(
    "scene-v0",
    ob_type="pixels",
    width=1024,
    height=1024,
    visualize_info=False,
    terminate_at_goal=True,
)
try:
    env.reset(seed=0)
    env.unwrapped.set_state(qpos, qvel, buttons)
    frame = env.unwrapped.render(camera="front_pixels")
    Image.fromarray(frame).save("/tmp/ogb_scene_highres/render_idx83_1024.png")
finally:
    env.close()
PY
cp /tmp/ogb_scene_highres/render_idx83_1024.png assets/datasets/scene_still.png
cp /tmp/ogb_scene_highres/render_idx83_1024.png paper/latex/images/scene_still.png
```

#### Diagnostics

```bash
sips -g pixelWidth -g pixelHeight assets/datasets/scene_still.png paper/latex/images/scene_still.png
.venv/bin/python -m py_compile make_task_media.py
```

#### Eval

```bash
cd paper/latex && latexmk -g -pdf -synctex=1 -interaction=nonstopmode -halt-on-error iclr_main.tex
```

#### Compare

```bash
git diff --stat -- assets/datasets/scene_still.png paper/latex/images/scene_still.png make_task_media.py
```

### Artifacts

* README Scene still: `assets/datasets/scene_still.png`
* Paper Scene still: `paper/latex/images/scene_still.png`
* Source render: `/tmp/ogb_scene_highres/render_idx83_1024.png`
* Regeneration path: `make_task_media.py`

### Result

Base:

Tracked Scene stills were 160x160 PNGs generated from 64x64
`visual-scene-play-v0` observations.

Candidate:

Rendered the same representative first-episode capped-frame index 83 from
dataset `qpos`, `qvel`, and `button_states` through MuJoCo at 1024x1024 using
the `front_pixels` camera.

Delta:

* Wins / losses: higher-resolution, paper-facing PNGs with the same Scene
  composition; file size remains small enough for the repository.
* Recall collapse? no

### Decision

Keep the 1024x1024 Scene render for README and paper. Keep the Scene GIF compact
at 160x160, and regenerate only the still at high resolution in
`make_task_media.py`.

### Notes

* The raw OGBench visual observations are only 64x64, so paper-quality output
  requires simulator rerendering from state rather than larger raw frames.
* A first local inspection attempt loaded the compressed observation member and
  was interrupted after confirming the render path; subsequent generation uses
  only `qpos`, `qvel`, and `button_states`.

## 2026-07-02 — jack-paper-polish-sigreg-probes-reruns

### Intent

Address Jack's paper comments, remove the remaining single-seed caveat for the
standard-task SIGReg baselines, add AC-MTM PushT probe evidence, and update the
ICLR manuscript so the controlled claims are three-seed, polished, and
submission-ready.

### Commands

#### Train

```bash
# No new training in this pass.
```

#### Inference

```bash
# PushT AC-MTM probe diagnostics were run through modal_app.py::probe, but the
# exact local wrapper commands were not recoverable after context compaction.
# App/results:
# ap-op40IN1lnzaUQM4alP0MoB -> pusht/mtm_action_nce_w030_e10_s3072, mean R2 0.6716
# ap-UawoqxEonSKdsd5ofWFzPj -> pusht/mtm_action_nce_w030_e10_s1, mean R2 0.6752
# ap-1rq9OwUpsvdTlCfXi2o2rG -> pusht/mtm_action_nce_w030_e10_s2, mean R2 0.6789
```

#### Diagnostics

```bash
mkdir -p /tmp/jackpaper-results
.venv/bin/modal volume get multi-future-lewm-cache tworoom/sigreg_tworoom_e10_s3072_n200_jackpaper_r3.txt /tmp/jackpaper-results/ --force
.venv/bin/modal volume get multi-future-lewm-cache tworoom/lewm_base_e10_s1/sigreg_tworoom_e10_s1_n200_jackpaper_r4.txt /tmp/jackpaper-results/ --force
.venv/bin/modal volume get multi-future-lewm-cache tworoom/lewm_base_e10_s2/sigreg_tworoom_e10_s2_n200_jackpaper_r4.txt /tmp/jackpaper-results/ --force
.venv/bin/modal volume get multi-future-lewm-cache pusht/sigreg_pusht_e10_s1_n200_jackpaper_r3.txt /tmp/jackpaper-results/ --force
.venv/bin/modal volume get multi-future-lewm-cache pusht/sigreg_pusht_e10_s2_n200_jackpaper_r3.txt /tmp/jackpaper-results/ --force
.venv/bin/modal volume get multi-future-lewm-cache cube/sigreg_cube_e10_s1_n200_jackpaper_r3.txt /tmp/jackpaper-results/ --force
.venv/bin/modal volume get multi-future-lewm-cache cube/sigreg_cube_e10_s2_n200_jackpaper_r3.txt /tmp/jackpaper-results/ --force
rg -n "success_rate" /tmp/jackpaper-results
```

#### Eval

```bash
.venv/bin/modal run --detach modal_app.py::evaluate --config-name tworoom --policy tworoom/lewm_base_e10_s0 --overrides 'eval.num_eval=200 eval.env_batch_size=10 output.save_video=false output.filename=sigreg_tworoom_e10_s3072_n200_jackpaper_r3.txt'
.venv/bin/modal run --detach modal_app.py::evaluate --config-name tworoom --policy tworoom/lewm_base_e10_s1/lewm_epoch_10 --overrides 'eval.num_eval=200 eval.env_batch_size=10 output.save_video=false output.filename=sigreg_tworoom_e10_s1_n200_jackpaper_r4.txt'
.venv/bin/modal run --detach modal_app.py::evaluate --config-name tworoom --policy tworoom/lewm_base_e10_s2/lewm_epoch_10 --overrides 'eval.num_eval=200 eval.env_batch_size=10 output.save_video=false output.filename=sigreg_tworoom_e10_s2_n200_jackpaper_r4.txt'
.venv/bin/modal run --detach modal_app.py::evaluate --config-name pusht --policy pusht/lewm_base_s1 --overrides 'eval.num_eval=200 eval.env_batch_size=10 output.save_video=false output.filename=sigreg_pusht_e10_s1_n200_jackpaper_r3.txt'
.venv/bin/modal run --detach modal_app.py::evaluate --config-name pusht --policy pusht/lewm_base_s2 --overrides 'eval.num_eval=200 eval.env_batch_size=10 output.save_video=false output.filename=sigreg_pusht_e10_s2_n200_jackpaper_r3.txt'
.venv/bin/modal run --detach modal_app.py::evaluate --config-name cube --policy cube/lewm_base_s1 --overrides 'eval.num_eval=200 eval.env_batch_size=10 output.save_video=false output.filename=sigreg_cube_e10_s1_n200_jackpaper_r3.txt'
.venv/bin/modal run --detach modal_app.py::evaluate --config-name cube --policy cube/lewm_base_s2 --overrides 'eval.num_eval=200 eval.env_batch_size=10 output.save_video=false output.filename=sigreg_cube_e10_s2_n200_jackpaper_r3.txt'
```

#### Compare

```bash
python3 - <<'PY'
rows = {
    'TwoRoom': [85.5, 86.0, 85.0],
    'PushT': [93.5, 93.0, 93.0],
    'Cube': [66.0, 66.5, 66.0],
}
for k, vals in rows.items():
    mean=sum(vals)/len(vals)
    pop=(sum((x-mean)**2 for x in vals)/len(vals))**0.5
    print(k, vals, f"mean={mean:.4f}", f"pop_std={pop:.4f}", f"paper={mean:.1f}+/-{pop:.1f}")
PY
```

### Artifacts

* TwoRoom seed 3072 eval: `tworoom/sigreg_tworoom_e10_s3072_n200_jackpaper_r3.txt`, app `ap-7NeURdrgy6QPgcZ0IxQkTD`
* TwoRoom seed 1 eval: `tworoom/lewm_base_e10_s1/sigreg_tworoom_e10_s1_n200_jackpaper_r4.txt`, app `ap-cy7fEpty1rVJDx9nde8JTf`
* TwoRoom seed 2 eval: `tworoom/lewm_base_e10_s2/sigreg_tworoom_e10_s2_n200_jackpaper_r4.txt`, app `ap-RTA2gqRSD8DNTAQetoU1K7`
* PushT seed 1 eval: `pusht/sigreg_pusht_e10_s1_n200_jackpaper_r3.txt`, app `ap-41hoTV7FgLSCaeqc9LeBx5`
* PushT seed 2 eval: `pusht/sigreg_pusht_e10_s2_n200_jackpaper_r3.txt`, app `ap-xhIEQFw10pcOAC9vdXOZcz`
* Cube seed 1 eval: `cube/sigreg_cube_e10_s1_n200_jackpaper_r3.txt`, app `ap-gscZBvdE8fVhkE8vkC2DIb`
* Cube seed 2 eval: `cube/sigreg_cube_e10_s2_n200_jackpaper_r3.txt`, app `ap-996NnMVPDXgWtMzvBsWWxy`
* Paper source: `paper/latex/iclr_main.tex`
* Figure scripts: `paper/latex/images/make_results_lewm_style.py`, `paper/latex/images/make_results_ablation_threeway.py`

### Result

Base:

The paper previously used single available matched SIGReg seeds for TwoRoom,
PushT, and Cube in the controlled standard-task tables.

Candidate:

Completed matched SIGReg 200-episode evals for training seeds `{3072,1,2}` on
the standard tasks:

| Task | SIGReg seeds | SIGReg mean +/- std | AC-MTM mean +/- std | Delta |
|---|---:|---:|---:|---:|
| TwoRoom | 85.5 / 86.0 / 85.0 | 85.5 +/- 0.4 | 90.7 +/- 0.6 | +5.2 |
| Reacher | 69.0 / 69.0 / 68.5 | 68.8 +/- 0.2 | 68.3 +/- 3.1 | -0.5 |
| PushT | 93.5 / 93.0 / 93.0 | 93.2 +/- 0.2 | 86.7 +/- 1.5 | -6.5 |
| OGB-Cube | 66.0 / 66.5 / 66.0 | 66.2 +/- 0.2 | 78.8 +/- 1.7 | +12.6 |

PushT AC-MTM probes over three seeds:

| Policy | state[4] orientation R2 | Mean R2 |
|---|---:|---:|
| `pusht/mtm_action_nce_w030_e10_s3072` | 0.5091 | 0.6716 |
| `pusht/mtm_action_nce_w030_e10_s1` | 0.5345 | 0.6752 |
| `pusht/mtm_action_nce_w030_e10_s2` | 0.4968 | 0.6789 |
| Mean | 0.514 | 0.675 |

Delta:

* Wins / losses: AC-MTM beats three-seed SIGReg on TwoRoom and Cube, matches on
  Reacher within noise, and trails on PushT.
* Recall collapse? no; AC-MTM remains non-collapsed in the probed PushT seeds.

### Decision

Keep the paper updates. The standard-task controlled comparison is now three-seed
for both SIGReg and AC-MTM, and the PushT mechanism section can cite AC-MTM
probe values directly rather than extrapolating from MTM-MSE.

### Notes

* The canonical AGENTS launch pattern uses `nohup .venv/bin/modal run --detach
  ... > /tmp/<job>.log 2>&1 &`. In this tool environment, backgrounded wrappers
  were reaped before producing usable logs or live apps. Valid reruns therefore
  used foreground `.venv/bin/modal run --detach ...` sessions kept open until
  completion.
* The first TwoRoom seed-1/2 shorthand relaunches with policies
  `tworoom/lewm_base_s1` and `tworoom/lewm_base_s2` failed because those paths do
  not exist. Corrected runs used exact checkpoint stems
  `tworoom/lewm_base_e10_s1/lewm_epoch_10` and
  `tworoom/lewm_base_e10_s2/lewm_epoch_10`.
* Valid jobs were verified by real progress logs plus Modal app state with
  `Tasks=1`; completion was verified from printed metrics and downloaded Modal
  volume result files.
* MuJoCo/EGL cleanup warnings appeared after Cube metrics were printed, but the
  apps completed and wrote result files.

## 2026-07-02 — jack-seed-caveat-and-surprise-diagnostics

### Intent

Remove the remaining paper caveats tied to one-seed rows and add measured
physical-understanding diagnostics. Specifically: fill the TwoRoom-long SIGReg
row for training seeds `{3072,1,2}`, fill the TwoRoom NoReg sanity check for
three training seeds, and test whether the learned latent dynamics assign higher
surprise to action-counterfactual and state-discontinuous transitions than to
dataset transitions on PushT and OGBench-Cube.

### Commands

#### Train

```bash
.venv/bin/modal run --detach modal_app.py::train_then_evaluate --config-name lewm --data tworoom --subdir tworoom/lewm_noreg_e10_s1 --overrides 'trainer.max_epochs=10 early_stopping.enabled=false seed=1 loss.sigreg.weight=0.0 wandb.config.name=tworoom_lewm_noreg_e10_s1' --eval-config-name tworoom --eval-policy tworoom/lewm_noreg_e10_s1/lewm_epoch_10 --eval-overrides 'eval.num_eval=200 eval.env_batch_size=10 seed=42 output.save_video=false output.filename=noreg_tworoom_e10_s1_n200_jackpaper_r1.txt'
.venv/bin/modal run --detach modal_app.py::train_then_evaluate --config-name lewm --data tworoom --subdir tworoom/lewm_noreg_e10_s2 --overrides 'trainer.max_epochs=10 early_stopping.enabled=false seed=2 loss.sigreg.weight=0.0 wandb.config.name=tworoom_lewm_noreg_e10_s2' --eval-config-name tworoom --eval-policy tworoom/lewm_noreg_e10_s2/lewm_epoch_10 --eval-overrides 'eval.num_eval=200 eval.env_batch_size=10 seed=42 output.save_video=false output.filename=noreg_tworoom_e10_s2_n200_jackpaper_r1.txt'
```

#### Inference

```bash
# Latent surprise diagnostics; run for PushT and Cube, SIGReg and AC-MTM,
# seeds 3072/1/2. Each writes JSON under
# .stable_worldmodel/surprise_diagnostics/.
```

#### Diagnostics

```bash
.venv/bin/modal app list | head -30
.venv/bin/modal app logs <app-id> --tail 80 --timestamps
.venv/bin/modal volume get multi-future-lewm-cache <result-path> /tmp/jackpaper-surprise/ --force
```

#### Eval

```bash
.venv/bin/modal run --detach modal_app.py::evaluate --config-name tworoom_long --policy tworoom/lewm_base_e10_s0 --overrides 'eval.num_eval=200 eval.env_batch_size=10 seed=42 output.save_video=false output.filename=sigreg_tworoom_long_e10_s3072_n200_jackpaper_r1.txt'
.venv/bin/modal run --detach modal_app.py::evaluate --config-name tworoom_long --policy tworoom/lewm_base_e10_s1/lewm_epoch_10 --overrides 'eval.num_eval=200 eval.env_batch_size=10 seed=42 output.save_video=false output.filename=sigreg_tworoom_long_e10_s1_n200_jackpaper_r1.txt'
.venv/bin/modal run --detach modal_app.py::evaluate --config-name tworoom_long --policy tworoom/lewm_base_e10_s2/lewm_epoch_10 --overrides 'eval.num_eval=200 eval.env_batch_size=10 seed=42 output.save_video=false output.filename=sigreg_tworoom_long_e10_s2_n200_jackpaper_r1.txt'
.venv/bin/modal run --detach modal_app.py::evaluate --config-name tworoom --policy tworoom/lewm_noreg/lewm_epoch_10 --overrides 'eval.num_eval=200 eval.env_batch_size=10 seed=42 output.save_video=false output.filename=noreg_tworoom_e10_s3072_n200_jackpaper_r1.txt'
```

#### Compare

```bash
# Pending job completion. Compare success means/stdevs and summarize surprise
# ratios/fractions greater than normal.
```

### Artifacts

* Pending TwoRoom-long SIGReg eval outputs:
  `tworoom/sigreg_tworoom_long_e10_s3072_n200_jackpaper_r1.txt`,
  `tworoom/lewm_base_e10_s1/sigreg_tworoom_long_e10_s1_n200_jackpaper_r1.txt`,
  `tworoom/lewm_base_e10_s2/sigreg_tworoom_long_e10_s2_n200_jackpaper_r1.txt`.
* Pending NoReg outputs:
  `tworoom/noreg_tworoom_e10_s3072_n200_jackpaper_r1.txt`,
  `tworoom/lewm_noreg_e10_s1/noreg_tworoom_e10_s1_n200_jackpaper_r1.txt`,
  `tworoom/lewm_noreg_e10_s2/noreg_tworoom_e10_s2_n200_jackpaper_r1.txt`.
* Pending surprise JSONs under `surprise_diagnostics/`.

### Result

Base:

Pending.

Candidate:

Pending.

Delta:

* Wins / losses: pending.
* Recall collapse? pending for new NoReg seeds.

### Decision

Pending.

### Notes

* This entry intentionally avoids entrypoint-level `--no-wait`.
* Because backgrounded local wrappers are unreliable in this tool environment,
  launch commands may be kept as foreground `modal run --detach` sessions and
  verified with `modal app list`, app logs, and output artifacts.

### Update — corrected surprise diagnostics

The initial surprise diagnostic used a shifted random permutation that could
leave a small number of examples paired with themselves. I fixed the helper to
sample a true derangement for batches larger than one and reran all twelve
PushT/Cube × SIGReg/AC-MTM × train-seed cells with unique `r2` JSON outputs.

Additional diagnostic command pattern:

```bash
.venv/bin/modal run --detach --name jp-surprise-r2-<task>-<model>-<seed> modal_app.py::surprise_diagnostics \
  --policy <policy-stem> \
  --dataset <pusht_expert_train|ogbench/cube_single_expert> \
  --n 4096 \
  --overrides '--seed 42 --output /workspace/long-horizon-world-model/.stable_worldmodel/surprise_diagnostics/jackpaper_<task>_<model>_<seed>_n4096_r2.json'
.venv/bin/modal volume get multi-future-lewm-cache surprise_diagnostics /tmp/jackpaper-surprise-r2 --force
```

Corrected surprise result, reported as mean per-clip invalid/normal prediction
error ratio with population standard deviation over train seeds:

| Task | Model | Action counterfactual | State discontinuity | Min invalid > normal |
|---|---|---:|---:|---:|
| PushT | SIGReg | 151.7 +/- 4.7x | 1246.0 +/- 34.1x | 99.95% |
| PushT | AC-MTM | 9.6 +/- 0.3x | 40.0 +/- 1.4x | 99.98% |
| OGB-Cube | SIGReg | 274.2 +/- 12.6x | 835.4 +/- 35.2x | 100.00% |
| OGB-Cube | AC-MTM | 82.3 +/- 1.6x | 434.4 +/- 5.6x | 100.00% |

Decision for the physical-understanding section: keep the diagnostic, but state
the claim narrowly as internal latent surprise/counterfactual consistency, not
public violation-of-expectation or out-of-distribution physical reasoning.

### Update — TwoRoom-long SIGReg three-seed completion

The matched TwoRoom-long SIGReg evals completed under the same 200-episode,
seed-42, CEM 300/30/30 protocol as MTM-MSE and AC-MTM.

Pulled artifacts:

* `tworoom/sigreg_tworoom_long_e10_s3072_n200_jackpaper_r1.txt`
  (local copy: `/tmp/jackpaper-seed-results/sigreg_tworoom_long_e10_s3072_n200_jackpaper_r1.txt`)
* `tworoom/lewm_base_e10_s1/sigreg_tworoom_long_e10_s1_n200_jackpaper_r1.txt`
  (local copy: `/tmp/jackpaper-seed-results/sigreg_tworoom_long_e10_s1_n200_jackpaper_r1.txt`)
* `tworoom/lewm_base_e10_s2/sigreg_tworoom_long_e10_s2_n200_jackpaper_r1.txt`
  (local copy: `/tmp/jackpaper-seed-results/sigreg_tworoom_long_e10_s2_n200_jackpaper_r1.txt`)

Result:

| Task | Model | Seed 3072 | Seed 1 | Seed 2 | Mean +/- std |
|---|---|---:|---:|---:|---:|
| TwoRoom-long 100/150 | SIGReg | 16.5 | 17.0 | 17.5 | 17.0 +/- 0.4 |

Decision: replace the previous single-seed `18.0` placeholder and remove the
paper dagger/caveat. AC-MTM remains above SIGReg by 7.2 points on this stress
test, while MTM-MSE remains 11.0 points above SIGReg.

### Update — NoReg three-seed completion

The TwoRoom NoReg anti-collapse control is now matched to the paper's
200-episode, seed-42, CEM 300/30/30 protocol across training seeds `3072,1,2`.

Additional training/evaluation commands:

```bash
.venv/bin/modal run --detach --name jp-noreg-s1-train-eval modal_app.py::train_then_evaluate \
  --config-name lewm \
  --data tworoom \
  --subdir tworoom/lewm_noreg_e10_s1 \
  --overrides 'trainer.max_epochs=10 early_stopping.enabled=false seed=1 loss.sigreg.weight=0.0 wandb.config.name=tworoom_lewm_noreg_e10_s1' \
  --eval-config-name tworoom \
  --eval-policy tworoom/lewm_noreg_e10_s1/lewm_epoch_10 \
  --eval-overrides 'eval.num_eval=200 eval.env_batch_size=10 seed=42 output.save_video=false output.filename=noreg_tworoom_e10_s1_n200_jackpaper_r1.txt'
.venv/bin/modal run --detach --name jp-noreg-s2-train-eval modal_app.py::train_then_evaluate \
  --config-name lewm \
  --data tworoom \
  --subdir tworoom/lewm_noreg_e10_s2 \
  --overrides 'trainer.max_epochs=10 early_stopping.enabled=false seed=2 loss.sigreg.weight=0.0 wandb.config.name=tworoom_lewm_noreg_e10_s2' \
  --eval-config-name tworoom \
  --eval-policy tworoom/lewm_noreg_e10_s2/lewm_epoch_10 \
  --eval-overrides 'eval.num_eval=200 eval.env_batch_size=10 seed=42 output.save_video=false output.filename=noreg_tworoom_e10_s2_n200_jackpaper_r1.txt'
.venv/bin/modal volume get multi-future-lewm-cache tworoom/lewm_noreg_e10_s1/noreg_tworoom_e10_s1_n200_jackpaper_r1.txt /tmp/jackpaper-seed-results/ --force
.venv/bin/modal volume get multi-future-lewm-cache tworoom/lewm_noreg_e10_s2/noreg_tworoom_e10_s2_n200_jackpaper_r1.txt /tmp/jackpaper-seed-results/ --force
```

Completed app IDs:

* Seed 3072 eval rerun: `ap-Jx8wY6saHM0v1hAY97LOqa`
* Seed 1 train+eval: `ap-qmObmTdW1IjObbaZceO3MX`
* Seed 2 train+eval: `ap-6W2SKD5qMtlBFfqyqd17WY`

Pulled artifacts:

* `tworoom/noreg_tworoom_e10_s3072_n200_jackpaper_r1.txt`
  (local copy: `/tmp/jackpaper-seed-results/noreg_tworoom_e10_s3072_n200_jackpaper_r1.txt`)
* `tworoom/lewm_noreg_e10_s1/noreg_tworoom_e10_s1_n200_jackpaper_r1.txt`
  (local copy: `/tmp/jackpaper-seed-results/noreg_tworoom_e10_s1_n200_jackpaper_r1.txt`)
* `tworoom/lewm_noreg_e10_s2/noreg_tworoom_e10_s2_n200_jackpaper_r1.txt`
  (local copy: `/tmp/jackpaper-seed-results/noreg_tworoom_e10_s2_n200_jackpaper_r1.txt`)

Result:

| Task | Model | Seed 3072 | Seed 1 | Seed 2 | Mean +/- std |
|---|---|---:|---:|---:|---:|
| TwoRoom 25/50 | NoReg | 30.5 | 28.0 | 25.5 | 28.0 +/- 2.0 |

Decision: keep NoReg only as the anti-collapse sanity check. It confirms that
plain next-latent prediction collapses and plans far below SIGReg (`85.5+/-0.4`),
MTM-MSE (`90.2+/-0.5`), and AC-MTM (`90.7+/-0.6`) under the same n=200
three-training-seed protocol. The paper's NoReg one-seed caveat is removed.

## 2026-07-03 — scene-mtm-mse-threeseed-and-random-baseline

### Intent

Fill the two gaps a reviewer would flag in the OGBench Visual Scene comparison:
(1) the MTM-MSE (non-contrastive inverse-regression, `lewm_masked`) ablation has
never been trained on Scene, so we cannot say whether the 80-vs-58 AC-MTM win
over SIGReg needs the contrastive form; (2) no random-policy floor exists for
Scene under the trajectory-goal protocol, which the paper's LeWM-style bar
chart needs. Train `lewm_masked` on `data=scene` for seeds {3072,1,2} with the
exact protocol used for the SIGReg and AC-MTM Scene runs (10 epochs, early
stopping off, eval n=50, eval seed 42, CEM 300/30/30), plus one random-policy
eval. Compare against SIGReg 58.0±2.0 and AC-MTM 80.0±2.0.

NOTE: an initial launch mistakenly used `lewm_ms_mtm` (the Direct-H MS-MTM
variant) instead of `lewm_masked` (the paper's MTM-MSE). Those three apps
(ap-S8DW5bty0qsi8PUW8U9J62, ap-UfiEI2Naizfj4NOyB4Kfon,
ap-H5LVy7Le0ee5MJd7RkpxkC) were stopped within minutes and relaunched with the
correct config.

### Commands

#### Train

```bash
# for s in 3072 1 2 (launched via tmux, one detached Modal app each):
.venv/bin/modal run --detach modal_app.py::train_then_evaluate \
  --config-name lewm_masked --data scene --subdir scene/lewm_masked_e10_s$s \
  --overrides 'seed=$s wandb.config.name=scene_lewm_masked_e10_s$s early_stopping.enabled=false' \
  --eval-config-name scene \
  --eval-overrides 'eval.num_eval=50 eval.env_batch_size=10 output.filename=scene_mtm_mse_e10_s${s}_n50.txt output.save_video=false'
```

#### Eval

```bash
.venv/bin/modal run --detach modal_app.py::evaluate --config-name scene --policy random \
  --overrides 'eval.num_eval=50 eval.env_batch_size=10 output.filename=scene_random_n50_jackpaper.txt output.save_video=false'
```

### Artifacts

* MTM-MSE run dirs: `.stable_worldmodel/scene/lewm_masked_e10_s{3072,1,2}`
* Eval outputs: `scene_mtm_mse_e10_s{3072,1,2}_n50.txt`, `scene_random_n50_jackpaper.txt`
* Apps: ap-8netvdmoWVk4DIsJmRZOQr, ap-4MKQnXVHxub0nDrHNOPo2D, ap-HogSMtBZHaTqwSr1soVCgV (MTM-MSE s3072/s1/s2 in launch order not yet mapped), ap-YqR7V30pHXj9TUJR8Vri0u (random)
* Launch logs: /tmp/scene_mtm_mse_e10_s{3072,1,2}_n50.log, /tmp/scene_random_n50.log
* Verified live: all four apps ephemeral(detached) with Tasks=1 at 10:58-11:00 IST.

### Result

Base:

Pending.

Candidate:

Pending.

Delta:

* Wins / losses: pending.
* Recall collapse? pending — watch for the Reacher-style MTM-MSE collapse mode on Scene.

### Decision

Pending completion.

### Notes

* Purpose is paper Figure "results_lewm_style" Scene panel density plus the
  contrastive-vs-regression question on Scene for the appendix threeway table.

### Update — random-policy Scene floor completed

The random-policy Scene eval (app ap-YqR7V30pHXj9TUJR8Vri0u) completed in ~7
minutes. NOTE: for `policy=random`, `eval.py` sets `results_path` to the repo
directory inside the container instead of the cache volume, so no txt artifact
landed on `multi-future-lewm-cache`; the metrics were recovered from the app
log (`/tmp/scene_random_n50.log`), which prints the full metrics dict.

Result: **random policy scores 52.0% (26/50)** on the Scene trajectory-goal
protocol (n=50, eval seed 42, goal offset 25, budget 50, CEM not used).
Per-batch success: 60/70/60/20/50.

Interpretation: like OGBench-Cube (48% random per the LeWM paper), the 25-step
trajectory-goal protocol has a high chance floor. This recalibrates the Scene
headline: SIGReg 58.0 is +6.0 over random; AC-MTM 80.0 is +28.0 over random.
Added to paper Table (tab:scene), Scene section text, and the main results
figure Scene panel. The benign MuJoCo/EGL cleanup tracebacks appeared after
metrics were printed, as in earlier Scene evals.

### Update — Scene MTM-MSE three-seed completion

All three `lewm_masked` (paper MTM-MSE) Scene train+eval runs completed and the
result files were pulled from the volume
(`scene/lewm_masked_e10_s{3072,1,2}/scene_mtm_mse_e10_s{3072,1,2}_n50.txt`,
local mirror `/tmp/scene_results_jackpaper/`). Benign MuJoCo/EGL renderer
cleanup tracebacks appeared after metrics printed, as with prior Scene evals.

Result (n=50 per seed, eval seed 42, epoch-10 checkpoints):

| Method | Seed 3072 | Seed 1 | Seed 2 | Mean +/- std (ddof=1) |
|---|---:|---:|---:|---:|
| SIGReg | 56.0 | 58.0 | 60.0 | 58.0 +/- 2.0 |
| MTM-MSE | 78.0 | 74.0 | 74.0 | 75.3 +/- 2.3 |
| AC-MTM | 80.0 | 78.0 | 82.0 | 80.0 +/- 2.0 |
| Random | — | — | — | 52.0 (single n=50 eval) |

Paired McNemar over the 150 matched episodes:

* MTM-MSE vs SIGReg: 39 wins / 13 losses, p ≈ 4.1e-4
* AC-MTM vs SIGReg: 40 wins / 7 losses, p ≈ 1.1e-6 (unchanged)
* AC-MTM vs MTM-MSE: 16 wins / 9 losses, p ≈ 0.23 (not significant)

Recall collapse? **No** — all three MTM-MSE Scene seeds are healthy (74-78%),
unlike the Reacher 2/3-seed collapse. Scene does not trigger the MTM-MSE
failure mode.

### Decision

Keep. Paper updated: MTM-MSE row added to the Scene table (tab:scene) and the
appendix three-way table; Scene panels added to both results figures
(`results_lewm_style.png`, `results_ablation_threeway.png`); Scene section,
intro finding (2)/(3), and Scene-audit appendix now state the honest framing —
the Scene win belongs to the inverse-dynamics family (both variants beat
SIGReg decisively), with the contrastive form adding a noise-level 4.7 points
on Scene on top of its Reacher reliability. Random-policy floor (52.0%) added
throughout.
