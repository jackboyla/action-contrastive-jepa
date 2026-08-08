# PushT / Reacher / TwoRoom / OGB-Cube — masked vs base vs reported (e10 + e100)

Planning success rate (%), all evals **n=200, seed 42, CEM n_steps=30** unless noted.
`base` = LeWM with SIGReg. `masked` = masked transition modeling (SIGReg off,
inverse-dynamics anti-collapse). `AC-CPC` = contrastive forward (SIGReg off, no
inverse head). `reported` = LeWM paper Fig 6.

## 2026-06-18 Reacher Correction

The Reacher masked-transition row below was originally based on one noncollapsed
seed. Two later masked seeds (`reacher/lewm_masked_s1`, `s2`) collapsed and scored
**11.5% / 13.5%**, while matched SIGReg seeds scored **69.0% / 68.5%**. The bad
runs are real latent collapse (`emb_std` near zero), not an eval bug. Treat
plain masked-transition on Reacher as seed-unstable/rejected unless a collapse
prevention change is added.

## 2026-06-22 Reacher Checkpoint Audit

Checked whether the Reacher collapse was just a bad epoch-10 checkpoint. It was
not for default MTM: seed 1 scored 12.0/12.0/10.0 at epochs 1/5/9 versus 11.5 at
epoch 10, and seed 2 scored 10.0/11.5/10.5 versus 13.5 at epoch 10. The healthy
default seed 3072 scored 64.5 at epoch 9 versus 68.0 at epoch 10.

For the pure-MTM lower-LR rescue (`optimizer.lr=3e-5`), epoch 9 is better than
epoch 10 across all three seeds: 68.5/74.0/65.5, mean 69.3, versus epoch-10 mean
64.7. Use this as the best Reacher MTM reporting checkpoint if we include a
schedule-sensitive Reacher row; do not frame it as a strong MTM win.

## Table 1 — e10 (matched 10-epoch schedule): the primary head-to-head

This is the apples-to-apples comparison: every model trained with the **same**
`max_epochs=10`, `early_stopping.enabled=false` recipe and evaluated at the final
**epoch 10** checkpoint. `masked` columns are the mean of 3 training seeds.

| Task | base (SIGReg) | masked | AC-CPC | reported LeWM | eval epoch | masked vs base |
|------|------:|------:|------:|------:|:--:|:--|
| **OGB-Cube** | 66.2 | **79.3** | — | 74 | 10 | **+13.1 win** (p<0.005, 3/3) |
| **TwoRoom — Long** | 17.0 | **28.0** | — | — | 10 | **+11.0 win** (p<0.05, 3/3) |
| TwoRoom — Protocol A | 85.5 | 90.2 | — | 87 | 10 | +4.7 (p 0.12–0.22) |
| Reacher | 68.8 mean (69.0/69.0/68.5) | 31.0 mean (68.0/11.5/13.5) | — | 86 max / ~82 mean | 10 | **seed-unstable collapse** |
| **PushT** | **93.2** | 85.5 | **62.5** | 96 | 10 | **-7.7 loss** (p<0.01, 3/3) |

## Table 2 — e100 (100-epoch reproduction schedule): only where it changed the verdict

The repo's `max_epochs=10` recipe collapses the cosine LR schedule (anneals LR→0
by epoch 10) and under-trains relative to the released 100-epoch checkpoints. We
reran on the corrected 100-epoch schedule **only for the tasks where it mattered**
— Reacher (to test the reproduction gap) and PushT (to test whether more training
rescues masked). Cube and TwoRoom were **not** rerun: masked already beats the
reported LeWM at e10, so a new baseline ceiling was unnecessary.

| Task | base (SIGReg) | masked | reported LeWM | eval epoch | note |
|------|------:|------:|------:|:--:|:--|
| Reacher | **81.5** | 75.0 | 86 max / ~82 mean | 30 | base **reproduces** the paper mean; masked −6.5 |
| PushT (masked only) | — *(not rerun; 93.2 @ e10)* | 81.0 | 96 | 15 | masked improved e10->e100 (85.5->... see below) but stays < base |
| OGB-Cube | not rerun | not rerun | 74 | — | masked e10 (79.3) already > reported (74) |
| TwoRoom | not rerun | not rerun | 87 | — | masked e10 already wins |

**Reacher e100 trajectory** (base / masked, n=200): ep15 82.0 / 69.5 · ep30 81.5 / 75.0.
Masked keeps improving with training but does not catch base; base lands on the
paper's ~82 mean. (Reported 86 is the **max** over seeds, not the mean — confirmed
in upstream issue #37.)

**PushT masked e100 trajectory** (n=200): ep10 78.0 · ep15 81.0. Stalled below the
e10 masked result (85.5) and far below base (93.2) — more training did **not** fix
PushT; the deficit is representational, not optimization (see probe below).

## PushT representational probe (why PushT resists SIGReg-free anti-collapse)

Linear-probe R² from the frozen latent to privileged physical state (n=4000, ridge
α=1), all three models at PushT epoch 10 with the identical probe script:

| Probe R² | base/SIGReg | masked | AC-CPC |
|----------|------:|------:|------:|
| agent pos (avg) | 0.946 | **0.999** | 0.777 |
| block pos (avg) | **0.979** | 0.937 | 0.740 |
| block orientation | **0.791** | 0.508 | 0.655 |
| **mean (7 dims)** | **0.701** | 0.674 | 0.564 |
| **planning SR** | **93.2%** | 85.5% | 62.5% |

**Planning SR is monotonic in mean probe R²** (0.701->93.2, 0.674->85.5,
0.564→62.5). SIGReg wins on PushT by producing the most *balanced,
high-decodability* latent (strong on agent **and** block position **and**
orientation). The SIGReg-free alternatives fail not by dropping one variable but by
yielding a less physically-organized latent overall — AC-CPC most of all.

## Headline

- **Masked transition modeling** is the better SIGReg-free alternative on some
  tasks: **wins** Cube (+13) and TwoRoom-Long (+11), **loses** PushT (−8 e10),
  and is **not robust on Reacher** after seeds 1/2 collapsed.
- **AC-CPC** (contrastive) does **not** rescue PushT (62.5%, worse than both); it
  confirms PushT specifically rewards SIGReg's balanced full-state latent.
- **Reacher "gap"** to the paper's 86 was a measurement artifact (86 = max, ~82 =
  mean) plus the e10 LR-schedule bug; the e100 base (81.5) reproduces the mean.

## Footnotes

- Binomial noise at n=200 is roughly ±5pp (1 s.d. ≈ 3.5pp at p≈0.85), so
  within-task deltas under ~5pp are not significant on their own; the masked Cube /
  TwoRoom-Long / PushT verdicts use paired McNemar across 3 seeds (p-values shown).
- e10 cells: final checkpoint of a 10-epoch run with early-stopping disabled, i.e.
  epoch 10. e100 cells: the exact milestone epoch listed (the 100-epoch LR schedule
  was preserved; runs were stopped early at the milestone via `stop_after_epoch`).
- "reported LeWM" is read off paper Fig 6 bars (Two-Room 87, Reacher 86, Push-T 96,
  OGB-Cube 74); the paper's eval protocol may differ from ours in minor ways.
