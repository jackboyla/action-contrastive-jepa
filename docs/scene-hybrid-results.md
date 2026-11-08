# Fixed Scene hybrid: completed results

All nine models reached epoch 10 and 68,970 updates. All 18 planning evaluations and nine probes completed successfully. The standalone Action-NCE models are shared with the sensitivity study, not independent replications.

## Planning

Successes out of the same 100 goals within each protocol; seeds are training seeds. Mean ± sample SD describes variation across three models, not independent goal samples.

| Objective | Seed | Short | Hard |
|---|---:|---:|---:|
| hybrid | 3072 | 75 | 28 |
| hybrid | 1 | 81 | 24 |
| hybrid | 2 | 80 | 22 |
| hybrid | Mean ± SD | 78.67 ± 3.21 | 24.67 ± 3.06 |
| nce | 3072 | 66 | 22 |
| nce | 1 | 69 | 19 |
| nce | 2 | 74 | 21 |
| nce | Mean ± SD | 69.67 ± 4.04 | 20.67 ± 1.53 |
| sigreg | 3072 | 64 | 18 |
| sigreg | 1 | 64 | 20 |
| sigreg | 2 | 60 | 18 |
| sigreg | Mean ± SD | 62.67 ± 2.31 | 18.67 ± 1.15 |

## Probes

The frozen protocol yielded 5,989 unique frames (3,593/1,197/1,199 train/validation/test) and 128 rollout windows. Ridge alpha was selected using validation only. These are episode-held-out probes from the encoder training collection, not encoder-OOD tests. R² below averages target dimensions within a factor; button columns are balanced accuracy. Rotation is a static-factor diagnostic here; color, lighting and other static factors were not probed.

| Objective | Seed | Cube position R² | Cube rotation R² | Drawer R² | Window R² | Button 0 | Button 1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| hybrid | 3072 | 0.8435 | -0.0484 | 0.9942 | 0.9956 | 0.9983 | 1.0000 |
| hybrid | 1 | 0.8390 | -0.0490 | 0.9950 | 0.9912 | 0.9991 | 1.0000 |
| hybrid | 2 | 0.8008 | -0.0529 | 0.9949 | 0.9840 | 1.0000 | 1.0000 |
| nce | 3072 | 0.6890 | -0.0267 | 0.5082 | 0.4656 | 0.8016 | 0.7664 |
| nce | 1 | 0.7073 | -0.0253 | 0.4794 | 0.5769 | 0.7905 | 0.7922 |
| nce | 2 | 0.6831 | -0.0235 | 0.4925 | 0.5101 | 0.7973 | 0.7472 |
| sigreg | 3072 | 0.5546 | -0.0958 | 0.9699 | 0.9869 | 1.0000 | 1.0000 |
| sigreg | 1 | 0.5376 | -0.0940 | 0.9743 | 0.9877 | 1.0000 | 1.0000 |
| sigreg | 2 | 0.5884 | -0.0874 | 0.9718 | 0.9839 | 0.9991 | 1.0000 |

Probe factor mean ± sample SD across training seeds:

| Objective | Position R² | Rotation R² | Drawer R² | Window R² | Button 0 | Button 1 |
|---|---:|---:|---:|---:|---:|---:|
| hybrid | 0.8278 ± 0.0234 | -0.0501 ± 0.0025 | 0.9947 ± 0.0004 | 0.9903 ± 0.0058 | 0.9991 ± 0.0009 | 1.0000 ± 0.0000 |
| nce | 0.6931 ± 0.0126 | -0.0252 ± 0.0016 | 0.4934 ± 0.0144 | 0.5175 ± 0.0560 | 0.7964 ± 0.0056 | 0.7686 ± 0.0226 |
| sigreg | 0.5602 ± 0.0258 | -0.0924 ± 0.0044 | 0.9720 ± 0.0022 | 0.9862 ± 0.0020 | 0.9997 ± 0.0005 | 1.0000 ± 0.0000 |

Across the three seeds, the hybrid retained drawer/window/button information much better than standalone NCE in these probes. Cube-rotation R² remained negative for every method and seed; the hybrid did not resolve that static-factor diagnostic. This does not measure general static-factor retention.

Latent rollout error below is normalized MSE divided by persistence error (lower is better), at 1/3/5/10 model steps, corresponding to 5/15/25/50 simulator steps. Each rollout begins with one observation.

| Objective | Seed | 1 step | 3 steps | 5 steps | 10 steps |
|---|---:|---:|---:|---:|---:|
| hybrid | 3072 | 0.0170 | 0.0102 | 0.0138 | 0.0154 |
| hybrid | 1 | 0.0188 | 0.0108 | 0.0124 | 0.0169 |
| hybrid | 2 | 0.0180 | 0.0113 | 0.0147 | 0.0167 |
| nce | 3072 | 0.0450 | 0.0223 | 0.0191 | 0.0261 |
| nce | 1 | 0.0460 | 0.0224 | 0.0209 | 0.0304 |
| nce | 2 | 0.0471 | 0.0219 | 0.0206 | 0.0276 |
| sigreg | 3072 | 0.0636 | 0.0428 | 0.0523 | 0.0863 |
| sigreg | 1 | 0.0616 | 0.0405 | 0.0488 | 0.0773 |
| sigreg | 2 | 0.0670 | 0.0454 | 0.0568 | 0.0924 |

## Audit and limits

Checks passed for all resolved planning configs, checkpoint identities, final epoch/update/no-resume records, finite training logs, positive additive hybrid loss terms, reconstructed goal identities, per-goal outcomes, probe splits/windows, saved finite arrays, and independently recomputed validation-selected probe metrics. The three shared NCE artifacts passed the predeclared/completed provenance gate again at final audit. Both hybrid objectives also have the integration gradient test recorded in the protocol.

The hybrid and NCE use the same masked-transition model, differing only in SIGReg weight after run identifiers are removed. The default SIGReg baseline uses the existing autoregressive LEWM without the inverse head. Thus the comparison to SIGReg also includes those model/objective differences; it is not a loss-only ablation on one identical model. All remaining resolved scientific settings match.

All checkpoints were final epoch 10; no hybrid tuning or test-based selection occurred. Three training seeds and one fixed 100-goal set per protocol do not establish statistical significance, robustness, or optimal hybrid coefficients. The original short task and harder task differ in both offset and budget. No paper source was edited.

The NCE seed1 scheduling trial completed both cells under one exclusive GPU0 lock: raw process makespan 1,170.805 s; short/hard process elapsed 692.939/1,140.293 s; loop times 685.977/1,131.461 s. Sampled total GPU memory peaked at 8,110 MiB, with minimum free GPU memory 24,000 MiB across 582 samples. Samples can miss transient peaks. Older seed3072 loop-only times (449.789/925.199 s) have different boundaries and seed, so no direct process speedup/slowdown is claimed. Later cells stayed serial. EGL destructor warnings appeared after successful outputs and exit 0; they were retained, not rerun. The intentional November8 pause remained in raw logs and was accounted for separately.

## Proposed manuscript addition

With fixed coefficients (Action-NCE 0.30 and SIGReg 0.09), the additive hybrid achieved 78.67% mean success on the short Scene protocol and 24.67% on the harder offset-100/budget-100 protocol across three training seeds, compared with 69.67%/20.67% for standalone Action-NCE and 62.67%/18.67% for standalone SIGReg. Each protocol used the same 100 goals across all models and final epoch-10 checkpoints. The standalone Action-NCE controls were shared with the sensitivity study; the default SIGReg baseline uses autoregressive LEWM rather than the masked-transition model. These results describe a fixed-coefficient combination, not an optimized hybrid or a significance test. Secondary held-out-episode probes showed stronger drawer/window/button decoding for the hybrid than standalone NCE, while cube-rotation R² remained negative for all methods.

## Artifact identities

| Objective | Seed | Final object SHA256 |
|---|---:|---|
| hybrid | 3072 | 4417682c71ac0b5ba8a77df64190f03fcac08d82a4a2bd87c6af5dfa276c822a |
| hybrid | 1 | 11812179df8754fd0915f77824fce57d83992f016970c3d9f6ab3993c914ed46 |
| hybrid | 2 | 8b1398fbc4efd4d5128da736af00d28ab42bcbe9bdb490991eadf861fd524b1a |
| nce | 3072 | 067fb34149e1e42c7bd6c09ca45c9ec08a2b623d7d6243d9aa8dee042f21bd68 |
| nce | 1 | 926e717d0ab932cfda43aa559caebdd95d8392decae50e63999b6783c18f3bd7 |
| nce | 2 | cf0b78693069d50c05299998fcfc10813d4ab6f12f0d82463c50ea51c5d81bff |
| sigreg | 3072 | fd8a95e98e7193f517e2d3ec3ac8ea1155ec27d12f05370bb86591912aed02ec |
| sigreg | 1 | 531182b60624fb391919fc5e257ee4e07aaef1b4e2dfaff83f79687955dc8afa |
| sigreg | 2 | 4259d3670a25e8847d866559c17117d4a673c7016b61b1c172697df74b56f264 |

Reproduction protocol and objective configuration: [Scene hybrid protocol](scene-hybrid-study.md). Local audit artifacts reside under the execution worktree `progress/evaluations/`: `final-study-audit.json`, `planning-config-outcome-audit.json`, `goal-identity-audit.json`, probe identity/metric audits, shared-NCE acceptance receipts and `nce1-pair-audit.json`. Raw logs, checkpoints, environment records and probe arrays are not committed.
