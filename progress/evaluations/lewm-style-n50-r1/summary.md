# LeWM-Style n=50 Rerun Summary

Date: 2026-06-18

All runs used seed 42, CEM 300/30/30, `eval.num_eval=50`,
`eval.env_batch_size=10`, `output.save_video=false`, exact checkpoint stems, and
unique output filenames.

## Main LeWM-Style Protocol

| Task | SIGReg checkpoint | SIGReg SR | MTM checkpoint | MTM SR | Delta |
|---|---|---:|---|---:|---:|
| TwoRoom | `tworoom/lewm_base_e100/lewm_epoch_30` | 88.0 | `tworoom/lewm_masked_e100/lewm_masked_epoch_30` | 96.0 | +8.0 |
| PushT | `pusht/lewm_base/lewm_epoch_10` | 96.0 | `pusht/lewm_masked/lewm_masked_epoch_10` | 90.0 | -6.0 |
| Reacher | `reacher/lewm_base/lewm_epoch_10` | 74.0 | `reacher/lewm_masked/lewm_masked_epoch_10` | 74.0 | 0.0 |
| OGBench-Cube | `cube/lewm_base/lewm_epoch_10` | 76.0 | `cube/lewm_masked/lewm_masked_epoch_10` | 86.0 | +10.0 |

Protocol details:

- TwoRoom main: goal offset 25, eval budget 50.
- PushT/Reacher/OGBench-Cube: goal offset 25, eval budget 50.

## TwoRoom-Long Stress Protocol

| Protocol | SIGReg SR | MTM SR | Delta |
|---|---:|---:|---:|
| `tworoom_long`, n=50, goal offset 100, eval budget 150 | 12.0 | 28.0 | +16.0 |
| `tworoom_long`, n=200, goal offset 100, eval budget 150 | 18.0 | 28.0 | +10.0 |

The n=200 row is the previously recorded controlled evidence; the n=50 row is
the 2026-06-18 rerun for context under the LeWM paper-text long protocol. Do not
mix this stress protocol into the LeWM Figure 6-style 25/50 comparison.
