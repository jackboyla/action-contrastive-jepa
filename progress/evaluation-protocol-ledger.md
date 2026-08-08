# Evaluation Protocol Ledger

This ledger records which evaluation protocol is used for paper-facing LeWM/MTM
comparisons, and why. It is intentionally separate from result tables so protocol
decisions remain auditable after reruns.

## Sources Checked

- LeWM paper text, Appendix F.1: states TwoRoom uses goal offset 100 and budget
  150, while PushT, Reacher, and OGBench-Cube use goal offset 25 and budget 50.
- LeWM released configs: `config/eval/*.yaml` default to `num_eval=50`.
- GitHub issue lucas-maes/le-wm#38: maintainer says the TwoRoom paper text is a
  typo; the intended TwoRoom result protocol is goal offset 25 and budget 50.
- GitHub issue lucas-maes/le-wm#67: another user reports the same TwoRoom
  paper/config mismatch and low success under 100/150.
- GitHub issue lucas-maes/le-wm#41: documents a paper/config mismatch for CEM
  iterations on Reacher and OGBench-Cube. Paper text says 10 CEM iterations for
  non-PushT tasks; released configs use 30 iterations for all tasks.
- GitHub issues lucas-maes/le-wm#37 and #62: Reacher reproduction is sensitive
  to training/eval seeds; use exact checkpoint stems and record eval seed.

## Paper-Facing Protocols

| Task | Main figure protocol | Rationale | Separate stress/sensitivity protocol |
|---|---|---|---|
| TwoRoom | `num_eval=50`, seed 42, goal offset 25, budget 50, CEM 300/30/30 | Maintainer-confirmed protocol in issue #38; reproduces released-checkpoint result scale | `tworoom_long`: goal offset 100, budget 150, `n=50` |
| PushT | `num_eval=50`, seed 42, goal offset 25, budget 50, CEM 300/30/30 | Released config and paper text agree on distance/budget; PushT uses CEM 30 in paper text | None unless reviewer asks |
| Reacher | `num_eval=50`, seed 42, goal offset 25, budget 50, CEM 300/30/30 | Use released config for main comparison; issue #41 records the CEM mismatch | Optional CEM-10 and eval-seed sweep if Reacher is central |
| OGBench-Cube | `num_eval=50`, seed 42, goal offset 25, budget 50, CEM 300/30/30 | Use released config for main comparison; issue #41 records the CEM mismatch | Optional CEM-10 sensitivity |

## Reporting Rules

- Main LeWM-style figure may include paper-reported external baselines, but must
  label those bars as reported values, not reruns.
- Our SIGReg and MTM bars must use the protocol above and exact checkpoint stems.
- The paper uses `n=50` throughout for planning success to avoid mixing
  protocols. Older `n=200` runs remain useful internal diagnostics, but they are
  not the paper-facing planning protocol.
- Use unique output filenames for every eval, especially Reacher, to avoid the
  prior `dmc_results.txt` race.
