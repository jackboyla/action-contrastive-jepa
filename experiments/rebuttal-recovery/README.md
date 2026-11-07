# Recovered historical evidence

This is an additive evidence set. It does not replace the prior
`experiments/rebuttal` bundle or report new training, encoder inference or planning.
All counts below come from archived outcomes. Configs come from the archived run
directories, not similarly named current presets.

| Request | Status | Evidence and limit |
|---|---|---|
| Archived Scene 40/50 | Recovered and verified | Nine original 50-episode files in `scene-short/`; Action-NCE seed 3072 has 40 successes. Training seeds 3072/1/2 give Action-NCE 40/39/41, SIGReg 28/29/30 and inverse-MSE 39/37/37. Evaluation seed 42, offset 25, budget 50, environment batch 10. These are the historical short-goal runs. |
| Seven C.2 sampling records | Recovered and verified | `sampling/` contains verbatim arrays and policy lines from all seven original evaluator logs. Each contains the same 100 unique sorted HDF5 row indices. No IDs were regenerated. Only start rows were printed. The nominal offset is 100; the current end-exclusive dataset loader consumes start + 99. Without the historical code/runtime identity, do not present inferred goal rows as archived observations. |
| Reacher variance-floor 66% | Recovered and verified, with provenance limits | Archived config: seed 3072, batch 128, ten epochs, inverse-MSE weight 1, variance weight .05 and gamma .1, SIGReg weight 0. Evaluation explicitly names `lewm_masked_vicreg_epoch_10` and contains 132/200 successes, environment batch 10. That object checkpoint's hash is recorded. The run metadata has no code commit; the historical binary has not been rerun. This supports variance-only configuration, not full VICReg or a multi-seed rescue. |
| Reacher SIGReg 76% / 100-epoch schedule | Partly recovered; exact evaluated checkpoint unavailable | Original vector contains 152/200 successes, environment batch 10. The file is named `lewm_base_e100_ep9_reacher_n200.txt`, but its policy names a directory, not an epoch. The ledger calls it epoch 10. Both epoch-9 and epoch-10 objects survive; their hashes identify candidates, not the object selected during evaluation. Saved config and training events confirm a 100-epoch horizon, seed 3072, batch 128, SIGReg .09. The config includes a later resume's stop-at-50 setting; it is not the original launch config. |
| Reacher inverse-MSE weight-1 batching | Partly recovered | Training configs for seeds 3072/1/2 all specify batch 128. The 68% seed-3072 evaluation records environment batch 10. The 13.5% seed-2 block records `env_batch_size: null`, which the evaluator resolves to all 200 episodes. The seed-1 11.5% outcome and batch are only ledger-supported; its original log was unavailable. Do not treat its presumed batch 200 as independently verified. |
| Initial Action-NCE weight-1 run | Recovered config and outcomes; directory-policy limit | Seed 1, training batch 128, ten epochs; evaluation batch 10 and 114/200 = 57%. This corrects the ledger's inconsistent parenthetical 113/200. Its policy is a directory; the surviving epoch-10 hash is a candidate identity, not a recorded load receipt. |
| C.1 probe/rank refit | Verified locally from saved arrays | All nine original NPZ files retain latents and factors. `probe-refit.json` records their hashes, reference hashes and CPU refits. All selected ridge alphas match; per-factor R² and button balanced accuracy match within 1e-5; participation ratios within 1e-8. No encoder ran. The large arrays are not duplicated in this compact delta; independent refitting requires those exact arrays. |
| Submitted OpenReview PDF | Unavailable after bounded search | See `submitted-pdf.json`. A local September 1 candidate survives, but neither its filename nor its creation date proves upload identity. The API note/edit endpoints and PDF fetch returned 403; the revisions page returned only an application shell. Do not substitute the revised manuscript PDF. |

The 76% archive cannot establish a matched three-seed E1 baseline. Fresh matched
controls remain necessary unless the actual evaluated checkpoint and original
launch provenance are recovered.

## Check the evidence

From any directory, run `python path/to/rebuttal-recovery/verify.py`. This checks
published hashes, all 15 recovered success vectors and the seven sampling arrays.
It needs only Python's standard library. `manifest.json` records source and
published hashes and the exact transformations: tracker settings removed from
training YAML, sampling excerpts, and the seed-2 block extracted from a file
containing several evaluations. Unresolved expressions in evaluator configs
remain unresolved. Hash equality establishes file identity, not a historical
runtime's environment or dataset identity.

To refit the saved probes, use NumPy, scikit-learn and their CPU dependencies:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python refit_probes.py /path/to/original-npz \
  /path/to/prior-bundle/results/rebuttal-scene refit.json
```

The original NPZ hashes are in `probe-refit.json`; the reference JSON hashes
identify the prior bundle's published files. This does not validate dataset
collection provenance, encoder out-of-distribution generalization or causality.

## Search boundary and remaining records

Recovery checked the main checkout's `progress/evaluations`, `progress/logs`,
`progress/reviews/sd-audit/raw`, `.stable_worldmodel`, temporary named run files,
repository history and the archived Modal Reacher directories. Read-only Modal
queries covered `lewm_masked_vicreg`, `lewm_base_e100`, `lewm_base`,
`lewm_masked`, `lewm_masked_s1`, `lewm_masked_s2` and `mtm_action_nce_e10_s1`;
config, metadata, checkpoint-state, named evaluation files and relevant surviving
checkpoint bytes were read. The two ledger-named weight-1 evaluation apps both
returned “No App with ID … found”. No job was launched.

Still missing: the seed-1 weight-1 original evaluation log/vector; an exact load
receipt for directory-based historical evaluations; the initial e100 launch
snapshot; recorded historical code/data/environment identities; and an
OpenReview revision-to-PDF identity. These are bounded search results, not claims
that the files never existed. Original sampling IDs for the historical Scene
short-goal and Reacher runs were not recovered and must not be confused with the
seven recovered C.2 arrays.

## Anonymous distribution

This directory contains no private discussion, author account links, machine
paths, tracker identities or authentication data. Share a file archive through an
anonymous supplement channel; a repository/PR URL reveals its owner. Git metadata,
local working records, full logs and checkpoint binaries are not part of this
review-facing delta. A separate array supplement is required to let reviewers
repeat the probe refit; the refit report alone does not provide that access.
