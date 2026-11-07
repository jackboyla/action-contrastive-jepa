# Action-block proximity

This CPU diagnostic measures how often the Action-NCE candidate pool contains
nearby action blocks. It does not measure behavioral equivalence, harmful false
negatives, or learned shortcuts.

## Protocol

For each dataset, use the first 100 full batches of 128 clips from the shuffled
training split, with seed 3072 and the training code's 90/10 clip split. Each clip
contains four observations five simulator steps apart. The inverse objective uses
the first three five-step action blocks, yielding 384 candidates and 383 negatives
per positive. Two negatives come from the same sampled window.

The script uses the actual `HDF5Dataset` action loading and project normalizer,
without loading pixels or a model. Normalize each raw action coordinate with the
whole dataset's mean and sample standard deviation, as training does, then cast to
float32 and flatten each block. Scene blocks have 25 coordinates; Reacher blocks
have 10. Neither dataset has a constant action coordinate. The diagnostic loader
uses no worker processes; the sampled clip IDs are saved and independently checked.

Distance is the root mean square coordinate difference between normalized blocks.
The thresholds are fixed at 0.05, 0.10 and 0.20, inclusive. Counts exclude each
positive's diagonal and count both directions of a negative pair. Cross-sample
pairs can share an episode. The same-episode subset is included in cross-sample
counts, not an additional disjoint group.

## Results

| Dataset | Pair group | Pairs | Distance ≤ .05 | ≤ .10 | ≤ .20 |
|---|---|---:|---:|---:|---:|
| Scene | Same window | 76,800 | 0 | 0 | 12 |
| Scene | Cross sample | 14,630,400 | 22 | 22 | 368 |
| Scene | Cross sample, same episode | 14,904 | 22 | 22 | 22 |
| Reacher | Same window | 76,800 | 0 | 0 | 0 |
| Reacher | Cross sample | 14,630,400 | 0 | 0 | 2 |
| Reacher | Cross sample, same episode | 1,332 | 0 | 0 | 0 |

Nearby action blocks are rare at these thresholds in these samples. The presence
of two same-window candidates does not by itself establish near duplication.
Proximity depends on normalization, block length and threshold; action distance
does not establish whether transitions have equivalent effects. These correlated
pairs are not independent trials, and these two datasets do not establish
robustness to other behavior policies. No episode-shortcut score is reported.

The JSON files retain per-batch counts, normalization statistics, resolved training
configurations, relevant dataset-column hashes, environment versions and source
hashes. Each NPZ stores `(clip_index, episode_index, start_step)` for every sampled
clip. `manifest.json` identifies the published files and the verified Reacher
dataset source. Its full-file hash came from dataset preparation; the diagnostic
checks action and episode-index columns. A full Scene-file hash is not provided.

## Reproduce and verify

Use the project Python environment and the datasets at their configured names
under `STABLEWM_HOME`, or pass `--cache-dir`. Run sequentially from the repository:

```bash
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=
python scripts/action_proximity.py --task scene --output /tmp/scene-proximity.json
python scripts/action_proximity.py --task reacher --output /tmp/reacher-proximity.json
python experiments/action-proximity/verify.py
python experiments/action-proximity/verify.py --cache-dir "${STABLEWM_HOME:-.stable_worldmodel}"
```

The script requires a `.json` output and refuses to overwrite either result file.
The first verification command checks hashes, sampled-ID shapes and count sums without datasets. With
`--cache-dir`, the verifier also reconstructs the split and shuffled sample,
checks action/episode hashes and normalization, and independently recounts all
distances using NumPy directly on the raw HDF5 rows. Both paths passed for the
saved files. Source hashes identify the executed script; the recorded Git commit
is its repository base and does not by itself identify uncommitted script bytes.
