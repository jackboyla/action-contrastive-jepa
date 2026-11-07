# Completed Scene appendix experiments

This bundle records the completed C.1 probes/rollouts, C.2 longer-goal evaluation,
and D.1 one-epoch sensitivity screen. It contains no results from subsequent
full-duration tuning, sensitivity, or hybrid studies.

`results/` contains original JSON summaries and per-episode evaluation text,
including embedded evaluation configurations. `${REPO_ROOT}` replaces identifying
absolute paths; it is a placeholder, not a Hydra expression to execute.
`manifest.json` records source and published hashes and every data transformation.
The nine NPZ files retain unchanged sample IDs, splits, rollout window IDs and
per-window errors; embeddings and state factors are omitted. JSON files retain
checkpoint SHA-256 identities. `configs/` contains recovered training configs.
Do not treat unresolved interpolations in archived configs as resolved values.

## Rerun

Use the repository's installation instructions and the Scene dataset at
`.stable_worldmodel/ogbench/visual_scene_play.h5`. Restore the checkpoints at the
relative paths recorded in the result JSON files and verify their SHA-256 first.
Weights and data are not distributed here. Exact historical environment/package
versions and dataset hashes have not been recovered; bitwise reproduction is
not established. The diagnostic code requires scipy and scikit-learn as well
as the project dependencies. Run from the repository root:

```bash
python experiments/rebuttal/scene_diagnostics.py \
  --checkpoint .stable_worldmodel/scene/lewm_sigreg_e10_s3072/lewm_epoch_10_object.ckpt \
  --output rerun/sigreg_3072.json
```

Repeat for seeds 3072, 1, 2 and methods `sigreg`, `masked`,
`masked_action_nce`; checkpoint prefixes are `lewm`, `lewm_masked`,
`lewm_masked_action_nce`, respectively. The script fixes the sampling seed at
20261106 and saves the actual split and rollout IDs. Probe episodes come from
the encoder training collection: this is not encoder out-of-distribution testing.
`python experiments/rebuttal/summarize_scene.py` recomputes the bundled C.1 summary
and asserts that all nine checkpoints used identical sample IDs.

C.2 uses frozen epoch-10 checkpoints, evaluation seed 43, and 100 episodes:

```bash
MUJOCO_GL=egl python eval.py --config-name scene \
  policy=scene/lewm_sigreg_e10_s3072/lewm_epoch_10 \
  eval.num_eval=100 eval.env_batch_size=10 seed=43 \
  eval.goal_offset_steps=100 eval.eval_budget=100 output.filename=rerun/longer.txt
```

Repeat for SIGReg and Action-NCE, seeds 3072/1/2. The pairing audit records the
100 shared initial-state indices. These shared tasks are not independent
repetitions across models. Raw files include the random baseline and runtime
reproduction checks; do not pool those different protocols. The recorded longer
goal means are 20% SIGReg and 21.3% Action-NCE; this does not retain the original
short-protocol margin.

D.1 training commands share the following options:

```bash
python train.py --config-name lewm_masked_action_nce data=scene seed=1 \
  trainer.max_epochs=1 num_workers=2 loader.prefetch_factor=2 \
  wandb.enabled=false early_stopping.enabled=false \
  logging.progress_refresh_steps=100 checkpoint.step_interval=2000 \
  subdir=rebuttal/scene_sensitivity_default_s1 \
  loader.batch_size=128 loss.masked.action_nce_temperature=0.1
MUJOCO_GL=egl python eval.py --config-name scene \
  policy=rebuttal/scene_sensitivity_default_s1/lewm_masked_action_nce_epoch_1 \
  eval.num_eval=50 eval.env_batch_size=10 seed=43 output.filename=rerun/default.txt
```

Repeat with `(name,batch,temperature)` equal to `(tau005,128,.05)`,
`(tau020,128,.2)`, `(batch64,64,.1)`, `(batch256,256,.1)`, changing both
subdirectory and policy. Batch changes also change optimizer update counts.
This is one training seed and one epoch, not full-training robustness. The
reason for choosing evaluation seed 43 instead of headline seed 42 is unresolved.

## Missing assets

The bundle does not supply weights, dataset contents/hash, a locked historical
environment, or recovered Reacher archive files. Sensitivity checkpoint hashes
and per-task IDs beyond those actually stored in the raw files are not invented.
Evaluation configs embedded in text may contain Hydra interpolations; training
configs are recovered snapshots, not a claim of fully resolved provenance.
See [historical audit](historical-audit.md) before coordinating any new runs.

Run `python experiments/rebuttal/check_integrity.py` for a CPU-only hash and
per-episode success-rate check. No training or evaluation is started.
