# Rebuttal studies

Three follow-up studies are in progress. Results and manuscript additions are pending audit; the protocols below are not evidence of improved performance.

| Study | Frozen design | Code branch |
|---|---|---|
| Fair coefficient tuning | Scene; SIGReg .03/.09/.27 and Action-NCE .1/.3/.9; equal one-epoch screening; selected coefficient confirmed for ten epochs with seeds 3072, 1, 2 | `t3/scene-sigreg-tuning` |
| Full-duration sensitivity | Scene; batches 64/128/256 at temperature .10, temperatures .05/.10/.20 at batch 128; coefficient .30; ten epochs and three seeds per setting | `t3/scene-action-nce-sensitivity` |
| Combined objective | Scene; Action-NCE .30 plus SIGReg .09; ten epochs, three seeds; matched standalone baselines | `t3/scene-hybrid-loss-ablation` |

[Draft PRs and current updates](https://github.com/jackboyla/action-contrastive-jepa/pulls) contain the study-specific commands, checks, results when available, and proposed paper changes. Branch publication is underway. The initial execution status was verified on November 7, 2026: sensitivity and hybrid training active; the first fair-tuning screen completed, with validation queued.

All studies use final checkpoints and report every predeclared setting. Training seeds repeat evaluation goals; they do not multiply the number of independent goals. The fair-tuning validation and final tasks share source trajectories and the training corpus, so that study is exploratory. Batch sensitivity holds epochs fixed, which changes optimizer-update counts. The hybrid experiment tests fixed coefficients, not an optimally tuned combination.

The [paper repository](https://github.com/jackboyla/action-contrastive-jepa-paper) contains the completed first-round revisions through commit `5ccc9a3`. These follow-up results will enter the manuscript only after verification. No replies have been submitted to OpenReview by the agents.
