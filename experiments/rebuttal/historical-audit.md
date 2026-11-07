# Reacher schedule and regularizer evidence

The [recovered archive](../rebuttal-recovery/README.md) supplies raw outcomes,
training configs and checkpoint identities for part of the historical record.
It distinguishes explicit checkpoint policies from directory policies that
selected the latest object at evaluation time. A surviving checkpoint hash
alone does not prove which object a historical directory policy loaded.

| Evidence | Recorded result | Provenance limit |
|---|---|---|
| SIGReg, ten-epoch cosine schedule | 69%, n=200 | Historical ledger |
| SIGReg, 100-epoch schedule, `reacher/lewm_base_e100` | 76%, 152/200 | Outcome recovered; filename says epoch 9, ledger says epoch 10, policy names a directory. Exact evaluated checkpoint unresolved. |
| SIGReg, epoch 30, seeds 3072/1/2 | 81.5/79.5/80.5% | Historical ledger; `lewm_base_e100`, `_s1`, `_s2` |
| Inverse-MSE, 100-epoch schedule, epoch 10 | 39.5% | Historical ledger; `lewm_masked_e100`, not Action-NCE |
| Inverse-MSE, epoch 30, seeds 3072/1/2 | 75/10.5/12% | Historical ledger; collapsed extra seeds recorded |
| Inverse-MSE + variance floor, seed 3072, ten epochs | 66%, 132/200 | Archived config, explicit epoch-10 policy, outcome vector and surviving checkpoint hash recovered; no evaluation-time digest or historical code commit |
| MSE + SIGReg .01, seed 1 | 60%, n=200 | Historical ledger; different regularizer and seed from the 66% run |
| Variance-floor screens, seed 1 | One epoch; inverse loss described as near baseline | Weights .005/.01/.05, gamma .05/.1; not full-duration planning evidence |

## Schedule comparison

A matched comparison needs fresh Action-NCE and SIGReg controls unless exact
historical provenance can establish equivalent runs. The single 76% point
cannot replace a three-seed control. The existing objectives also retain their
respective model paths; this is not an isolated loss swap on an identical model.

`train.py` exposes `runtime.stop_after_epoch` independently of
`trainer.max_epochs`. Use a 100-epoch horizon and stop after ten completed epochs
to preserve that learning-rate curve. Setting `max_epochs=10` changes it.
Historical inverse-MSE runs do not fill an Action-NCE comparison. A runtime and
schedule change cannot establish the cause of the entire reproduction gap.

## Variance floor versus full VICReg

The recovered 66% run used inverse-MSE weight 1, variance weight .05, gamma .1
and SIGReg weight 0, with training seed 3072 and batch 128. Its evaluation names
the epoch-10 policy explicitly and uses batch 10. This establishes a historical
single-seed variance-floor result, not full VICReg or a matched multi-seed rescue.
Keep the 60% MSE+SIGReg result and one-epoch screens separate. These observations
do not support a general claim that variance regularization cannot work.
