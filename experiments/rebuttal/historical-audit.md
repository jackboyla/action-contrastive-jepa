# Reacher schedule and regularizer evidence

This is a historical evidence audit, not a new experiment. Source: the June
research ledger, current configs and loss implementation. Reacher historical
weights, exact resolved configs and evaluator logs were not found in the local
`.stable_worldmodel` archive. Ledger statements below are not independently
recomputed results.

| Evidence | Recorded result | Provenance limit |
|---|---|---|
| SIGReg, ten-epoch cosine schedule | 69%, n=200 | Historical ledger |
| SIGReg, 100-epoch schedule at epoch 10, `reacher/lewm_base_e100` | 76%, n=200 | Ledger explicitly says exact launch/eval commands and local metadata unavailable |
| SIGReg, epoch 30, seeds 3072/1/2 | 81.5/79.5/80.5% | `lewm_base_e100`, `_s1`, `_s2`; epoch-30 stems `lewm_epoch_30` |
| Inverse-MSE, 100-epoch schedule, epoch 10 | 39.5% | `lewm_masked_e100`; **not Action-NCE** |
| Inverse-MSE, epoch 30, seeds 3072/1/2 | 75/10.5/12% | `lewm_masked_e100`, `_s1`, `_s2`; collapsed extra seeds recorded |
| `lewm_masked_vicreg`, seed 3072, ten epochs | 66%, n=200 | Config name alone does not establish full VICReg |
| MSE + SIGReg .01, seed 1 | 60%, n=200 | Different regularizer and seed from the 66% run |
| Variance-floor screens, seed 1 | One epoch; inverse loss described as near baseline | Weights .005/.01/.05, gamma .05/.1; not full-duration planning evidence |

## E1: feasible, but baseline parity remains unverified

The missing cell is **Action-NCE**, 100-epoch cosine schedule, stopped after ten
completed epochs, for training seeds 3072/1/2. Historical inverse-MSE runs do not
fill it. `train.py` exposes `runtime.stop_after_epoch` independently of
`trainer.max_epochs`; this supports the intended experiment without shortening
the scheduler. Setting `max_epochs=10` would change the experiment.

Before launch, recover archived baseline configs, checkpoint identities, data
split, optimizer/scheduler settings and evaluation task IDs, or predeclare a
fresh matched SIGReg control. The historical single 76% point is not a verified
three-seed control. Do not attribute the entire reproduction gap to schedule,
and do not say E1 is running.

Proposed fixed protocol: Reacher, both objectives, seeds 3072/1/2, released
100-epoch scheduler, stop at epoch 10, no checkpoint selection, same data split,
architecture, optimizer, batch size and evaluation tasks. Freeze coefficient and
temperature before launch. Use n=200 matched evaluation episodes with recorded
seed and task IDs, and report each seed plus paired episode differences. Confirm
the stop callback saves epoch 10 and scheduler horizon remains 100 in a CPU
configuration check first. Resource/time estimates from the historical menu
are not validated workstation measurements. Parent coordination is required;
no runs were launched here.

## E4: distinguish variance-only from full VICReg

The current `config/train/lewm_masked_vicreg.yaml` inherits `lewm_masked` and
adds `loss.variance_reg.weight=.05`, `gamma=.1`. `train.py` applies `VarianceReg`
to embeddings. This is a variance floor, not the full VICReg objective. The
June 10 launch entry also describes a variance lower bound. “Full” may mean
ten training epochs; it must not imply full VICReg. Recovery of the actual
historical resolved config is still needed to prove this identity for the 66%
checkpoint. The name alone is insufficient.

Recommended reply: cite the 66% result as a single-seed historical variance-floor
diagnostic, explicitly qualified by the missing archived config; describe the
one-epoch screens as preliminary. Keep 60% MSE+SIGReg separate. Neither supports
a general claim that variance regularization cannot work. Defer new E4 training
until provenance recovery and parent budget coordination. E5 proximity counts
and E6 evaluation-seed sweep remain separate recommendations; neither was run.
