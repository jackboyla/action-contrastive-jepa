# AGENTS.md

This file contains project-specific instructions for AI coding agents working in this repository.

## Experiment log

Maintain a chronological append-only research ledger at:

```text
progress/experiment-log.md
```

Create it if missing.

Every meaningful training, inference, evaluation, comparison, or failed run should get an entry.

Use this structure:

````md
## YYYY-MM-DD — <RUN_ID>

### Intent

Explain what this run is testing and what baseline or previous run it should be compared against.

### Commands

#### Train

```bash
<exact training command, if applicable>
````

#### Inference

```bash
<exact inference command, if applicable>
```

#### Diagnostics

```bash
<exact diagnostic or quality command, if applicable>
```

#### Eval

```bash
<exact eval command, if applicable>
```

#### Compare

```bash
<exact compare command, if applicable>
```

### Artifacts

* Train metadata: `finetuning/results/<RUN_ID>/train_result.json`
* GPU metrics: `finetuning/results/<RUN_ID>/gpu_metrics.csv`
* Predictions: `finetuning/results/<RUN_ID>/predictions.jsonl`
* Eval summary: `progress/evaluations/<RUN_ID>/summary.json`
* Per-dialogue eval: `progress/evaluations/<RUN_ID>/concept_per_dialog.jsonl`

### Result

Base:


Candidate:


Delta:

* Wins / losses:
* Recall collapse? yes/no

### Decision

Keep / reject / rerun / modify X.

### Notes

* Repeated-output diagnostics? Treat as debugging signal, not automatic rejection.
* Modal/vLLM issues?
* GPU utilization or memory issue?
* Anything to change next?

## Modal CLI reference

Correct entrypoint names and flags for `modal run modal_app.py::`:

| Action | Entrypoint | Key flags |
|---|---|---|
| Train | `train` | `--config-name`, `--data`, `--subdir`, `--overrides` |
| Evaluate | `evaluate` | `--config-name`, `--policy <path>` |
| Probe | `probe` | `--policy <path>` |

### Canonical launch pattern (ALWAYS use this)

```bash
nohup .venv/bin/modal run --detach modal_app.py::<entrypoint> <args> > /tmp/<job>.log 2>&1 &
```

- The `modal` binary is at `.venv/bin/modal` — it is NOT on PATH. Bare `modal` fails with `command not found`.
- Use the Modal CLI's global `run --detach` so the remote app survives a local client disconnect.
- Use `nohup ... &` so the local wrapper survives and logs are written to `/tmp/<job>.log`.
- Do NOT pass entrypoint-level `--no-wait`.

### NEVER use `--no-wait`

`--no-wait` makes the local entrypoint return immediately, and Modal can then **kill the spawned task**. The result is an app that shows `stopped` with `0 tasks` and NOTHING runs. This has silently wasted hours. Always keep the entrypoint blocking and use the global `modal run --detach` option instead.

### MANDATORY: verify every job is actually running

After launching ANY Modal job, you MUST confirm it is live before reporting status or moving on. A spawned app that exits is NOT a running job. Verify with BOTH:

1. **Log shows real progress**, not just `Initialized`/`Spawned`:
   ```bash
   tail -5 /tmp/<job>.log    # training: a progress bar line; eval: container logs
   ```
2. **App is actually executing tasks** (`State=running`/`ephemeral` with `Tasks>=1`, NOT `stopped` with `0`):
   ```bash
   .venv/bin/modal app list | head -20
   ```

If the app is `stopped` with `0 tasks`, the job did NOT run — fix and relaunch. Never claim a job is running based only on `✓ Initialized` or `✓ App completed` — those print even when the spawned task was killed. Confirm an output artifact (checkpoint, `.txt` result, metrics line) before declaring success.

### `--policy` path convention (evaluate / probe)

`AutoCostModel` resolves `--policy` like this:
- If the path is a **directory** → it globs `*_object.ckpt` inside and loads the newest.
- Otherwise → it appends `_object.ckpt` to whatever you passed.

So pass the **stem WITHOUT `_object.ckpt`**, e.g. `pusht/lewm_ms_mtm/lewm_ms_mtm_epoch_10` (NOT `..._epoch_10_object.ckpt`, which double-suffixes to `..._object.ckpt_object.ckpt` and fails with `Checkpoint path does not exist`). Or pass the directory `pusht/lewm_ms_mtm` to auto-pick the latest epoch.

### Other gotchas

- `::eval` does not exist — use `::evaluate`.
- `--checkpoint` does not exist — use `--policy`.
- Cube uses data config `ogb` (`--data ogb`); the eval config is named `cube` (`--config-name cube`). They differ.

When in doubt: `grep "@app.local_entrypoint" -A1 modal_app.py` lists all valid entrypoints.

````
