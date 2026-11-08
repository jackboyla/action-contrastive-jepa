# Local Reacher continuation

`scripts/run_reacher_parallel.py` continues the six frozen cells from
[the Reacher protocol](reacher-schedule.md) using one training or evaluation per
local GPU. It adopts the first live training without signaling or restarting it.
Each cell must finish training, pass the epoch-10/config/step-count audit, and
finish evaluation before its GPU lane takes another cell. Final reporting uses
the existing task-identity and outcome audit.

The machine-local handoff JSON records `root`, `cache`, `results`, `logs`,
`python`, `manifest`, per-device `locks`, and PID/start-time/parent/command/cwd
identities under `adopt` for the original `parent`, `flock`, `wrapper`, and
`trainer`. It also pins the manifest, data audit, successful whole-study gate,
and dispatcher hashes in `sha256`, plus `serial_script` and
`serial_script_sha256`. Keep this receipt and runtime records under `progress/`.

Before activating that receipt, add this boundary adapter to the serial
entrypoint, before calling its existing main function:

```python
if handoff.exists():
    assert len(sys.argv) == 3
    from run_reacher_parallel import bridge
    bridge(handoff, sys.argv[1], sys.argv[2])
    raise SystemExit(0)
```

The already-loaded serial parent and training wrapper continue normally. Future
serial stage requests validate their ancestry, release only their inherited GPU
lock, and await the new dispatcher's receipts. They never launch scientific
commands. The dispatcher checks the full dependency gate, source/runtime/data
identity and current process tree before claiming ownership.

Run it in a durable session using the same Python environment and checkout:

```bash
python -u scripts/run_reacher_parallel.py --handoff progress/e1-parallel-handoff.json
```

`parallel-status.json` is authoritative for both lanes; the original
`status.json` describes only the serial adapter. Per-stage claims and the owner
receipt prevent silent retries, including after a crash. A failed or lost owner
requires explicit inspection and adoption; restarting the command cannot rerun
a cell. GPU locks remain exclusive, and only CUDA/EGL placement differs from the
frozen environment. The dispatcher enforces the existing 6 GiB output limit and
10 GiB free-space floor. Measure aggregate throughput before assuming a speedup:
both lanes read the same dataset and can compete for storage bandwidth.
