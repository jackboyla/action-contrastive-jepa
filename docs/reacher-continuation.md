# Reacher continuation at a natural boundary

`scripts/continue_reacher.py` transfers the frozen six-cell Reacher study from
the original two-lane dispatcher to the corrected dispatcher. It requires a
separately reviewed machine-local request and approval for the live handoff.
Publishing or merging this code does not install or invoke it. Keep this code
outside the pinned scientific checkout; it runs scientific commands in the
original checkout with the original environment, dataset and manifest.

The request JSON contains:

```json
{
  "handoff": "/absolute/path/to/original-handoff.json",
  "handoff_sha256": "reviewed SHA256",
  "owner_sha256": "SHA256 of original dispatch/owner.json",
  "code_sha256": {
    "/absolute/publication/scripts/continue_reacher.py": "reviewed SHA256",
    "/absolute/publication/scripts/run_reacher_parallel.py": "reviewed SHA256",
    "/absolute/publication/scripts/reacher_schedule.py": "reviewed SHA256"
  }
}
```

Use the shared, frozen Python environment after the request and implementation
have received handoff review:

```bash
/path/to/shared/python /absolute/publication/scripts/continue_reacher.py control /absolute/path/to/request.json
```

Run that command in a durable session with a log under `progress/`. Do not run
the `keeper` or `successor` roles yourself. They use private inherited sockets
and descriptors. No new scientific options or retry flags are accepted.

## Transfer and evidence

The controller reserves `dispatch/continuation/` once. It validates the frozen
sources, runtime, dataset, budgets and entire sensitivity success certificate.
A separate keeper inherits the original GPU-lock descriptors and acquires each
lock normally when the existing stage releases it. It retains every acquired
lock if the controller dies, including during partial fencing. It does not
signal scientific jobs. Acquiring one lock while the other GPU finishes may
leave that GPU idle temporarily.

Before retirement, the controller requires the incumbent's kernel `flock`
waiter records for each pending lane, matching `waiting_lock` state and claims,
and no conflicting post-lock health, launch, log or scientific-output evidence.
While unclaimed work remains, both lanes must reach their pre-launch flocks.
It tracks scientific roots and descendants through recorded identities,
ancestry, commands and process groups/sessions, and checks CUDA/EGL inventory.
The original training wrapper and flock must have exited naturally. A missing
launch receipt or empty log does not authorize reuse of an attempt.

Only the exact idle incumbent and the recognized legacy queue controls qualify
for retirement. Every target receives a verified pidfd before the first signal.
Retirement uses SIGTERM through those pidfds, never numeric-PID or group signals;
there is no forced-kill escalation. Unknown legacy descendants and stale
identities block the transfer. An instance that escaped all recorded ancestry,
command and session/group identities cannot be proven absent by this mechanism;
the live handoff review must establish that the frozen scientific process model
fits this scope. Uncertain external or untracked descendants prohibit arming.

The successor acquires the same `dispatch/owner.lock` after the old controls
exit. It revalidates the certificate, receipts and scientific audits, reconstructs
the same lane assignments and manifest suffix, and writes its immutable ready
receipt. Only then may the keeper and controller release their GPU-lock copies.
The successor requires a final socket message before dispatching. It uses the
corrected subreaper supervisor and existing checkpoint/config/task/outcome audits.

Original owner, attempt, success and audit receipts stay in place. Completed
stages never rerun. A fenced unlaunched claim gets a separate adoption receipt;
the original claim is neither deleted nor rewritten. All completed cells produce
an `all-complete.json` audit without retirement or a successor. Failure or an
ambiguous attempt never authorizes a replay.

The generation directory records the barrier owner, each acquired fence,
certificate, retirements, successor ownership/readiness, release, transfer and
completion or failure. `parallel-status.json` becomes authoritative for the
successor only after dispatch begins; `dispatch/owner.json` remains historical.
The new owner identity lives in `dispatch/continuation/owner-acquired.json`.

## Failure limits

A crash after fencing can leave the keeper holding GPU locks indefinitely.
Failure to write the keeper's error receipt also retains those locks.
After retirement, failure leaves the generation blocked until explicit evidence
reconciliation; invoking the controller again refuses the existing generation.
Do not delete receipts or the generation directory, unlock another process's
descriptor, or retry a cell to recover. The keeper catches ordinary termination
signals; killing the keeper itself with SIGKILL is outside this guarantee.
A controller loss after verified readiness/release but before the final socket
message leaves the successor unable to dispatch. Once that message is consumed,
the single successor continues independently of the controller.

The existing experiment monitor must retain its cadence. After an approved live
transfer, update its ownership references using the generation receipts. This
command does not create, change or disable a monitor.
