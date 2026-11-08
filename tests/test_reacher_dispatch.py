"""Real process/lock checks for adopting the serial Reacher queue."""
import json
import fcntl
import subprocess
import sys
import time
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
from run_reacher_parallel import identity, reserve, same_process, sha256


@pytest.mark.parametrize('success', [True, False])
def test_legacy_bridge_releases_only_its_lock_and_obeys_receipt(tmp_path, success):
    lock = tmp_path / 'gpu.lock'
    other_lock = tmp_path / 'other.lock'
    results = tmp_path / 'results'
    records = results / 'dispatch'
    records.mkdir(parents=True)
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'jobs': [{'label': 'cell'}]}))
    handoff = tmp_path / 'handoff.json'
    script = tmp_path / 'serial.py'
    script.write_text(f'import sys\nsys.path.insert(0, {str(SCRIPTS)!r})\n'
                      f'from run_reacher_parallel import bridge\n'
                      f'bridge({str(handoff)!r}, sys.argv[1], sys.argv[2])\n')
    saved = identity(__import__('os').getpid())
    handoff.write_text(json.dumps(dict(manifest=str(manifest), results=str(results),
        adopt={'parent': saved}, locks={'1': str(lock)}, python=sys.executable,
        serial_script=str(script), serial_script_sha256=sha256(script))))
    (records / 'owner.json').write_text(json.dumps({'process': saved}))
    with other_lock.open('a') as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        proc = subprocess.Popen(['flock', str(lock), sys.executable, str(script), 'cell', 'train'],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            # Wait until the adapter has released its inherited descriptor.
            deadline = time.monotonic() + 10
            while True:
                assert proc.poll() is None
                with lock.open('a') as probe:
                    try:
                        fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        # Avoid mistaking the pre-exec gap for the release.
                        children = Path(f'/proc/{proc.pid}/task/{proc.pid}/children').read_text().split()
                        if children and Path(f'/proc/{children[0]}/fd/3').resolve() == lock:
                            break
                    except BlockingIOError:
                        pass
                assert time.monotonic() < deadline
                time.sleep(.05)
            with other_lock.open('a') as untouched:
                with pytest.raises(BlockingIOError):
                    fcntl.flock(untouched, fcntl.LOCK_EX | fcntl.LOCK_NB)
            target = results / 'cell-train-exit.json' if success else records / 'cell-train-failure.json'
            target.write_text(json.dumps({'exit': 0} if success else {'error': 'failed training'}))
            _, error = proc.communicate(timeout=15)
            assert (proc.returncode == 0) is success, error
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()


def test_claim_and_process_identity_never_allow_silent_retry(tmp_path):
    claim = tmp_path / 'attempt.json'
    reserve(claim, {'state': 'attempted'})
    with pytest.raises(FileExistsError):
        reserve(claim, {'state': 'retry'})
    saved = identity(__import__('os').getpid())
    assert same_process(saved)
    saved['starttime'] = str(int(saved['starttime']) + 1)
    assert not same_process(saved)


@pytest.mark.parametrize('mode', ['receipt_failure', 'owner_sigterm', 'owner_sigkill', 'exit_0', 'exit_7'])
def test_stage_lock_outlives_owner_and_escaped_descendant(tmp_path, mode):
    """Real CPU processes/flocks; only machine health/GPU inventory are stubbed."""
    import os
    import signal

    child = tmp_path / 'science.py'
    child.write_text('''import json, os, subprocess, sys, time
from pathlib import Path
root = Path(sys.argv[1])
if sys.argv[2] == 'leaf':
    while not (root/'release-leaf').exists():
        time.sleep(.01)
    sys.exit(0)
leaf = subprocess.Popen([sys.executable, __file__, str(root), 'leaf'],
    close_fds=True, start_new_session=True)
(root/'tree.tmp').write_text(json.dumps({'child': os.getpid(), 'leaf': leaf.pid}))
(root/'tree.tmp').replace(root/'tree.json')
while not (root/'release-child').exists():
    time.sleep(.01)
sys.exit(int(sys.argv[2]))
''')
    # No real workstation/GPU commands run in this regression.
    binpath = tmp_path / 'bin'
    binpath.mkdir()
    for name in ('hostname', 'uptime', 'df', 'free', 'nvidia-smi', 'docker', 'tmux', 'ps'):
        path = binpath / name
        path.write_text('#!/bin/sh\ncase "$*" in *uuid*) case "$*" in *compute*) ;; *) echo TEST-GPU ;; esac ;; esac\n')
        path.chmod(0o755)
    worker = tmp_path / 'worker.py'
    worker.write_text(f'''import sys, time
from pathlib import Path
sys.path.insert(0, {str(SCRIPTS)!r})
import run_reacher_parallel as code
root = Path({str(tmp_path)!r})
d = code.Dispatcher.__new__(code.Dispatcher)
d.root = root
for name in ('cache','logs','records'):
    path = root / name
    path.mkdir()
    setattr(d, name, path)
d.out = d.records
d.cfg = {{'python':sys.executable, 'locks':{{'0':str(root/'gpu.lock')}}}}
d.manifest = {{'environment':{{}}}}
d.check = lambda: None
d.budget = lambda: 0
d.state = lambda *a, **k: None
d.audit_training = lambda *a: None
original = code.write
def write(path, value):
    if str(path).endswith('-launch.json'):
        original(root/'supervisor.json', value['process'])
        deadline = time.monotonic() + 10
        while not (root/'tree.json').exists():
            assert time.monotonic() < deadline
            time.sleep(.01)
        if {mode!r} == 'receipt_failure':
            raise OSError('injected receipt failure')
    original(path, value)
code.write = write
d.run_stage({{'label':'cell', 'subdir':'unused',
             'train':[{str(child)!r}, str(root), {'7' if mode == 'exit_7' else '0'!r}]}}, 'train', 0)
''')
    with (tmp_path / 'owner.log').open('w') as log:
        owner = subprocess.Popen([sys.executable, str(worker)], stdout=log,
                                 stderr=subprocess.STDOUT, env=dict(os.environ, PATH=str(binpath)))
        saved = {}
        try:
            deadline = time.monotonic() + 15
            while not (tmp_path / 'tree.json').exists():
                assert owner.poll() is None, (tmp_path / 'owner.log').read_text()
                assert time.monotonic() < deadline
                time.sleep(.02)
            tree = json.loads((tmp_path / 'tree.json').read_text())
            saved['leaf'] = identity(tree['leaf'])
            saved['supervisor'] = json.loads((tmp_path / 'supervisor.json').read_text())
            if mode.startswith('owner_sig'):
                owner.send_signal(signal.SIGTERM if mode == 'owner_sigterm' else signal.SIGKILL)
                owner.wait(timeout=5)
                assert identity(tree['child'])
            # Make the scientific root exit while its detached, close_fds child survives.
            (tmp_path / 'release-child').touch()
            deadline = time.monotonic() + 10
            while True:
                try:
                    identity(tree['child'])
                except (ProcessLookupError, FileNotFoundError):
                    break
                assert time.monotonic() < deadline
                time.sleep(.02)
            assert identity(tree['leaf'])['starttime'] == saved['leaf']['starttime']
            with (tmp_path / 'gpu.lock').open('a') as competitor:
                with pytest.raises(BlockingIOError):
                    fcntl.flock(competitor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            assert not (tmp_path/'records/cell-train-exit.json').exists()
            (tmp_path / 'release-leaf').touch()
            if mode.startswith('owner_sig'):
                assert owner.returncode < 0
            else:
                owner.wait(timeout=20)
                output = (tmp_path / 'owner.log').read_text()
                assert (owner.returncode == 0) == (mode == 'exit_0'), output
                if mode == 'receipt_failure':
                    assert 'injected receipt failure' in output
                if mode == 'exit_7':
                    assert 'cell train exit 7' in output
            # Last descendant exit must release the lock, without a restart.
            deadline = time.monotonic() + 10
            while True:
                with (tmp_path / 'gpu.lock').open('a') as competitor:
                    try:
                        fcntl.flock(competitor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except BlockingIOError:
                        pass
                assert time.monotonic() < deadline
                time.sleep(.02)
            assert not (tmp_path/'records/cell-train-exit.json').exists() or mode == 'exit_0'
        finally:
            # Release only this test's cooperative children, even after a failed assertion.
            (tmp_path / 'release-child').touch()
            (tmp_path / 'release-leaf').touch()
            if owner.poll() is None:
                owner.terminate()
                owner.wait(timeout=5)
            for value in saved.values():
                deadline = time.monotonic() + 10
                while True:
                    try:
                        current = identity(value['pid'])
                    except (FileNotFoundError, ProcessLookupError):
                        break
                    if current['starttime'] != value['starttime']:
                        break
                    assert time.monotonic() < deadline
                    time.sleep(.02)
