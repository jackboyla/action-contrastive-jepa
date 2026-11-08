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
