"""CPU-only subprocess regressions for the complete E6 worker boundary."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest
from experiments.e6 import runner


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / 'experiments/e6/runner.py'


def setup_tools(tmp_path, stall=None):
    bindir = tmp_path / 'bin'
    bindir.mkdir()
    for command, should_stall in [('python', stall == 'packages'), ('nvidia-smi', stall == 'gpu')]:
        script = bindir / command
        script.write_text(
            f'#!{sys.executable}\n'
            'import os, subprocess, sys, time\n'
            'from pathlib import Path\n'
            "if Path(sys.argv[0]).name == 'python' and sys.argv[1:4] != ['-m', 'pip', 'freeze']:\n"
            f'    os.execv({sys.executable!r}, [{sys.executable!r}, *sys.argv[1:]])\n'
            + ("Path('setup.pid').write_text(str(os.getpid()))\n"
               "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
               "Path('descendant.pid').write_text(str(child.pid))\n"
               'time.sleep(60)\n' if should_stall else "print('test metadata')\n")
        )
        script.chmod(0o755)
    return bindir


@pytest.mark.skipif(os.name != 'posix', reason='E6 workers use POSIX process groups')
@pytest.mark.parametrize('stall', ['checkpoint', 'packages', 'gpu'])
def test_deadline_bounds_setup_before_evaluator(tmp_path, stall):
    """Use real blocking I/O/commands, with an independent test safety timeout."""
    bindir = setup_tools(tmp_path, stall)
    checkpoint = None
    if stall == 'checkpoint':
        checkpoint = 'blocked.ckpt'
        os.mkfifo(tmp_path / checkpoint)
    (tmp_path / 'eval.py').write_text("from pathlib import Path\nPath('evaluator-started').touch()\n")
    job = dict(name='setup-regression', checkpoint=checkpoint, sha256='unused',
               episodes=1, eval_seed=42, method='test')
    harness = f'''
import importlib.util, json, time
from pathlib import Path
spec = importlib.util.spec_from_file_location('runner', {str(RUNNER)!r})
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
original_popen = runner.subprocess.Popen
def record_group(*args, **kwargs):
    child = original_popen(*args, **kwargs)
    if kwargs.get('start_new_session'):
        with Path('worker-groups.txt').open('a') as stream:
            stream.write(str(child.pid) + '\\n')
    return child
runner.subprocess.Popen = record_group
try:
    runner.run_job({job!r}, Path('.'), Path('output'), time.time() + .5)
except TimeoutError:
    print('deadline enforced', flush=True)
else:
    raise AssertionError('worker returned without enforcing its setup deadline')
'''
    process = subprocess.Popen(
        [sys.executable, '-c', harness], cwd=tmp_path,
        env={**os.environ, 'PATH': str(bindir) + os.pathsep + os.environ['PATH']},
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True,
    )
    try:
        try:
            output, _ = process.communicate(timeout=4)
        except subprocess.TimeoutExpired:
            pytest.fail('Worker setup outlived its deadline (independent 4s safety timeout)')
        assert process.returncode == 0, output
        assert 'deadline enforced' in output
        assert not (tmp_path / 'evaluator-started').exists()
        # A setup tool can spawn children. Neither it nor that child may keep running.
        for name in ('setup.pid', 'descendant.pid'):
            if (tmp_path / name).exists():
                pid = int((tmp_path / name).read_text())
                result = subprocess.run(['ps', '-o', 'stat=', '-p', str(pid)],
                                        capture_output=True, text=True)
                assert not result.stdout.strip() or result.stdout.strip().startswith('Z')
    finally:
        groups = tmp_path / 'worker-groups.txt'
        if groups.exists():
            for group in groups.read_text().splitlines():
                try:
                    os.killpg(int(group), signal.SIGKILL)
                except ProcessLookupError:
                    pass
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def test_worker_returns_unchanged_success_record(tmp_path, monkeypatch):
    bindir = setup_tools(tmp_path)
    (tmp_path / 'eval.py').write_text(
        'import json, sys\nfrom pathlib import Path\n'
        "args = dict(arg.lstrip('+').split('=', 1) for arg in sys.argv[1:] if '=' in arg)\n"
        "Path(args['output.metadata_path']).write_text(json.dumps({'metrics': {'episode_successes': [True]}}))\n"
        "Path(args['output.filename']).write_text('test outcome')\n"
        "print('evaluator completed', flush=True)\n"
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('PATH', str(bindir) + os.pathsep + os.environ['PATH'])
    job = dict(name='success', checkpoint=None, episodes=1, eval_seed=42, method='random')
    record = runner.run_job(job, tmp_path, tmp_path / 'output', time.time() + 10)
    assert record == json.loads((tmp_path / 'output/runtime.json').read_text())
    assert record['job'] == job and record['returncode'] == 0
    assert record['protocol_sha256'] == runner.sha256(runner.PROTOCOL)
    assert 'seed=42' in record['command'] and '+eval.random_action_seed=42' in record['command']
    assert 'evaluator completed' in (tmp_path / 'output/eval.log').read_text()


def test_worker_rejects_expired_deadline_before_spawning(tmp_path, monkeypatch):
    def unexpected_spawn(*args, **kwargs):
        pytest.fail('An expired worker must not launch setup or an evaluator')

    monkeypatch.setattr(runner.subprocess, 'Popen', unexpected_spawn)
    with pytest.raises(TimeoutError):
        runner.run_job(runner.jobs()[-1], tmp_path, tmp_path / 'expired',
                       deadline_unix=time.time() - 1)


def test_worker_deadline_stops_real_evaluator_subprocess(tmp_path, monkeypatch):
    # A real CPU sleep exercises launch, log forwarding and process-group cleanup.
    bindir = setup_tools(tmp_path)
    (tmp_path / 'eval.py').write_text("import time\nprint('evaluator started', flush=True)\ntime.sleep(60)\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('PATH', str(bindir) + os.pathsep + os.environ['PATH'])
    original_popen = runner.subprocess.Popen
    children = []

    def record_spawn(*args, **kwargs):
        child = original_popen(*args, **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(runner.subprocess, 'Popen', record_spawn)
    started = time.monotonic()
    with pytest.raises(TimeoutError):
        runner.run_job(runner.jobs()[-1], tmp_path, tmp_path / 'bounded',
                       deadline_unix=time.time() + 1)
    assert time.monotonic() - started < 10
    assert len(children) == 1 and children[0].poll() is not None
    assert 'evaluator started' in (tmp_path / 'bounded/eval.log').read_text()


@pytest.mark.parametrize('failure', ['constructor', 'start'])
def test_guard_initialization_failure_reaps_spawned_worker(tmp_path, monkeypatch, failure):
    original_popen = runner.subprocess.Popen
    children = []

    def record_spawn(*args, **kwargs):
        child = original_popen(*args, **kwargs)
        children.append(child)
        return child

    class BrokenTimer:
        def __init__(self, *args):
            if failure == 'constructor':
                raise RuntimeError('guard initialization failed')

        def start(self):
            raise RuntimeError('guard initialization failed')

        def cancel(self):
            pass

    monkeypatch.setattr(runner.subprocess, 'Popen', record_spawn)
    monkeypatch.setattr(runner.threading, 'Timer', BrokenTimer)
    with pytest.raises(RuntimeError, match='guard initialization failed'):
        runner.run_job(runner.jobs()[-1], tmp_path, tmp_path / 'unused', time.time() + 10)
    assert len(children) == 1 and children[0].poll() is not None
