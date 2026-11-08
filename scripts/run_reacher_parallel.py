"""Continue frozen Reacher cells on two local GPUs, adopting an existing training.

The handoff JSON binds the live serial process tree, frozen artifacts, and paths.
Install the legacy entrypoint adapter before starting this dispatcher. It releases
only the lock inherited from its own flock parent, then waits for stage receipts.
The already-running training and its wrapper never execute that adapter.
"""
import argparse
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

import yaml

from reacher_schedule import audit_run, sha256, summarize

TEMP_DIR = None


def write(path, value):
    path = Path(path)
    # Keep rename temporaries outside the legacy wrapper's live budget scan.
    with tempfile.NamedTemporaryFile(mode='w', dir=TEMP_DIR or path.parent,
                                     delete=False) as stream:
        stream.write(json.dumps(value, indent=2) + '\n')
        temporary = Path(stream.name)
    temporary.replace(path)


def identity(pid):
    path = Path('/proc') / str(pid)
    fields = (path / 'stat').read_text().rsplit(')', 1)[1].split()
    if fields[0] == 'Z':
        raise ProcessLookupError(pid)
    return dict(pid=pid, starttime=fields[19], ppid=int(fields[1]),
                cwd=str((path / 'cwd').resolve()),
                cmdline=(path / 'cmdline').read_bytes().decode().split('\0')[:-1])


def same_process(saved):
    try:
        return identity(saved['pid']) == saved
    except (FileNotFoundError, ProcessLookupError):
        return False


def reserve(path, value):
    """An attempt is permanent: absence of a success receipt never permits retry."""
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def release_inherited_lock(lock_path):
    # flock and its child share this open file description; do not reopen/unlock
    # the pathname, which would not release the parent's existing lock.
    matches = []
    for fd in Path('/proc/self/fd').iterdir():
        try:
            if fd.resolve() == Path(lock_path).resolve():
                matches.append(int(fd.name))
        except FileNotFoundError:
            pass
    assert len(matches) == 1, f'Expected one inherited GPU lock, got {matches}'
    fcntl.flock(matches[0], fcntl.LOCK_UN)


def bridge(handoff, label, stage):
    cfg = json.loads(Path(handoff).read_text())
    assert stage in ('train', 'eval')
    manifest = json.loads(Path(cfg['manifest']).read_text())
    assert label in {job['label'] for job in manifest['jobs']}
    assert same_process(cfg['adopt']['parent']), 'Legacy parent identity changed'
    parent = identity(os.getppid())
    assert parent['ppid'] == cfg['adopt']['parent']['pid']
    assert parent['cmdline'] == ['flock', cfg['locks']['1'], cfg['python'],
                                 cfg['serial_script'], label, stage]
    assert sha256(cfg['serial_script']) == cfg['serial_script_sha256']
    release_inherited_lock(cfg['locks']['1'])
    results = Path(cfg['results'])
    receipt = results / f'{label}-{stage}-exit.json'
    while True:
        if receipt.exists():
            assert json.loads(receipt.read_text())['exit'] == 0
            return
        failure = results / 'dispatch' / f'{label}-{stage}-failure.json'
        if failure.exists():
            raise RuntimeError(failure.read_text())
        owner_path = results / 'dispatch' / 'owner.json'
        assert owner_path.exists(), 'Parallel dispatcher never established ownership'
        assert same_process(json.loads(owner_path.read_text())['process']), 'Parallel owner lost'
        time.sleep(10)


def locked_child(lock_fd, command):
    """Hold the inherited lock until even orphaned, exec'd descendants exit."""
    import ctypes
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0) != 0:  # Linux PR_SET_CHILD_SUBREAPER
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    os.fstat(lock_fd)
    # A group cancellation reaches the scientific child too. Keep its lock
    # while reaping descendants; caught handlers reset to defaults on exec.
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, lambda *_: None)
    proc = subprocess.Popen(command, pass_fds=(lock_fd,))
    try:
        returncode = proc.wait()
    finally:
        while True:
            try:
                os.waitpid(-1, 0)
            except ChildProcessError:
                break
    return returncode if returncode >= 0 else 128 - returncode


class Dispatcher:
    def __init__(self, handoff):
        global TEMP_DIR
        self.handoff = Path(handoff)
        self.cfg = json.loads(self.handoff.read_text())
        self.root = Path(self.cfg['root'])
        TEMP_DIR = self.root / 'progress' / 'e1-dispatch-tmp'
        TEMP_DIR.mkdir(exist_ok=True)
        self.cache = Path(self.cfg['cache'])
        self.out = Path(self.cfg['results'])
        self.logs = Path(self.cfg['logs'])
        self.records = self.out / 'dispatch'
        self.records.mkdir(exist_ok=True)
        self.manifest = json.loads(Path(self.cfg['manifest']).read_text())
        self.data = json.loads((self.out / 'data-audit.json').read_text())
        self.guard = threading.RLock()
        self.failed = threading.Event()
        self.pending = list(self.manifest['jobs'][1:])
        self.states = {}

    def check(self):
        for path, expected in self.cfg['sha256'].items():
            assert sha256(path) == expected, f'Handoff artifact changed: {path}'
        for path, expected in self.manifest['source_sha256'].items():
            assert sha256(self.root / path) == expected, f'Frozen source changed: {path}'
        assert sha256(self.cfg['serial_script']) == self.cfg['serial_script_sha256']
        ready = json.loads((self.out / 'dataset-ready.json').read_text())
        stat = Path(ready['path']).stat()
        assert stat.st_size == ready['stat']['size'] and stat.st_mtime_ns == ready['stat']['mtime_ns']
        runtime = json.loads((self.out / 'runtime.json').read_text())
        assert sys.version == runtime['python']
        assert all(importlib.metadata.version(k) == v for k, v in runtime['packages'].items())
        self.budget()

    def budget(self):
        files = list(self.out.rglob('*')) + list(self.logs.glob('e1*'))
        for job in self.manifest['jobs']:
            files.extend((self.cache / job['subdir']).rglob('*'))
        size = 0
        for path in files:
            try:
                if path.is_file():
                    size += path.stat().st_size
            except FileNotFoundError:
                pass  # The legacy wrapper atomically replaces its own receipts.
        assert size < 6 * 2**30, f'Output budget exceeded: {size}'
        import shutil
        assert shutil.disk_usage(self.cache).free >= 10 * 2**30, 'Disk floor reached'
        return size

    def state(self, label, **details):
        with self.guard:
            self.states[label] = dict(utc=time.time(), **details)
            write(self.out / 'parallel-status.json', dict(
                utc=time.time(), owner_pid=os.getpid(), blocked=self.failed.is_set(),
                jobs=self.states, completed_trainings=len(list(self.out.glob('*-train-exit.json'))),
                completed_evaluations=len(list(self.out.glob('*-eval-exit.json')))))

    def audit_training(self, job):
        folder = self.cache / job['subdir']
        config = yaml.safe_load((folder / 'config.yaml').read_text())
        for key in ('action_dim', 'observation_dim'):
            config['wm'].pop(key)
        assert config == self.manifest['configs'][job['label']]['train']
        assert json.loads((folder / 'run_metadata.json').read_text())['resume_checkpoint'] is None
        events = [json.loads(v) for v in (folder / 'events.jsonl').read_text().splitlines()]
        assert [v['epoch'] for v in events if v['event'] == 'train_epoch_end'] == list(range(10))
        ends = [v for v in events if v['event'] == 'fit_end']
        assert len(ends) == 1 and ends[0]['global_step'] == self.data['optimizer_steps']
        checkpoint = folder / f'{job["model"]}_epoch_10_object.ckpt'
        assert list(folder.glob('*_object.ckpt')) == [checkpoint]
        assert checkpoint.stat().st_size > 1_000_000
        result = dict(optimizer_steps=ends[0]['global_step'], checkpoint_sha256=sha256(checkpoint))
        path = self.records / f'{job["label"]}-training-audit.json'
        if path.exists():
            assert json.loads(path.read_text()) == result
        else:
            write(path, result)
        return result

    def adopt(self, job):
        saved = self.cfg['adopt']
        for key in ('parent', 'flock', 'wrapper', 'trainer'):
            assert same_process(saved[key]), f'Cannot adopt changed {key}'
        assert saved['flock']['ppid'] == saved['parent']['pid']
        assert saved['wrapper']['ppid'] == saved['flock']['pid']
        assert saved['trainer']['ppid'] == saved['wrapper']['pid']
        assert saved['trainer']['cmdline'] == [self.cfg['python'], *job['train']]
        reserve(self.records / f'{job["label"]}-train-claim.json', dict(adopted=saved))
        receipt = self.out / f'{job["label"]}-train-exit.json'
        while not receipt.exists():
            assert same_process(saved['wrapper']), 'Adopted wrapper lost without success receipt'
            self.budget()
            self.state(job['label'], stage='train', gpu=1, adopted=True, pid=saved['trainer']['pid'])
            time.sleep(15)
        assert json.loads(receipt.read_text())['exit'] == 0
        self.audit_training(job)

    def claim_stage(self, job, stage, gpu):
        reserve(self.records / f'{job["label"]}-{stage}-claim.json', dict(gpu=gpu, utc=time.time()))

    def run_stage(self, job, stage, gpu):
        self.check()
        label = job['label']
        self.claim_stage(job, stage, gpu)
        self.state(label, stage=stage, gpu=gpu, status='waiting_lock')
        with Path(self.cfg['locks'][str(gpu)]).open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            self.budget()
            uuid = subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-gpu=uuid',
                                            '--format=csv,noheader'], text=True).strip()
            compute = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                               '--format=csv,noheader'], text=True)
            assert not any(line.startswith(uuid) for line in compute.splitlines()), 'GPU has existing compute'
            health = {}
            for command in [['hostname'], ['uptime'], ['df', '-h', str(self.cache)],
                            ['free', '-h'], ['nvidia-smi'], ['docker', 'ps'], ['tmux', 'ls'],
                            ['ps', '-eo', 'pid,ppid,pgid,stat,etime,%cpu,%mem,cmd']]:
                result = subprocess.run(command, capture_output=True, text=True)
                health[' '.join(command)] = dict(exit=result.returncode,
                                                output=result.stdout + result.stderr)
            write(self.records / f'{label}-{stage}-health.json', health)
            env = dict(os.environ, **self.manifest['environment'],
                       STABLEWM_HOME=str(self.cache), PYTHONPATH=str(self.root))
            env.update(CUDA_VISIBLE_DEVICES=str(gpu), MUJOCO_EGL_DEVICE_ID=str(gpu))
            locked = [self.cfg['python'], str(Path(__file__).resolve()),
                      '--lock-fd', str(lock.fileno()), '--']
            if stage == 'train':
                assert not (self.cache / job['subdir']).exists(), 'Refuse training restart'
            else:
                self.audit_training(job)
                subprocess.run(locked + [self.cfg['python'], str(Path(__file__).resolve()), '--egl', str(gpu)],
                               env=env, cwd=self.root, pass_fds=(lock.fileno(),),
                               start_new_session=True, check=True)
            command = [self.cfg['python'], *job[stage]]
            started = time.time()
            with (self.logs / f'{label}-{stage}.log').open('x') as log:
                proc = None
                try:
                    proc = subprocess.Popen(locked + command, cwd=self.root, env=env, stdout=log,
                                            stderr=subprocess.STDOUT, start_new_session=True,
                                            pass_fds=(lock.fileno(),))
                    write(self.records / f'{label}-{stage}-launch.json', dict(
                        process=identity(proc.pid), gpu=gpu, uuid=uuid, command=command,
                        environment={k: env[k] for k in self.manifest['environment']}, utc=started))
                    while proc.poll() is None:
                        self.budget()
                        self.state(label, stage=stage, gpu=gpu, status='running', pid=proc.pid)
                        smi = subprocess.check_output(['nvidia-smi'], text=True)
                        members = {str(proc.pid)}
                        for row in subprocess.check_output(['ps', '-eo', 'pid,pgid'], text=True).splitlines()[1:]:
                            pid, pgid = row.split()
                            if pgid == str(proc.pid):
                                members.add(pid)
                        for line in smi.splitlines():
                            row = line.split()
                            if len(row) > 5 and row[4] in members and row[5] in ('C', 'G', 'C+G'):
                                assert row[1] == str(gpu), f'Wrong GPU: {line}'
                        with (self.logs / f'e1-placement-gpu{gpu}.log').open('a') as placement:
                            placement.write(f'{time.time()} {label} {stage}\n{smi}\n')
                        time.sleep(15)
                except BaseException:
                    # This is only the child started above, never the adopted tree.
                    if proc is not None:
                        if proc.poll() is None:
                            os.killpg(proc.pid, signal.SIGTERM)
                        proc.wait()
                    raise
            assert proc.returncode == 0, f'{label} {stage} exit {proc.returncode}'
            if stage == 'train':
                self.audit_training(job)
            write(self.out / f'{label}-{stage}-exit.json', dict(
                exit=0, seconds=time.time() - started, gpu=gpu))

    def lane(self, gpu, adopted=None):
        job, stage = adopted, 'train'
        try:
            while True:
                if job is None:
                    with self.guard:
                        if self.failed.is_set() or not self.pending:
                            return
                        job = self.pending.pop(0)
                stage = 'train'
                if adopted is job:
                    self.adopt(job)
                else:
                    self.run_stage(job, stage, gpu)
                stage = 'eval'
                self.run_stage(job, stage, gpu)
                audit = audit_run(job, self.cache, self.out, self.manifest['configs'][job['label']],
                                  self.data['tasks'], self.data['optimizer_steps'])
                write(self.out / f'{job["label"]}-audit.json', audit)
                self.state(job['label'], stage='complete', gpu=gpu)
                job = None
        except BaseException as error:
            self.failed.set()
            label = job['label'] if job else f'gpu{gpu}'
            write(self.records / f'{label}-{stage}-failure.json', dict(error=repr(error), utc=time.time()))
            self.state(label, stage=stage, gpu=gpu, status='blocked', error=repr(error))
            raise

    def run(self):
        with (self.records / 'owner.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            assert not (self.records / 'owner.json').exists(), 'Existing owner: explicit recovery required'
            self.check()
            gate = json.loads((self.out / 'sensitivity-gate.json').read_text())
            assert gate['status'] == 'verified_success' and gate['trainings'] == gate['evaluations'] == 15
            assert all(sha256(p) == h for p, h in gate['sha256'].items())
            for saved in self.cfg['adopt'].values():
                assert same_process(saved), 'Adoption identity changed before ownership transfer'
            reserve(self.records / 'owner.json', dict(process=identity(os.getpid()),
                                                      handoff_sha256=sha256(self.handoff), utc=time.time()))
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [executor.submit(self.lane, 1, self.manifest['jobs'][0]),
                           executor.submit(self.lane, 0)]
                for future in futures:
                    future.result()
            results = [json.loads((self.out / f'{j["label"]}-audit.json').read_text())
                       for j in self.manifest['jobs']]
            write(self.out / 'summary.json', summarize(results))
            self.state('dispatcher', stage='complete')


def egl(gpu):
    import mujoco
    model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><light pos="0 0 3"/>'
                                          '<geom type="sphere" size=".1"/></worldbody></mujoco>')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    with mujoco.Renderer(model, height=64, width=64) as renderer:
        renderer.update_scene(data)
        pixels = renderer.render()
        assert pixels.max() > 0
        smi = subprocess.check_output(['nvidia-smi'], text=True)
        rows = [line.split() for line in smi.splitlines() if len(line.split()) > 5
                and line.split()[4] == str(os.getpid())]
        assert rows and all(row[1] == str(gpu) for row in rows), smi
        print(json.dumps(dict(egl_gpu=gpu, pid=os.getpid(), pixel_sum=int(pixels.sum()))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--handoff', type=Path)
    parser.add_argument('--egl', type=int)
    parser.add_argument('--lock-fd', type=int)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.lock_fd is not None:
        assert args.command[0] == '--'
        raise SystemExit(locked_child(args.lock_fd, args.command[1:]))
    if args.egl is not None:
        egl(args.egl)
    else:
        assert args.handoff is not None
        Dispatcher(args.handoff).run()
