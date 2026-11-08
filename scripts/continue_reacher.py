"""One reviewed E1 boundary handoff; no retry/recovery or scientific overrides."""
import argparse
import ctypes
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import fcntl
import json
import os
from pathlib import Path
import select
import signal
import socket
import subprocess
import sys
import time

import run_reacher_parallel as parallel
from reacher_schedule import audit_run, sha256, summarize


def open_pidfd(pid):
    # The shared Python build omits os.pidfd_open; glibc exposes the same API.
    libc = ctypes.CDLL(None, use_errno=True)
    fd = libc.pidfd_open(ctypes.c_int(pid), ctypes.c_uint(0))
    if fd < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return fd


def signal_pidfd(fd, sig):
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.pidfd_send_signal(ctypes.c_int(fd), ctypes.c_int(sig), None, ctypes.c_uint(0)) < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))


class Waiting(Exception):
    """The incumbent has not reached both pre-launch lock boundaries yet."""


def event(folder, name, **values):
    parallel.reserve(Path(folder) / f'{name}.json', dict(utc=time.time(), **values))


def send(channel, value):
    channel.sendall(json.dumps(value).encode() + b'\n')


def receive(channel):
    data = bytearray()
    while not data.endswith(b'\n'):
        part = channel.recv(1)
        if not part:
            raise EOFError('Handoff channel closed')
        data.extend(part)
        assert len(data) <= 65536
    return json.loads(data)


def inventory():
    """Linux process inventory, used only during an explicitly invoked handoff."""
    rows = {}
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        try:
            row = parallel.identity(int(path.name))
            fields = (path / 'stat').read_text().rsplit(')', 1)[1].split()
            row.update(pgid=int(fields[2]), sid=int(fields[3]))
            rows[row['pid']] = row
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            pass
    return rows


def descendants(pid, rows):
    found, pending = set(), [pid]
    while pending:
        parent = pending.pop()
        children = {p for p, row in rows.items() if row['ppid'] == parent} - found
        found.update(children)
        pending.extend(children)
    return found


def alive(saved, rows):
    row = rows.get(saved['pid'])
    return row is not None and row['starttime'] == saved['starttime']


def holds(pid, path):
    expected = Path(path).stat()
    for fd in Path(f'/proc/{pid}/fd').iterdir():
        try:
            stat = fd.stat()
            info = Path(f'/proc/{pid}/fdinfo/{fd.name}').read_text()
            if (stat.st_dev, stat.st_ino) == (expected.st_dev, expected.st_ino) and 'FLOCK  ADVISORY  WRITE' in info:
                return True
        except FileNotFoundError:
            pass
    return False


def blocked_gpus(pid, locks):
    # Kernel waiter records establish the blocked flock without ptrace access.
    wanted = {}
    for gpu, path in locks.items():
        stat = Path(path).stat()
        wanted[(os.major(stat.st_dev), os.minor(stat.st_dev), stat.st_ino)] = int(gpu)
    found = set()
    for line in Path('/proc/locks').read_text().splitlines():
        row = line.split()
        if len(row) < 7 or row[1:5] != ['->', 'FLOCK', 'ADVISORY', 'WRITE'] or int(row[5]) != pid:
            continue
        major, minor, inode = row[6].split(':')
        key = (int(major, 16), int(minor, 16), int(inode))
        if key in wanted:
            found.add(wanted[key])
    return found


def gpu_idle():
    text = subprocess.check_output(['nvidia-smi'], text=True)
    for line in text.splitlines():
        row = line.split()
        assert not (len(row) > 5 and row[1] in ('0', '1') and row[5] in ('C', 'G', 'C+G')), text


def legacy_controls(d, rows):
    commands = [d.old['cmdline'], d.cfg['adopt']['parent']['cmdline']]
    for job in d.manifest['jobs']:
        for stage in ('train', 'eval'):
            bridge = [d.cfg['python'], d.cfg['serial_script'], job['label'], stage]
            commands.extend([bridge, ['flock', d.cfg['locks']['1'], *bridge]])
    return {pid for pid, row in rows.items() if row['cmdline'] in commands}


def snapshot(d):
    paths = [d.records / 'owner.json']
    paths += list(d.records.glob('*-claim.json')) + list(d.records.glob('*-training-audit.json'))
    paths += list(d.out.glob('*-exit.json')) + list(d.out.glob('*-audit.json'))
    return {str(p): sha256(p) for p in paths}


def unchanged(hashes):
    assert all(sha256(path) == expected for path, expected in hashes.items()), 'Receipt or code changed'


class Continuation(parallel.Dispatcher):
    def __init__(self, request):
        self.request_path = Path(request).resolve()
        self.request_hash = sha256(self.request_path)
        self.request = json.loads(self.request_path.read_text())
        assert sha256(self.request['handoff']) == self.request['handoff_sha256']
        super().__init__(self.request['handoff'])
        self.generation = self.records / 'continuation'
        self.old = json.loads((self.records / 'owner.json').read_text())['process']
        assert sha256(self.records / 'owner.json') == self.request['owner_sha256']
        self.inherited_claims = {}

    def check(self):
        assert sha256(self.request_path) == self.request_hash
        assert sha256(self.request['handoff']) == self.request['handoff_sha256']
        assert sha256(self.records / 'owner.json') == self.request['owner_sha256']
        unchanged(self.request['code_sha256'])
        for path in (__file__, parallel.__file__, sys.modules['reacher_schedule'].__file__):
            assert str(Path(path).resolve()) in self.request['code_sha256']
        super().check()

    def budget(self):
        size = super().budget()
        for name in ('lightning_logs', 'outputs'):
            size += sum(p.stat().st_size for p in (self.root / name).rglob('*') if p.is_file())
        assert size < 6 * 2**30, f'Output budget including side artifacts exceeded: {size}'
        return size

    def frontier(self, states, blocked):
        assert not list(self.records.glob('*-failure.json')), 'Failed attempt: no automatic recovery'
        lanes, pending, completed, claims = {}, [], [], {}
        untouched = False
        for job in self.manifest['jobs']:
            label = job['label']
            receipts = [self.out / f'{label}-{s}-exit.json' for s in ('train', 'eval')]
            for stage, receipt in zip(('train', 'eval'), receipts):
                if receipt.exists():
                    assert json.loads(receipt.read_text())['exit'] == 0
                    assert (self.records / f'{label}-{stage}-claim.json').exists()
            train_claim = self.records / f'{label}-train-claim.json'
            if not train_claim.exists():
                assert not any(receipt.exists() for receipt in receipts)
                assert not (self.cache / job['subdir']).exists()
                assert not list(self.records.glob(f'{label}-*'))
                assert not list(self.logs.glob(f'{label}-*'))
                assert not (self.out / f'{label}.txt').exists()
                untouched = True
                pending.append(job)
                continue
            assert not untouched, 'Claim order is not a manifest prefix'
            claim = json.loads(train_claim.read_text())
            gpu = 1 if 'adopted' in claim else claim['gpu']
            if receipts[0].exists():
                self.audit_training(job)
            if receipts[1].exists():
                assert receipts[0].exists()
                assert json.loads((self.records / f'{label}-eval-claim.json').read_text())['gpu'] == gpu
                result = audit_run(job, self.cache, self.out, self.manifest['configs'][label],
                                   self.data['tasks'], self.data['optimizer_steps'])
                path = self.out / f'{label}-audit.json'
                if not path.exists():
                    raise Waiting(f'{label} audit is not committed')
                assert json.loads(path.read_text()) == result
                completed.append(result)
                continue
            stage = 'eval' if receipts[0].exists() else 'train'
            path = self.records / f'{label}-{stage}-claim.json'
            if not path.exists():
                raise Waiting(f'{label} has not claimed {stage}')
            claim = json.loads(path.read_text())
            assert claim.get('gpu') == gpu, 'Cannot move an assigned lane'
            # Old code creates health/log/output evidence after the exact blocked flock.
            assert not (self.records / f'{label}-{stage}-health.json').exists(), 'Ambiguous post-lock attempt'
            assert not (self.records / f'{label}-{stage}-launch.json').exists(), 'Ambiguous launch'
            assert not (self.logs / f'{label}-{stage}.log').exists(), 'Ambiguous spawn log'
            assert not ((self.cache / job['subdir']).exists() if stage == 'train'
                        else (self.out / f'{label}.txt').exists()), 'Ambiguous scientific output'
            state = states.get(label, {})
            if not (state.get('status') == 'waiting_lock' and state.get('stage') == stage
                    and state.get('gpu') == gpu and gpu in blocked):
                raise Waiting(f'{label} not fenced at its pre-launch flock')
            assert gpu not in lanes
            lanes[gpu] = dict(job=job, stage=stage)
            claims[f'{label}-{stage}'] = sha256(path)
        if pending and set(lanes) != {0, 1}:
            raise Waiting('Both lanes must reach pre-launch flocks before pending work transfers')
        return dict(lanes=lanes, pending=pending, completed=completed, claims=claims)

    def idle_actors(self, rows, tracked):
        for key in ('trainer', 'wrapper', 'flock'):
            if alive(self.cfg['adopt'][key], rows):
                raise Waiting(f'Original {key} has not exited naturally')
        if any(alive(saved, rows) for saved in tracked.values()):
            raise Waiting('Scientific process/descendant still alive')
        assert parallel.same_process(self.old), 'Incumbent identity lost'
        assert holds(self.old['pid'], self.records / 'owner.lock')
        assert not descendants(self.old['pid'], rows), 'Incumbent has an unclassified child'
        parent = self.cfg['adopt']['parent']
        actors = []
        assert alive(parent, rows), 'Legacy parent identity lost; explicit recovery required'
        if alive(parent, rows):
            assert parallel.same_process(parent)
            actors.append(parent)
            family = descendants(parent['pid'], rows)
            for pid in sorted(family):
                row = rows[pid]
                cmd = row['cmdline']
                valid = False
                for job in self.manifest['jobs']:
                    for stage in ('train', 'eval'):
                        bridge = [self.cfg['python'], self.cfg['serial_script'], job['label'], stage]
                        if row['ppid'] == parent['pid'] and cmd == ['flock', self.cfg['locks']['1'], *bridge]:
                            valid = True
                        if cmd == bridge and row['ppid'] in family:
                            valid = rows[row['ppid']]['cmdline'] == ['flock', self.cfg['locks']['1'], *bridge]
                assert valid, f'Unclassified legacy descendant {pid}'
                actors.append(parallel.identity(pid))
        actors.append(self.old)
        assert legacy_controls(self, rows) == {actor['pid'] for actor in actors}, 'Unclassified legacy controller'
        return actors

    def claim_stage(self, job, stage, gpu):
        key = f'{job["label"]}-{stage}'
        if key not in self.inherited_claims:
            return super().claim_stage(job, stage, gpu)
        expected = self.inherited_claims.pop(key)
        path = self.records / f'{key}-claim.json'
        assert sha256(path) == expected and json.loads(path.read_text())['gpu'] == gpu
        assert not (self.out / f'{key}-exit.json').exists()
        assert not (self.records / f'{key}-health.json').exists()
        assert not (self.records / f'{key}-launch.json').exists()
        assert not (self.logs / f'{key}.log').exists()
        event(self.generation, f'{key}-adopted', original_claim_sha256=expected)

    def continue_lane(self, gpu, first):
        job, stage = (first['job'], first['stage']) if first else (None, 'train')
        try:
            while True:
                if job is None:
                    with self.guard:
                        if self.failed.is_set() or not self.pending:
                            return
                        job = self.pending.pop(0)
                    stage = 'train'
                if stage == 'train':
                    self.run_stage(job, 'train', gpu)
                stage = 'eval'
                self.run_stage(job, stage, gpu)
                result = audit_run(job, self.cache, self.out, self.manifest['configs'][job['label']],
                                   self.data['tasks'], self.data['optimizer_steps'])
                parallel.write(self.out / f'{job["label"]}-audit.json', result)
                self.state(job['label'], stage='complete', gpu=gpu)
                job = None
        except BaseException as error:
            self.failed.set()
            event(self.generation, f'gpu{gpu}-failure', label=job['label'] if job else None,
                  stage=stage, error=repr(error))
            self.state(job['label'] if job else f'gpu{gpu}', stage=stage, gpu=gpu,
                       status='blocked', error=repr(error))
            raise


def track_science(d, rows, tracked):
    roots = [d.cfg['adopt']['trainer']]
    roots += [json.loads(p.read_text())['process'] for p in d.records.glob('*-launch.json')]
    commands = [[d.cfg['python'], *job[stage]] for job in d.manifest['jobs'] for stage in ('train', 'eval')]
    root_ids = {r['pid'] for r in roots}
    for pid, row in rows.items():
        if row['cmdline'] in commands or row['sid'] in root_ids or row['pgid'] in root_ids:
            tracked[(pid, row['starttime'])] = row
    for root in roots:
        if alive(root, rows):
            for pid in {root['pid']} | descendants(root['pid'], rows):
                tracked[(pid, rows[pid]['starttime'])] = rows[pid]


def spawn_role(d, role, fds):
    parent, child = socket.socketpair()
    proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), role,
                             str(d.request_path), '--channel', str(child.fileno()),
                             '--locks', *map(str, fds)],
                            pass_fds=(*fds, child.fileno()), start_new_session=True)
    child.close()
    return proc, parent


def keeper(d, channel, locks):
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, lambda *_: None)
    tracked = {}
    released = False
    try:
        assert len(locks) == 2
        for gpu, fd in enumerate(locks):
            assert Path(f'/proc/self/fd/{fd}').samefile(d.cfg['locks'][str(gpu)])
        event(d.generation, 'barrier-owner', process=parallel.identity(os.getpid()))
        for gpu, fd in enumerate(locks):
            while True:
                track_science(d, inventory(), tracked)
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    d.budget()
                    time.sleep(.25)
            event(d.generation, f'fenced-{gpu}')
        send(channel, {'holding': True, 'tracked': list(tracked.values())})
        message = receive(channel)
        complete = d.generation / 'all-complete.json'
        if complete.exists() and message == {'complete': sha256(complete)}:
            d.check()
            state = json.loads((d.out / 'parallel-status.json').read_text())
            frontier = d.frontier(state['jobs'], set())
            rows = inventory()
            track_science(d, rows, tracked)
            assert len(frontier['completed']) == len(d.manifest['jobs'])
            assert not any(alive(saved, rows) for saved in tracked.values())
        else:
            ready = json.loads((d.generation / 'ready.json').read_text())
            assert message == {'release': sha256(d.generation / 'ready.json')}
            assert parallel.same_process(ready['process'])
            assert holds(ready['process']['pid'], d.records / 'owner.lock')
            assert ready['certificate_sha256'] == sha256(d.generation / 'certificate.json')
        for fd in locks:
            os.close(fd)
        released = True
        send(channel, {'released': True})
    except BaseException as error:
        if released:
            event(d.generation, 'barrier-released-controller-lost', error=repr(error))
            return
        try:
            event(d.generation, 'barrier-blocked', error=repr(error))
        finally:
            # Receipt storage may fail too; it must not release an acquired fence.
            while True:
                time.sleep(60)


def successor(d, channel):
    certificate = json.loads((d.generation / 'certificate.json').read_text())
    with (d.records / 'owner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        d.check()
        unchanged(certificate['receipt_sha256'])
        rows = inventory()
        assert not any(alive(actor, rows) for actor in certificate['actors'])
        assert not legacy_controls(d, rows), 'Legacy control process remains'
        event(d.generation, 'owner-acquired', process=parallel.identity(os.getpid()))
        states = {value['job']['label']: dict(status='waiting_lock', stage=value['stage'], gpu=int(gpu))
                  for gpu, value in certificate['frontier']['lanes'].items()}
        expected = d.frontier(states, {int(gpu) for gpu in certificate['frontier']['lanes']})
        assert json.loads(json.dumps(expected)) == certificate['frontier']
        d.inherited_claims = certificate['frontier']['claims'].copy()
        d.pending = certificate['frontier']['pending']
        event(d.generation, 'ready', process=parallel.identity(os.getpid()),
              certificate_sha256=sha256(d.generation / 'certificate.json'))
        send(channel, {'ready': sha256(d.generation / 'ready.json')})
        assert receive(channel) == {'go': sha256(d.generation / 'ready.json')}
        d.check()
        unchanged(certificate['receipt_sha256'])
        event(d.generation, 'dispatch-started')
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(d.continue_lane, gpu, certificate['frontier']['lanes'].get(str(gpu)))
                       for gpu in (0, 1)]
            for future in futures:
                future.result()
        results = [json.loads((d.out / f'{job["label"]}-audit.json').read_text()) for job in d.manifest['jobs']]
        parallel.write(d.out / 'summary.json', summarize(results))
        d.state('dispatcher', stage='complete')
        event(d.generation, 'complete')


def control(d):
    d.generation.mkdir()  # A crashed attempt cannot silently start over.
    d.check()
    gate = json.loads((d.out / 'sensitivity-gate.json').read_text())
    assert gate['status'] == 'verified_success' and gate['trainings'] == gate['evaluations'] == 15
    unchanged(gate['sha256'])
    event(d.generation, 'attempt', request_sha256=d.request_hash)
    with ExitStack() as stack:
        locks = [stack.enter_context(Path(d.cfg['locks'][str(gpu)]).open('a')) for gpu in (0, 1)]
        guard, guard_channel = spawn_role(d, 'keeper', [lock.fileno() for lock in locks])
        stack.callback(guard_channel.close)
        held = receive(guard_channel)
        assert held['holding'] is True
        tracked = {(row['pid'], row['starttime']): row for row in held['tracked']}
        event(d.generation, 'guarded', pid=guard.pid)
        state = json.loads((d.out / 'parallel-status.json').read_text())
        if all((d.out / f'{job["label"]}-eval-exit.json').exists() for job in d.manifest['jobs']):
            rows = inventory()
            track_science(d, rows, tracked)
            frontier = d.frontier(state['jobs'], set())
            assert not any(alive(saved, rows) for saved in tracked.values())
            event(d.generation, 'all-complete', summary=summarize(frontier['completed']))
            send(guard_channel, {'complete': sha256(d.generation / 'all-complete.json')})
            assert receive(guard_channel) == {'released': True}
            assert guard.wait(timeout=10) == 0
            return
        while True:
            d.check()
            rows = inventory()
            track_science(d, rows, tracked)
            state = json.loads((d.out / 'parallel-status.json').read_text())
            assert not state['blocked']
            try:
                assert parallel.same_process(d.old), 'Incumbent identity lost'
                blocked = blocked_gpus(d.old['pid'], d.cfg['locks'])
                frontier = d.frontier(state['jobs'], blocked)
                actors = d.idle_actors(rows, tracked)
                gpu_idle()
                break
            except Waiting:
                time.sleep(.25)
        certificate = dict(request_sha256=d.request_hash, actors=actors, frontier=frontier,
                           receipt_sha256=snapshot(d), scientific_instances=list(tracked.values()))
        event(d.generation, 'certificate', **certificate)
        # Open every pidfd before the first retirement; validate the exact saved identity.
        pidfds = []
        for actor in actors:
            fd = open_pidfd(actor['pid'])
            stack.callback(os.close, fd)
            assert parallel.same_process(actor), 'Stale retirement identity'
            pidfds.append((actor, fd))
        assert guard.poll() is None, 'Barrier keeper lost'
        assert d.idle_actors(inventory(), tracked) == actors
        unchanged(certificate['receipt_sha256'])
        event(d.generation, 'retirement-started')
        for index, (actor, fd) in enumerate(pidfds):
            if not select.select([fd], [], [], 0)[0]:
                signal_pidfd(fd, signal.SIGTERM)
            assert select.select([fd], [], [], 10)[0], 'Idle controller did not exit; no escalation'
            event(d.generation, f'retired-{index}', process=actor)
        assert not legacy_controls(d, inventory()), 'Legacy control process appeared during retirement'
        event(d.generation, 'retired')
        proc, channel = spawn_role(d, 'successor', [])
        stack.callback(channel.close)
        expected = receive(channel)
        ready = json.loads((d.generation / 'ready.json').read_text())
        assert expected == {'ready': sha256(d.generation / 'ready.json')}
        assert ready['process']['pid'] == proc.pid and parallel.same_process(ready['process'])
        assert holds(proc.pid, d.records / 'owner.lock')
        send(guard_channel, {'release': sha256(d.generation / 'ready.json')})
        assert receive(guard_channel) == {'released': True}
        assert guard.wait(timeout=10) == 0
        for lock in locks:
            lock.close()
        event(d.generation, 'released')
        send(channel, {'go': sha256(d.generation / 'ready.json')})
        event(d.generation, 'transferred', successor=ready['process'])


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('role', choices=('control', 'keeper', 'successor'))
    parser.add_argument('request', type=Path)
    parser.add_argument('--channel', type=int)
    parser.add_argument('--locks', type=int, nargs='*', default=[])
    args = parser.parse_args()
    d = Continuation(args.request)
    try:
        if args.role == 'control':
            control(d)
        else:
            with socket.socket(fileno=args.channel) as channel:
                if args.role == 'keeper':
                    keeper(d, channel, args.locks)
                else:
                    successor(d, channel)
    except BaseException as error:
        event(d.generation, f'{args.role}-failed', error=repr(error))
        raise


if __name__ == '__main__':
    main()
