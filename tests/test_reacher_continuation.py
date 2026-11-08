"""Isolated handoff integration; science/audits and global inventory are stubbed."""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
import continue_reacher as code


def wait_for(predicate, timeout=10):
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline
        time.sleep(.01)


@pytest.fixture
def study(tmp_path, request):
    root = tmp_path
    active = getattr(request, 'param', '') == 'active'
    if active: (root/'with-science').touch()
    for name in ('records', 'logs', 'cache', 'bin'):
        (root / name).mkdir()
    for name in ('hostname', 'uptime', 'df', 'free', 'nvidia-smi', 'docker', 'tmux', 'ps'):
        path = root / 'bin' / name
        path.write_text('#!/bin/sh\ncase "$*" in *uuid*) case "$*" in *compute*) ;; *) echo TEST-GPU ;; esac ;; esac\n')
        path.chmod(0o755)
    worker = root / 'worker.py'
    worker.write_text(f'''import sys,os,time,json,fcntl,threading,subprocess
from pathlib import Path
sys.path.insert(0, {str(SCRIPTS)!r})
import continue_reacher as c
root=Path(__file__).resolve().parent

def record_pid():
    path=root/f'pid-{{os.getpid()}}.tmp'
    path.write_text(json.dumps(c.parallel.identity(os.getpid())))
    path.replace(path.with_suffix('.json'))
record_pid()
if sys.argv[1]=='legacy':
    while True: time.sleep(1)
if sys.argv[1]=='science':
    # This scratch scientific child must exit naturally, without any signal.
    signal=__import__('signal')
    signal.signal(signal.SIGTERM,lambda *_: (root/'science-signalled').touch())
    (root/'science-ready').touch()
    while not (root/'finish-science').exists(): time.sleep(.01)
    (root/'science-exit').touch()
    sys.exit(0)
if sys.argv[1]=='incumbent':
    with (root/'records/owner.lock').open('a') as owner:
        fcntl.flock(owner,fcntl.LOCK_EX)
        c.parallel.reserve(root/'records/owner.json',{{'process':c.parallel.identity(os.getpid())}})
        def lane(gpu):
            if gpu==0 and (root/'with-science').exists():
                with (root/'gpu0.lock').open('a') as lock:
                    fcntl.flock(lock,fcntl.LOCK_EX)
                    subprocess.run([sys.executable,str(__file__),'science'],check=True)
            while not (root/'allow-boundary').exists(): time.sleep(.01)
            name=f'cell{{1-gpu}}'
            c.parallel.reserve(root/f'records/{{name}}-eval-claim.json',{{'gpu':gpu}})
            (root/f'waiting-{{gpu}}').write_text('waiting')
            with (root/f'gpu{{gpu}}.lock').open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX)
                (root/f'incumbent-advanced-{{gpu}}').touch()
                while True: time.sleep(1)
        threads=[threading.Thread(target=lane,args=(gpu,)) for gpu in (0,1)]
        for t in threads:t.start()
        for t in threads:t.join()
    sys.exit(0)

# Test-specific wrapper, never a production switch. Inventory reads only these
# scratch processes, not any workstation experiment or resource.
c.__file__=str(__file__)
def inventory():
    rows={{}}
    for path in root.glob('pid-*.json'):
        saved=json.loads(path.read_text())
        try:
            row=c.parallel.identity(saved['pid'])
            if row['starttime']!=saved['starttime']:continue
            fields=Path(f'/proc/{{row["pid"]}}/stat').read_text().rsplit(')',1)[1].split()
            row.update(pgid=int(fields[2]),sid=int(fields[3]))
            rows[row['pid']]=row
        except (FileNotFoundError,ProcessLookupError):pass
    return rows
c.inventory=inventory
c.gpu_idle=lambda:None
def init(self,request):
    self.request_path=Path(request);self.request_hash=c.sha256(request)
    self.request=json.loads(self.request_path.read_text())
    self.root=root;self.records=root/'records';self.out=self.records
    self.logs=root/'logs';self.cache=root/'cache';self.generation=self.records/'continuation'
    self.cfg=self.request['cfg'];self.manifest=self.request['manifest'];self.data={{'tasks':[], 'optimizer_steps':1}}
    self.old=json.loads((self.records/'owner.json').read_text())['process']
    self.inherited_claims={{}};self.guard=threading.RLock();self.failed=threading.Event();self.pending=[];self.states={{}}
c.Continuation.__init__=init
c.Continuation.check=lambda self:None
c.Continuation.budget=lambda self:0
c.Continuation.audit_training=lambda self,job:{{'audited':job['label']}}
c.audit_run=lambda job,*a:{{'label':job['label']}}
c.summarize=lambda rows:{{'labels':[r['label'] for r in rows]}}
original_event=c.event
def event(folder,name,**values):
    original_event(folder,name,**values)
    fault=root/'fault'
    if fault.exists() and fault.read_text()==name:
        (root/f'paused-{{name}}').touch()
        while not (root/f'release-{{name}}').exists():time.sleep(.01)
c.event=event
# Reuse real claim handling; scientific commands themselves are cooperative CPU stubs.
def run_stage(self,job,stage,gpu):
    self.claim_stage(job,stage,gpu)
    with (root/f'gpu{{gpu}}.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        c.parallel.reserve(self.records/f'{{job["label"]}}-{{stage}}-launch.json',{{'gpu':gpu}})
        c.parallel.reserve(self.out/f'{{job["label"]}}-{{stage}}-exit.json',{{'exit':0}})
c.Continuation.run_stage=run_stage
c.main()
''')
    env = dict(os.environ, PATH=str(root/'bin'))
    processes = []
    def start(*args):
        log = (root/f'process-{len(processes)}.log').open('w')
        proc = subprocess.Popen([sys.executable, str(worker), *args], stdout=log,
                                stderr=subprocess.STDOUT, env=env)
        log.close()
        processes.append(proc)
        return proc
    # Hold both test locks before incumbent claims its waiting stages.
    held = [(root/f'gpu{gpu}.lock').open('a') for gpu in (0,1)]
    for gpu, lock in enumerate(held):
        if not (active and gpu == 0): fcntl.flock(lock,fcntl.LOCK_EX)
    legacy = start('legacy')
    incumbent = start('incumbent')
    wait_for(lambda: (root/'records/owner.json').exists())
    if active: wait_for(lambda: (root/'science-ready').exists())
    jobs = [dict(label=f'cell{i}',subdir=f'cell{i}',train=['stub'],eval=['stub']) for i in range(6)]
    for i in (0,1):
        code.parallel.reserve(root/f'records/cell{i}-train-claim.json',{'adopted':{}} if i == 0 else {'gpu':0})
        code.parallel.reserve(root/f'records/cell{i}-train-exit.json',{'exit':0})
    state = {'blocked':False,'jobs':{f'cell{i}':{'status':'waiting_lock','stage':'eval','gpu':1-i} for i in (0,1)}}
    (root/'records/parallel-status.json').write_text(json.dumps(state))
    (root/'records/sensitivity-gate.json').write_text(json.dumps({'status':'verified_success','trainings':15,'evaluations':15,'sha256':{}}))
    dead = dict(pid=99999999,starttime='0',ppid=0,cwd='',cmdline=[])
    science = next((json.loads(p.read_text()) for p in root.glob('pid-*.json')
                    if json.loads(p.read_text())['cmdline'][-1] == 'science'), dead)
    request = dict(cfg={'python':sys.executable,'serial_script':str(worker),
        'locks':{str(i):str(root/f'gpu{i}.lock') for i in (0,1)},
        'adopt':dict(parent=code.parallel.identity(legacy.pid),trainer=science,wrapper=dead,flock=dead)},
        manifest={'jobs':jobs,'configs':{j['label']:{} for j in jobs},'environment':{}})
    (root/'request.json').write_text(json.dumps(request))
    original = {str(p):p.read_bytes() for p in (root/'records').glob('*') if p.is_file() and p.name != 'parallel-status.json'}
    try:
        yield root, start, held, incumbent, legacy, original
    finally:
        for lock in held: lock.close()
        for path in root.glob('paused-*'):
            (root/path.name.replace('paused-','release-',1)).touch()
        (root/'finish-science').touch()
        # Only exact recorded scratch instances are cleaned up; never a process group.
        for path in root.glob('pid-*.json'):
            saved=json.loads(path.read_text())
            try:
                fd=code.open_pidfd(saved['pid'])
                if code.alive(saved,{saved['pid']:code.parallel.identity(saved['pid'])}):
                    code.signal_pidfd(fd,signal.SIGKILL)
                os.close(fd)
            except (ProcessLookupError,FileNotFoundError):pass
        for proc in processes:
            proc.wait(timeout=5)


def launch_controller(study, fault=None):
    root,start,held,*_=study
    if fault: (root/'fault').write_text(fault)
    proc=start('control',str(root/'request.json'))
    # Wait until this controller is queued on the first scratch flock before release.
    wait_for(lambda: (root/'records/continuation/attempt.json').exists())
    for lock in held: lock.close()
    if not (root/'with-science').exists() and not (root/'all-done').exists() and fault != 'fenced-0':
        wait_for(lambda: (root/'records/continuation/fenced-1.json').exists())
        (root/'allow-boundary').touch()
    return proc


def test_takeover(study):
    root,start,held,incumbent,legacy,original=study
    proc=launch_controller(study)
    wait_for(lambda:(root/'records/continuation/complete.json').exists(),20)
    assert proc.wait(timeout=5)==0
    assert incumbent.wait(timeout=5)==-signal.SIGTERM
    assert legacy.wait(timeout=5)==-signal.SIGTERM
    assert len(list((root/'records').glob('*-eval-exit.json')))==6
    cert=json.loads((root/'records/continuation/certificate.json').read_text())
    assert cert['frontier']['lanes']['1']['job']['label']=='cell0'
    assert cert['frontier']['lanes']['0']['job']['label']=='cell1'
    assert [j['label'] for j in cert['frontier']['pending']]==['cell2','cell3','cell4','cell5']
    for path,data in original.items():
        assert Path(path).read_bytes()==data


def assert_locked(root, expected):
    for gpu in (0,1):
        with (root/f'gpu{gpu}.lock').open('a') as lock:
            if expected:
                with pytest.raises(BlockingIOError):
                    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            else:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)


@pytest.mark.parametrize('phase', ['fenced-0','guarded','certificate','retired-0','retired',
                                    'owner-acquired','ready','released'])
def test_controller_crash_never_replays_or_releases_before_ready(study, phase):
    root,start,held,incumbent,legacy,original=study
    proc=launch_controller(study,phase)
    wait_for(lambda:(root/f'paused-{phase}').exists())
    proc.kill()
    proc.wait(timeout=5)
    (root/f'release-{phase}').touch()
    gen=root/'records/continuation'
    if phase != 'released':
        wait_for(lambda:(gen/'barrier-blocked.json').exists())
        assert_locked(root,True)
    else:
        assert_locked(root,False)
    assert not (gen/'dispatch-started.json').exists()
    assert not list((root/'records').glob('*-launch.json'))
    for path,data in original.items(): assert Path(path).read_bytes()==data
    retry=start('control',str(root/'request.json'))
    assert retry.wait(timeout=5)!=0
    assert not (gen/'dispatch-started.json').exists()


@pytest.mark.parametrize('evidence', ['launch','empty-log','output'])
def test_ambiguous_attempt_blocks_without_retirement(study, evidence):
    root,start,held,incumbent,legacy,original=study
    if evidence=='launch': (root/'records/cell0-eval-launch.json').write_text(json.dumps({'process':{'pid':99999999,'starttime':'0'}}))
    elif evidence=='empty-log': (root/'logs/cell0-eval.log').touch()
    else: (root/'records/cell0.txt').write_text('partial output')
    proc=launch_controller(study)
    assert proc.wait(timeout=5)!=0
    gen=root/'records/continuation'
    wait_for(lambda:(gen/'barrier-blocked.json').exists())
    assert_locked(root,True)
    assert not (gen/'retirement-started.json').exists()
    assert incumbent.poll() is None and legacy.poll() is None


@pytest.mark.parametrize('actor', ['parent','owner'])
def test_stale_identity_never_receives_signal(study, actor):
    root,start,held,incumbent,legacy,original=study
    path=root/('request.json' if actor=='parent' else 'records/owner.json')
    value=json.loads(path.read_text())
    saved=value['cfg']['adopt']['parent'] if actor=='parent' else value['process']
    saved['starttime']=str(int(saved['starttime'])+1)
    path.write_text(json.dumps(value))
    proc=launch_controller(study)
    assert proc.wait(timeout=5)!=0
    gen=root/'records/continuation'
    wait_for(lambda:(gen/'barrier-blocked.json').exists())
    assert_locked(root,True)
    assert not (gen/'retirement-started.json').exists()
    assert incumbent.poll() is None and legacy.poll() is None


@pytest.mark.parametrize('study', ['active'], indirect=True)
def test_current_science_exits_naturally_before_fence_and_retirement(study):
    root,start,held,incumbent,legacy,original=study
    proc=launch_controller(study)
    assert not (root/'science-exit').exists()
    assert not (root/'records/continuation/fenced-0.json').exists()
    assert not (root/'science-signalled').exists()
    (root/'finish-science').touch()
    wait_for(lambda:(root/'science-exit').exists())
    wait_for(lambda:(root/'records/continuation/fenced-1.json').exists())
    (root/'allow-boundary').touch()
    wait_for(lambda:(root/'records/continuation/complete.json').exists(),20)
    assert proc.wait(timeout=5)==0
    assert not (root/'science-signalled').exists()


def test_all_complete_audits_without_successor_or_retirement(study):
    root,start,held,incumbent,legacy,original=study
    for i in range(6):
        for stage in ('train','eval'):
            for suffix,value in [('claim',{'gpu':1-i if i<2 else i%2}),('exit',{'exit':0})]:
                path=root/f'records/cell{i}-{stage}-{suffix}.json'
                if not path.exists():code.parallel.reserve(path,value)
        code.parallel.reserve(root/f'records/cell{i}-audit.json',{'label':f'cell{i}'})
    (root/'all-done').touch()
    proc=launch_controller(study)
    assert proc.wait(timeout=5)==0
    gen=root/'records/continuation'
    assert (gen/'all-complete.json').exists()
    assert (gen/'barrier-owner.json').exists()
    assert not (gen/'owner-acquired.json').exists()
    assert not (gen/'retirement-started.json').exists()
    assert incumbent.poll() is None and legacy.poll() is None
    assert_locked(root,False)


def test_unclassified_legacy_process_blocks_retirement(study):
    root,start,held,incumbent,legacy,original=study
    extra=start('legacy')
    wait_for(lambda:(root/f'pid-{extra.pid}.json').exists())
    proc=launch_controller(study)
    assert proc.wait(timeout=5)!=0
    wait_for(lambda:(root/'records/continuation/barrier-blocked.json').exists())
    assert_locked(root,True)
    assert extra.poll() is None and incumbent.poll() is None and legacy.poll() is None
    assert not (root/'records/continuation/retirement-started.json').exists()


def test_training_audit_preserves_existing_receipt(tmp_path):
    d=code.parallel.Dispatcher.__new__(code.parallel.Dispatcher)
    d.cache=tmp_path/'cache';d.records=tmp_path/'records';d.records.mkdir()
    job={'label':'cell','subdir':'cell','model':'model'}
    folder=d.cache/'cell';folder.mkdir(parents=True)
    (folder/'config.yaml').write_text('wm:\n  action_dim: 2\n  observation_dim: 6\n')
    (folder/'run_metadata.json').write_text('{"resume_checkpoint":null}')
    (folder/'events.jsonl').write_text('\n'.join(json.dumps(e) for e in
        [dict(event='train_epoch_end',epoch=i) for i in range(10)]+[dict(event='fit_end',global_step=1)]))
    with (folder/'model_epoch_10_object.ckpt').open('wb') as f:f.truncate(1_000_001)
    d.manifest={'configs':{'cell':{'train':{'wm':{}}}}};d.data={'optimizer_steps':1}
    d.audit_training(job)
    path=d.records/'cell-training-audit.json';original=path.read_bytes();stamp=path.stat().st_mtime_ns
    d.audit_training(job)
    assert path.read_bytes()==original and path.stat().st_mtime_ns==stamp


def test_request_handoff_owner_and_publication_hashes_remain_pinned(tmp_path, monkeypatch):
    root = tmp_path
    (root / 'progress').mkdir()
    out = root / 'out'
    (out / 'dispatch').mkdir(parents=True)
    dataset = root / 'dataset'
    dataset.write_bytes(b'synthetic')
    serial = root / 'serial.py'
    serial.write_text('# synthetic bridge\n')
    manifest = root / 'manifest.json'
    manifest.write_text(json.dumps({'jobs': [], 'source_sha256': {}}))
    (out / 'data-audit.json').write_text('{}')
    (out / 'dataset-ready.json').write_text(json.dumps({'path': str(dataset), 'stat': {
        'size': dataset.stat().st_size, 'mtime_ns': dataset.stat().st_mtime_ns}}))
    (out / 'runtime.json').write_text(json.dumps({'python': sys.version, 'packages': {}}))
    owner = out / 'dispatch/owner.json'
    owner.write_text('{"process": {"pid": 99999999, "starttime": "0"}}')
    handoff = root / 'handoff.json'
    handoff.write_text(json.dumps({'root': str(root), 'cache': str(root), 'results': str(out),
        'logs': str(root), 'manifest': str(manifest), 'serial_script': str(serial),
        'serial_script_sha256': code.sha256(serial), 'sha256': {str(manifest): code.sha256(manifest)}}))
    request = root / 'request.json'
    paths = [code.__file__, code.parallel.__file__, sys.modules['reacher_schedule'].__file__]
    publication = {str(Path(path).resolve()): code.sha256(path) for path in paths}
    extra = root / 'extra-reviewed-file'
    extra.write_text('pinned')
    publication[str(extra)] = code.sha256(extra)
    request.write_text(json.dumps({'handoff': str(handoff), 'handoff_sha256': code.sha256(handoff),
        'owner_sha256': code.sha256(owner), 'code_sha256': publication}))
    # Exercise the actual scientific/runtime hash checks without resource polling.
    monkeypatch.setattr(code.Continuation, 'budget', lambda self: 0)
    d = code.Continuation(request)
    d.check()
    for path in (request, handoff, owner, manifest, extra):
        original = path.read_bytes()
        path.write_bytes(original + b' ')
        with pytest.raises(AssertionError):
            d.check()
        path.write_bytes(original)
    d.check()
