"""Frozen E4 commands and evidence gates; no scores control dispatch."""
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import threading
import time

from experiments.e6.runner import sha256

PROTOCOL = Path(__file__).with_name('protocol.json')
P = json.loads(PROTOCOL.read_text())


def jobs():
    return [dict(name=f'{arm}_s{seed}', arm=arm, seed=seed, weight=weight)
            for seed in P['seeds'] for arm, weight in P['arms'].items()]


def commands(job, output, cache):
    subdir = str(output.relative_to(cache) / 'train')
    train = ['train.py', '--config-name', 'lewm_masked', 'data=reacher',
             f'seed={job["seed"]}', f'subdir={subdir}',
             'trainer.max_epochs=10', 'runtime.stop_after_epoch=null',
             'loader.batch_size=128', f'num_workers={P["loader_workers"]}', 'loader.prefetch_factor=2',
             'loss.masked.inverse_weight=1.0', 'loss.sigreg.weight=0.0',
             f'+loss.variance_reg.weight={job["weight"]}', '+loss.variance_reg.gamma=0.1',
             'resume.mode=never', 'early_stopping.enabled=false',
             'checkpoint.enabled=false', 'checkpoint.object_epoch_interval=10',
             'wandb.enabled=false', 'logging.progress_refresh_steps=100',
             f'+runtime.audit_path={output / "finiteness.json"}']
    evaluate = ['eval.py', '--config-name', 'reacher',
                f'policy={subdir}/lewm_masked_epoch_10', 'seed=42',
                'eval.num_eval=200', 'eval.env_batch_size=10',
                'eval.goal_offset_steps=25', 'eval.eval_budget=50',
                'plan_config.horizon=5', 'plan_config.receding_horizon=5',
                'plan_config.action_block=5', 'solver.num_samples=300',
                'solver.n_steps=30', 'solver.topk=30', 'solver.batch_size=1',
                'output.save_video=false', 'output.save_planning_artifacts=false',
                f'output.filename={output / "outcomes.txt"}',
                f'+output.metadata_path={output / "tasks.json"}']
    return dict(train=train, eval=evaluate)


def source_hashes():
    paths = list(Path('.').glob('*.py')) + list(Path('config').rglob('*.yaml'))
    paths += list(Path('experiments/e4').glob('*.py')) + [PROTOCOL, Path('experiments/e6/runner.py')]
    return {str(p): sha256(p) for p in sorted(paths)}


def preflight(cache, output):
    import h5py
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    path = cache / P['dataset']
    if path.stat().st_size != P['dataset_bytes'] or sha256(path) != P['dataset_sha256']:
        raise ValueError('Canonical Reacher dataset identity mismatch')
    with h5py.File(path) as f:
        schema = {k: dict(shape=list(v.shape), dtype=str(v.dtype)) for k,v in f.items()}
    assert schema == P['dataset_schema']
    print('E4 DATA IDENTITY/SCHEMA OK', flush=True)
    OmegaConf.register_new_resolver('eval', eval, replace=True)
    configs = {}
    for job in jobs():
        configs[job['name']] = {}
        for stage, cmd in commands(job, output / job['name'], cache).items():
            with initialize_config_dir(config_dir=str(Path('config',stage).resolve()), version_base=None):
                cfg = compose(config_name=cmd[2], overrides=cmd[3:])
            configs[job['name']][stage] = OmegaConf.to_container(cfg, resolve=True)
    return dict(configs=configs, dataset_sha256=P['dataset_sha256'],
                source_sha256=source_hashes())


def run_command(command, output, stage, deadline, commit=lambda: None):
    with (output / f'{stage}.log').open('w') as log:
        process = subprocess.Popen(['python', *command], stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, start_new_session=True)
        stopped = threading.Event()
        reason = []
        def guard():
            last_commit = time.time()
            while not stopped.wait(5):
                try:
                    if time.time()-last_commit >= 60:
                        commit()
                        last_commit = time.time()
                    size = sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
                    if time.time() >= deadline or size > P['output_bytes_per_cell'] - 16*1024**2:
                        reason.append('deadline/output cap')
                except Exception as error:
                    reason.append(repr(error))
                if reason:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    return
        thread = threading.Thread(target=guard, daemon=True)
        thread.start()
        try:
            for line in process.stdout:
                print(line, end='', flush=True)
                log.write(line)
                log.flush()
            code = process.wait()
            if code or reason:
                raise RuntimeError(f'{stage} exit={code}: {reason}')
        finally:
            stopped.set()
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            thread.join()


def audit_training(output):
    import torch
    from omegaconf import OmegaConf
    frozen=json.loads((output.parent/'frozen.json').read_text())
    actual=OmegaConf.to_container(OmegaConf.load(output/'train/config.yaml'),resolve=True)
    actual['wm'].pop('action_dim'); actual['wm'].pop('observation_dim')
    assert actual==frozen['configs'][output.name]['train'], 'Training config changed'
    events = [json.loads(x) for x in (output/'train/events.jsonl').read_text().splitlines()]
    ends = [e for e in events if e['event']=='fit_end']
    assert len(ends)==1 and ends[0]['global_step']==P['optimizer_steps']
    assert [e['epoch'] for e in events if e['event']=='train_epoch_end']==list(range(10))
    assert json.loads((output/'train/run_metadata.json').read_text())['resume_checkpoint'] is None
    finite = json.loads((output/'finiteness.json').read_text())
    assert not finite['nonfinite'] and finite['backward_checks']==P['optimizer_steps']
    checkpoint = output/'train/lewm_masked_epoch_10_object.ckpt'
    assert list((output/'train').glob('*_object.ckpt')) == [checkpoint]
    model = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert all(bool(torch.isfinite(v).all()) for v in model.state_dict().values())
    return dict(checkpoint_sha256=sha256(checkpoint), finiteness=finite,
                initialization=json.loads((output/'initialization.json').read_text()))


def audit_tasks(output, cache):
    data = json.loads((output/'tasks.json').read_text())
    frozen=json.loads((output.parent/'frozen.json').read_text())
    expected=frozen['configs'][output.name]['eval']
    expected['solver']['device']='cuda'
    assert data['config']==expected, 'Evaluation config changed'
    raw = data['metrics']['episode_successes']
    assert len(raw)==200 and all(type(v) is bool for v in raw)
    assert data['start_rows']==P['expected_start_rows']
    chunks = data['consumed_chunks']
    assert len(chunks)==200 and all('id' in c and 'qpos' in c for c in chunks)
    # Canonical id identifies a trajectory, not a row. Check the full endpoint.
    import h5py
    import numpy as np
    with h5py.File(cache/P['dataset']) as f:
        for endpoint, rows in [('start',data['start_rows']),('goal',data['goal_rows'])]:
            for key in ('id','ep_idx','step_idx','qpos'):
                assert np.array_equal(f[key][rows], [c[key][endpoint] for c in chunks]), (key,endpoint)
    assert [c['step_idx']['start'] for c in chunks] == data['start_steps']
    assert [c['ep_idx']['start'] for c in chunks] == data['episode_ids']
    return dict(successes=sum(raw), percent=sum(raw)/2, outcomes=raw,
                tasks={k:data[k] for k in ('start_rows','episode_ids','start_steps','consumed_chunks')})


def run_job(job, cache, output, deadline, commit):
    if time.time() >= deadline:
        raise TimeoutError('Late E4 worker refused')
    # Existing output means an interrupted or duplicate call. Never restart it.
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    record = dict(job=job, started_unix=started, task_id=os.getenv('MODAL_TASK_ID'),
                  source_sha256=source_hashes(), python=platform.python_version(),
                  protocol_sha256=sha256(PROTOCOL), dataset_sha256=P['dataset_sha256'])
    for name, cmd in [('packages.txt',['python','-m','pip','freeze']),('gpu.txt',['nvidia-smi'])]:
        (output/name).write_text(subprocess.check_output(cmd,text=True))
    record['packages_sha256'] = sha256(output/'packages.txt')
    import stable_pretraining
    import stable_worldmodel
    record['runtime_source_sha256'] = {
        package.__name__+'/'+str(p.relative_to(Path(package.__file__).parent)):sha256(p)
        for package in (stable_pretraining,stable_worldmodel)
        for p in Path(package.__file__).parent.rglob('*.py')}
    record['runtime_source_sha256'] = dict(sorted(record['runtime_source_sha256'].items()))
    record['gpu_name'] = subprocess.check_output(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],text=True).strip()
    if 'A100' not in record['gpu_name'] or '40GB' not in record['gpu_name']:
        raise RuntimeError('Wrong GPU allocation')
    cmd = commands(job,output,cache)
    record['commands'] = cmd
    def save():
        (output/'runtime.json').write_text(json.dumps(record,indent=2)+'\n')
        commit()
    save()
    try:
        record['state']='training'; save()
        run_command(cmd['train'],output,'train',deadline,commit)
        record['training'] = audit_training(output)
        record['state']='evaluating'; save()
        run_command(cmd['eval'],output,'eval',deadline,commit)
        record['evaluation'] = audit_tasks(output,cache)
        record['state']='complete'
    except BaseException as error:
        record.update(state='failed', error=repr(error))
        raise
    finally:
        record['elapsed_seconds']=time.time()-started
        record['artifact_sha256']={str(p.relative_to(output)):sha256(p)
            for p in output.rglob('*') if p.is_file() and p.name!='runtime.json'}
        save()
    return record


def check_pair(records):
    a,b=records
    for key in ('source_sha256','runtime_source_sha256','packages_sha256','gpu_name'):
        assert a[key]==b[key], f'Runtime pair mismatch: {key}'
    assert a['training']['initialization']==b['training']['initialization'], 'Unmatched initialization/split/RNG'
    assert a['evaluation']['tasks']==b['evaluation']['tasks'], 'Unmatched consumed tasks'


def project(records, prior_gpu_seconds=0):
    spent = sum(r['elapsed_seconds'] for r in records)
    return prior_gpu_seconds + P['shutdown_reserve_seconds'] + spent + (6-len(records))*max(
        (r['elapsed_seconds'] for r in records), default=0)*1.25
