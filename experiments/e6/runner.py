"""Frozen E6 job specification and runtime provenance."""
import hashlib
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import threading
import time

PROTOCOL = Path(__file__).with_name('protocol.json')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def jobs():
    protocol = json.loads(PROTOCOL.read_text())
    result = []
    for seed in protocol['evaluation_seeds']:
        for model in protocol['models']:
            result.append(dict(model, eval_seed=seed, episodes=protocol['model_episodes'],
                               name=f"{model['method']}_train{model['training_seed']}_eval{seed}"))
    result.append(dict(name='random_eval42_n200', eval_seed=42, episodes=200,
                       method='random', checkpoint=None))
    return result


def verify_assets(cache):
    protocol = json.loads(PROTOCOL.read_text())
    assets = [(protocol['dataset'], protocol['dataset_sha256'])]
    for model in protocol['models']:
        assets.extend([(model['checkpoint'], model['sha256']),
                       (str(Path(model['checkpoint']).parent / 'config.yaml'),
                        model['training_config_sha256'])])
    for path, expected in assets:
        actual = sha256(cache / path)
        if actual != expected:
            raise ValueError(f'Asset identity mismatch: {path}: {actual} != {expected}')
        print(f'IDENTITY OK {path} {actual}', flush=True)
    return dict(assets)


def _run_job(job, cache, output, deadline_unix):
    if time.time() >= deadline_unix:
        raise TimeoutError('E6 attempt deadline expired before evaluation')
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = job['checkpoint']
    if checkpoint and sha256(cache / checkpoint) != job['sha256']:
        raise ValueError(f'Checkpoint identity mismatch: {checkpoint}')
    policy = checkpoint.removesuffix('_object.ckpt') if checkpoint else 'random'
    command = ['python', 'eval.py', '--config-name=scene', f'policy={policy}',
               f"seed={job['eval_seed']}", f"eval.num_eval={job['episodes']}",
               'eval.env_batch_size=10', 'eval.goal_offset_steps=25', 'eval.eval_budget=50',
               'plan_config.horizon=5', 'plan_config.receding_horizon=5',
               'plan_config.action_block=5', 'solver.num_samples=300', 'solver.n_steps=30',
               'solver.topk=30', 'solver.batch_size=1',
               'output.save_video=false', 'output.save_planning_artifacts=false',
               f'output.filename={output / "outcomes.txt"}',
               f'+output.metadata_path={output / "tasks.json"}']
    if not checkpoint:
        command.append('+eval.random_action_seed=42')
    record = dict(job=job, command=command, protocol_sha256=sha256(PROTOCOL),
                  python=platform.python_version(), platform=platform.platform(),
                  started_unix=time.time(),
                  environment={k: os.environ[k] for k in (
                      'MODAL_TASK_ID', 'MODAL_CLOUD_PROVIDER', 'CUDA_VISIBLE_DEVICES',
                      'MUJOCO_GL', 'STABLEWM_HOME', 'OMP_NUM_THREADS') if k in os.environ},
                  source_sha256={p: sha256(p) for p in ('eval.py', 'jepa.py', 'module.py', 'modal_app.py',
                                                        'experiments/e6/runner.py') if Path(p).is_file()})
    for name, cmd in [('packages.txt', ['python', '-m', 'pip', 'freeze']),
                      ('gpu.txt', ['nvidia-smi'])]:
        result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (output / name).write_text(result.stdout)
    (output / 'runtime.json').write_text(json.dumps(record, indent=2) + '\n')
    print('E6 EVALUATING', job['name'], command, flush=True)
    if time.time() >= deadline_unix:
        raise TimeoutError('E6 attempt deadline expired during setup')
    # The child shares stdout with Modal; retain an independent raw log as well.
    with (output / 'eval.log').open('w') as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True,
                                   env={**os.environ, 'PYTHONUNBUFFERED': '1'})
        def stop_evaluator():
            try:
                process.kill()
            except ProcessLookupError:
                pass
        try:
            for line in process.stdout:
                print(line, end='', flush=True)
                log.write(line)
                log.flush()
            record['returncode'] = process.wait()
        except BaseException:
            stop_evaluator()
            process.wait()
            raise
    record['elapsed_seconds'] = time.time() - record['started_unix']
    (output / 'runtime.json').write_text(json.dumps(record, indent=2) + '\n')
    if record['returncode']:
        raise RuntimeError(f"{job['name']} exited {record['returncode']}")
    result = json.loads((output / 'tasks.json').read_text())
    assert len(result['metrics']['episode_successes']) == job['episodes']
    return record


def run_job(job, cache, output, deadline_unix):
    """Bound local setup and evaluation together, including blocking file/tool I/O.

    Modal startup, volume reload/commit and provider teardown remain outside this
    local process boundary. This does not change the attempt deadline or budget.
    """
    if time.time() >= deadline_unix:
        raise TimeoutError('E6 attempt deadline expired before worker setup')
    command = [sys.executable, os.path.abspath(__file__), '--worker',
               json.dumps(job), os.path.abspath(cache), os.path.abspath(output), str(deadline_unix)]
    expired = threading.Event()

    def stop_worker():
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    def expire():
        expired.set()
        stop_worker()

    timer = None
    record = None
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, start_new_session=True,
                               env={**os.environ, 'PYTHONUNBUFFERED': '1'})
    try:
        timer = threading.Timer(max(0, deadline_unix - time.time()), expire)
        timer.daemon = True
        timer.start()
        for line in process.stdout:
            if line.startswith('E6_WORKER_RESULT '):
                record = json.loads(line.removeprefix('E6_WORKER_RESULT '))
            else:
                print(line, end='', flush=True)
        returncode = process.wait()
    finally:
        if timer is not None:
            timer.cancel()
        # Also reap/terminate on caller cancellation or a setup/evaluation failure.
        stop_worker()
        process.wait()
        process.stdout.close()
    if expired.is_set() or (returncode and time.time() >= deadline_unix):
        raise TimeoutError('E6 attempt deadline terminated worker setup/evaluation process group')
    if returncode:
        raise RuntimeError(f"{job['name']} worker exited {returncode}")
    if record is None or record.get('job') != job or record.get('returncode') != 0:
        raise RuntimeError(f"{job['name']} worker returned no valid completion record")
    return record


if __name__ == '__main__':
    if len(sys.argv) != 6 or sys.argv[1] != '--worker':
        raise SystemExit('Internal E6 worker entrypoint requires job, cache, output and deadline')
    result = _run_job(json.loads(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]), float(sys.argv[5]))
    print('E6_WORKER_RESULT ' + json.dumps(result), flush=True)
