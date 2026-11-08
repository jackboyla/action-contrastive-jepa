"""Run two predeclared evaluation commands under one exclusive GPU lock.

The JSON spec supplies lock, cwd, env, jobs (argv/log), and report paths.
Commands and environment are passed unchanged. No retries or process killing.
"""
import argparse
import fcntl
import json
import re
from pathlib import Path
import subprocess
import time


def resources():
    line = subprocess.check_output([
        'nvidia-smi', '--id=0', '--query-gpu=memory.used,memory.free',
        '--format=csv,noheader,nounits'], text=True).strip()
    used, free = map(int, line.split(','))
    ram = next(int(line.split()[1]) // 1024 for line in
               Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))
    return {'gpu_used_mib': used, 'gpu_free_mib': free, 'ram_available_mib': ram}


def headroom(sample):
    # 8 GiB is a conservative admission allowance, not a measured peak bound.
    return sample['gpu_free_mib'] >= 8192 + 4096 and sample['ram_available_mib'] >= 16384


def placement(pid):
    table = subprocess.check_output(['nvidia-smi'], text=True)
    rows = [line for line in table.splitlines() if re.search(rf'\s{pid}\s', line)]
    verified = any(re.search(rf'\|\s+0\s+N/A\s+N/A\s+{pid}\s+C\+G\s', line) for line in rows)
    return {'gpu0_compute_and_graphics': verified, 'rows': rows}


def execute(spec, sample=resources, locate=placement, interval=2, stagger=30):
    assert len(spec['jobs']) == 2
    report = Path(spec['report'])
    assert not report.exists(), 'Existing trial receipt requires explicit audit; never retry automatically'
    state = {'status': 'waiting_lock', 'jobs': [], 'commands': [job['argv'] for job in spec['jobs']],
             'admission_allowance_mib': 8192, 'required_headroom_mib': 4096}
    report.parent.mkdir(parents=True, exist_ok=True)

    def save():
        tmp = report.with_suffix('.tmp')
        tmp.write_text(json.dumps(state, indent=2) + '\n')
        tmp.replace(report)

    save()
    processes, logs = [], []
    with open(spec['lock'], 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state.update(status='running', lock_acquired=time.time(), peak_gpu_used_mib=0,
                     headroom_risk=False, mode='serial')
        save()

        def start(index):
            job = spec['jobs'][index]
            # Existing nonempty logs or outputs cannot be silently overwritten.
            assert not Path(job['output']).exists(), job['output']
            logpath = Path(job['log'])
            assert not logpath.exists() or logpath.stat().st_size == 0, str(logpath)
            log = logpath.open('w'); logs.append(log)
            proc = subprocess.Popen(job['argv'], cwd=spec['cwd'], env=spec['env'],
                                    stdout=log, stderr=subprocess.STDOUT,
                                    pass_fds=(lock.fileno(),))
            processes.append(proc)
            state['jobs'].append({'pid': proc.pid, 'launched': time.time(), 'exit_code': None})
            save()

        def tick():
            now = time.time()
            for proc, receipt in zip(processes, state['jobs']):
                code = proc.poll()
                index = processes.index(proc)
                if code is None and 'placement' not in receipt and 'CEM solve time:' in Path(spec['jobs'][index]['log']).read_text():
                    receipt['placement'] = locate(proc.pid)
                if code is not None and receipt['exit_code'] is None:
                    receipt.update(exit_code=code, ended=now,
                                   elapsed_seconds=now-receipt['launched'])
            r = sample()
            state['peak_gpu_used_mib'] = max(state['peak_gpu_used_mib'], r['gpu_used_mib'])
            state['headroom_risk'] |= r['gpu_free_mib'] < 4096
            with report.with_suffix('.samples.jsonl').open('a') as stream:
                stream.write(json.dumps({'time': now, **r})+'\n')
            save()
            return r

        try:
            initial = tick(); state['preflight'] = initial
            start(0)
            warmed = False
            while processes[0].poll() is None:
                r = tick()
                elapsed = time.time()-state['jobs'][0]['launched']
                warmed = 'CEM solve time:' in Path(spec['jobs'][0]['log']).read_text()
                if elapsed >= stagger and warmed:
                    break
                if elapsed >= 120:
                    break
                time.sleep(interval)
            r = tick()
            if warmed and processes[0].poll() is None:
                state['jobs'][0]['placement'] = locate(processes[0].pid)
            parallel = (processes[0].poll() is None and warmed and headroom(initial)
                        and headroom(r) and not state['headroom_risk']
                        and state['jobs'][0].get('placement', {}).get('gpu0_compute_and_graphics', False))
            state['mode'] = 'parallel' if parallel else 'serial'
            save()
            if not parallel:
                while processes[0].poll() is None:
                    tick(); time.sleep(interval)
                tick()
                if processes[0].returncode:
                    raise RuntimeError('First evaluation failed; second not launched')
            start(1)
            while any(p.poll() is None for p in processes):
                tick(); time.sleep(interval)
            tick()
            state['status'] = 'complete' if all(p.returncode == 0 for p in processes) else 'failed'
        except BaseException as error:
            state.update(status='failed', error=repr(error))
            raise
        finally:
            # Even errors/interrupts must not release the lock while a sibling runs.
            for proc, receipt in zip(processes, state['jobs']):
                code = proc.wait()
                if receipt['exit_code'] is None:
                    end = time.time()
                    receipt.update(exit_code=code, ended=end, elapsed_seconds=end-receipt['launched'])
            for log in logs:
                log.close()
            state['ended'] = time.time()
            if state['jobs']:
                state['makespan_seconds'] = max(j['ended'] for j in state['jobs'])-state['jobs'][0]['launched']
            save()
    return state


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('spec', type=Path)
    args = parser.parse_args()
    result = execute(json.loads(args.spec.read_text()))
    raise SystemExit(0 if result['status'] == 'complete' else 1)
