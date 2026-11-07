"""Frozen E1 commands and complete-result audit; see docs/reacher-schedule.md."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import statistics

import yaml
from omegaconf import OmegaConf

SEEDS = (3072, 1, 2)
METHODS = {'sigreg': 'lewm', 'action_nce': 'lewm_masked_action_nce'}


def jobs(results):
    runs = []
    for seed in SEEDS:
        for method, model in METHODS.items():
            label = f'e1_reacher_{method}_s{seed}'
            subdir = f'rebuttal/{label}'
            runs.append(dict(label=label, method=method, seed=seed, model=model,
                subdir=subdir, train=['train.py', '--config-name', model,
                    'data=reacher', f'seed={seed}', f'subdir={subdir}',
                    'trainer.max_epochs=100', 'runtime.stop_after_epoch=10',
                    'early_stopping.enabled=false', 'checkpoint.enabled=false',
                    'checkpoint.object_epoch_interval=10', 'wandb.enabled=false',
                    'num_workers=2', 'loader.prefetch_factor=2',
                    'logging.progress_refresh_steps=100'],
                eval=['eval.py', '--config-name', 'reacher',
                    f'policy={subdir}/{model}_epoch_10',
                    'eval.num_eval=200', 'eval.env_batch_size=10', 'seed=42',
                    'output.save_video=false',
                    f'output.filename={Path(results).resolve() / (label + ".txt")}']))
    return runs


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def array(text, key, boolean=False):
    match = re.search(r"'" + key + r"': array\(\[(.*?)\]", text, re.S)
    if not match:
        raise ValueError(f'Missing {key}')
    tokens = [v.strip() for v in match[1].replace('\n', '').split(',') if v.strip()]
    if boolean:
        assert all(v in ('True', 'False') for v in tokens)
        return [v == 'True' for v in tokens]
    return [int(v) for v in tokens]


def audit_run(job, cache, results, frozen, expected_tasks, expected_steps):
    folder = Path(cache) / job['subdir']
    cfg = yaml.safe_load((folder / 'config.yaml').read_text())
    normalized = copy.deepcopy(cfg)
    # train.py infers only these two dimensions after opening the dataset.
    for key in ('action_dim', 'observation_dim'):
        normalized['wm'].pop(key)
    assert normalized == frozen['train'], f'Training config mismatch: {job["label"]}'
    metadata = json.loads((folder / 'run_metadata.json').read_text())
    assert metadata['resume_checkpoint'] is None
    events = [json.loads(v) for v in (folder / 'events.jsonl').read_text().splitlines()]
    ends = [v for v in events if v['event'] == 'fit_end']
    epochs = [v['epoch'] for v in events if v['event'] == 'train_epoch_end']
    assert len(ends) == 1 and epochs == list(range(10))
    assert ends[0]['global_step'] == expected_steps
    checkpoint = folder / f'{job["model"]}_epoch_10_object.ckpt'
    assert checkpoint.stat().st_size > 1_000_000
    assert list(folder.glob('*_object.ckpt')) == [checkpoint]
    path = Path(results) / (job['label'] + '.txt')
    text = path.read_text()
    assert text.count('==== CONFIG ====') == 1
    ecfg = yaml.safe_load(text.split('==== CONFIG ====')[1].split('==== RESULTS ====')[0])
    # eval.py resolves the device but writes other interpolations as YAML.
    expected_eval = copy.deepcopy(frozen['eval'])
    expected_eval['solver']['device'] = 'cuda'
    ecfg = OmegaConf.to_container(OmegaConf.create(ecfg), resolve=True)
    assert ecfg == expected_eval, f'Evaluation config mismatch: {job["label"]}'
    tasks = {k: array(text, k) for k in ('start_row_indices', 'start_episode_ids', 'start_steps')}
    assert all(len(v) == 200 for v in tasks.values())
    assert len(set(tasks['start_row_indices'])) == 200
    assert tasks == expected_tasks, f'Task identity mismatch: {job["label"]}'
    outcomes = array(text, 'episode_successes', boolean=True)
    assert len(outcomes) == 200
    rate = float(re.search(r"'success_rate': ([\d.]+)", text).group(1))
    assert abs(rate - sum(outcomes) / 2) < 1e-7
    starts = [v for v in events if v['event'] == 'fit_start']
    assert len(starts) == 1
    return dict(label=job['label'], method=job['method'], seed=job['seed'],
                successes=sum(outcomes), n=200, percent=rate, outcomes=outcomes,
                optimizer_steps=expected_steps, checkpoint_sha256=sha256(checkpoint),
                training_seconds=ends[0]['time'] - starts[0]['time'],
                evaluation_seconds=float(re.search(r'evaluation_time: ([\d.]+)', text).group(1)),
                files={str(p): sha256(p) for p in [path, folder / 'config.yaml',
                    folder / 'events.jsonl', folder / 'run_metadata.json']})


def summarize(runs):
    assert {(r['method'], r['seed']) for r in runs} == {(m, s) for m in METHODS for s in SEEDS}
    assert len(runs) == 6
    summary = {m: dict(mean=statistics.mean(r['percent'] for r in runs if r['method'] == m),
                sample_sd=statistics.stdev(r['percent'] for r in runs if r['method'] == m))
               for m in METHODS}
    pairs = []
    for seed in SEEDS:
        a = next(r for r in runs if r['method'] == 'action_nce' and r['seed'] == seed)
        b = next(r for r in runs if r['method'] == 'sigreg' and r['seed'] == seed)
        pairs.append(dict(seed=seed, delta_pp=a['percent'] - b['percent'],
            nce_only=sum(x and not y for x, y in zip(a['outcomes'], b['outcomes'], strict=True)),
            sigreg_only=sum(y and not x for x, y in zip(a['outcomes'], b['outcomes'], strict=True))))
    return dict(runs=runs, summary=summary, paired_outcomes=pairs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--audit', type=Path, help='Frozen manifest JSON (otherwise print commands)')
    parser.add_argument('--cache', type=Path)
    parser.add_argument('--data-audit', type=Path)
    args = parser.parse_args()
    if args.audit:
        manifest = json.loads(args.audit.read_text())
        data = json.loads(args.data_audit.read_text())
        output = summarize([audit_run(j, args.cache, args.results,
            manifest['configs'][j['label']], data['tasks'], data['optimizer_steps'])
            for j in manifest['jobs']])
    else:
        output = jobs(args.results)
    print(json.dumps(output, indent=2))
