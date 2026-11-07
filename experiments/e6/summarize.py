"""Audit raw E6 tasks and summarize each axis without pooling shared episodes."""
import argparse
import json
from pathlib import Path
import statistics

from experiments.e6.runner import jobs, PROTOCOL, sha256


def summarize(root):
    rows, matched = [], {}
    for job in jobs():
        directory = root / job['name']
        if not (directory / 'tasks.json').exists():
            rows.append(dict(job=job['name'], state='missing'))
            continue
        runtime = json.loads((directory / 'runtime.json').read_text())
        data = json.loads((directory / 'tasks.json').read_text())
        assert runtime['returncode'] == 0, job['name']
        assert runtime['job'] == job
        assert runtime['protocol_sha256'] == sha256(PROTOCOL)
        config = data['config']
        assert config['seed'] == job['eval_seed']
        assert config['eval']['num_eval'] == job['episodes']
        for key, value in [('env_batch_size', 10), ('goal_offset_steps', 25), ('eval_budget', 50)]:
            assert config['eval'][key] == value
        for key, value in [('num_samples', 300), ('n_steps', 30), ('topk', 30), ('batch_size', 1)]:
            assert config['solver'][key] == value
        assert config['plan_config'] == dict(horizon=5, receding_horizon=5, action_block=5)
        policy = job['checkpoint'].removesuffix('_object.ckpt') if job['checkpoint'] else 'random'
        assert config['policy'] == policy
        successes = data['metrics']['episode_successes']
        assert len(successes) == job['episodes'] and all(type(x) is bool for x in successes)
        assert len(data['start_rows']) == job['episodes']
        assert len(set(data['start_rows'])) == job['episodes']
        assert data['goal_rows'] == [x + 25 for x in data['start_rows']]
        ids = [data[key] for key in ('start_rows', 'goal_rows', 'episode_ids', 'start_steps')]
        if job['checkpoint']:
            assert matched.setdefault(job['eval_seed'], ids) == ids
        else:
            assert config['eval']['random_action_seed'] == 42
        rows.append(dict(job=job['name'], state='complete', method=job['method'],
                         training_seed=job.get('training_seed'), evaluation_seed=job['eval_seed'],
                         successes=sum(successes), episodes=job['episodes'],
                         percent=100 * sum(successes) / job['episodes'],
                         evaluation_seconds=data['evaluation_seconds']))
    groups = {}
    for row in rows:
        if row['state'] != 'complete' or row['method'] == 'random':
            continue
        for axis, key in [('across_training_seeds', 'evaluation_seed'),
                          ('across_evaluation_seeds', 'training_seed')]:
            group = f"{axis}/{row['method']}/{row[key]}"
            groups.setdefault(group, []).append(row['percent'])
    variability = {key: dict(n=len(values), mean=statistics.mean(values),
                             sample_sd=statistics.stdev(values) if len(values) > 1 else None)
                   for key, values in groups.items()}
    return dict(complete=sum(r['state'] == 'complete' for r in rows), expected=28,
                rows=rows, variability=variability,
                note='Descriptive seed variability; shared tasks are not independent training repetitions.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.root), indent=2))
