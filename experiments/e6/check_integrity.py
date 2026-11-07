"""Verify the published E6 evidence and summaries without GPUs or model data."""
import hashlib
import json
import math
from pathlib import Path
import re

from experiments.e6.runner import jobs, PROTOCOL, sha256
from experiments.e6.summarize import summarize


def check(root):
    manifest = json.loads((root / 'manifest.json').read_text())
    protocol = json.loads(PROTOCOL.read_text())
    assert manifest['final'] and manifest['complete'] == manifest['expected'] == 28
    assert manifest['protocol_sha256'] == sha256(PROTOCOL)
    assert [cell['job'] for cell in manifest['cells']] == jobs()
    assert len({cell['call_id'] for cell in manifest['cells']}) == 28
    assert {p.parent.name for p in root.glob('*/tasks.json')} == {j['name'] for j in jobs()}
    seconds, ends, events = [], [], []
    for cell in manifest['cells']:
        name = cell['job']['name']
        assert cell['state'] == 'complete', name
        for file in cell['files'].values():
            assert sha256(root / file['path']) == file['sha256'], file['path']
        runtime = json.loads((root / name / 'runtime.json').read_text())
        data = json.loads((root / name / 'tasks.json').read_text())
        text = (root / name / 'outcomes.txt').read_text()
        match = re.search(r"'episode_successes': array\(\[(.*?)\]", text, re.S)
        assert match, name
        outcomes = [v == 'True' for v in re.findall(r'True|False', match[1])]
        assert outcomes == data['metrics']['episode_successes'], name
        rate = 100 * sum(outcomes) / len(outcomes)
        assert math.isclose(float(re.search(r"'success_rate': ([0-9.]+)", text)[1]), rate)
        assert math.isclose(data['metrics']['success_rate'], rate)
        for metric, key in [('start_row_indices', 'start_rows'),
                            ('start_episode_ids', 'episode_ids'), ('start_steps', 'start_steps')]:
            assert data['metrics'][metric] == data[key], (name, key)
        ids = {key: data[key] for key in ('start_rows', 'goal_rows', 'episode_ids', 'start_steps')}
        digest = hashlib.sha256(json.dumps(ids, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        key = f"{cell['job']['eval_seed']}_{cell['job']['episodes']}"
        assert digest == manifest['independent_task_ids_sha256'][key], name
        assert runtime['environment']['MODAL_TASK_ID'] == cell['worker_task_id']
        assert runtime['source_sha256'] == manifest['sources'][cell['source_commit']]
        assert runtime['elapsed_seconds'] == cell['worker_seconds']
        assert data['evaluation_seconds'] <= runtime['elapsed_seconds'] <= protocol['timeout_seconds']
        assert 'L4' in (root / name / 'gpu.txt').read_text(), name
        start, elapsed = runtime['started_unix'], runtime['elapsed_seconds']
        end = start + elapsed
        assert end <= manifest['deadline_unix'], name
        events.extend([(start, 1), (end, -1)])
        ends.append(end)
        seconds.append(elapsed)
    assert manifest['accepted_worker_seconds'] == sum(seconds)
    assert manifest['last_worker_end_unix'] == max(ends)
    accounted = sum(seconds) + manifest['prior_gpu_seconds']
    assert accounted == manifest['budget_accounted_seconds'] <= protocol['budget_gpu_hours'] * 3600
    concurrent = 0
    for _, change in sorted(events):
        concurrent += change
        assert 0 <= concurrent <= protocol['concurrency']
    assert concurrent == 0
    for partial in manifest['incomplete_evaluations']:
        file = partial['runtime']
        assert sha256(root / file['path']) == file['sha256']
        runtime = json.loads((root / file['path']).read_text())
        assert 'returncode' not in runtime and 'elapsed_seconds' not in runtime
        assert runtime['job']['name'] == partial['job']
        assert runtime['environment']['MODAL_TASK_ID'] == partial['worker_task_id']
        assert runtime['source_sha256'] == manifest['sources'][partial['source_commit']]
        lines, counts = partial['raw_completed_batch_lines'], partial['batch_successes']
        assert len(lines) == len(counts)
        for line, count in zip(lines, counts):
            assert float(re.search(r'success_rate=([0-9.]+)%', line)[1]) == 100 * count / partial['batch_episodes']
    summary = summarize(root)
    assert summary['complete'] == 28
    recorded = json.loads((root / 'summary.json').read_text())
    actual_stats, recorded_stats = summary.pop('variability'), recorded.pop('variability')
    assert summary == recorded
    assert actual_stats.keys() == recorded_stats.keys()
    for key, actual in actual_stats.items():
        expected = recorded_stats[key]
        assert actual.keys() == expected.keys() == {'n', 'mean', 'sample_sd'}
        assert actual['n'] == expected['n'] == 3
        # statistics.stdev differs by a final rounding bit between Python versions.
        for metric in ('mean', 'sample_sd'):
            assert math.isclose(actual[metric], expected[metric], rel_tol=1e-12, abs_tol=1e-12)
    print('28/28 cells: hashes, raw outcomes, task/config/source provenance, bounds and both seed summaries agree.')


if __name__ == '__main__':
    check(Path(__file__).with_name('results'))
