"""Check the recovery delta without models, datasets, GPU access or network."""
import hashlib
import json
from pathlib import Path
import re

root = Path(__file__).resolve().parent
manifest = json.loads((root / 'manifest.json').read_text())
for entry in manifest:
    assert hashlib.sha256((root / entry['path']).read_bytes()).hexdigest() == entry['sha256'], entry['path']
    if entry['transformation'] == 'none':
        assert entry['sha256'] == entry['source_sha256']

outcomes = {}
for path in sorted([*root.glob('scene-short/*.txt'), *root.glob('reacher/**/*.txt')]):
    text = path.read_text()
    assert text.count('==== CONFIG ====') == text.count('==== RESULTS ====') == 1, path
    array = re.search(r"'episode_successes': array\(\[([\s\S]*?)\]", text)[1]
    values = re.findall(r'True|False', array)
    n = int(re.search(r'^  num_eval: (\d+)$', text, re.M)[1])
    assert len(values) == n, path
    successes = values.count('True')
    rate = float(re.search(r"'success_rate': ([\d.]+)", text)[1])
    assert abs(100 * successes / n - rate) < 1e-10, path
    outcomes[str(path.relative_to(root))] = {'successes': successes, 'episodes': n}
assert len(list(root.glob('scene-short/*.txt'))) == 9
assert outcomes['scene-short/scene_action_nce_e10_s3072_n50.txt']['successes'] == 40
assert outcomes['reacher/lewm_masked_vicreg/lewm_masked_vicreg_ep10_reacher_n200.txt']['successes'] == 132
assert outcomes['reacher/lewm_base_e100_ep9_reacher_n200.txt']['successes'] == 152

arrays = []
for path in sorted(root.glob('sampling/*.txt')):
    text = path.read_text()
    match = re.search(r'valid starting points found for evaluation\.\s*\[([\d\s]+)\]', text)
    ids = [int(value) for value in match[1].split()]
    assert len(ids) == len(set(ids)) == 100 and ids == sorted(ids), path
    assert 'episodes=100' in text and 'eval_budget=100' in text, path
    arrays.append(ids)
assert len(arrays) == 7 and all(ids == arrays[0] for ids in arrays)
print(json.dumps({'manifest_files': len(manifest), 'outcomes': outcomes,
                  'original_sampling_records': len(arrays), 'shared_unique_ids': 100}, indent=2))
