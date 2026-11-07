"""Check published hashes and success rates without loading models or data."""
import hashlib
import json
import re
from pathlib import Path

root = Path(__file__).resolve().parent
for record in json.loads((root / 'manifest.json').read_text()):
    path = root / record['file']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == record['sha256'], path
for path in (root / 'results').rglob('*.txt'):
    text = path.read_text()
    match = re.search(r"'episode_successes': array\(\[(.*?)\]", text, re.S)
    if match:
        outcomes = re.findall('True|False', match[1])
        rate = float(re.search(r"'success_rate': ([0-9.]+)", text)[1])
        assert abs(100 * outcomes.count('True') / len(outcomes) - rate) < 1e-6, path
print('Published hashes and recorded episode success rates agree.')
