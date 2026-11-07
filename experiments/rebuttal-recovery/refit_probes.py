"""Refit saved Scene probe arrays on CPU; never load a model or dataset."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import balanced_accuracy_score, r2_score
from sklearn.preprocessing import StandardScaler


def refit(path, reference):
    with np.load(path, allow_pickle=False) as arrays:
        z, splits = arrays['latents'], arrays['split']
        train, val, test = [np.flatnonzero(splits == i) for i in range(3)]
        zs = StandardScaler().fit(z[train]).transform(z)
        probes = {}
        for name in reference['probes']:
            y = arrays[name]
            trials = []
            for alpha in [.01, .1, 1., 10., 100., 1000.]:
                model = Ridge(alpha=alpha).fit(zs[train], y[train])
                trials.append((r2_score(y[val], model.predict(zs[val])), alpha, model))
            _, alpha, model = max(trials, key=lambda item: item[0])
            prediction = model.predict(zs[test])
            result = {'alpha': alpha, 'r2': r2_score(y[test], prediction, multioutput='raw_values').tolist()}
            if name.startswith('button'):
                result['balanced_accuracy'] = balanced_accuracy_score(y[test].ravel(), prediction.ravel() >= .5)
            expected = reference['probes'][name]
            assert alpha == expected['alpha'], (path.name, name, 'alpha')
            np.testing.assert_allclose(result['r2'], expected['r2'], rtol=0, atol=1e-5)
            if 'balanced_accuracy' in result:
                np.testing.assert_allclose(result['balanced_accuracy'], expected['balanced_accuracy'], rtol=0, atol=1e-5)
            probes[name] = result
        centered = z[test].astype(np.float64) - z[test].mean(0)
        covariance = centered.T @ centered / (len(test) - 1)
        trace = float(np.trace(covariance))
        rank = trace ** 2 / float(np.square(covariance).sum()) if trace > 0 else 0.
        np.testing.assert_allclose(rank, reference['participation_ratio'], rtol=0, atol=1e-8)
        return {'array_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'probe_split_sizes': [len(train), len(val), len(test)],
                'probes': probes, 'participation_ratio': rank}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('arrays', type=Path)
    parser.add_argument('references', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    results = {}
    for method in ['sigreg', 'masked', 'masked_action_nce']:
        for seed in [3072, 1, 2]:
            name = f'{method}_{seed}'
            ref_path = args.references / f'{name}.json'
            results[name] = refit(args.arrays / f'{name}.npz', json.loads(ref_path.read_text()))
            results[name]['reference_sha256'] = hashlib.sha256(ref_path.read_bytes()).hexdigest()
            print(name, 'matched', flush=True)
    args.output.write_text(json.dumps(results, indent=2) + '\n')


if __name__ == '__main__':
    main()
