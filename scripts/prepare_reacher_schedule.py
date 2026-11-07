"""Freeze E1 commands, configs and dataset-derived audit inputs before training."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import h5py
from hydra import compose, initialize_config_dir
import numpy as np
from omegaconf import OmegaConf

from reacher_schedule import jobs, sha256

ROOT = Path(__file__).resolve().parents[1]


def dataset_inputs(path, train, evaluation):
    before = path.stat()
    with h5py.File(path, 'r') as dataset:
        lengths = dataset['ep_len'][:]
        column = 'episode_idx' if 'episode_idx' in dataset else 'ep_idx'
        episodes = dataset[column][:]
        steps = dataset['step_idx'][:]

    # Match eval.py's per-episode maximum step without scanning every row per episode.
    unique, inverse = np.unique(episodes, return_inverse=True)
    maximum = np.full(len(unique), -1, dtype=steps.dtype)
    np.maximum.at(maximum, inverse, steps)
    valid = np.flatnonzero(steps <= maximum[inverse] - evaluation['eval']['goal_offset_steps'])
    rng = np.random.default_rng(evaluation['seed'])
    indices = np.sort(valid[rng.choice(len(valid), size=evaluation['eval']['num_eval'], replace=False)])

    data = train['data']['dataset']
    clips = int(np.maximum(lengths - data['num_steps'] * data['frameskip'] + 1, 0).sum())
    fraction = train['train_split']
    split = [math.floor(clips * f) for f in (fraction, 1 - fraction)]
    # torch.random_split assigns the fractional remainder to the first split.
    train_clips = split[0] + min(clips - sum(split), 1)
    batches = train_clips // train['loader']['batch_size']

    print(f'Hashing {path} ({before.st_size} bytes)', flush=True)
    digest = sha256(path)
    after = path.stat()
    fields = ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
    if any(getattr(before, key) != getattr(after, key) for key in fields):
        raise RuntimeError('Dataset changed during preparation')
    return dict(
        dataset_path=str(path), dataset_sha256=digest,
        dataset_stat={key: getattr(after, key) for key in fields},
        clips=clips, train_clips=train_clips,
        optimizer_steps=batches * train['runtime']['stop_after_epoch'],
        scheduler_steps=batches * train['trainer']['max_epochs'],
        tasks=dict(start_row_indices=indices.tolist(),
                   start_episode_ids=episodes[indices].tolist(),
                   start_steps=steps[indices].tolist()),
    )


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--data-audit', type=Path, required=True)
    args = parser.parse_args()
    outputs = [args.manifest.resolve(), args.data_audit.resolve()]
    if outputs[0] == outputs[1] or any(path.exists() for path in outputs):
        parser.error('Manifest and data-audit must be distinct, new files')

    runs = jobs(args.results)
    cache = args.cache.resolve()
    for job in runs:
        if (cache / job['subdir']).exists() or (args.results / (job['label'] + '.txt')).exists():
            parser.error(f'Run already exists: {job["label"]}; prepare before training')

    OmegaConf.register_new_resolver('eval', eval)
    configs = {}
    for job in runs:
        configs[job['label']] = {}
        for stage in ('train', 'eval'):
            command = job[stage]
            with initialize_config_dir(config_dir=str(ROOT / 'config' / stage), version_base=None):
                cfg = compose(config_name=command[2], overrides=command[3:])
            configs[job['label']][stage] = OmegaConf.to_container(cfg, resolve=True)

    paths = [ROOT / name for name in (
        'train.py', 'eval.py', 'utils.py', 'jepa.py', 'module.py', 'project_paths.py',
        'scripts/reacher_schedule.py', 'scripts/prepare_reacher_schedule.py',
    )]
    paths += sorted((ROOT / 'config').rglob('*.yaml'))
    source = {str(path.relative_to(ROOT)): sha256(path) for path in paths}
    reference = configs[runs[0]['label']]
    dataset = cache / (reference['train']['data']['dataset']['name'] + '.h5')
    data = dataset_inputs(dataset, reference['train'], reference['eval'])
    if any(sha256(ROOT / name) != digest for name, digest in source.items()):
        raise RuntimeError('Source changed during preparation')
    manifest = dict(
        frozen_utc=datetime.now(timezone.utc).isoformat(),
        git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        jobs=runs, configs=configs, source_sha256=source,
        cache=str(cache), dataset_sha256=data['dataset_sha256'],
        environment={key: os.environ.get(key) for key in (
            'STABLEWM_HOME', 'PYTHONPATH', 'CUDA_VISIBLE_DEVICES', 'MUJOCO_GL',
            'MUJOCO_EGL_DEVICE_ID', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
        )},
        runtime_python=sys.executable, python_version=sys.version,
        packages={dist.metadata['Name']: dist.version for dist in importlib.metadata.distributions()
                  if dist.metadata['Name']},
    )
    for path, payload in zip(outputs, (manifest, data)):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x') as handle:
            json.dump(payload, handle, indent=2)
            handle.write('\n')
        print(path, flush=True)


if __name__ == '__main__':
    main()
