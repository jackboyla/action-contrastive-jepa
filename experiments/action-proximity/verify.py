"""Check the bundle, optionally recounting distances from the original datasets."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
THRESHOLDS = (.05, .1, .2)


def recount(result, ids, cache_dir):
    import h5py
    import hdf5plugin  # noqa: F401
    import torch
    from torch.utils.data import DataLoader, random_split

    torch.set_num_threads(2)
    with h5py.File(cache_dir / (result["dataset"]["name"] + ".h5"), "r") as f:
        raw, lengths, offsets = f["action"][:], f["ep_len"][:], f["ep_offset"][:]
    for array, key in [(raw, "action_sha256"), (lengths, "ep_len_sha256"), (offsets, "ep_offset_sha256")]:
        assert hashlib.sha256(np.ascontiguousarray(array).view(np.uint8)).hexdigest() == result["dataset"][key]
    valid = torch.from_numpy(raw)
    valid = valid[~torch.isnan(valid).any(dim=1)]
    mean, std = valid.mean(0).numpy(), valid.std(0).numpy()
    np.testing.assert_array_equal(mean, np.asarray(result["normalization"]["mean"], dtype=mean.dtype))
    np.testing.assert_array_equal(std, np.asarray(result["normalization"]["sample_std"], dtype=std.dtype))
    assert np.all(std > 0), "These saved datasets had no constant action dimensions"
    clips = np.asarray([(ep, start) for ep, length in enumerate(lengths)
                        for start in range(int(length)-20+1)], dtype=np.int64)
    np.testing.assert_array_equal(clips[ids[:, :, 0]], ids[:, :, 1:])
    generator = torch.Generator().manual_seed(3072)
    train, _ = random_split(range(len(clips)), [.9, 1-.9], generator=generator)
    loader = DataLoader(train, batch_size=128, shuffle=True, drop_last=True, generator=generator, num_workers=0)
    for number, sampled in enumerate(loader):
        if number == 100:
            break
        np.testing.assert_array_equal(sampled.numpy(), ids[number, :, 0])
    window = np.repeat(np.arange(128), 3)
    same_window = (window[:, None] == window[None, :]) & ~np.eye(384, dtype=bool)
    cross_sample = window[:, None] != window[None, :]
    for number, batch in enumerate(ids):
        rows = offsets[batch[:, 1]] + batch[:, 2]
        actions = raw[rows[:, None] + np.arange(15)[None, :]]
        assert np.isfinite(actions).all()
        blocks = ((actions - mean) / std).astype(np.float32).reshape(384, -1).astype(np.float64)
        diff = blocks[:, None] - blocks[None, :]
        squared = np.einsum("ijk,ijk->ij", diff, diff) / blocks.shape[-1]
        episodes = np.repeat(batch[:, 1], 3)
        masks = {"same_window": same_window, "cross_sample": cross_sample,
                 "cross_sample_same_episode": cross_sample & (episodes[:, None] == episodes[None, :])}
        counts = {name: {"denominator": int(mask.sum()), "numerators": {
            str(t): int(((squared <= t*t) & mask).sum()) for t in THRESHOLDS}} for name, mask in masks.items()}
        assert counts == result["per_batch"][number], (result["task"], number)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, help="Recount using the original HDF5 action columns")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    for task in ("scene", "reacher"):
        result = json.loads((ROOT / f"{task}.json").read_text())
        assert result["source_sha256"]["scripts/action_proximity.py"] == manifest["files"]["../../scripts/action_proximity.py"]
        ids = np.load(ROOT / f"{task}.npz")["clip_episode_start"]
        assert ids.shape == (100, 128, 3)
        assert len(result["per_batch"]) == 100
        assert result["seed"] == 3072 and result["batch_size"] == 128
        assert result["candidates"] == 384 and result["negatives_per_positive"] == 383
        for group, total in result["totals"].items():
            assert sum(row[group]["denominator"] for row in result["per_batch"]) == total["denominator"]
            for threshold, count in total["numerators"].items():
                assert 0 <= count <= total["denominator"]
                assert sum(row[group]["numerators"][threshold] for row in result["per_batch"]) == count
        assert result["totals"]["same_window"]["denominator"] == 100 * 384 * 2
        assert result["totals"]["cross_sample"]["denominator"] == 100 * 384 * 381
        if args.cache_dir:
            recount(result, ids, args.cache_dir)
    print(json.dumps({"tasks": 2, "batches_per_task": 100,
                      "hashes_and_counts": "verified", "raw_recount": bool(args.cache_dir)}))


if __name__ == "__main__":
    main()
