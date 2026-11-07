"""Measure action-block proximity in the fixed Scene/Reacher training sample."""
import argparse
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from stable_worldmodel.data import HDF5Dataset
from torch.utils.data import DataLoader, Dataset, random_split
from utils import get_column_normalizer


class IndexedActions(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]
        ep, start = self.dataset.clip_indices[index]
        return {"action": item["action"], "index": index, "episode": ep, "start": start}


def counts(blocks, episodes):
    batch, transitions, dimension = blocks.shape
    x = blocks.flatten(0, 1).double()
    window = torch.arange(batch).repeat_interleave(transitions)
    episode = episodes.repeat_interleave(transitions)
    negative = ~torch.eye(len(x), dtype=torch.bool)
    masks = {
        "same_window": (window[:, None] == window[None, :]) & negative,
        "cross_sample": window[:, None] != window[None, :],
        "cross_sample_same_episode": (window[:, None] != window[None, :])
        & (episode[:, None] == episode[None, :]),
    }
    distance = torch.cdist(x, x, compute_mode="donot_use_mm_for_euclid_dist") / math.sqrt(dimension)
    return {
        name: {"denominator": int(mask.sum()), "numerators": {
            str(t): int(((distance <= t) & mask).sum()) for t in (0.05, 0.10, 0.20)
        }} for name, mask in masks.items()
    }


def sha_array(array):
    return hashlib.sha256(np.ascontiguousarray(array).view(np.uint8)).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["scene", "reacher"], required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path(os.environ.get("STABLEWM_HOME", ROOT / ".stable_worldmodel")))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.suffix != ".json":
        raise ValueError("--output must end in .json; sampled IDs use the matching .npz path")
    if args.output.exists() or args.output.with_suffix(".npz").exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    with initialize_config_dir(config_dir=str(ROOT / "config/train"), version_base=None):
        cfg = compose(config_name="lewm_masked_action_nce", overrides=[
            f"data={args.task}", "seed=3072", "subdir=e5_action_proximity"
        ])
    resolved = OmegaConf.to_container(cfg, resolve=True)
    ds_args = dict(resolved["data"]["dataset"])
    original_keys = ds_args["keys_to_load"]
    ds_args.update(keys_to_load=["action"], keys_to_cache=["action"], cache_dir=args.cache_dir)
    dataset = HDF5Dataset(**ds_args)
    assert dataset.num_steps == 4 and dataset.frameskip == 5
    assert int(cfg.loader.batch_size) == 128 and float(cfg.train_split) == 0.9
    raw = dataset.get_col_data("action")
    raw_tensor = torch.from_numpy(np.array(raw))
    valid = raw_tensor[~torch.isnan(raw_tensor).any(dim=1)]
    if not torch.isfinite(valid).all():
        raise ValueError("Nonfinite action normalization data")
    mean, std = valid.mean(0), valid.std(0)
    constant = std == 0
    assert torch.isfinite(mean).all() and torch.isfinite(std).all()
    dataset.transform = get_column_normalizer(dataset, "action", "action")
    generator = torch.Generator().manual_seed(int(cfg.seed))
    train, _ = random_split(IndexedActions(dataset), [float(cfg.train_split), 1-float(cfg.train_split)], generator=generator)
    loader = DataLoader(train, batch_size=128, shuffle=True, drop_last=True, generator=generator, num_workers=0)
    started = time.monotonic()
    records, ids = [], []
    for number, batch in enumerate(loader):
        if number >= 100:
            break
        if torch.isinf(batch["action"]).any():
            raise ValueError("Action normalizer produced infinity; inspect zero-variance coordinates")
        action = torch.nan_to_num(batch["action"], 0.0)
        assert action.shape[:2] == (128, 4) and torch.isfinite(action).all()
        if number == 0:
            ep, start = int(batch["episode"][0]), int(batch["start"][0])
            offset = int(dataset.offsets[ep]) + start
            original = raw_tensor[offset:offset + dataset.span]
            if not torch.isfinite(original).all():
                raise ValueError("Nonfinite raw action in first sampled clip")
            expected = torch.nan_to_num(((original-mean)/std).float(), 0.0).reshape(4, -1)
            torch.testing.assert_close(action[0], expected, rtol=0, atol=0)
        for ep, start in zip(batch["episode"], batch["start"]):
            offset = int(dataset.offsets[int(ep)]) + int(start)
            if not torch.isfinite(raw_tensor[offset:offset + 15]).all():
                raise ValueError("Nonfinite raw action in sampled transition")
        records.append(counts(action[:, :-1], batch["episode"]))
        ids.append(np.stack([batch[k].numpy() for k in ("index", "episode", "start")], axis=-1))
    assert len(records) == 100
    totals = {name: {"denominator": sum(r[name]["denominator"] for r in records),
                     "numerators": {t: sum(r[name]["numerators"][t] for r in records)
                                    for t in records[0][name]["numerators"]}}
              for name in records[0]}
    stat = dataset.h5_path.stat()
    sources = {"train.py": ROOT / "train.py", "utils.py": ROOT / "utils.py",
               "stable_worldmodel/data/dataset.py": Path(inspect.getsourcefile(HDF5Dataset)),
               "scripts/action_proximity.py": Path(__file__)}
    result = {
        "task": args.task, "seed": 3072, "batches": 100, "batch_size": 128,
        "transitions_per_window": 3, "candidates": 384, "negatives_per_positive": 383,
        "distance": "RMS across flattened normalized action-block coordinates",
        "ordered_pairs": True, "threshold_inclusive": True,
        "totals": totals, "per_batch": records, "resolved_train_config": resolved,
        "omitted_columns": [k for k in original_keys if k != "action"],
        "normalization": {"mean": mean.tolist(), "sample_std": std.tolist(),
                          "constant_dimensions": torch.where(constant)[0].tolist(),
                          "constant_treatment": "training NaN-to-zero; original RMS denominator retained"},
        "dataset": {"name": ds_args["name"], "bytes": stat.st_size,
                    "action_sha256": sha_array(raw), "ep_len_sha256": sha_array(dataset.lengths),
                    "ep_offset_sha256": sha_array(dataset.offsets), "full_file_hash": None},
        "source_sha256": {name: hashlib.sha256(p.read_bytes()).hexdigest() for name, p in sources.items()},
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "environment": {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__,
                        "stable_worldmodel": importlib.metadata.version("stable-worldmodel")},
        "measurement_seconds": time.monotonic() - started,
        "scope": "Action proximity only; does not establish behavioral equivalence or learned shortcuts."
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output.with_suffix(".npz"), clip_episode_start=np.stack(ids))
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({"task": args.task, "totals": totals, "seconds": result["measurement_seconds"]}))


if __name__ == "__main__":
    main()
