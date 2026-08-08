"""Linear-probe a world-model encoder for privileged physical state.

Encodes a sample of dataset frames with a checkpoint's encoder and fits a
ridge regression from the latent to each dimension of the privileged `state`
(for PushT: agent x/y, block x/y, block angle). Higher R^2 = that quantity is
more linearly decodable from the latent. Used to test *which* state variables a
representation encodes (e.g. does masked under-encode the PushT block?).

Reads only the sampled rows via HDF5Dataset.get_row_data (memory-safe), and
prints `PROBE_R2 ...` lines so results are recoverable from logs.
"""

import os
import sys

os.environ.setdefault("MUJOCO_GL", "glfw" if sys.platform == "darwin" else "egl")

import argparse

import numpy as np
import stable_pretraining as spt
import torch
from torchvision.transforms import v2 as transforms

from project_paths import configure_stablewm_home

configure_stablewm_home()

import stable_worldmodel as swm
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score


def find_jepa(module):
    if hasattr(module, "encode") and hasattr(module, "get_cost"):
        return module
    for child in getattr(module, "children", lambda: [])():
        found = find_jepa(child)
        if found is not None:
            return found
    return None


def to_hwc(im: np.ndarray) -> np.ndarray:
    # torchvision ToImage expects HWC; transpose if stored CHW.
    if im.ndim == 3 and im.shape[0] in (1, 3) and im.shape[0] < im.shape[-1]:
        return np.transpose(im, (1, 2, 0))
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--dataset", default="pusht_expert_train")
    ap.add_argument("--state-key", default="state")
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--alpha", type=float, default=1.0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    jepa = find_jepa(swm.policy.AutoCostModel(args.policy))
    assert jepa is not None, "could not locate JEPA in checkpoint"
    jepa = jepa.to(device).eval()

    ds = swm.data.HDF5Dataset(
        args.dataset,
        num_steps=1,
        frameskip=1,
        keys_to_load=["pixels", args.state_key],
        keys_to_cache=[args.state_key],
    )
    state_all = np.asarray(ds.get_col_data(args.state_key))
    n_total = len(state_all)
    rng = np.random.default_rng(args.seed)
    idx = np.sort(rng.choice(n_total, size=min(args.n, n_total), replace=False))

    rows = ds.get_row_data([int(i) for i in idx])
    pixels = np.asarray(rows["pixels"])
    Y = np.asarray(rows[args.state_key], dtype=np.float64)

    tf = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=args.img_size),
        ]
    )

    Z = []
    bs = 64
    with torch.inference_mode():
        for i in range(0, len(pixels), bs):
            ims = [tf(to_hwc(im)) for im in pixels[i : i + bs]]
            t = torch.stack(ims).unsqueeze(1).to(device)  # (b, 1, C, H, W)
            z = jepa.encode({"pixels": t})["emb"][:, 0]
            Z.append(z.float().cpu().numpy())
    Z = np.concatenate(Z, 0).astype(np.float64)

    n = len(Z)
    perm = rng.permutation(n)
    n_tr = int(0.7 * n)
    tr, te = perm[:n_tr], perm[n_tr:]

    print(f"POLICY {args.policy} | n={n} latent_dim={Z.shape[1]} state_dim={Y.shape[1]}", flush=True)
    r2s = []
    for d in range(Y.shape[1]):
        reg = Ridge(alpha=args.alpha).fit(Z[tr], Y[tr, d])
        r2 = float(r2_score(Y[te, d], reg.predict(Z[te])))
        r2s.append(r2)
        print(f"  PROBE_R2 state[{d}] = {r2:.4f}", flush=True)
    print(f"  PROBE_R2 MEAN = {float(np.mean(r2s)):.4f}", flush=True)


if __name__ == "__main__":
    main()
