"""Compare orientation decodability: planning CLS latent vs raw patch tokens.

The planner scores ‖z_CLS - z_goal_CLS‖², so it only "sees" what the CLS/projector
latent encodes. The probe found block orientation collapsed in that CLS latent. This
script asks the prerequisite question for any frozen-encoder test-time adaptation:
is orientation still recoverable from the encoder's PATCH tokens (spatially resolved,
196×D) even though it collapsed in the single CLS token?

For each privileged state dim it fits a ridge probe from three representations:
  - cls    : projector(CLS)            — exactly what planning uses (baseline)
  - patch_mean : mean-pooled patch tokens (D)         — pooling washes out spatial info
  - patch_grid : patches pooled to a small spatial grid then flattened (G*G*D)
                 — preserves coarse spatial layout, where orientation lives

If patch_grid >> cls for the orientation dims, the information survives in the encoder
and an adapter/TTA module has a real target to route into the planning latent. If
patch_grid ≈ cls (both collapsed), orientation is gone everywhere a frozen-encoder
module could reach it, and TTA cannot recover it.

Prints PROBE_R2 lines so results are recoverable from Modal logs.
"""

import os
import sys

os.environ.setdefault("MUJOCO_GL", "glfw" if sys.platform == "darwin" else "egl")

import argparse

import numpy as np
import stable_pretraining as spt
import torch
import torch.nn.functional as F
from einops import rearrange
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
    if im.ndim == 3 and im.shape[0] in (1, 3) and im.shape[0] < im.shape[-1]:
        return np.transpose(im, (1, 2, 0))
    return im


def encode_reps(jepa, pixels_batch, grid, device):
    """Return (cls, patch_mean, patch_grid) numpy reps for a batch of HWC frames."""
    out = jepa.encoder(pixels_batch, interpolate_pos_encoding=True)
    hidden = out.last_hidden_state  # (B, 1+P, D)
    cls_tok = hidden[:, 0]
    cls = jepa.projector(cls_tok)  # exactly what encode() feeds the planner
    patches = hidden[:, 1:]  # (B, P, D)

    patch_mean = patches.mean(dim=1)

    # reshape patch tokens to a square map then adaptive-pool to grid x grid
    b, p, d = patches.shape
    side = int(round(p ** 0.5))
    if side * side == p:
        pmap = rearrange(patches, "b (h w) d -> b d h w", h=side, w=side)
        pooled = F.adaptive_avg_pool2d(pmap, output_size=(grid, grid))  # (B, D, g, g)
        patch_grid = rearrange(pooled, "b d h w -> b (h w d)")
    else:
        # non-square token count: fall back to mean only
        patch_grid = patch_mean
    return (
        cls.float().cpu().numpy(),
        patch_mean.float().cpu().numpy(),
        patch_grid.float().cpu().numpy(),
    )


def probe_rep(name, Z, Y, tr, te, alpha):
    Z = Z.astype(np.float64)
    print(f"REP {name} | dim={Z.shape[1]} alpha={alpha}", flush=True)
    r2s = []
    for d in range(Y.shape[1]):
        reg = Ridge(alpha=alpha).fit(Z[tr], Y[tr, d])
        r2 = float(r2_score(Y[te, d], reg.predict(Z[te])))
        r2s.append(r2)
        print(f"  PROBE_R2 {name} state[{d}] = {r2:.4f}", flush=True)
    print(f"  PROBE_R2 {name} MEAN = {float(np.mean(r2s)):.4f}", flush=True)
    return r2s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--dataset", default="pusht_expert_train")
    ap.add_argument("--state-key", default="state")
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--grid", type=int, default=4)
    ap.add_argument("--alpha-cls", type=float, default=1.0)
    ap.add_argument("--alpha-patch", type=float, default=10.0)
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

    cls_list, pmean_list, pgrid_list = [], [], []
    bs = 64
    with torch.inference_mode():
        for i in range(0, len(pixels), bs):
            ims = [tf(to_hwc(im)) for im in pixels[i : i + bs]]
            t = torch.stack(ims).to(device)  # (b, C, H, W)
            cls, pmean, pgrid = encode_reps(jepa, t, args.grid, device)
            cls_list.append(cls)
            pmean_list.append(pmean)
            pgrid_list.append(pgrid)

    cls = np.concatenate(cls_list, 0)
    patch_mean = np.concatenate(pmean_list, 0)
    patch_grid = np.concatenate(pgrid_list, 0)

    n = len(cls)
    perm = rng.permutation(n)
    n_tr = int(0.7 * n)
    tr, te = perm[:n_tr], perm[n_tr:]

    print(
        f"POLICY {args.policy} | n={n} state_dim={Y.shape[1]} "
        f"cls_dim={cls.shape[1]} patch_grid_dim={patch_grid.shape[1]} grid={args.grid}",
        flush=True,
    )
    probe_rep("cls", cls, Y, tr, te, args.alpha_cls)
    probe_rep("patch_mean", patch_mean, Y, tr, te, args.alpha_patch)
    probe_rep("patch_grid", patch_grid, Y, tr, te, args.alpha_patch)


if __name__ == "__main__":
    main()
