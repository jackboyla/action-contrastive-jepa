"""Latent surprise diagnostics for action-conditioned JEPA world models.

The diagnostic measures whether the one-step predictor assigns higher latent
error to counterfactual or impossible transitions than to dataset transitions.
It is intentionally model-facing: it probes the same encoder, action encoder,
and autoregressive predictor used by MPC, without introducing a separate
simulator benchmark.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "glfw" if sys.platform == "darwin" else "egl")

import numpy as np
import stable_pretraining as spt
import torch
from torch.utils.data._utils.collate import default_collate

from project_paths import configure_stablewm_home
from utils import get_column_normalizer, get_img_preprocessor

configure_stablewm_home()

import stable_worldmodel as swm


def find_jepa(module):
    if hasattr(module, "encode") and hasattr(module, "predict"):
        return module
    for child in getattr(module, "children", lambda: [])():
        found = find_jepa(child)
        if found is not None:
            return found
    return None


def scalar_stats(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "median": float(np.median(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
    }


def ratio_stats(numerator: np.ndarray, denominator: np.ndarray) -> dict[str, float]:
    eps = 1e-12
    ratios = np.asarray(numerator, dtype=np.float64) / (
        np.asarray(denominator, dtype=np.float64) + eps
    )
    return {
        "mean": float(ratios.mean()),
        "median": float(np.median(ratios)),
        "p10": float(np.percentile(ratios, 10)),
        "p90": float(np.percentile(ratios, 90)),
    }


def shifted_permutation(size: int, rng: np.random.Generator) -> torch.Tensor:
    if size <= 1:
        return torch.arange(size)
    index = np.arange(size)
    for _ in range(128):
        perm = rng.permutation(size)
        if not np.any(perm == index):
            break
    else:
        perm = np.roll(index, 1)
    return torch.as_tensor(perm, dtype=torch.long)


def visual_perturb(pixels: torch.Tensor) -> torch.Tensor:
    """Appearance-only color/brightness perturbation in normalized image space."""

    perturbed = pixels.clone()
    if perturbed.size(-3) >= 3:
        perturbed[..., 0, :, :] = perturbed[..., 0, :, :] + 0.35
        perturbed[..., 1, :, :] = perturbed[..., 1, :, :] - 0.20
        perturbed[..., 2, :, :] = perturbed[..., 2, :, :] + 0.10
    else:
        perturbed = perturbed + 0.25
    return perturbed


def build_dataset(args):
    raw_dataset = swm.data.HDF5Dataset(
        args.dataset,
        frameskip=args.frameskip,
        num_steps=args.history_size + args.num_preds,
        keys_to_load=["pixels", "action"],
        keys_to_cache=["action"],
    )
    transform = spt.data.transforms.Compose(
        get_img_preprocessor(source="pixels", target="pixels", img_size=args.img_size),
        get_column_normalizer(raw_dataset, "action", "action"),
    )
    return swm.data.HDF5Dataset(
        args.dataset,
        frameskip=args.frameskip,
        num_steps=args.history_size + args.num_preds,
        keys_to_load=["pixels", "action"],
        keys_to_cache=["action"],
        transform=transform,
    )


def output_path(args) -> Path:
    if args.output:
        path = Path(args.output)
    else:
        safe_policy = args.policy.replace("/", "__")
        safe_dataset = args.dataset.replace("/", "__")
        path = (
            Path(swm.data.utils.get_cache_dir())
            / "surprise_diagnostics"
            / f"{safe_policy}__{safe_dataset}__n{args.n}_seed{args.seed}.json"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def run(args) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    jepa = find_jepa(swm.policy.AutoCostModel(args.policy))
    if jepa is None:
        raise RuntimeError(f"could not locate JEPA model in policy {args.policy!r}")
    jepa = jepa.to(device).eval()
    jepa.requires_grad_(False)

    dataset = build_dataset(args)
    rng = np.random.default_rng(args.seed)
    n = min(args.n, len(dataset))
    indices = rng.choice(len(dataset), size=n, replace=False)

    normal_errors: list[np.ndarray] = []
    action_errors: list[np.ndarray] = []
    future_errors: list[np.ndarray] = []
    visual_errors: list[np.ndarray] = []

    with torch.inference_mode():
        for start in range(0, n, args.batch_size):
            batch_indices = indices[start : start + args.batch_size]
            rows = [dataset[int(i)] for i in batch_indices]
            batch = default_collate(rows)
            pixels = batch["pixels"].to(device)
            action = torch.nan_to_num(batch["action"].to(device), 0.0)

            encoded = jepa.encode({"pixels": pixels, "action": action})
            emb = encoded["emb"]
            act_emb = encoded["act_emb"]

            ctx_emb = emb[:, : args.history_size]
            ctx_act = act_emb[:, : args.history_size]
            pred = jepa.predict(ctx_emb, ctx_act)
            pred_len = pred.size(1)
            target = emb[:, args.num_preds : args.num_preds + pred_len]

            normal = (pred - target).pow(2).mean(dim=(1, 2))

            perm = shifted_permutation(pred.size(0), rng).to(device)
            wrong_action = action[perm, : args.history_size]
            wrong_act_emb = jepa.action_encoder(wrong_action)
            action_pred = jepa.predict(ctx_emb, wrong_act_emb)
            action_cf = (action_pred - target).pow(2).mean(dim=(1, 2))

            wrong_future = target[perm]
            future_cf = (pred - wrong_future).pow(2).mean(dim=(1, 2))

            target_pixels = pixels[:, args.num_preds : args.num_preds + pred_len]
            visual_target = jepa.encode({"pixels": visual_perturb(target_pixels)})["emb"]
            visual_cf = (pred - visual_target).pow(2).mean(dim=(1, 2))

            normal_errors.append(normal.detach().cpu().numpy())
            action_errors.append(action_cf.detach().cpu().numpy())
            future_errors.append(future_cf.detach().cpu().numpy())
            visual_errors.append(visual_cf.detach().cpu().numpy())

    normal = np.concatenate(normal_errors)
    action = np.concatenate(action_errors)
    future = np.concatenate(future_errors)
    visual = np.concatenate(visual_errors)

    result = {
        "policy": args.policy,
        "dataset": args.dataset,
        "n": int(n),
        "seed": int(args.seed),
        "history_size": int(args.history_size),
        "num_preds": int(args.num_preds),
        "frameskip": int(args.frameskip),
        "img_size": int(args.img_size),
        "errors": {
            "normal": scalar_stats(normal),
            "counterfactual_action": scalar_stats(action),
            "state_discontinuity": scalar_stats(future),
            "appearance_perturbation": scalar_stats(visual),
        },
        "ratios_vs_normal": {
            "counterfactual_action": ratio_stats(action, normal),
            "state_discontinuity": ratio_stats(future, normal),
            "appearance_perturbation": ratio_stats(visual, normal),
        },
        "fraction_greater_than_normal": {
            "counterfactual_action": float((action > normal).mean()),
            "state_discontinuity": float((future > normal).mean()),
            "appearance_perturbation": float((visual > normal).mean()),
        },
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--history-size", type=int, default=3)
    parser.add_argument("--num-preds", type=int, default=1)
    parser.add_argument("--frameskip", type=int, default=5)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    result = run(args)
    path = output_path(args)
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    ratios = result["ratios_vs_normal"]
    fractions = result["fraction_greater_than_normal"]
    print(
        "SURPRISE "
        f"policy={result['policy']} dataset={result['dataset']} n={result['n']} "
        f"normal={result['errors']['normal']['mean']:.6g} "
        f"action={result['errors']['counterfactual_action']['mean']:.6g} "
        f"future={result['errors']['state_discontinuity']['mean']:.6g} "
        f"visual={result['errors']['appearance_perturbation']['mean']:.6g} "
        f"action_ratio={ratios['counterfactual_action']['mean']:.3f} "
        f"future_ratio={ratios['state_discontinuity']['mean']:.3f} "
        f"visual_ratio={ratios['appearance_perturbation']['mean']:.3f} "
        f"action_gt={fractions['counterfactual_action']:.3f} "
        f"future_gt={fractions['state_discontinuity']:.3f} "
        f"visual_gt={fractions['appearance_perturbation']:.3f} "
        f"output={path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
