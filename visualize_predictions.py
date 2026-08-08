"""Visualize LeWM latent predictions on offline task trajectories.

The original paper visualizes decoded latent rollouts using an auxiliary
decoder. This script only renders predicted frames when a decoder checkpoint is
provided; otherwise it writes latent-space metrics and skips prediction images.
"""

import csv
import html
import json
import os
import sys
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MUJOCO_GL", "glfw" if sys.platform == "darwin" else "egl")

from project_paths import configure_stablewm_home

configure_stablewm_home()

import hydra
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from PIL import Image, ImageDraw
from torchvision.transforms import v2 as transforms

from decoder import build_decoder_from_checkpoint


ROOT = Path(__file__).resolve().parent


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_eval_cfg(task: str) -> DictConfig:
    path = ROOT / "config" / "eval" / f"{task}.yaml"
    if not path.exists():
        choices = sorted(p.stem for p in (ROOT / "config" / "eval").glob("*.yaml"))
        raise FileNotFoundError(f"unknown task '{task}'. Available tasks: {choices}")
    return OmegaConf.load(path)


def infer_context_size(model: torch.nn.Module, cfg: DictConfig) -> int:
    if cfg.context_size is not None:
        return int(cfg.context_size)
    future_predictor = getattr(model, "future_predictor", None)
    if future_predictor is not None and hasattr(future_predictor, "num_context"):
        return int(future_predictor.num_context)
    predictor = getattr(model, "predictor", None)
    if predictor is not None and hasattr(predictor, "pos_embedding"):
        return int(predictor.pos_embedding.size(1))
    return 3


def infer_horizon(model: torch.nn.Module, cfg: DictConfig, eval_cfg: DictConfig) -> int:
    if cfg.horizon is not None:
        return int(cfg.horizon)
    future_predictor = getattr(model, "future_predictor", None)
    if future_predictor is not None and hasattr(future_predictor, "horizon"):
        return int(future_predictor.horizon)
    return int(eval_cfg.plan_config.get("horizon", 5))


def infer_rollout_mode(model: torch.nn.Module, cfg: DictConfig) -> str:
    if cfg.rollout_mode != "auto":
        return str(cfg.rollout_mode)
    return str(getattr(model, "rollout_mode", "autoregressive"))


def make_img_preprocessor(img_size: int):
    stats = spt.data.dataset_stats.ImageNet
    return transforms.Compose(
        [
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**stats),
            transforms.Resize(size=(img_size, img_size), antialias=True),
        ]
    )


def preprocess_pixels(pixels: torch.Tensor, img_size: int) -> torch.Tensor:
    """Normalize channel-first uint8 frames while preserving leading dimensions."""
    shape = pixels.shape
    flat = pixels.reshape(-1, *shape[-3:])
    flat = make_img_preprocessor(img_size)(flat)
    return flat.reshape(*shape[:-3], *flat.shape[-3:])


def sample_trajectory_starts(
    dataset: swm.data.HDF5Dataset,
    *,
    num_rollouts: int,
    seq_env_steps: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    lengths = np.asarray(dataset.lengths)
    valid_counts = np.maximum(lengths - seq_env_steps, 0)
    valid_eps = np.flatnonzero(valid_counts > 0)
    if len(valid_eps) == 0:
        raise ValueError(
            f"no episodes are long enough for {seq_env_steps} environment steps"
        )

    probs = valid_counts[valid_eps] / valid_counts[valid_eps].sum()
    episodes = rng.choice(valid_eps, size=num_rollouts, replace=True, p=probs)
    starts = np.array(
        [rng.integers(0, valid_counts[ep]) for ep in episodes], dtype=np.int64
    )
    return episodes.astype(np.int64), starts


def load_trajectories(
    dataset: swm.data.HDF5Dataset,
    *,
    episodes: np.ndarray,
    starts: np.ndarray,
    seq_len: int,
    frameskip: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    ends = starts + seq_len * frameskip
    chunks = dataset.load_chunk(episodes, starts, ends)
    pixels = torch.stack([chunk["pixels"] for chunk in chunks])
    actions = torch.stack([chunk["action"] for chunk in chunks])
    actions = torch.nan_to_num(actions, nan=0.0)
    if pixels.size(1) != seq_len:
        raise RuntimeError(f"expected {seq_len} frames, got {pixels.size(1)}")
    if actions.size(1) != seq_len:
        raise RuntimeError(f"expected {seq_len} action blocks, got {actions.size(1)}")
    return pixels, actions


@torch.inference_mode()
def predict_rollouts(
    model: torch.nn.Module,
    *,
    pixels: torch.Tensor,
    actions: torch.Tensor,
    context_size: int,
    horizon: int,
    img_size: int,
    mode: str,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    pixels = preprocess_pixels(pixels, img_size).to(device)
    actions = actions.to(device)

    target_emb = model.encode({"pixels": pixels})["emb"]
    action_sequence = actions[:, : context_size + horizon - 1].unsqueeze(1)
    info = {"pixels": pixels[:, None, :context_size]}

    if mode == "direct_horizon":
        predicted = model.rollout_direct(info, action_sequence)
    elif mode == "autoregressive":
        predicted = model.rollout(info, action_sequence, history_size=context_size)
    else:
        raise ValueError(
            "rollout_mode must be 'auto', 'autoregressive', or 'direct_horizon'"
        )

    pred_emb = predicted["predicted_emb"][
        :, 0, context_size : context_size + horizon
    ]
    tgt_emb = target_emb[:, context_size : context_size + horizon]
    return pred_emb.detach().cpu(), tgt_emb.detach().cpu()


def denormalize_imagenet(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
    return (x * std + mean).clamp(0.0, 1.0)


def decoder_output_to_rgb(x: torch.Tensor, target_space: str) -> torch.Tensor:
    if target_space == "imagenet_normalized":
        return denormalize_imagenet(x.float())
    if target_space == "rgb":
        return x.float().clamp(0.0, 1.0)
    return x.float().clamp(0.0, 1.0)


@torch.inference_mode()
def decode_latents(
    decoder_path: str | Path,
    pred_emb: torch.Tensor,
    *,
    device: torch.device | str,
    batch_size: int,
) -> torch.Tensor:
    decoder, checkpoint = build_decoder_from_checkpoint(decoder_path)
    decoder = decoder.to(device).eval()
    target_space = str(checkpoint.get("target_space", "rgb"))
    decoded = []
    flat = pred_emb.reshape(-1, pred_emb.size(-1))
    for start in range(0, flat.size(0), batch_size):
        batch = flat[start : start + batch_size].to(device)
        rgb = decoder_output_to_rgb(decoder(batch), target_space)
        decoded.append(rgb.detach().cpu())
    return torch.cat(decoded, dim=0).reshape(
        pred_emb.size(0), pred_emb.size(1), 3, decoder.img_size, decoder.img_size
    )


def frame_to_image(frame: torch.Tensor, cell_size: int) -> Image.Image:
    arr = frame.detach().cpu().numpy()
    if arr.shape[0] in (1, 3):
        arr = np.moveaxis(arr, 0, -1)
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.dtype != np.uint8:
        if arr.max() <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    image = Image.fromarray(arr)
    return image.resize((cell_size, cell_size), Image.Resampling.BILINEAR)


def draw_text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str) -> None:
    draw.multiline_text(xy, text, fill=(30, 30, 30), spacing=3)


def save_rollout_grid(
    *,
    context: torch.Tensor,
    target: torch.Tensor,
    pred_frames: torch.Tensor,
    mse: np.ndarray,
    cosine: np.ndarray,
    path: Path,
    cell_size: int,
    pred_label: str,
) -> None:
    columns = context.size(0) + target.size(0)
    label_w = 92
    header_h = 32
    caption_h = 42
    row_h = cell_size + caption_h
    width = label_w + columns * cell_size
    height = header_h + 2 * row_h
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    for col in range(columns):
        x = label_w + col * cell_size + 6
        label = (
            f"ctx {col + 1}"
            if col < context.size(0)
            else f"h+{col - context.size(0) + 1}"
        )
        draw_text(draw, (x, 8), label)

    rows = [
        ("Target", torch.cat([context, target], dim=0)),
        (pred_label, torch.cat([context, pred_frames], dim=0)),
    ]
    for row_idx, (label, frames) in enumerate(rows):
        y = header_h + row_idx * row_h
        draw_text(draw, (10, y + cell_size // 2 - 8), label)
        for col, frame in enumerate(frames):
            x = label_w + col * cell_size
            canvas.paste(frame_to_image(frame, cell_size), (x, y))
            if row_idx == 1 and col >= context.size(0):
                h = col - context.size(0)
                caption = f"mse {mse[h]:.3f}\ncos {cosine[h]:.3f}"
                draw_text(draw, (x + 5, y + cell_size + 4), caption)

    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def make_video_frame(
    target: torch.Tensor,
    pred: torch.Tensor,
    *,
    title: str,
    cell_size: int,
) -> np.ndarray:
    label_h = 34
    width = cell_size * 2
    height = cell_size + label_h
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw_text(draw, (8, 8), "target")
    draw_text(draw, (cell_size + 8, 8), title)
    canvas.paste(frame_to_image(target, cell_size), (0, label_h))
    canvas.paste(frame_to_image(pred, cell_size), (cell_size, label_h))
    return np.asarray(canvas)


def save_rollout_gif(
    *,
    target: torch.Tensor,
    pred_frames: torch.Tensor,
    mse: np.ndarray,
    path: Path,
    cell_size: int,
    fps: int,
    loop: int = 0,
) -> None:
    if fps <= 0:
        raise ValueError("fps must be positive")

    frames = []
    for h in range(target.size(0)):
        title = f"pred h+{h + 1} mse {mse[h]:.3f}"
        frames.append(
            make_video_frame(
                target[h], pred_frames[h], title=title, cell_size=cell_size
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = max(1, round(1000 / fps))
    pil_frames = [Image.fromarray(frame) for frame in frames]
    pil_frames[0].save(
        path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=loop,
        disposal=2,
    )


def save_metrics_csv(
    *,
    mse: np.ndarray,
    cosine: np.ndarray,
    episodes: Iterable[int],
    starts: Iterable[int],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["rollout", "episode", "start_step", "horizon", "mse", "cosine_distance"]
        )
        for rollout_idx, (episode, start) in enumerate(zip(episodes, starts)):
            for h in range(mse.shape[1]):
                writer.writerow(
                    [
                        rollout_idx,
                        int(episode),
                        int(start),
                        h + 1,
                        mse[rollout_idx, h],
                        cosine[rollout_idx, h],
                    ]
                )


def save_error_plot(mse: np.ndarray, cosine: np.ndarray, path: Path) -> None:
    horizons = np.arange(1, mse.shape[1] + 1)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), constrained_layout=True)
    for ax, values, title, ylabel in [
        (axes[0], mse, "Latent MSE", "MSE"),
        (axes[1], cosine, "Cosine Distance", "1 - cosine"),
    ]:
        mean = values.mean(axis=0)
        std = values.std(axis=0)
        ax.plot(horizons, mean, marker="o", color="#1f77b4")
        ax.fill_between(
            horizons, mean - std, mean + std, color="#1f77b4", alpha=0.2
        )
        ax.set_title(title)
        ax.set_xlabel("Prediction horizon")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def clear_rollout_artifacts(output_dir: Path) -> None:
    for pattern in ("rollout_*.png", "rollout_*.gif"):
        for path in output_dir.glob(pattern):
            if path.is_file():
                path.unlink()


def write_index(
    *,
    cfg: DictConfig,
    metadata: dict,
    output_dir: Path,
    grid_paths: list[Path],
    gif_paths: list[Path | None],
) -> None:
    cards = []
    for grid_path, gif_path in zip(grid_paths, gif_paths):
        grid_rel = html.escape(grid_path.name)
        gif_rel = html.escape(gif_path.name) if gif_path else ""
        gif_html = f'<img src="{gif_rel}" alt="rollout gif">' if gif_path else ""
        cards.append(
            f"""
            <section>
              <h2>{html.escape(grid_path.stem)}</h2>
              <img src="{grid_rel}" alt="rollout grid">
              {gif_html}
            </section>
            """
        )

    body = (
        "\n".join(cards)
        if cards
        else "<p>No decoded prediction frames were rendered.</p>"
    )
    config_text = html.escape(OmegaConf.to_yaml(cfg))
    metadata_text = html.escape(json.dumps(metadata, indent=2))
    renderer = html.escape(metadata.get("renderer", "none"))
    (output_dir / "index.html").write_text(
        f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>LeWM Prediction Visualization</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 28px; color: #1f2933; }}
    img {{ max-width: 100%; border: 1px solid #d8dee4; margin: 8px 0 20px; }}
    section {{ margin-bottom: 28px; }}
    pre {{ background: #f6f8fa; padding: 12px; overflow-x: auto; }}
  </style>
</head>
<body>
  <h1>LeWM Prediction Visualization</h1>
  <p>Predicted latents are rendered with: {renderer}.</p>
  <img src="horizon_errors.png" alt="horizon errors">
  {body}
  <h2>Metadata</h2>
  <pre>{metadata_text}</pre>
  <h2>Config</h2>
  <pre>{config_text}</pre>
</body>
</html>
""",
        encoding="utf-8",
    )


def default_output_dir(cfg: DictConfig) -> Path:
    if cfg.output_dir is not None:
        return Path(str(cfg.output_dir)).expanduser().resolve()
    policy_name = str(cfg.policy).replace("/", "__")
    return Path(
        swm.data.utils.get_cache_dir(), "visualizations", str(cfg.task), policy_name
    )


@hydra.main(
    version_base=None, config_path="./config/visualize", config_name="predictions"
)
def run(cfg: DictConfig) -> None:
    if str(cfg.policy) == "random":
        raise ValueError(
            "visualization requires a model checkpoint policy, not 'random'"
        )

    eval_cfg = load_eval_cfg(str(cfg.task))
    dataset_name = str(cfg.dataset_name or eval_cfg.eval.dataset_name)
    frameskip = int(cfg.frameskip or eval_cfg.plan_config.get("action_block", 1))
    img_size = int(cfg.img_size or eval_cfg.eval.get("img_size", 224))
    device = resolve_device(str(cfg.device))
    cache_dir = Path(cfg.cache_dir or swm.data.utils.get_cache_dir())
    output_dir = default_output_dir(cfg)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model '{cfg.policy}' on {device}")
    model = swm.policy.AutoCostModel(str(cfg.policy), cache_dir=cache_dir)
    model = model.to(device).eval()
    model.requires_grad_(False)
    if hasattr(model, "interpolate_pos_encoding"):
        model.interpolate_pos_encoding = True

    mode = infer_rollout_mode(model, cfg)
    context_size = infer_context_size(model, cfg)
    horizon = infer_horizon(model, cfg, eval_cfg)
    if mode == "direct_horizon":
        max_horizon = getattr(
            getattr(model, "future_predictor", None), "horizon", horizon
        )
        if horizon > max_horizon:
            raise ValueError(
                f"direct_horizon checkpoint supports horizon {max_horizon}, got {horizon}"
            )

    seq_len = context_size + horizon
    seq_env_steps = seq_len * frameskip

    print(f"Loading dataset '{dataset_name}' from {cache_dir}")
    dataset = swm.data.HDF5Dataset(
        dataset_name,
        frameskip=frameskip,
        keys_to_load=["pixels", "action"],
        keys_to_cache=["action"],
        cache_dir=cache_dir,
    )

    episodes, starts = sample_trajectory_starts(
        dataset,
        num_rollouts=int(cfg.num_rollouts),
        seq_env_steps=seq_env_steps,
        seed=int(cfg.seed),
    )
    pixels, actions = load_trajectories(
        dataset,
        episodes=episodes,
        starts=starts,
        seq_len=seq_len,
        frameskip=frameskip,
    )

    pred_emb, tgt_emb = predict_rollouts(
        model,
        pixels=pixels,
        actions=actions,
        context_size=context_size,
        horizon=horizon,
        img_size=img_size,
        mode=mode,
        device=device,
    )
    mse = (pred_emb - tgt_emb).pow(2).mean(dim=-1).numpy()
    cosine = (1.0 - F.cosine_similarity(pred_emb, tgt_emb, dim=-1)).numpy()

    pred_frames = None
    pred_label = None
    if cfg.decoder.path is not None:
        print(f"Decoding predictions with '{cfg.decoder.path}'")
        pred_frames = decode_latents(
            cfg.decoder.path,
            pred_emb,
            device=device,
            batch_size=int(cfg.decoder.batch_size),
        )
        pred_label = "Decoded"
    else:
        print("No decoder.path provided; skipping prediction frame rendering")

    save_metrics_csv(
        mse=mse,
        cosine=cosine,
        episodes=episodes,
        starts=starts,
        path=output_dir / "horizon_errors.csv",
    )
    save_error_plot(mse, cosine, output_dir / "horizon_errors.png")
    clear_rollout_artifacts(output_dir)

    grid_paths = []
    gif_paths = []
    if pred_frames is not None:
        for idx in range(pixels.size(0)):
            grid_path = output_dir / f"rollout_{idx:02d}.png"
            save_rollout_grid(
                context=pixels[idx, :context_size],
                target=pixels[idx, context_size:],
                pred_frames=pred_frames[idx],
                mse=mse[idx],
                cosine=cosine[idx],
                path=grid_path,
                cell_size=int(cfg.render.cell_size),
                pred_label=pred_label,
            )
            grid_paths.append(grid_path)

            gif_path = output_dir / f"rollout_{idx:02d}.gif"
            if cfg.render.save_gif:
                save_rollout_gif(
                    target=pixels[idx, context_size:],
                    pred_frames=pred_frames[idx],
                    mse=mse[idx],
                    path=gif_path,
                    cell_size=int(cfg.render.cell_size),
                    fps=int(cfg.render.fps),
                    loop=int(cfg.render.get("loop", 0)),
                )
                gif_paths.append(gif_path)
            else:
                gif_paths.append(None)

    metadata = {
        "task": str(cfg.task),
        "policy": str(cfg.policy),
        "dataset_name": dataset_name,
        "frameskip": frameskip,
        "img_size": img_size,
        "device": device,
        "rollout_mode": mode,
        "context_size": context_size,
        "horizon": horizon,
        "renderer": "decoder" if cfg.decoder.path is not None else "none",
        "decoder_path": None if cfg.decoder.path is None else str(cfg.decoder.path),
        "episodes": episodes.tolist(),
        "starts": starts.tolist(),
        "mean_mse": mse.mean(axis=0).tolist(),
        "mean_cosine_distance": cosine.mean(axis=0).tolist(),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    write_index(
        cfg=cfg,
        metadata=metadata,
        output_dir=output_dir,
        grid_paths=grid_paths,
        gif_paths=gif_paths,
    )

    print(f"Saved prediction visualization to {output_dir}")


if __name__ == "__main__":
    run()
