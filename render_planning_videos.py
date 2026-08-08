"""Render saved eval planning latents with an auxiliary LeWM decoder."""

from __future__ import annotations

import argparse
import html
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw

from decoder import build_decoder_from_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decode eval planning_artifacts.pt files into imagined videos."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Task folder or artifact root containing task_* folders.",
    )
    parser.add_argument(
        "--decoder",
        type=Path,
        required=True,
        help="Decoder checkpoint produced by train_decoder.py.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Torch device. Defaults to cuda, then mps, then cpu.",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-tasks", type=int, default=None)
    return parser.parse_args()


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def denormalize_imagenet(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
    return (x * std + mean).clamp(0.0, 1.0)


def decoder_output_to_rgb(x: torch.Tensor, target_space: str) -> torch.Tensor:
    if target_space == "imagenet_normalized":
        return denormalize_imagenet(x.float())
    return x.float().clamp(0.0, 1.0)


def find_task_dirs(root: Path) -> list[Path]:
    root = root.expanduser().resolve()
    if root.is_file():
        root = root.parent
    if (root / "planning_artifacts.pt").exists() or (
        root / "planning_latents.pt"
    ).exists():
        return [root]
    return [
        path
        for path in sorted(root.rglob("task_*"))
        if path.is_dir()
        and (
            (path / "planning_artifacts.pt").exists()
            or (path / "planning_latents.pt").exists()
        )
    ]


def load_planning_payload(task_dir: Path) -> dict:
    artifact_path = (
        task_dir / "planning_artifacts.pt"
        if (task_dir / "planning_artifacts.pt").exists()
        else task_dir / "planning_latents.pt"
    )
    payload = torch.load(artifact_path, map_location="cpu", weights_only=False)
    if not payload.get("replans"):
        raise ValueError(f"No replans found in {artifact_path}")
    return payload


def executed_horizon(record: dict) -> int:
    for key in (
        "executed_action_plan",
        "executed_latent_action_plan",
        "executed_decoded_action_plan",
    ):
        value = record.get(key)
        if torch.is_tensor(value):
            return int(value.shape[0])
    if "executed_horizon" in record:
        return int(record["executed_horizon"])
    return int(as_latent_sequence(record["state_plan_latents"]).shape[0])


def as_latent_sequence(latents: torch.Tensor) -> torch.Tensor:
    while latents.ndim > 2 and latents.size(0) == 1:
        latents = latents[0]
    if latents.ndim != 2:
        raise ValueError(f"expected latent sequence shaped (T, D), got {latents.shape}")
    return latents


def build_executed_latent_sequence(replans: list[dict]) -> torch.Tensor:
    segments = []
    for record in replans:
        state_latents = as_latent_sequence(record["state_plan_latents"])
        keep = min(executed_horizon(record), state_latents.size(0))
        if keep > 0:
            segments.append(state_latents[:keep])
    if not segments:
        raise ValueError("No executed latent segments found to render.")
    return torch.cat(segments, dim=0)


@torch.inference_mode()
def decode_latents(
    *,
    decoder: torch.nn.Module,
    target_space: str,
    latents: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    frames = []
    for start in range(0, latents.size(0), batch_size):
        batch = latents[start : start + batch_size].to(device)
        rgb = decoder_output_to_rgb(decoder(batch), target_space)
        rgb = rgb.mul(255).round().byte().permute(0, 2, 3, 1).cpu().numpy()
        frames.append(rgb)
    return np.concatenate(frames, axis=0)


def write_video(path: Path, frames: np.ndarray, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(path, fps=fps, codec="libx264") as writer:
        for frame in frames:
            writer.append_data(frame)


def load_real_rollout_frames(task_dir: Path) -> np.ndarray | None:
    rollout_path = task_dir / "rollout.mp4"
    if not rollout_path.exists():
        return None
    frames = imageio.mimread(rollout_path)
    if not frames:
        return None
    video = np.stack(frames)
    if video.shape[-1] == 4:
        video = video[..., :3]
    return video


def resize_nearest(frames: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    target_h, target_w = target_hw
    if frames.shape[1] == target_h and frames.shape[2] == target_w:
        return frames
    y_idx = np.clip(
        np.round(np.linspace(0, frames.shape[1] - 1, target_h)).astype(np.int64),
        0,
        frames.shape[1] - 1,
    )
    x_idx = np.clip(
        np.round(np.linspace(0, frames.shape[2] - 1, target_w)).astype(np.int64),
        0,
        frames.shape[2] - 1,
    )
    return frames[:, y_idx][:, :, x_idx]


def expand_imagined_frames(
    imagined_frames: np.ndarray, *, action_block: int, target_len: int | None
) -> np.ndarray:
    expanded = np.repeat(imagined_frames, repeats=max(1, action_block), axis=0)
    if target_len is None:
        return expanded
    if len(expanded) >= target_len:
        return expanded[:target_len]
    pad = np.repeat(expanded[-1:], repeats=target_len - len(expanded), axis=0)
    return np.concatenate([expanded, pad], axis=0)


def build_comparison_video(
    imagined_frames: np.ndarray, real_frames: np.ndarray
) -> np.ndarray:
    imagined_frames = resize_nearest(imagined_frames, real_frames.shape[1:3])
    frame_count = min(len(imagined_frames), len(real_frames))
    return np.concatenate(
        [imagined_frames[:frame_count], real_frames[:frame_count]], axis=2
    )


def save_replan_grid(
    *,
    decoded_replans: list[np.ndarray],
    replans: list[dict],
    path: Path,
) -> None:
    if not decoded_replans:
        return
    cell_h, cell_w = decoded_replans[0].shape[1:3]
    label_w = 110
    header_h = 30
    row_h = cell_h + 22
    columns = max(frames.shape[0] for frames in decoded_replans)
    canvas = Image.new(
        "RGB",
        (label_w + columns * cell_w, header_h + len(decoded_replans) * row_h),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    for col in range(columns):
        draw.text((label_w + col * cell_w + 6, 8), f"h+{col + 1}", fill=(30, 30, 30))

    for row, (frames, record) in enumerate(zip(decoded_replans, replans, strict=True)):
        y = header_h + row * row_h
        env_step = int(record.get("env_step", row))
        draw.text((8, y + cell_h // 2 - 8), f"step {env_step}", fill=(30, 30, 30))
        for col, frame in enumerate(frames):
            image = Image.fromarray(frame)
            canvas.paste(image, (label_w + col * cell_w, y))

    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def render_task(
    *,
    task_dir: Path,
    decoder: torch.nn.Module,
    target_space: str,
    device: torch.device,
    batch_size: int,
    fps: int,
    overwrite: bool,
) -> dict:
    imagined_video_path = task_dir / "imagined_trajectory.mp4"
    comparison_video_path = task_dir / "imagined_vs_real.mp4"
    grid_path = task_dir / "imagined_replans.png"
    summary_path = task_dir / "imagined_summary.pt"

    if (
        not overwrite
        and imagined_video_path.exists()
        and grid_path.exists()
        and summary_path.exists()
    ):
        return {"task_dir": str(task_dir), "skipped": True}

    payload = load_planning_payload(task_dir)
    replans = payload["replans"]
    action_block = int(payload.get("action_block", payload.get("real_action_block", 1)))

    executed_latents = build_executed_latent_sequence(replans)
    imagined_frames = decode_latents(
        decoder=decoder,
        target_space=target_space,
        latents=executed_latents,
        device=device,
        batch_size=batch_size,
    )

    decoded_replans = [
        decode_latents(
            decoder=decoder,
            target_space=target_space,
            latents=as_latent_sequence(record["state_plan_latents"]),
            device=device,
            batch_size=batch_size,
        )
        for record in replans
    ]

    real_frames = load_real_rollout_frames(task_dir)
    imagined_video_frames = expand_imagined_frames(
        imagined_frames,
        action_block=action_block,
        target_len=None if real_frames is None else len(real_frames),
    )
    write_video(imagined_video_path, imagined_video_frames, fps=fps)

    if real_frames is not None:
        write_video(
            comparison_video_path,
            build_comparison_video(imagined_video_frames, real_frames),
            fps=fps,
        )

    save_replan_grid(decoded_replans=decoded_replans, replans=replans, path=grid_path)
    summary = {
        "task_dir": str(task_dir),
        "imagined_video": str(imagined_video_path),
        "comparison_video": str(comparison_video_path)
        if comparison_video_path.exists()
        else None,
        "replan_grid": str(grid_path),
        "imagined_frame_count": int(imagined_video_frames.shape[0]),
        "latent_frame_count": int(imagined_frames.shape[0]),
        "action_block": action_block,
        "replan_horizons": [
            int(as_latent_sequence(record["state_plan_latents"]).shape[0])
            for record in replans
        ],
        "executed_horizons": [executed_horizon(record) for record in replans],
        "skipped": False,
    }
    torch.save(summary, summary_path)
    return summary


def write_index(root: Path, summaries: list[dict]) -> None:
    if not summaries:
        return
    common_root = root if root.is_dir() else root.parent
    cards = []
    for summary in summaries:
        task_dir = Path(summary["task_dir"])
        grid = task_dir / "imagined_replans.png"
        video = task_dir / "imagined_trajectory.mp4"
        rel_grid = html.escape(str(grid.relative_to(common_root)))
        rel_video = html.escape(str(video.relative_to(common_root)))
        cards.append(
            f"""
            <section>
              <h2>{html.escape(task_dir.name)}</h2>
              <video src="{rel_video}" controls loop muted></video>
              <img src="{rel_grid}" alt="decoded replans">
            </section>
            """
        )

    (common_root / "imagined_index.html").write_text(
        f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>LeWM Imagined Planning Rollouts</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 28px; color: #1f2933; }}
    video, img {{ display: block; max-width: 100%; border: 1px solid #d8dee4; margin: 8px 0 22px; }}
    section {{ margin-bottom: 30px; }}
  </style>
</head>
<body>
  <h1>LeWM Imagined Planning Rollouts</h1>
  {"".join(cards)}
</body>
</html>
""",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    device = torch.device(resolve_device(args.device))
    decoder, checkpoint = build_decoder_from_checkpoint(args.decoder)
    decoder = decoder.to(device).eval()
    decoder.requires_grad_(False)
    target_space = str(checkpoint.get("target_space", "rgb"))

    task_dirs = find_task_dirs(args.input)
    if args.max_tasks is not None:
        task_dirs = task_dirs[: args.max_tasks]
    if not task_dirs:
        raise FileNotFoundError(f"No planning artifacts found under {args.input}")

    print(f"Found {len(task_dirs)} task folders.")
    summaries = []
    for idx, task_dir in enumerate(task_dirs, start=1):
        print(f"[{idx}/{len(task_dirs)}] Rendering {task_dir}")
        summaries.append(
            render_task(
                task_dir=task_dir,
                decoder=decoder,
                target_space=target_space,
                device=device,
                batch_size=int(args.batch_size),
                fps=int(args.fps),
                overwrite=bool(args.overwrite),
            )
        )

    write_index(args.input.expanduser().resolve(), summaries)


if __name__ == "__main__":
    main()
