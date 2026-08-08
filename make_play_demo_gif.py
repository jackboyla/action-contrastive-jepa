#!/usr/bin/env python3
"""Generate the README GIF for the interactive ``play.py`` demo.

The GIF is built from a scripted Reacher episode using the same ``Episode``
logic as the real demo. It intentionally avoids screen recording so it can run
headlessly as long as MuJoCo offscreen rendering works.

Usage:
    MUJOCO_GL=glfw .venv/bin/python make_play_demo_gif.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from play import Episode, TASKS


PANEL = 260
MARGIN = 24
TOP = 68
GAP = 28
WIDTH = 820
HEIGHT = 505

BG = (18, 18, 22)
FG = (232, 232, 238)
DIM = (150, 150, 162)
ACC = (110, 170, 255)
WARN = (236, 184, 75)
OK = (96, 216, 128)
PANEL_BG = (28, 34, 42)
BORDER = (55, 94, 132)
BAR_BG = (45, 45, 54)

FONT_CANDIDATES = [
    "/System/Library/Fonts/Menlo.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
]


def load_font(size: int) -> ImageFont.ImageFont:
    for path in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


F_TITLE = load_font(28)
F_MED = load_font(18)
F_SMALL = load_font(14)
F_TINY = load_font(12)


def text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], s: str, font=F_SMALL, fill=FG):
    draw.text(xy, s, font=font, fill=fill)


def panelize(arr: np.ndarray) -> Image.Image:
    img = Image.fromarray(np.asarray(arr, dtype=np.uint8))
    return img.resize((PANEL, PANEL), Image.Resampling.BILINEAR)


def draw_key(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    label: str,
    *,
    active: bool = False,
):
    fill = ACC if active else (36, 36, 44)
    outline = (175, 205, 255) if active else (75, 75, 84)
    draw.rounded_rectangle((x, y, x + 42, y + 30), radius=6, fill=fill, outline=outline, width=2)
    bbox = draw.textbbox((0, 0), label, font=F_TINY)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    label_fill = (10, 14, 20) if active else FG
    draw.text((x + 21 - tw / 2, y + 15 - th / 2 - 1), label, font=F_TINY, fill=label_fill)


def draw_bars(draw: ImageDraw.ImageDraw, action: np.ndarray, x: int, y: int):
    for i, value in enumerate(action):
        bx = x + i * 72
        draw.rectangle((bx, y, bx + 58, y + 18), fill=BAR_BG)
        draw.line((bx + 29, y, bx + 29, y + 18), fill=DIM, width=1)
        width = int(abs(float(value)) * 29)
        if value >= 0:
            draw.rectangle((bx + 29, y, bx + 29 + max(1, width), y + 18), fill=ACC)
        else:
            draw.rectangle((bx + 29 - max(1, width), y, bx + 29, y + 18), fill=ACC)
        text(draw, (bx + 21, y + 22), f"a{i}", F_TINY, DIM)


def draw_frame(
    obs: np.ndarray,
    goal: np.ndarray,
    steps: int,
    budget: int,
    distance: float,
    action: np.ndarray,
    solved: bool,
) -> Image.Image:
    frame = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(frame)
    text(draw, (MARGIN, 18), "TASK  REACHER", F_TITLE, ACC)
    text(draw, (MARGIN + 245, 24), "TURN-BASED   goal:rollout   mag:0.6", F_MED, DIM)

    left_x = MARGIN
    right_x = MARGIN + PANEL + GAP
    for x in (left_x, right_x):
        draw.rectangle(
            (x - 6, TOP - 6, x + PANEL + 6, TOP + PANEL + 6),
            fill=PANEL_BG,
            outline=BORDER,
            width=4,
        )
    frame.paste(panelize(obs), (left_x, TOP))
    frame.paste(panelize(goal), (right_x, TOP))
    text(draw, (left_x, TOP + PANEL + 10), "OBSERVATION (what the model sees)", F_TINY, DIM)
    text(draw, (right_x, TOP + PANEL + 10), "GOAL", F_TINY, DIM)

    hud_y = TOP + PANEL + 36
    text(draw, (MARGIN, hud_y), f"step {steps:02d} / {budget}", F_MED, FG)
    if solved:
        text(draw, (MARGIN + 170, hud_y), "GOAL REACHED  ✓", F_MED, OK)
    else:
        text(draw, (MARGIN + 170, hud_y), f"max|dqpos| rad: {distance:.3f}", F_MED, WARN)

    action_text = np.array2string(action, precision=2, suppress_small=True, floatmode="fixed")
    text(draw, (MARGIN, hud_y + 32), "action " + action_text, F_MED, FG)
    draw_bars(draw, action, MARGIN + 220, hud_y + 34)

    text(
        draw,
        (MARGIN, hud_y + 70),
        "Torque the 2-joint arm so it matches the goal arm configuration.",
        F_SMALL,
        DIM,
    )
    text(
        draw,
        (MARGIN, hud_y + 92),
        "Left/Right: shoulder torque     Up/Down: wrist torque",
        F_SMALL,
        FG,
    )
    text(
        draw,
        (MARGIN, hud_y + 116),
        "R reset   N next task   Space no-op   T realtime   [ ] magnitude   H help   Esc quit",
        F_TINY,
        DIM,
    )

    key_x = WIDTH - MARGIN - 135
    key_y = hud_y + 16
    draw_key(draw, key_x + 46, key_y, "↑", active=action[1] > 0.05)
    draw_key(draw, key_x, key_y + 34, "←", active=action[0] < -0.05)
    draw_key(draw, key_x + 46, key_y + 34, "↓", active=action[1] < -0.05)
    draw_key(draw, key_x + 92, key_y + 34, "→", active=action[0] > 0.05)
    text(draw, (key_x - 2, key_y + 72), "held keys repeat", F_TINY, DIM)
    return frame


def build_frames(seed: int, gain: float, hold_frames: int) -> list[Image.Image]:
    task = TASKS["reacher"]
    episode = Episode(task, seed=seed, goal_kind="rollout", steps_ahead=25, budget=task.default_budget)
    frames: list[Image.Image] = []
    last_action = np.zeros(task.action_dim, dtype=np.float32)
    try:
        for _ in range(24):
            unwrapped = episode.env.unwrapped
            err = unwrapped.env.task.target_qpos - unwrapped.env.physics.data.qpos
            action = np.clip(gain * err, -1, 1).astype(np.float32)
            frames.append(
                draw_frame(
                    episode.obs_img,
                    episode.goal_img,
                    episode.steps,
                    episode.budget,
                    episode.distance,
                    action,
                    episode.solved,
                )
            )
            last_action = action
            if episode.solved:
                break
            episode.step(action)
        frames.append(
            draw_frame(
                episode.obs_img,
                episode.goal_img,
                episode.steps,
                episode.budget,
                episode.distance,
                np.zeros_like(last_action),
                episode.solved,
            )
        )
        frames.extend(frames[-1].copy() for _ in range(hold_frames))
        return frames
    finally:
        episode.close()


def save_gif(frames: list[Image.Image], out: Path, duration_ms: int):
    out.parent.mkdir(parents=True, exist_ok=True)
    palette_frames = [f.convert("P", palette=Image.Palette.ADAPTIVE, colors=128) for f in frames]
    palette_frames[0].save(
        out,
        save_all=True,
        append_images=palette_frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
        disposal=2,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("assets/play_demo.gif"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gain", type=float, default=0.6)
    parser.add_argument("--duration-ms", type=int, default=110)
    parser.add_argument("--hold-frames", type=int, default=5)
    args = parser.parse_args()

    frames = build_frames(seed=args.seed, gain=args.gain, hold_frames=args.hold_frames)
    save_gif(frames, args.out, duration_ms=args.duration_ms)
    print(f"wrote {args.out} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
