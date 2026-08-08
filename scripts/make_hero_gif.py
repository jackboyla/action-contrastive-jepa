"""Build the README hero animation.

Composites the five benchmark task animations into one strip and labels each
panel with the AC-MTM planning-success delta against the SIGReg (LeWM) baseline
reported in the paper.

Usage:
    python scripts/make_hero_gif.py --out assets/hero.gif
"""

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageSequence

TASKS = [
    ("tworoom", "TwoRoom", +5.2),
    ("reacher", "Reacher", -0.5),
    ("pusht", "PushT", -6.5),
    ("cube", "OGB-Cube", +12.6),
    ("scene", "OGB-Scene", +22.0),
]

TITLE = "AC-MTM: anti-collapse from the transitions, not from a Gaussian prior"
SUBTITLE = "planning success delta (points) vs. SIGReg (LeWM) — three seeds, matched CEM planner"

BG = (13, 17, 23)
PANEL_EDGE = (48, 54, 66)
TITLE_FG = (230, 237, 243)
SUB_FG = (125, 133, 144)
LABEL_FG = (201, 209, 217)
WIN = (63, 185, 80)
LOSS = (248, 81, 73)

PANEL = 160
GUTTER = 16
MARGIN = 26
TOP = 74
LABEL_BAND = 64

FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")


def load_font(name: str, size: int) -> ImageFont.FreeTypeFont:
    path = FONT_DIR / name
    if path.exists():
        return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def read_frames(path: Path) -> list[Image.Image]:
    with Image.open(path) as im:
        return [f.convert("RGB").copy() for f in ImageSequence.Iterator(im)]


def rounded(img: Image.Image, radius: int) -> Image.Image:
    mask = Image.new("L", img.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, img.width - 1, img.height - 1], radius, fill=255)
    out = Image.new("RGB", img.size, BG)
    out.paste(img, (0, 0), mask)
    return out


def build(assets: Path, out: Path, colors: int) -> None:
    clips = {name: read_frames(assets / f"{name}.gif") for name, _, _ in TASKS}
    n_frames = max(len(v) for v in clips.values())

    width = 2 * MARGIN + len(TASKS) * PANEL + (len(TASKS) - 1) * GUTTER
    height = TOP + PANEL + LABEL_BAND

    f_title = load_font("DejaVuSans-Bold.ttf", 19)
    f_sub = load_font("DejaVuSans.ttf", 13)
    f_task = load_font("DejaVuSans-Bold.ttf", 15)
    f_delta = load_font("DejaVuSans-Bold.ttf", 20)

    frames = []
    for i in range(n_frames):
        canvas = Image.new("RGB", (width, height), BG)
        draw = ImageDraw.Draw(canvas)
        draw.text((MARGIN, 20), TITLE, font=f_title, fill=TITLE_FG)
        draw.text((MARGIN, 46), SUBTITLE, font=f_sub, fill=SUB_FG)

        for col, (name, label, delta) in enumerate(TASKS):
            x = MARGIN + col * (PANEL + GUTTER)
            clip = clips[name]
            panel = clip[i % len(clip)].resize((PANEL, PANEL), Image.LANCZOS)
            canvas.paste(rounded(panel, 10), (x, TOP))
            draw.rounded_rectangle(
                [x, TOP, x + PANEL - 1, TOP + PANEL - 1], 10, outline=PANEL_EDGE, width=1
            )

            ty = TOP + PANEL + 12
            draw.text((x + PANEL / 2, ty), label, font=f_task, fill=LABEL_FG, anchor="ma")
            text = f"{delta:+.1f}"
            draw.text(
                (x + PANEL / 2, ty + 21),
                text,
                font=f_delta,
                fill=WIN if delta > 0 else LOSS,
                anchor="ma",
            )

        frames.append(canvas.quantize(colors=colors, method=Image.MEDIANCUT, dither=Image.FLOYDSTEINBERG))

    out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        out,
        save_all=True,
        append_images=frames[1:],
        duration=80,
        loop=0,
        optimize=True,
        disposal=2,
    )
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB, {len(frames)} frames, {width}x{height})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", type=Path, default=Path("assets/datasets"))
    parser.add_argument("--out", type=Path, default=Path("assets/hero.gif"))
    parser.add_argument("--colors", type=int, default=128)
    args = parser.parse_args()
    build(args.assets, args.out, args.colors)


if __name__ == "__main__":
    main()
