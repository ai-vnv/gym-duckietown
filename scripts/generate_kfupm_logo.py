#!/usr/bin/env python3
"""Generate a placeholder KFUPM logo PNG.

Produces a circular badge ("KFUPM" + tagline) using PIL, suitable as a
floor decal in the simulator. Drop the real KFUPM PNG into the same path
later (e.g. ``assets/kfupm_logo.png``) and the rest of the pipeline picks
it up unchanged.

Usage:
    .venv/bin/python scripts/generate_kfupm_logo.py \\
        -o assets/kfupm_logo.png --size 1024
"""
from __future__ import annotations

import argparse
import os
import sys

from PIL import Image, ImageDraw, ImageFont

# KFUPM brand-ish colors
GREEN = (0, 106, 78, 255)        # ~#006A4E
GREEN_DARK = (0, 70, 52, 255)
GOLD = (212, 175, 55, 255)
WHITE = (255, 255, 255, 255)
TRANSPARENT = (0, 0, 0, 0)


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    candidates = (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    )
    for path in candidates:
        if os.path.isfile(path):
            try:
                return ImageFont.truetype(path, size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def make_logo(side: int = 1024) -> Image.Image:
    img = Image.new("RGBA", (side, side), TRANSPARENT)
    draw = ImageDraw.Draw(img)

    # outer disk (green) with gold ring
    pad = int(side * 0.04)
    draw.ellipse((pad, pad, side - pad, side - pad), fill=GREEN, outline=GOLD, width=int(side * 0.015))
    inner_pad = int(side * 0.12)
    draw.ellipse(
        (inner_pad, inner_pad, side - inner_pad, side - inner_pad),
        outline=GOLD,
        width=int(side * 0.006),
    )

    # main wordmark
    big = _load_font(int(side * 0.22))
    text = "KFUPM"
    bbox = draw.textbbox((0, 0), text, font=big, stroke_width=0)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    draw.text(
        ((side - tw) / 2 - bbox[0], (side - th) / 2 - bbox[1] - int(side * 0.04)),
        text,
        fill=WHITE,
        font=big,
    )

    # tagline
    small = _load_font(int(side * 0.045))
    tagline = "King Fahd University of Petroleum & Minerals"
    bbox = draw.textbbox((0, 0), tagline, font=small)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    draw.text(
        ((side - tw) / 2 - bbox[0], side * 0.62 - bbox[1]),
        tagline,
        fill=GOLD,
        font=small,
    )

    return img


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("-o", "--output", default="assets/kfupm_logo.png")
    p.add_argument("--size", type=int, default=1024, help="Output side length in px (square).")
    args = p.parse_args()

    img = make_logo(args.size)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    img.save(args.output, "PNG")
    print(f"wrote {args.output} ({args.size}x{args.size})", file=sys.stderr)


if __name__ == "__main__":
    main()
