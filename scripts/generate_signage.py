#!/usr/bin/env python3
"""Generate procedural signage PNGs used by scenarios.

Outputs (default into ``assets/``):

  - kfupm_logo.png         floor-decal logo (delegated to generate_kfupm_logo)
  - stop_sign.png          red octagon-ish "STOP" billboard
  - traffic_light_red.png
  - traffic_light_yellow.png
  - traffic_light_green.png
  - billboard_jisr3.png    "Happy JISR3-ing!" celebratory billboard

Each PNG is RGBA with transparent background so it composites cleanly on
the GL quad. The traffic-light variants share a layout and differ only in
which lamp is "lit".

Usage:
    .venv/bin/python scripts/generate_signage.py [--out-dir assets/]
"""
from __future__ import annotations

import argparse
import math
import os
import sys

from PIL import Image, ImageDraw, ImageFont


WHITE = (255, 255, 255, 255)
BLACK = (10, 10, 10, 255)
TRANSPARENT = (0, 0, 0, 0)
RED = (210, 30, 30, 255)
YELLOW = (240, 200, 40, 255)
GREEN = (60, 200, 80, 255)
DIM = (60, 60, 60, 255)


def _font(size: int):
    for path in (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ):
        if os.path.isfile(path):
            try:
                return ImageFont.truetype(path, size=size)
            except OSError:
                pass
    return ImageFont.load_default()


# ----------------------------------------------------------------------------
# Stop sign
# ----------------------------------------------------------------------------


def make_stop_sign(side: int = 512) -> Image.Image:
    img = Image.new("RGBA", (side, side), TRANSPARENT)
    draw = ImageDraw.Draw(img)

    cx, cy = side / 2, side / 2
    r = side * 0.46
    # octagon vertices: 8 points, starting top-left of the top edge
    pts = []
    for i in range(8):
        angle = math.pi * (i / 4.0) + math.pi / 8.0
        pts.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))
    draw.polygon(pts, fill=RED, outline=WHITE)

    # white "ring"
    r2 = r * 0.85
    inner = []
    for i in range(8):
        angle = math.pi * (i / 4.0) + math.pi / 8.0
        inner.append((cx + r2 * math.cos(angle), cy + r2 * math.sin(angle)))
    draw.line(inner + [inner[0]], fill=WHITE, width=max(2, side // 80))

    font = _font(int(side * 0.32))
    bbox = draw.textbbox((0, 0), "STOP", font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    draw.text(
        ((side - tw) / 2 - bbox[0], (side - th) / 2 - bbox[1]),
        "STOP",
        fill=WHITE,
        font=font,
    )
    return img


# ----------------------------------------------------------------------------
# Traffic light (3 variants)
# ----------------------------------------------------------------------------


def make_traffic_light(state: str, w: int = 256, h: int = 640) -> Image.Image:
    if state not in ("red", "yellow", "green"):
        raise ValueError(f"unknown state: {state!r}")
    img = Image.new("RGBA", (w, h), TRANSPARENT)
    draw = ImageDraw.Draw(img)

    # housing
    pad = w * 0.06
    draw.rounded_rectangle(
        (pad, pad, w - pad, h - pad),
        radius=int(w * 0.12),
        fill=BLACK,
        outline=(40, 40, 40, 255),
        width=max(2, int(w * 0.02)),
    )

    # three lamps stacked
    cx = w / 2
    lamp_r = w * 0.30
    spacing = (h - 2 * pad) / 3
    for i, color_name in enumerate(("red", "yellow", "green")):
        cy = pad + spacing * (i + 0.5)
        lit = state == color_name
        color = {"red": RED, "yellow": YELLOW, "green": GREEN}[color_name]
        fill = color if lit else DIM
        draw.ellipse(
            (cx - lamp_r, cy - lamp_r, cx + lamp_r, cy + lamp_r),
            fill=fill,
            outline=(15, 15, 15, 255),
            width=max(1, int(w * 0.012)),
        )
        if lit:
            # subtle halo: bright ring just inside
            draw.ellipse(
                (cx - lamp_r * 0.85, cy - lamp_r * 0.85, cx + lamp_r * 0.85, cy + lamp_r * 0.85),
                outline=WHITE,
                width=max(1, int(w * 0.015)),
            )
    return img


# ----------------------------------------------------------------------------
# JISR3 billboard
# ----------------------------------------------------------------------------


def make_pole(w: int = 64, h: int = 512) -> Image.Image:
    """A tall thin vertical metallic-gray gradient (used as a pole texture)."""
    img = Image.new("RGBA", (w, h), TRANSPARENT)
    px = img.load()
    # vertical center band with horizontal gradient for a metallic look
    band_w = max(2, int(w * 0.5))
    x0 = (w - band_w) // 2
    x1 = x0 + band_w
    for y in range(h):
        for x in range(x0, x1):
            # bright in the middle of the band, darker at edges
            t = (x - x0) / max(1, band_w - 1)
            v = int(70 + 130 * (1.0 - abs(2 * t - 1)))  # 70..200..70
            px[x, y] = (v, v, v, 255)
    return img


def make_jisr3_billboard(
    text: str = "Happy JISR3-ing!",
    subtitle: str = "AI V&V Lab, IRC-SML, KFUPM-SDAIA JRCAI",
    w: int = 1024,
    h: int = 512,
) -> Image.Image:
    img = Image.new("RGBA", (w, h), TRANSPARENT)
    draw = ImageDraw.Draw(img)
    # background panel: KFUPM green with gold border
    border = int(min(w, h) * 0.025)
    draw.rounded_rectangle(
        (border, border, w - border, h - border),
        radius=int(min(w, h) * 0.07),
        fill=(0, 106, 78, 255),
        outline=(212, 175, 55, 255),
        width=border,
    )

    # Auto-fit the wordmark: shrink the font until the rendered text fits
    # within the inner panel with comfortable side margins.
    max_text_w = int(w * 0.80)
    max_text_h = int(h * 0.45)
    font_size = int(h * 0.40)
    while font_size > 12:
        big_font = _font(font_size)
        bbox = draw.textbbox((0, 0), text, font=big_font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        if tw <= max_text_w and th <= max_text_h:
            break
        font_size = int(font_size * 0.9)
    draw.text(
        ((w - tw) / 2 - bbox[0], (h - th) / 2 - bbox[1] - int(h * 0.06)),
        text,
        fill=(255, 255, 255, 255),
        font=big_font,
    )

    # Auto-fit subtitle the same way as the wordmark so longer text scales down.
    sub_max_w = int(w * 0.86)
    sub_max_h = int(h * 0.18)
    sub_size = int(h * 0.12)
    while sub_size > 10:
        small_font = _font(sub_size)
        sub_bbox = draw.textbbox((0, 0), subtitle, font=small_font)
        tw = sub_bbox[2] - sub_bbox[0]
        sub_th = sub_bbox[3] - sub_bbox[1]
        if tw <= sub_max_w and sub_th <= sub_max_h:
            break
        sub_size = int(sub_size * 0.92)
    draw.text(
        ((w - tw) / 2 - sub_bbox[0], h * 0.72 - sub_bbox[1]),
        subtitle,
        fill=(212, 175, 55, 255),
        font=small_font,
    )
    return img


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", default="assets")
    p.add_argument("--billboard-text", default="Happy JISR3-ing!")
    p.add_argument(
        "--billboard-subtitle",
        default="AI V&V Lab, IRC-SML, KFUPM-SDAIA JRCAI",
    )
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    paths = {
        "stop_sign": os.path.join(args.out_dir, "stop_sign.png"),
        "tl_red": os.path.join(args.out_dir, "traffic_light_red.png"),
        "tl_yellow": os.path.join(args.out_dir, "traffic_light_yellow.png"),
        "tl_green": os.path.join(args.out_dir, "traffic_light_green.png"),
        "billboard": os.path.join(args.out_dir, "billboard_jisr3.png"),
        "pole": os.path.join(args.out_dir, "pole.png"),
    }
    make_stop_sign().save(paths["stop_sign"], "PNG")
    make_traffic_light("red").save(paths["tl_red"], "PNG")
    make_traffic_light("yellow").save(paths["tl_yellow"], "PNG")
    make_traffic_light("green").save(paths["tl_green"], "PNG")
    make_jisr3_billboard(args.billboard_text, args.billboard_subtitle).save(paths["billboard"], "PNG")
    make_pole().save(paths["pole"], "PNG")
    for name, path in paths.items():
        print(f"  {name:12s} -> {path}", file=sys.stderr)


if __name__ == "__main__":
    main()
