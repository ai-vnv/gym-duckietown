# coding=utf-8
"""
Visual scene presets for DuckietownEnv (kwargs passed to Simulator).

The simulator draws a colored horizon quad and tints the “infinite ground” plane.
Road / tile textures still come from duckietown-world data; presets only change
ambient outdoor *feel* (warm desert sky + sand), not full asset replacement.

**Arabian / desert outdoor** — late-afternoon warm sky, sand-colored ground,
domain randomization off so colors stay stable for experiments.

**Mining / off-road pit** — dusty sky, pale sand plane, asphalt tiles remapped to
grass texture and heavily tinted so reads as dark haul dirt; lane tiles (straight /
curves) get a separate brownish path tint; ground scatter triangles use white sand,
brown sand, and gravel tones. There is no vertical geometry: use a winding map
(e.g. ``zigzag_dists``) to mimic switchback “pit” driving.
"""

from __future__ import annotations

from typing import Any, Dict

# RGB in [0, 1]. Warm amber sky, sun-baked sand (readable in RGB camera).
ARABIAN_DESERT_OUTDOOR: Dict[str, Any] = {
    "domain_rand": False,
    "color_sky": (0.90, 0.70, 0.46),
    "color_ground": (0.72, 0.56, 0.34),
}

# Open-pit / haul-road vibe: dark non-asphalt fill, brownish tracked lanes, sand+gravel clutter.
MINING_PIT_OUTDOOR: Dict[str, Any] = {
    "domain_rand": False,
    "color_sky": (0.46, 0.48, 0.50),
    "color_ground": (0.90, 0.86, 0.78),
    # Asphalt cells use grass photo atlas tinted down (no separate gravel atlas in gd1).
    "texture_kind_remap": {"asphalt": "grass"},
    "tile_rgb_mult": (1.0, 1.0, 1.0),
    "tile_kind_rgb_mult": {
        "asphalt": (0.38, 0.40, 0.34),
        "straight": (0.58, 0.52, 0.44),
        "curve_left": (0.58, 0.52, 0.44),
        "curve_right": (0.58, 0.52, 0.44),
        "grass": (0.52, 0.48, 0.42),
        "floor": (0.5, 0.46, 0.42),
        "3way_left": (0.55, 0.5, 0.44),
        "3way_right": (0.55, 0.5, 0.44),
        "4way": (0.55, 0.5, 0.44),
    },
    "mining_ground_scatter": True,
    "num_tris_distractors": 48,
}
