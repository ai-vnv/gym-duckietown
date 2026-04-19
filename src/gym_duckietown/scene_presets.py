# coding=utf-8
"""
Visual scene presets for DuckietownEnv (kwargs passed to Simulator).

The simulator draws a colored horizon quad and tints the “infinite ground” plane.
Road / tile textures still come from duckietown-world data; presets only change
ambient outdoor *feel* (warm desert sky + sand), not full asset replacement.

**Arabian / desert outdoor** — late-afternoon warm sky, sand-colored ground,
domain randomization off so colors stay stable for experiments.

**Mining / off-road pit** — dusty sky, pale sand plane, large ``asphalt`` fill cells
remapped to the neutral ``floor`` photo (not grass — grass atlas stays vivid green).
Lane tiles keep ``straight`` / ``curve_*`` asphalt textures with a brown dirt tint.
Ground scatter triangles use white sand, brown sand, and gravel tones. Flat
geometry only; use ``zigzag_dists`` for winding “pit” paths.
"""

from __future__ import annotations

from typing import Any, Dict

# RGB in [0, 1]. Warm amber sky, sun-baked sand (readable in RGB camera).
ARABIAN_DESERT_OUTDOOR: Dict[str, Any] = {
    "domain_rand": False,
    "color_sky": (0.90, 0.70, 0.46),
    "color_ground": (0.72, 0.56, 0.34),
}

# Open-pit / haul-road vibe: grey spoil pad (floor texture) instead of green grass fill,
# brown-tinted lane strips, pale sand horizon, sand/gravel scatter triangles.
MINING_PIT_OUTDOOR: Dict[str, Any] = {
    "domain_rand": False,
    "color_sky": (0.58, 0.52, 0.46),
    "color_ground": (0.92, 0.88, 0.80),
    # Map "asphalt" fill to floor atlas — avoids vivid grass; still flat photo tiles.
    "texture_kind_remap": {"asphalt": "floor"},
    "tile_rgb_mult": (1.0, 1.0, 1.0),
    "tile_kind_rgb_mult": {
        "asphalt": (0.48, 0.43, 0.36),
        "straight": (0.44, 0.38, 0.30),
        "curve_left": (0.44, 0.38, 0.30),
        "curve_right": (0.44, 0.38, 0.30),
        "grass": (0.35, 0.38, 0.30),
        "floor": (0.52, 0.46, 0.40),
        "3way_left": (0.48, 0.42, 0.36),
        "3way_right": (0.48, 0.42, 0.36),
        "4way": (0.48, 0.42, 0.36),
    },
    "mining_ground_scatter": True,
    "num_tris_distractors": 72,
}
