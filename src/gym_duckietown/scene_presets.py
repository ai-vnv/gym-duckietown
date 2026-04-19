# coding=utf-8
"""
Visual scene presets for DuckietownEnv (kwargs passed to Simulator).

The simulator draws a colored horizon quad and tints the “infinite ground” plane.
Road / tile textures still come from duckietown-world data; presets only change
ambient outdoor *feel* (warm desert sky + sand), not full asset replacement.

**Arabian / desert outdoor** — late-afternoon warm sky, sand-colored ground,
domain randomization off so colors stay stable for experiments.
"""

from __future__ import annotations

from typing import Any, Dict

# RGB in [0, 1]. Warm amber sky, sun-baked sand (readable in RGB camera).
ARABIAN_DESERT_OUTDOOR: Dict[str, Any] = {
    "domain_rand": False,
    "color_sky": (0.90, 0.70, 0.46),
    "color_ground": (0.72, 0.56, 0.34),
}
