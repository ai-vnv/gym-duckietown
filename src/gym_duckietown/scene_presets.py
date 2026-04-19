# coding=utf-8
"""
Visual scene presets for :class:`gym_duckietown.envs.duckietown_env.DuckietownEnv`
(kwargs forwarded to :class:`gym_duckietown.simulator.Simulator`).

The simulator draws a horizon quad and tints the infinite ground plane. Road tile
textures normally come from **duckietown-world** atlases. This module adds optional
overrides:

* ``custom_tile_texture_paths`` — on-disk PNG/JPG per tile kind (see ``MINING_PIT_OUTDOOR_GL``).
* ``custom_tile_textures`` — in-memory ``uint8`` RGB(A) or ``callable(rng) -> array`` per kind
  (see ``MINING_PIT_OUTDOOR_PROG``), uploaded via :func:`gym_duckietown.graphics.load_texture_from_rgba`.

**Arabian / desert outdoor** — warm sky + sand ground tint; ``domain_rand=False`` for stable colors.

**Mining / off-road pit** — remaps large ``asphalt`` pads to ``floor`` textures, brown lane tints,
pale horizon, sand/gravel floor scatter. Flat geometry; pair with ``zigzag_dists`` for long switchbacks.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

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

# Same as ``MINING_PIT_OUTDOOR`` but large pads use generated ``spoil_brown.png`` / sand PNGs
# in OpenGL (run ``python scripts/generate_mining_assets.py`` first). Lane tiles
# (``straight`` / ``curve_*``) still use stock textures so yellow lines stay visible.
_MINING_GL_TEXTURES: Dict[str, str] = {
    "floor": "assets/mining_generated/spoil_brown.png",
    "grass": "assets/mining_generated/sand_white.png",
}

MINING_PIT_OUTDOOR_GL: Dict[str, Any] = {
    **MINING_PIT_OUTDOOR,
    "custom_tile_texture_paths": _MINING_GL_TEXTURES,
}


def _rng_integers(rng: Any, low: int, high: int, size: Any) -> np.ndarray:
    """``[low, high)`` integers; works with :class:`numpy.random.Generator` and legacy ``RandomState``."""
    if hasattr(rng, "integers"):
        return np.asarray(rng.integers(low, high, size=size, endpoint=False), dtype=np.int32)
    return np.asarray(rng.randint(low, high, size=size), dtype=np.int32)


def _rng_normal(rng: Any, size: Any) -> np.ndarray:
    """Standard normal samples shaped like ``size`` (tuple or int)."""
    if hasattr(rng, "standard_normal"):
        return np.asarray(rng.standard_normal(size=size), dtype=np.float32)
    if isinstance(size, tuple):
        return np.asarray(rng.randn(*size), dtype=np.float32)
    return np.asarray(rng.randn(int(size)), dtype=np.float32)


def _mining_floor_rgba(rng: Any, size: int) -> np.ndarray:
    h = w = int(size)
    r = _rng_integers(rng, 128, 172, (h, w))
    g = _rng_integers(rng, 112, 158, (h, w))
    b = _rng_integers(rng, 86, 128, (h, w))
    rgb = np.stack([r, g, b], axis=-1).astype(np.float32)
    rgb += _rng_normal(rng, (h, w, 3)) * 9.0
    return np.clip(rgb, 0, 255).astype(np.uint8)


def _mining_grass_rgba(rng: Any, size: int) -> np.ndarray:
    h = w = int(size)
    r = _rng_integers(rng, 200, 235, (h, w))
    g = _rng_integers(rng, 185, 220, (h, w))
    b = _rng_integers(rng, 155, 195, (h, w))
    rgb = np.stack([r, g, b], axis=-1).astype(np.float32)
    rgb += _rng_normal(rng, (h, w, 3)) * 14.0
    return np.clip(rgb, 0, 255).astype(np.uint8)


def mining_procedural_tile_textures(size: int = 256) -> Dict[str, Any]:
    """
    Build ``custom_tile_textures`` for mining-style pads (no files).

    Callables receive the simulator ``np_random`` (``numpy.random.Generator`` or legacy ``RandomState``).
    """
    sz = int(size)

    def floor_fn(rng: Any) -> np.ndarray:
        return _mining_floor_rgba(rng, sz)

    def grass_fn(rng: Any) -> np.ndarray:
        return _mining_grass_rgba(rng, sz)

    return {"floor": floor_fn, "grass": grass_fn}


# Same look goals as ``MINING_PIT_OUTDOOR`` but ``floor`` / ``grass`` tiles use procedural RGB.
MINING_PIT_OUTDOOR_PROG: Dict[str, Any] = {
    **MINING_PIT_OUTDOOR,
    "custom_tile_textures": mining_procedural_tile_textures(256),
}


def mining_gl_texture_paths(
    spoil: Optional[str] = None,
    grass_sand: Optional[str] = None,
) -> Dict[str, str]:
    """
    Build ``custom_tile_texture_paths`` with defaults under ``assets/mining_generated/``.

    Override any path with an absolute file path or a repo-relative string.
    """
    out: Dict[str, str] = dict(_MINING_GL_TEXTURES)
    if spoil is not None:
        out["floor"] = spoil
    if grass_sand is not None:
        out["grass"] = grass_sand
    return out
