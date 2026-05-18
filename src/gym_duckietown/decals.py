"""Floor-decal support: stick a PNG onto the ground as a textured GL quad.

A "decal" is a flat textured quad rendered just above the floor plane. It
becomes part of the GL scene, so it appears in every camera view (driver,
BEV, side mirrors, top-rear, etc.) without per-view compositing.

Typical use:

    sim = env.unwrapped
    sim.add_floor_decal(
        image_path="/path/to/kfupm_logo.png",
        position=(map_center_x, map_center_z),
        size=0.5,           # meters along the longer side
        rotation_deg=0.0,   # CCW around the y axis
    )

The texture is loaded lazily via :func:`get_decal_texture` (LRU-cached) so
the same PNG bound to multiple decals shares one GPU upload.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Tuple


# Default eye-of-pixel y-offset above the ground plane to avoid z-fighting
# against the asphalt tiles. 1 mm is plenty given the simulator's scale.
DEFAULT_DECAL_LIFT = 0.001


@dataclass
class FloorDecal:
    """A textured floor quad.

    Attributes
    ----------
    image_path : str
        Absolute path to a PNG (with alpha) / JPG file.
    position : tuple[float, float]
        World ``(x, z)`` center of the quad in meters.
    size : float
        Length of the quad's longer side in meters. The shorter side is
        scaled to preserve the image aspect ratio.
    rotation_deg : float
        Yaw (CCW around the world y axis) in degrees.
    lift : float
        Meters above the ground plane (small positive, to avoid z-fighting).
    """

    image_path: str
    position: Tuple[float, float]
    size: float = 0.5
    rotation_deg: float = 0.0
    lift: float = DEFAULT_DECAL_LIFT

    def __post_init__(self) -> None:
        if not self.image_path:
            raise ValueError("FloorDecal.image_path must be a non-empty string")
        if self.size <= 0:
            raise ValueError(f"FloorDecal.size must be > 0, got {self.size}")
        if self.lift < 0:
            raise ValueError(f"FloorDecal.lift must be >= 0, got {self.lift}")


@lru_cache(maxsize=64)
def get_decal_texture(image_path: str):
    """Load a PNG/JPG once and return a Pyglet texture (cached)."""
    import pyglet  # local: avoid GL at import time

    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Decal image not found: {image_path}")
    return pyglet.image.load(image_path).get_texture()


def _aspect(image_path: str) -> float:
    """Width / height of the texture (used to scale the shorter side)."""
    tex = get_decal_texture(image_path)
    return float(tex.width) / float(tex.height)


def draw_decal(decal: FloorDecal) -> None:
    """Issue GL calls to draw one decal. Assumes a GL context is current."""
    import pyglet
    from pyglet import gl

    tex = get_decal_texture(decal.image_path)
    ratio = _aspect(decal.image_path)
    if ratio >= 1.0:
        half_w = decal.size / 2.0
        half_h = (decal.size / ratio) / 2.0
    else:
        half_w = (decal.size * ratio) / 2.0
        half_h = decal.size / 2.0

    cx, cz = float(decal.position[0]), float(decal.position[1])
    y = float(decal.lift)

    gl.glPushAttrib(gl.GL_ENABLE_BIT | gl.GL_CURRENT_BIT)
    gl.glEnable(gl.GL_TEXTURE_2D)
    gl.glBindTexture(tex.target, tex.id)
    # premultiplied alpha is uncommon for ad-hoc PNGs; use straight alpha blend
    gl.glEnable(gl.GL_BLEND)
    gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
    gl.glColor4f(1.0, 1.0, 1.0, 1.0)

    gl.glPushMatrix()
    gl.glTranslatef(cx, y, cz)
    gl.glRotatef(float(decal.rotation_deg), 0.0, 1.0, 0.0)

    gl.glBegin(gl.GL_QUADS)
    # Texture coords assume Pyglet's default (origin bottom-left); we flip v
    # so the image displays right-side-up when viewed from above.
    gl.glTexCoord2f(0.0, 1.0); gl.glVertex3f(-half_w, 0.0, -half_h)
    gl.glTexCoord2f(1.0, 1.0); gl.glVertex3f(+half_w, 0.0, -half_h)
    gl.glTexCoord2f(1.0, 0.0); gl.glVertex3f(+half_w, 0.0, +half_h)
    gl.glTexCoord2f(0.0, 0.0); gl.glVertex3f(-half_w, 0.0, +half_h)
    gl.glEnd()

    gl.glPopMatrix()
    gl.glDisable(gl.GL_BLEND)
    gl.glPopAttrib()


def draw_decals(decals) -> None:
    """Draw every decal in an iterable. Cheap no-op if empty."""
    for d in decals:
        draw_decal(d)
