"""Billboard support: a PNG rendered as a *vertical* textured GL quad.

Where :mod:`gym_duckietown.decals` puts an image *on* the floor, a Billboard
stands *up*. Useful for traffic signs, scoreboards, sponsor placards, etc.

The quad's local frame:
  - bottom-center sits at ``(position[0], y_base, position[1])`` in world coords
  - the quad's plane is x-axis (width) by y-axis (height)
  - ``rotation_deg`` yaws the quad around the world y axis (face direction)

The texture is cached via :func:`get_billboard_texture` (LRU).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Tuple


@dataclass
class Billboard:
    """A vertical textured quad in the world.

    Attributes
    ----------
    image_path : str
        Absolute path to a PNG/JPG.
    position : tuple[float, float]
        World ``(x, z)`` of the quad's bottom-center anchor.
    width : float
        Horizontal extent of the quad (meters).
    height : float
        Vertical extent of the quad (meters).
    y_base : float
        Height in meters of the bottom edge above the ground plane.
    rotation_deg : float
        Yaw around world y axis (degrees, CCW). Quad's normal at zero yaw
        faces +z.
    enabled : bool
        If False, the billboard is skipped at render time. Lets a higher-
        level controller toggle visibility (e.g. traffic-light state swap).
    single_sided : bool
        If True (default), back-face culling is enabled while drawing this
        quad so it is only visible from the side its texture faces. This is
        what you want for any billboard with text/symbols (STOP signs,
        banners, traffic lights) — the alternative shows the texture
        mirrored from behind, which z-fights badly when two such signs are
        placed back-to-back. Set to False for symmetric textures (poles)
        that should be visible from any angle.
    """

    image_path: str
    position: Tuple[float, float]
    width: float = 0.4
    height: float = 0.3
    y_base: float = 0.0
    rotation_deg: float = 0.0
    enabled: bool = True
    single_sided: bool = True

    def __post_init__(self) -> None:
        if not self.image_path:
            raise ValueError("Billboard.image_path must be a non-empty string")
        if self.width <= 0 or self.height <= 0:
            raise ValueError(
                f"Billboard width/height must be > 0, got width={self.width} height={self.height}"
            )
        if self.y_base < 0:
            raise ValueError(f"Billboard.y_base must be >= 0, got {self.y_base}")


@lru_cache(maxsize=64)
def get_billboard_texture(image_path: str):
    """Load a PNG/JPG once and return a Pyglet texture (cached)."""
    import pyglet

    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Billboard image not found: {image_path}")
    return pyglet.image.load(image_path).get_texture()


def draw_billboard(b: Billboard) -> None:
    """Issue GL calls to draw one billboard. Assumes a current GL context."""
    from pyglet import gl

    if not b.enabled:
        return
    tex = get_billboard_texture(b.image_path)

    half_w = b.width / 2.0
    h = b.height

    gl.glPushAttrib(gl.GL_ENABLE_BIT | gl.GL_CURRENT_BIT | gl.GL_POLYGON_BIT)
    gl.glEnable(gl.GL_TEXTURE_2D)
    gl.glBindTexture(tex.target, tex.id)
    gl.glEnable(gl.GL_BLEND)
    gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
    # Make billboards self-illuminating so they read well under any lighting.
    gl.glDisable(gl.GL_LIGHTING)
    if b.single_sided:
        # Cull the back face. Our vertex winding (BL, BR, TR, TL) is CCW
        # when viewed from +Z, so +Z is the front face. After a yaw rotation
        # around Y, the +Z normal rotates with the quad — the front always
        # faces wherever ``rotation_deg`` points.
        gl.glEnable(gl.GL_CULL_FACE)
        gl.glCullFace(gl.GL_BACK)
        gl.glFrontFace(gl.GL_CCW)
    gl.glColor4f(1.0, 1.0, 1.0, 1.0)

    gl.glPushMatrix()
    gl.glTranslatef(float(b.position[0]), float(b.y_base), float(b.position[1]))
    gl.glRotatef(float(b.rotation_deg), 0.0, 1.0, 0.0)

    gl.glBegin(gl.GL_QUADS)
    # Quad in local x-y plane, facing +z. Vertices CCW when viewed from +z.
    gl.glTexCoord2f(0.0, 0.0); gl.glVertex3f(-half_w, 0.0, 0.0)
    gl.glTexCoord2f(1.0, 0.0); gl.glVertex3f(+half_w, 0.0, 0.0)
    gl.glTexCoord2f(1.0, 1.0); gl.glVertex3f(+half_w, h, 0.0)
    gl.glTexCoord2f(0.0, 1.0); gl.glVertex3f(-half_w, h, 0.0)
    gl.glEnd()

    gl.glPopMatrix()
    gl.glPopAttrib()


def draw_billboards(billboards) -> None:
    for b in billboards:
        draw_billboard(b)
