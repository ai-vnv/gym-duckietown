"""Top-down trajectory panel: cached map raster + per-frame polyline overlay.

Avoids matplotlib in the hot path so long renders stay tractable. The map
tile background is rasterized once in ``__init__`` and copied per frame; only
the polyline and current-position dot are redrawn.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np


class TrajectoryPanel:
    """Renders a top-down map + trajectory polyline at fixed resolution.

    Coordinates: world is (x, z) with z increasing into the map. The panel
    image is HxW with y-down; ``world_to_px`` linearly maps world (x, z) to
    pixel (col, row) using a single uniform scale chosen to fit the whole
    grid plus a small padding margin.

    The panel can be reused across frames; per-frame ``render(xs, zs)`` is
    constant-time relative to the trajectory length (it is O(N) in the
    polyline draw but very cheap).
    """

    TITLE = "trajectory"
    BG_COLOR = (26, 26, 26)
    TILE_FILL = (58, 58, 58)
    TILE_EDGE = (90, 90, 90)
    PATH_COLOR = (74, 210, 255)
    HEAD_COLOR = (90, 90, 255)
    TITLE_COLOR = (238, 238, 238)
    TITLE_BAND_HEIGHT = 26

    def __init__(self, sim, h: int, w: int, pad: float = 0.15):
        import cv2  # local import; cv2 must not load before GL on macOS

        self.h = int(h)
        self.w = int(w)
        self.pad = float(pad)
        ts = float(sim.road_tile_size)
        self.world_w = sim.grid_width * ts + 2 * self.pad
        self.world_h = sim.grid_height * ts + 2 * self.pad

        usable_w = self.w - 12
        usable_h = self.h - (self.TITLE_BAND_HEIGHT + 6)
        self.scale = min(usable_w / self.world_w, usable_h / self.world_h)
        self.off_x = (self.w - self.world_w * self.scale) / 2.0
        self.off_y = (
            (self.h - self.TITLE_BAND_HEIGHT) - self.world_h * self.scale
        ) / 2.0 + self.TITLE_BAND_HEIGHT

        bg = np.full((self.h, self.w, 3), self.BG_COLOR, dtype=np.uint8)
        for i in range(sim.grid_width):
            for j in range(sim.grid_height):
                tile = sim._get_tile(i, j)
                if tile is None or not tile.get("drivable", False):
                    continue
                p0 = self.world_to_px(i * ts, j * ts)
                p1 = self.world_to_px((i + 1) * ts, (j + 1) * ts)
                cv2.rectangle(bg, p0, p1, self.TILE_FILL, thickness=-1)
                cv2.rectangle(bg, p0, p1, self.TILE_EDGE, thickness=1)

        cv2.putText(
            bg,
            self.TITLE,
            (8, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            self.TITLE_COLOR,
            1,
            lineType=16,
        )
        self._background = bg

    def world_to_px(self, x: float, z: float) -> Tuple[int, int]:
        """Map world (x, z) to image pixel (col, row)."""
        px = int(round((x + self.pad) * self.scale + self.off_x))
        py = int(round((z + self.pad) * self.scale + self.off_y))
        return px, py

    def render(self, xs: Sequence[float], zs: Sequence[float]) -> np.ndarray:
        """Return an HxWx3 uint8 RGB frame for the given trajectory."""
        import cv2

        img = self._background.copy()
        if len(xs) >= 2:
            pts = np.array(
                [self.world_to_px(x, z) for x, z in zip(xs, zs)], dtype=np.int32
            ).reshape(-1, 1, 2)
            cv2.polylines(
                img, [pts], isClosed=False, color=self.PATH_COLOR, thickness=2, lineType=16
            )
        if len(xs) >= 1:
            cx, cy = self.world_to_px(xs[-1], zs[-1])
            cv2.circle(img, (cx, cy), 4, self.HEAD_COLOR, thickness=-1, lineType=16)
        return img

    @property
    def background(self) -> np.ndarray:
        """Read-only view of the cached map background (for inspection / tests)."""
        return self._background


def sliding_window(xs: List[float], zs: List[float], window: int) -> Tuple[List[float], List[float]]:
    """Return the last ``window`` entries of ``xs``/``zs`` (in place: trims).

    Returns the same lists for chainability. ``window`` <= 0 disables the trim.
    """
    if window and len(xs) > window:
        del xs[: len(xs) - window]
        del zs[: len(zs) - window]
    return xs, zs
