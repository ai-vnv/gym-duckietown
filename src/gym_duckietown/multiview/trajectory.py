"""Top-down trajectory panel: cached map raster + per-frame polyline overlay.

Avoids matplotlib in the hot path so long renders stay tractable. The map
tile background is rasterized once in ``__init__`` and copied per frame; only
the polyline and current-position dot are redrawn.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

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
    PATH_COLOR = (74, 210, 255)       # newest segment colour
    PATH_FADE_COLOR = (58, 58, 58)    # oldest segments dim toward the tile fill
    HEAD_COLOR = (90, 90, 255)
    TITLE_COLOR = (238, 238, 238)
    METRIC_COLOR = (230, 230, 230)
    TITLE_BAND_HEIGHT = 26
    METRIC_STRIP_HEIGHT = 56          # bottom area reserved for dashboard text
    FADE_BUCKETS = 40                 # color-bucket count for polyline fade

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

    def render(
        self,
        xs: Sequence[float],
        zs: Sequence[float],
        metrics: Optional[dict] = None,
    ) -> np.ndarray:
        """Return an HxWx3 uint8 RGB frame for the given trajectory.

        ``metrics`` is an optional ordered mapping of ``label -> value_str``
        rendered as a dashboard strip at the bottom of the panel.
        """
        import cv2

        img = self._background.copy()
        n = len(xs)
        if n >= 2:
            pts = np.array(
                [self.world_to_px(x, z) for x, z in zip(xs, zs)], dtype=np.int32
            )
            # Bucketed colour fade: split the polyline into chunks and draw
            # each chunk with a colour interpolated between PATH_FADE_COLOR
            # (oldest) and PATH_COLOR (newest). Much cheaper than per-segment
            # drawing for long trajectories.
            n_buckets = min(self.FADE_BUCKETS, n - 1)
            bucket_size = max(1, (n - 1) // n_buckets)
            for b in range(n_buckets):
                i0 = b * bucket_size
                i1 = (b + 1) * bucket_size if b < n_buckets - 1 else (n - 1)
                # extend by 1 so consecutive buckets share an endpoint -> no gaps
                i1_inclusive = min(i1 + 1, n)
                # Normalise on the bucket's *upper* index so the newest
                # bucket lands exactly on PATH_COLOR (t=1.0).
                t = i1 / max(1, n - 1)
                col = (
                    int(self.PATH_FADE_COLOR[0] * (1 - t) + self.PATH_COLOR[0] * t),
                    int(self.PATH_FADE_COLOR[1] * (1 - t) + self.PATH_COLOR[1] * t),
                    int(self.PATH_FADE_COLOR[2] * (1 - t) + self.PATH_COLOR[2] * t),
                )
                segment = pts[i0:i1_inclusive].reshape(-1, 1, 2)
                cv2.polylines(
                    img, [segment], isClosed=False, color=col, thickness=2, lineType=16
                )
        if n >= 1:
            cx, cy = self.world_to_px(xs[-1], zs[-1])
            cv2.circle(img, (cx, cy), 4, self.HEAD_COLOR, thickness=-1, lineType=16)

        if metrics:
            self._draw_metrics(img, metrics)
        return img

    def _draw_metrics(self, img: np.ndarray, metrics: dict) -> None:
        """Semi-transparent dashboard strip at the bottom of the panel."""
        import cv2

        h, w = img.shape[:2]
        strip_h = self.METRIC_STRIP_HEIGHT
        # Dim background strip
        overlay = img.copy()
        cv2.rectangle(overlay, (0, h - strip_h), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, img, 0.45, 0, dst=img)
        # Two columns of metric text
        items = list(metrics.items())
        col_w = w // 2
        for idx, (label, val) in enumerate(items[:6]):
            row = idx // 2
            col = idx % 2
            x = 8 + col * col_w
            y = h - strip_h + 16 + row * 16
            cv2.putText(
                img,
                f"{label}: {val}",
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                self.METRIC_COLOR,
                1,
                lineType=16,
            )

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
