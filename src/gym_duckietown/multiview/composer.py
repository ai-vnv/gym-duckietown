"""Compose a list of labeled panels into a single video frame.

The composer is layout-agnostic: pass an iterable of rows, each a list of
``(label_or_none, image)`` pairs, and it resizes every panel to
``(tile_w, tile_h)``, stamps a label band on top (unless label is ``None``),
and concatenates row-by-row.
"""
from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

PanelSpec = Tuple[Optional[str], np.ndarray]


def draw_label(img_bgr: np.ndarray, text: str, band_height: int = 26) -> None:
    """Stamp a dark label band + caption on the top of an in-place BGR image."""
    import cv2

    cv2.rectangle(img_bgr, (0, 0), (img_bgr.shape[1], band_height), (0, 0, 0), -1)
    cv2.putText(
        img_bgr,
        text,
        (8, band_height - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (240, 240, 240),
        1,
        lineType=16,
    )


def _tile(label: Optional[str], img: np.ndarray, tile_w: int, tile_h: int) -> np.ndarray:
    import cv2

    resized = cv2.resize(img, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
    if label is None:
        return resized
    bgr = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
    draw_label(bgr, label)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def compose_frame(
    rows: Iterable[Sequence[PanelSpec]],
    tile_h: int,
    tile_w: int,
) -> np.ndarray:
    """Concatenate ``rows`` of labeled panels into one RGB frame.

    Each panel is resized to ``(tile_w, tile_h)``; the output shape is
    ``(tile_h * n_rows, tile_w * cols_per_row, 3)``. All rows must have the
    same number of panels.
    """
    rendered_rows: List[np.ndarray] = []
    cols_per_row: Optional[int] = None
    for row in rows:
        row_list = list(row)
        if cols_per_row is None:
            cols_per_row = len(row_list)
        elif len(row_list) != cols_per_row:
            raise ValueError(
                f"compose_frame: all rows must have the same number of panels "
                f"(got {len(row_list)}, expected {cols_per_row})"
            )
        tiles = [_tile(label, img, tile_w, tile_h) for label, img in row_list]
        rendered_rows.append(np.concatenate(tiles, axis=1))
    if not rendered_rows:
        raise ValueError("compose_frame: at least one row is required")
    return np.concatenate(rendered_rows, axis=0)


def default_layout(views: dict, traj_img: np.ndarray) -> List[List[PanelSpec]]:
    """Standard 3x2 layout used by the CLI.

    Top row: driver, BEV, trajectory (trajectory has its own title baked in,
    so label is ``None`` to avoid double-stamping).
    Bottom row: left, top-rear, right.
    """
    return [
        [("driver", views["driver"]), ("BEV (map)", views["bev"]), (None, traj_img)],
        [("left", views["left"]), ("top-rear", views["top_rear"]), ("right", views["right"])],
    ]
