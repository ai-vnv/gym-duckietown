"""Unit tests for the layout composer."""
from __future__ import annotations

import numpy as np
import pytest

from gym_duckietown.multiview.composer import compose_frame, default_layout, draw_label


def _solid(h: int, w: int, color=(255, 0, 0)) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[..., :] = color
    return img


def test_compose_shape_matches_grid():
    rows = [
        [("a", _solid(50, 80)), ("b", _solid(50, 80)), (None, _solid(50, 80))],
        [("c", _solid(50, 80)), ("d", _solid(50, 80)), ("e", _solid(50, 80))],
    ]
    out = compose_frame(rows, tile_h=120, tile_w=160)
    assert out.shape == (240, 480, 3)
    assert out.dtype == np.uint8


def test_compose_rejects_unequal_row_widths():
    rows = [
        [("a", _solid(10, 10)), ("b", _solid(10, 10))],
        [("c", _solid(10, 10))],
    ]
    with pytest.raises(ValueError):
        compose_frame(rows, tile_h=10, tile_w=10)


def test_compose_rejects_empty_input():
    with pytest.raises(ValueError):
        compose_frame([], tile_h=10, tile_w=10)


def test_unlabeled_panel_has_no_label_band():
    """If label is None, the top band should NOT be the dark label rectangle."""
    out = compose_frame(
        [[(None, _solid(100, 100, color=(255, 255, 255)))]],
        tile_h=100,
        tile_w=100,
    )
    # No label means the top band stays the panel's color (white).
    assert tuple(out[2, 50]) == (255, 255, 255)


def test_labeled_panel_has_dark_label_band():
    out = compose_frame(
        [[("hello", _solid(100, 100, color=(255, 255, 255)))]],
        tile_h=100,
        tile_w=100,
    )
    # Label band darkens the top 26 px to (0, 0, 0).
    assert tuple(out[2, 50]) == (0, 0, 0)


def test_draw_label_in_place():
    img = _solid(50, 100, color=(200, 200, 200))
    draw_label(img, "x")
    # Top of image should now be black (label band).
    assert tuple(img[2, 50]) == (0, 0, 0)
    # Bottom row should be untouched.
    assert tuple(img[-1, 50]) == (200, 200, 200)


def test_default_layout_keys_and_shape():
    views = {
        "driver": _solid(60, 80),
        "bev": _solid(60, 80),
        "left": _solid(60, 80),
        "right": _solid(60, 80),
        "top_rear": _solid(60, 80),
    }
    traj = _solid(60, 80)
    layout = default_layout(views, traj)
    assert len(layout) == 2
    assert all(len(row) == 3 for row in layout)
    # Trajectory entry has label=None (already self-titled).
    labels_top = [lab for lab, _ in layout[0]]
    assert labels_top[2] is None
