"""Unit tests for :class:`gym_duckietown.multiview.TrajectoryPanel`."""
from __future__ import annotations

import numpy as np
import pytest

from gym_duckietown.multiview import TrajectoryPanel
from gym_duckietown.multiview.trajectory import sliding_window


def test_panel_shape_matches_request(stub_sim):
    panel = TrajectoryPanel(stub_sim, h=240, w=320)
    img = panel.render([], [])
    assert img.shape == (240, 320, 3)
    assert img.dtype == np.uint8


def test_world_to_px_in_bounds_for_all_drivable_tiles(stub_sim):
    panel = TrajectoryPanel(stub_sim, h=240, w=320)
    ts = stub_sim.road_tile_size
    for i in range(stub_sim.grid_width):
        for j in range(stub_sim.grid_height):
            for x, z in ((i * ts, j * ts), ((i + 1) * ts, (j + 1) * ts)):
                px, py = panel.world_to_px(x, z)
                assert 0 <= px < panel.w
                assert 0 <= py < panel.h


def test_background_drawn_only_on_drivable_tiles(stub_sim):
    panel = TrajectoryPanel(stub_sim, h=240, w=320)
    bg = panel.background
    # Center tile is non-drivable in the stub; sample its centroid pixel.
    ts = stub_sim.road_tile_size
    cx, cy = panel.world_to_px(1.5 * ts, 1.5 * ts)
    # Non-drivable area should be background color (26, 26, 26).
    assert tuple(bg[cy, cx]) == TrajectoryPanel.BG_COLOR
    # A border (drivable) tile centroid should be tile-fill color.
    bx, by = panel.world_to_px(0.5 * ts, 0.5 * ts)
    assert tuple(bg[by, bx]) == TrajectoryPanel.TILE_FILL


def test_trajectory_draws_path_color_along_route(stub_sim):
    panel = TrajectoryPanel(stub_sim, h=240, w=320)
    ts = stub_sim.road_tile_size
    xs = np.linspace(0.2, 2.0, 30).tolist()
    zs = np.full_like(xs, 0.5 * ts).tolist()
    img = panel.render(xs, zs)
    # Look along the row where z = 0.5*ts; we should find at least one pixel
    # tinted in PATH_COLOR (the polyline is 2px thick).
    target = np.array(TrajectoryPanel.PATH_COLOR, dtype=np.uint8)
    found = np.any(np.all(img == target, axis=-1))
    assert found, "expected to find path-colored pixels along the trajectory"


def test_head_marker_drawn_at_last_point(stub_sim):
    panel = TrajectoryPanel(stub_sim, h=240, w=320)
    xs = [0.1, 0.5]
    zs = [0.1, 0.4]
    img = panel.render(xs, zs)
    cx, cy = panel.world_to_px(xs[-1], zs[-1])
    # Some pixel inside the 4px-radius head disc should be HEAD_COLOR.
    head_color = np.array(TrajectoryPanel.HEAD_COLOR, dtype=np.uint8)
    patch = img[max(0, cy - 3) : cy + 4, max(0, cx - 3) : cx + 4]
    assert np.any(np.all(patch == head_color, axis=-1))


def test_empty_trajectory_renders_just_background(stub_sim):
    panel = TrajectoryPanel(stub_sim, h=120, w=160)
    img = panel.render([], [])
    assert np.array_equal(img, panel.background)


def test_sliding_window_trims_in_place():
    xs = list(range(10))
    zs = list(range(10, 20))
    sliding_window(xs, zs, 4)
    assert xs == [6, 7, 8, 9]
    assert zs == [16, 17, 18, 19]


def test_sliding_window_zero_is_noop():
    xs = [1.0, 2.0]
    zs = [3.0, 4.0]
    sliding_window(xs, zs, 0)
    assert xs == [1.0, 2.0]
    assert zs == [3.0, 4.0]


@pytest.mark.parametrize("size", [(120, 160), (240, 320), (480, 640)])
def test_panel_scales_to_arbitrary_size(stub_sim, size):
    h, w = size
    panel = TrajectoryPanel(stub_sim, h=h, w=w)
    assert panel.background.shape == (h, w, 3)
    assert panel.scale > 0
