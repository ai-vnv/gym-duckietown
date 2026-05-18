"""Unit tests for the floor-decal data model (no GL).

The actual GL ``draw_decal`` path is not unit-tested here because it
requires a live OpenGL context; it's exercised end-to-end by the render
CLI smoke run.
"""
from __future__ import annotations

import pytest

from gym_duckietown.decals import DEFAULT_DECAL_LIFT, FloorDecal


def test_floor_decal_defaults():
    d = FloorDecal(image_path="/tmp/x.png", position=(1.0, 2.0))
    assert d.size == 0.5
    assert d.rotation_deg == 0.0
    assert d.lift == DEFAULT_DECAL_LIFT
    assert d.position == (1.0, 2.0)


def test_floor_decal_custom_values():
    d = FloorDecal(
        image_path="/tmp/x.png",
        position=(0.3, 0.4),
        size=1.25,
        rotation_deg=45.0,
        lift=0.005,
    )
    assert d.size == 1.25
    assert d.rotation_deg == 45.0
    assert d.lift == 0.005


def test_empty_image_path_rejected():
    with pytest.raises(ValueError):
        FloorDecal(image_path="", position=(0.0, 0.0))


@pytest.mark.parametrize("bad_size", [0.0, -0.1, -1.0])
def test_non_positive_size_rejected(bad_size):
    with pytest.raises(ValueError):
        FloorDecal(image_path="/tmp/x.png", position=(0, 0), size=bad_size)


def test_negative_lift_rejected():
    with pytest.raises(ValueError):
        FloorDecal(image_path="/tmp/x.png", position=(0, 0), lift=-0.01)


def test_get_decal_texture_raises_for_missing_file():
    from gym_duckietown.decals import get_decal_texture

    with pytest.raises(FileNotFoundError):
        get_decal_texture("/tmp/definitely-not-a-real-decal-file.png")
