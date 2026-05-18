"""Unit tests for the :class:`Billboard` data model (no GL)."""
from __future__ import annotations

import pytest

from gym_duckietown.billboards import Billboard, get_billboard_texture


def test_billboard_defaults():
    b = Billboard(image_path="/tmp/x.png", position=(1.0, 2.0))
    assert b.width == 0.4
    assert b.height == 0.3
    assert b.y_base == 0.0
    assert b.rotation_deg == 0.0
    assert b.enabled is True
    assert b.single_sided is True  # signs default to back-face-culled


def test_billboard_can_be_double_sided():
    b = Billboard(image_path="/tmp/x.png", position=(0, 0), single_sided=False)
    assert b.single_sided is False


def test_billboard_custom_values():
    b = Billboard(
        image_path="/tmp/x.png",
        position=(0.3, 0.4),
        width=1.0,
        height=0.5,
        y_base=0.2,
        rotation_deg=90.0,
        enabled=False,
    )
    assert b.width == 1.0 and b.height == 0.5
    assert b.y_base == 0.2
    assert b.rotation_deg == 90.0
    assert b.enabled is False


def test_empty_image_path_rejected():
    with pytest.raises(ValueError):
        Billboard(image_path="", position=(0, 0))


@pytest.mark.parametrize("w,h", [(0.0, 0.3), (0.4, 0.0), (-0.1, 0.3), (0.3, -0.1)])
def test_non_positive_dims_rejected(w, h):
    with pytest.raises(ValueError):
        Billboard(image_path="/tmp/x.png", position=(0, 0), width=w, height=h)


def test_negative_y_base_rejected():
    with pytest.raises(ValueError):
        Billboard(image_path="/tmp/x.png", position=(0, 0), y_base=-0.01)


def test_get_billboard_texture_raises_for_missing_file():
    with pytest.raises(FileNotFoundError):
        get_billboard_texture("/tmp/definitely-not-a-real-billboard.png")
