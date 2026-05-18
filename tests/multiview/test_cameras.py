"""Unit tests for camera-rig helpers (state-restoration semantics).

These tests don't boot the real simulator; they verify the *contract* of the
yaw/height override using a stub sim that records what state was visible at
render time.
"""
from __future__ import annotations

import math

import pytest

from gym_duckietown.multiview import render_ego_yaw, render_mode


def test_render_mode_does_not_perturb_state(stub_sim):
    angle_before = stub_sim.cur_angle
    height_before = stub_sim.cam_height
    render_mode(stub_sim, "ego")
    assert stub_sim.cur_angle == angle_before
    assert stub_sim.cam_height == height_before


def test_render_ego_yaw_applies_offset_during_render(stub_sim):
    stub_sim.cur_angle = 1.0
    render_ego_yaw(stub_sim, math.pi / 2)
    # The stub records (angle, height) seen at render time.
    seen_angle, seen_height, _ = stub_sim.render_calls[-1]
    assert seen_angle == pytest.approx(1.0 + math.pi / 2)
    assert seen_height == pytest.approx(stub_sim.cam_height)


def test_render_ego_yaw_restores_state(stub_sim):
    stub_sim.cur_angle = 0.7
    stub_sim.cam_height = 0.12
    render_ego_yaw(stub_sim, math.pi, extra_height=0.3)
    assert stub_sim.cur_angle == pytest.approx(0.7)
    assert stub_sim.cam_height == pytest.approx(0.12)


def test_render_ego_yaw_applies_height_during_render(stub_sim):
    stub_sim.cam_height = 0.10
    render_ego_yaw(stub_sim, 0.0, extra_height=0.5)
    _, seen_height, _ = stub_sim.render_calls[-1]
    assert seen_height == pytest.approx(0.60)


def test_render_ego_yaw_restores_on_exception(stub_sim):
    """If the renderer raises, original state must still be restored."""

    def _boom(*_a, **_kw):
        raise RuntimeError("synthetic GL failure")

    stub_sim.cur_angle = 0.5
    stub_sim.cam_height = 0.108
    stub_sim._render_img = _boom  # type: ignore[assignment]
    with pytest.raises(RuntimeError):
        render_ego_yaw(stub_sim, math.pi, extra_height=0.3)
    assert stub_sim.cur_angle == pytest.approx(0.5)
    assert stub_sim.cam_height == pytest.approx(0.108)


def test_zero_extra_height_does_not_override_height_at_all(stub_sim):
    """Sanity: passing extra_height=0 leaves cam_height untouched during render."""
    stub_sim.cam_height = 0.42
    render_ego_yaw(stub_sim, math.pi / 4, extra_height=0.0)
    _, seen_height, _ = stub_sim.render_calls[-1]
    assert seen_height == pytest.approx(0.42)


def test_render_ego_yaw_applies_pitch_during_render(stub_sim):
    stub_sim.cam_angle = [12.0, 0.0, 0.0]
    render_ego_yaw(stub_sim, 0.0, pitch_deg_delta=-8.0)
    _, _, seen_pitch = stub_sim.render_calls[-1]
    assert seen_pitch == pytest.approx(4.0)


def test_render_ego_yaw_restores_pitch(stub_sim):
    stub_sim.cam_angle = [12.0, 0.0, 0.0]
    render_ego_yaw(stub_sim, 0.0, pitch_deg_delta=-8.0)
    assert stub_sim.cam_angle[0] == pytest.approx(12.0)


def test_render_ego_yaw_restores_pitch_on_exception(stub_sim):
    def _boom(*_a, **_kw):
        raise RuntimeError("synthetic GL failure")

    stub_sim.cam_angle = [20.0, 1.0, 2.0]
    stub_sim._render_img = _boom  # type: ignore[assignment]
    with pytest.raises(RuntimeError):
        render_ego_yaw(stub_sim, 0.0, pitch_deg_delta=-5.0)
    assert stub_sim.cam_angle == [20.0, 1.0, 2.0]
