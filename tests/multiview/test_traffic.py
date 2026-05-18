"""Unit tests for stop-sign / traffic-light rule logic and the wrapper controller."""
from __future__ import annotations

import numpy as np
import pytest

from gym_duckietown.traffic import (
    RuleAwareController,
    StopSign,
    TrafficLight,
)


# ----------------------------------------------------------------------------
# StopSign
# ----------------------------------------------------------------------------


def test_stop_sign_out_of_range_does_not_apply():
    ss = StopSign(position=(0.0, 0.0), trigger_radius=0.3)
    assert ss.update(agent_xz=(2.0, 2.0), speed=0.5, t=0.0) is False
    assert ss.triggered is False


def test_stop_sign_in_range_demands_stop_until_satisfied():
    ss = StopSign(
        position=(0.0, 0.0), trigger_radius=0.5, required_stop_time_s=0.5, stop_speed_threshold=0.05
    )
    # entering at speed: must hold
    assert ss.update((0.1, 0.0), speed=0.4, t=0.0) is True
    # slowed down — clock starts
    assert ss.update((0.1, 0.0), speed=0.0, t=0.1) is True
    # ... half a second later it is satisfied
    assert ss.update((0.1, 0.0), speed=0.0, t=0.6) is False
    assert ss.satisfied is True


def test_stop_sign_resets_after_leaving():
    ss = StopSign(position=(0.0, 0.0), trigger_radius=0.4, required_stop_time_s=0.2)
    # satisfy on first pass
    ss.update((0.1, 0.0), speed=0.0, t=0.0)
    ss.update((0.1, 0.0), speed=0.0, t=0.3)
    assert ss.satisfied is True
    # leave
    ss.update((5.0, 5.0), speed=0.5, t=0.4)
    assert ss.satisfied is False
    assert ss.triggered is False
    # second approach must hold again
    assert ss.update((0.1, 0.0), speed=0.6, t=0.5) is True


def test_stop_sign_never_stopping_keeps_holding():
    ss = StopSign(position=(0.0, 0.0), trigger_radius=0.4, required_stop_time_s=0.3)
    for t in (0.0, 0.05, 0.1, 0.2, 0.5, 1.0):
        assert ss.update((0.1, 0.0), speed=0.5, t=t) is True


# ----------------------------------------------------------------------------
# TrafficLight
# ----------------------------------------------------------------------------


def test_traffic_light_color_phases():
    tl = TrafficLight(position=(0, 0), cycle_s=10.0, green_frac=0.5, yellow_frac=0.1)
    assert tl.color_at(0.0) == "green"
    assert tl.color_at(4.9) == "green"
    assert tl.color_at(5.0) == "yellow"
    assert tl.color_at(5.9) == "yellow"
    assert tl.color_at(6.0) == "red"
    assert tl.color_at(9.9) == "red"
    # wraps
    assert tl.color_at(10.0) == "green"


def test_traffic_light_phase_offset():
    tl = TrafficLight(position=(0, 0), cycle_s=10.0, green_frac=0.5, yellow_frac=0.1, phase_offset_s=5.0)
    # at t=0, raw phase = -5 mod 10 = 5 -> yellow
    assert tl.color_at(0.0) == "yellow"


def test_traffic_light_out_of_range():
    tl = TrafficLight(position=(0, 0), trigger_radius=0.3, cycle_s=10.0)
    assert tl.speed_multiplier((2.0, 2.0), t=0.0) is None


def test_traffic_light_multipliers():
    tl = TrafficLight(
        position=(0, 0), trigger_radius=1.0, cycle_s=10.0, green_frac=0.5, yellow_frac=0.1,
        yellow_slowdown=0.25,
    )
    assert tl.speed_multiplier((0.1, 0), t=0.0) == pytest.approx(1.0)
    assert tl.speed_multiplier((0.1, 0), t=5.5) == pytest.approx(0.25)
    assert tl.speed_multiplier((0.1, 0), t=8.0) == pytest.approx(0.0)


def test_traffic_light_invalid_fracs_rejected():
    with pytest.raises(ValueError):
        TrafficLight(position=(0, 0), green_frac=0.7, yellow_frac=0.5)
    with pytest.raises(ValueError):
        TrafficLight(position=(0, 0), yellow_slowdown=1.5)


# ----------------------------------------------------------------------------
# RuleAwareController
# ----------------------------------------------------------------------------


class _ConstBase:
    def __init__(self, v=0.5, omega=0.1):
        self.v = v
        self.omega = omega

    def __call__(self, obs):  # noqa: ARG002
        return np.array([self.v, self.omega], dtype=np.float32)


def test_rule_controller_no_rules_is_pass_through():
    base = _ConstBase(v=0.6, omega=0.2)
    ctrl = RuleAwareController(base=base, agent_pose_fn=lambda: (0.0, 0.0, 0.6), dt=0.05)
    out = ctrl(None)
    assert np.allclose(out, [0.6, 0.2])


def test_rule_controller_stop_sign_zeroes_velocity_and_steering():
    """Diff-drive: with v=0 the bot will spin in place unless we also zero omega."""
    base = _ConstBase(v=0.7, omega=0.4)
    ss = StopSign(position=(0, 0), trigger_radius=0.5, required_stop_time_s=0.2)
    pose = [(0.1, 0.0, 0.7)]  # in range, moving
    ctrl = RuleAwareController(
        base=base,
        agent_pose_fn=lambda: pose[0],
        stop_signs=[ss],
        dt=0.05,
    )
    out = ctrl(None)
    assert out[0] == pytest.approx(0.0)
    assert out[1] == pytest.approx(0.0)  # forced to 0 to prevent spin-in-place


def test_rule_controller_traffic_light_red_zeroes_velocity_and_steering():
    base = _ConstBase(v=0.5, omega=0.3)
    tl = TrafficLight(
        position=(0, 0), trigger_radius=1.0, cycle_s=10.0, green_frac=0.0, yellow_frac=0.0,
        # all red
    )
    ctrl = RuleAwareController(
        base=base, agent_pose_fn=lambda: (0.1, 0.0, 0.5), traffic_lights=[tl], dt=0.05
    )
    out = ctrl(None)
    assert out[0] == pytest.approx(0.0)
    assert out[1] == pytest.approx(0.0)


def test_rule_controller_yellow_keeps_steering():
    """Partial slowdown (yellow) must pass omega through unchanged."""
    base = _ConstBase(v=0.5, omega=0.3)
    tl = TrafficLight(
        position=(0, 0), trigger_radius=1.0, cycle_s=10.0, green_frac=0.0, yellow_frac=1.0,
        yellow_slowdown=0.4,
    )
    ctrl = RuleAwareController(
        base=base, agent_pose_fn=lambda: (0.1, 0.0, 0.5), traffic_lights=[tl], dt=0.05
    )
    out = ctrl(None)
    assert out[0] == pytest.approx(0.20)  # 0.5 * 0.4
    assert out[1] == pytest.approx(0.30)  # unchanged


def test_rule_controller_strictest_rule_wins():
    """If stop sign demands stop AND traffic light is green, v stays 0."""
    base = _ConstBase(v=0.5, omega=0.0)
    ss = StopSign(position=(0, 0), trigger_radius=0.5, required_stop_time_s=99.0)  # never satisfied here
    tl = TrafficLight(
        position=(0, 0), trigger_radius=1.0, cycle_s=10.0, green_frac=1.0, yellow_frac=0.0
    )
    ctrl = RuleAwareController(
        base=base,
        agent_pose_fn=lambda: (0.1, 0.0, 0.5),
        stop_signs=[ss],
        traffic_lights=[tl],
        dt=0.05,
    )
    out = ctrl(None)
    assert out[0] == pytest.approx(0.0)


def test_rule_controller_advances_time():
    base = _ConstBase()
    ctrl = RuleAwareController(base=base, agent_pose_fn=lambda: (0, 0, 0), dt=0.1)
    assert ctrl.t == 0.0
    ctrl(None)
    assert ctrl.t == pytest.approx(0.1)
    ctrl(None)
    assert ctrl.t == pytest.approx(0.2)
