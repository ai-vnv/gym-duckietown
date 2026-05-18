"""Unit tests for :class:`gym_duckietown.multiview.SmoothedController`."""
from __future__ import annotations

import numpy as np
import pytest

from gym_duckietown.multiview import SmoothedController


def test_first_call_returns_gain_scaled_raw(constant_policy):
    policy = constant_policy(v=0.5, omega=0.4)
    ctrl = SmoothedController(policy, alpha=0.7, omega_gain=2.0)
    out = ctrl(None)
    assert np.allclose(out, [0.5, 0.8])


def test_ema_blending_matches_formula(constant_policy):
    policy = constant_policy(v=1.0, omega=1.0)
    ctrl = SmoothedController(policy, alpha=0.5, omega_gain=1.0)
    a0 = ctrl(None)  # bootstraps from raw -> [1, 1]
    # change the policy output and verify EMA blending
    policy.v, policy.omega = 0.0, 0.0
    a1 = ctrl(None)
    expected = 0.5 * a0 + 0.5 * np.array([0.0, 0.0], dtype=np.float32)
    assert np.allclose(a1, expected)
    a2 = ctrl(None)
    expected = 0.5 * a1 + 0.5 * np.array([0.0, 0.0], dtype=np.float32)
    assert np.allclose(a2, expected)


def test_alpha_zero_no_smoothing(constant_policy):
    policy = constant_policy(v=0.3, omega=0.6)
    ctrl = SmoothedController(policy, alpha=0.0, omega_gain=1.0)
    ctrl(None)
    policy.v, policy.omega = 0.9, -0.4
    out = ctrl(None)
    assert np.allclose(out, [0.9, -0.4])


def test_reset_clears_state(constant_policy):
    policy = constant_policy(v=1.0, omega=1.0)
    ctrl = SmoothedController(policy, alpha=0.8, omega_gain=1.0)
    ctrl(None)
    ctrl.reset()
    policy.v, policy.omega = 0.2, 0.2
    out = ctrl(None)
    # After reset, the first call should be raw (no blending against stale prev).
    assert np.allclose(out, [0.2, 0.2])


def test_omega_gain_applied_before_smoothing(constant_policy):
    policy = constant_policy(v=0.0, omega=0.5)
    ctrl = SmoothedController(policy, alpha=0.0, omega_gain=3.0)
    out = ctrl(None)
    assert np.allclose(out, [0.0, 1.5])


@pytest.mark.parametrize("bad_alpha", [-0.1, 1.0, 1.5])
def test_alpha_out_of_range_raises(constant_policy, bad_alpha):
    with pytest.raises(ValueError):
        SmoothedController(constant_policy(), alpha=bad_alpha)


def test_does_not_mutate_internal_state_via_returned_array(constant_policy):
    """Returned array must be a copy: caller mutating it shouldn't poison EMA."""
    policy = constant_policy(v=0.5, omega=0.5)
    ctrl = SmoothedController(policy, alpha=0.5, omega_gain=1.0)
    out = ctrl(None)
    out[0] = 999.0
    assert np.allclose(ctrl._prev, [0.5, 0.5])
