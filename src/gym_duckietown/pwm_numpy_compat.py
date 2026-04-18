# coding=utf-8
"""
duckietown-world's pwm_dynamics integrates accelerations that are shape (2, 1).
On recent NumPy, longit_prev + dt * x_dot_dot[0] becomes a 1-element array, so
linear = [longitudinal, lateral] is inhomogeneous and se2_from_linear_angular fails.

Patch DynamicModel.integrate to coerce accelerations to Python floats before use.
"""

from __future__ import annotations

import geometry as geo
import numpy as np

from duckietown_world.world_duckietown.generic_kinematics import GenericKinematicsSE2
from duckietown_world.world_duckietown.pwm_dynamics import DynamicModel, PWMCommands


def _scalar2(x_dot_dot) -> tuple[float, float]:
    flat = np.asarray(x_dot_dot, dtype=np.float64).reshape(-1)
    return float(flat[0]), float(flat[1])


def _patched_integrate(self: DynamicModel, dt: float, commands: PWMCommands) -> DynamicModel:
    linear_angular_prev = geo.linear_angular_from_se2(self.v0)
    linear_prev = linear_angular_prev[0]
    longit_prev = float(linear_prev[0])
    lateral_prev = float(linear_prev[1])
    angular_prev = float(linear_angular_prev[1])

    x_dot_dot = self.model(commands, self.parameters, u=longit_prev, w=angular_prev)
    ax, ay = _scalar2(x_dot_dot)
    longitudinal = longit_prev + dt * ax
    angular = angular_prev + dt * ay
    lateral = 0.0

    linear = [longitudinal, lateral]
    commands_se2 = geo.se2_from_linear_angular(linear, angular)
    s1 = GenericKinematicsSE2.integrate(self, dt, commands_se2)

    c1 = s1.q0, s1.v0
    t1 = s1.t0

    d = self.parameters.wheel_distance
    Rr = self.parameters.wheel_radius_right
    Rl = self.parameters.wheel_radius_left
    M = np.array([[Rr / d, -Rl / d], [Rr / 2, Rl / 2]])
    anglin = np.array((angular, longitudinal), dtype=np.float64)
    MInv = np.linalg.inv(M)
    wRL = MInv @ anglin
    wRL = np.asarray(wRL).reshape(-1)
    wR = float(wRL[0])
    wL = float(wRL[1])

    axis_left_rad = self.axis_left_rad + wL * dt
    axis_right_rad = self.axis_right_rad + wR * dt

    return DynamicModel(
        self.parameters, c1, t1, axis_left_rad=axis_left_rad, axis_right_rad=axis_right_rad
    )


def apply_patch() -> None:
    DynamicModel.integrate = _patched_integrate  # type: ignore[assignment]
