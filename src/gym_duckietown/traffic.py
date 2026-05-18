"""Traffic-rule primitives: stop signs, traffic lights, and a rule-aware
controller wrapper that modulates an underlying policy's velocity command.

The duckietown simulator's reward function does not enforce signage rules,
so "obeying" a stop sign or red light is a *controller-side* concern: this
module wraps any base controller (e.g. :class:`SmoothedController`) and
gates its velocity output based on agent proximity to each rule.

Everything in this module is pure Python — no GL, no env. The
``RuleAwareController`` does need access to the simulator only to read the
agent's current pose; if you don't want that coupling, drive the rules
manually by passing positions to ``StopSign.update`` / ``TrafficLight.color_at``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt
from typing import Any, Callable, List, Optional, Tuple

import numpy as np


# ----------------------------------------------------------------------------
# Stop sign
# ----------------------------------------------------------------------------


@dataclass
class StopSign:
    """A stop sign that requires the agent to come to a full stop briefly.

    The rule is satisfied for the current approach once the agent has spent
    at least ``required_stop_time_s`` seconds with speed below
    ``stop_speed_threshold``. Leaving the trigger radius resets the state so
    the next lap re-triggers the rule.

    Attributes
    ----------
    position : tuple[float, float]
        World ``(x, z)`` of the sign in meters.
    trigger_radius : float
        Distance (m) at which the rule starts applying.
    required_stop_time_s : float
        How long the agent must be effectively stationary to satisfy the rule.
    stop_speed_threshold : float
        Speed below which the agent counts as "stopped".
    """

    position: Tuple[float, float]
    trigger_radius: float = 0.35
    required_stop_time_s: float = 1.0
    stop_speed_threshold: float = 0.05

    # state — not part of the user-facing config but useful for visualization
    triggered: bool = field(default=False, init=False)
    stopped_at_time: Optional[float] = field(default=None, init=False)
    satisfied: bool = field(default=False, init=False)

    def reset(self) -> None:
        self.triggered = False
        self.stopped_at_time = None
        self.satisfied = False

    def update(self, agent_xz: Tuple[float, float], speed: float, t: float) -> bool:
        """Update internal state; return True if the agent must still hold."""
        dx = agent_xz[0] - self.position[0]
        dz = agent_xz[1] - self.position[1]
        dist = sqrt(dx * dx + dz * dz)

        if dist > self.trigger_radius:
            # Out of range — reset so next approach re-triggers.
            self.reset()
            return False

        self.triggered = True
        if self.satisfied:
            # Already satisfied for this approach — let the agent through.
            return False

        if speed <= self.stop_speed_threshold:
            if self.stopped_at_time is None:
                self.stopped_at_time = t
            elif t - self.stopped_at_time >= self.required_stop_time_s:
                self.satisfied = True
                return False
        return True


# ----------------------------------------------------------------------------
# Traffic light
# ----------------------------------------------------------------------------


@dataclass
class TrafficLight:
    """A simple periodic green / yellow / red traffic light.

    Phase fractions sum to <= 1; the remainder is red. The cycle starts at
    ``-phase_offset_s`` so multiple lights can be desynchronized.

    Speed handling:
      - green  → speed_multiplier = 1.0
      - yellow → speed_multiplier = ``yellow_slowdown`` (default 0.3)
      - red    → speed_multiplier = 0.0 (stop)
    """

    position: Tuple[float, float]
    trigger_radius: float = 0.45
    cycle_s: float = 12.0
    green_frac: float = 0.55
    yellow_frac: float = 0.10
    phase_offset_s: float = 0.0
    yellow_slowdown: float = 0.3

    def __post_init__(self) -> None:
        if self.green_frac + self.yellow_frac > 1.0:
            raise ValueError(
                f"green_frac + yellow_frac must be <= 1, got "
                f"{self.green_frac} + {self.yellow_frac}"
            )
        if not (0.0 <= self.yellow_slowdown <= 1.0):
            raise ValueError(f"yellow_slowdown must be in [0, 1], got {self.yellow_slowdown}")

    def color_at(self, t: float) -> str:
        """Return ``"green"``, ``"yellow"``, or ``"red"`` for time ``t``."""
        phase = ((t - self.phase_offset_s) % self.cycle_s) / self.cycle_s
        if phase < self.green_frac:
            return "green"
        if phase < self.green_frac + self.yellow_frac:
            return "yellow"
        return "red"

    def speed_multiplier(self, agent_xz: Tuple[float, float], t: float) -> Optional[float]:
        """Speed multiplier in [0, 1], or ``None`` if rule doesn't apply (out of range)."""
        dx = agent_xz[0] - self.position[0]
        dz = agent_xz[1] - self.position[1]
        dist = sqrt(dx * dx + dz * dz)
        if dist > self.trigger_radius:
            return None
        c = self.color_at(t)
        if c == "green":
            return 1.0
        if c == "yellow":
            return self.yellow_slowdown
        return 0.0


# ----------------------------------------------------------------------------
# Rule-aware controller
# ----------------------------------------------------------------------------


class RuleAwareController:
    """Wraps a base controller; gates its velocity command via traffic rules.

    Steering (``action[1]``) is passed through unchanged. Velocity
    (``action[0]``) is multiplied by the strictest applicable rule's speed
    multiplier:

    * Each :class:`StopSign` in range that has not yet been satisfied forces
      v = 0.
    * Each :class:`TrafficLight` in range contributes a multiplier
      (red=0, yellow=``yellow_slowdown``, green=1).

    Time is tracked via a ``dt`` per call; you can override per call by
    passing ``t=``.

    Parameters
    ----------
    base : callable
        ``base(obs) -> np.ndarray([v, omega])``. Typically a
        :class:`SmoothedController`.
    agent_pose_fn : callable
        ``() -> (x, z, speed)``. The pipeline passes one bound to the sim.
    stop_signs, traffic_lights : sequences
        Lists of rule objects. Either may be empty.
    dt : float
        Time-step advance per call (seconds). Used when ``t`` is not passed.
    """

    def __init__(
        self,
        base: Callable[[Any], np.ndarray],
        agent_pose_fn: Callable[[], Tuple[float, float, float]],
        stop_signs: Optional[List[StopSign]] = None,
        traffic_lights: Optional[List[TrafficLight]] = None,
        dt: float = 1.0 / 20.0,
        decel_time_s: float = 0.6,
        accel_time_s: float = 0.4,
    ) -> None:
        self.base = base
        self.agent_pose_fn = agent_pose_fn
        self.stop_signs: List[StopSign] = list(stop_signs or [])
        self.traffic_lights: List[TrafficLight] = list(traffic_lights or [])
        self.dt = float(dt)
        self.decel_time = float(decel_time_s)
        self.accel_time = float(accel_time_s)
        self.t = 0.0

        # Live state, exposed for the dashboard + telemetry log.
        self.effective_mult: float = 1.0           # smoothed gate multiplier
        self.target_mult: float = 1.0              # raw mult from rules this step
        self.brake: float = 0.0                    # 0..1 brake-pedal proxy
        self.in_stop_zone: List[bool] = []         # per stop sign
        self.stop_satisfied: List[bool] = []       # per stop sign
        self.stop_clock: List[Optional[float]] = []
        self.in_tl_zone: List[bool] = []           # per traffic light
        self.tl_color: List[str] = []              # per traffic light

    def reset(self) -> None:
        self.t = 0.0
        self.effective_mult = 1.0
        self.brake = 0.0
        for ss in self.stop_signs:
            ss.reset()

    def __call__(self, observation: Any, t: Optional[float] = None) -> np.ndarray:
        action = np.asarray(self.base(observation), dtype=np.float32).copy()
        advance_time = t is None
        if t is None:
            t = self.t

        x, z, speed = self.agent_pose_fn()
        pose = (float(x), float(z))

        # Per-rule state for telemetry
        self.in_stop_zone = []
        self.stop_satisfied = []
        self.stop_clock = []
        self.in_tl_zone = []
        self.tl_color = []

        target_mult = 1.0
        for ss in self.stop_signs:
            dx = pose[0] - ss.position[0]
            dz = pose[1] - ss.position[1]
            in_zone = (dx * dx + dz * dz) ** 0.5 <= ss.trigger_radius
            self.in_stop_zone.append(in_zone)
            if ss.update(pose, float(speed), float(t)):
                target_mult = 0.0
            self.stop_satisfied.append(ss.satisfied)
            self.stop_clock.append(ss.stopped_at_time)
        for tl in self.traffic_lights:
            dx = pose[0] - tl.position[0]
            dz = pose[1] - tl.position[1]
            in_zone = (dx * dx + dz * dz) ** 0.5 <= tl.trigger_radius
            self.in_tl_zone.append(in_zone)
            self.tl_color.append(tl.color_at(float(t)))
            f = tl.speed_multiplier(pose, float(t))
            if f is not None:
                target_mult = min(target_mult, f)
        self.target_mult = target_mult

        # Smoothly slew effective_mult toward target_mult — eliminates the
        # abrupt zero-everything stop and creates a real braking phase.
        if target_mult < self.effective_mult:
            step = self.dt / max(1e-3, self.decel_time)
            new_mult = max(target_mult, self.effective_mult - step)
        else:
            step = self.dt / max(1e-3, self.accel_time)
            new_mult = min(target_mult, self.effective_mult + step)
        # Brake intensity: how fast the gate is dropping (0 .. 1).
        self.brake = max(0.0, (self.effective_mult - new_mult) / max(step, 1e-3))
        self.effective_mult = new_mult

        action[0] = float(action[0]) * self.effective_mult
        # Diff-drive: if effective mult is near zero, also gate omega to
        # avoid in-place spinning while the rule asks for a halt.
        if self.effective_mult < 0.05:
            action[1] = 0.0
        if advance_time:
            self.t += self.dt
        return action


def make_sim_pose_fn(sim) -> Callable[[], Tuple[float, float, float]]:
    """Return a function reading ``(x, z, speed)`` from a duckietown sim."""

    def _read():
        return (
            float(sim.cur_pos[0]),
            float(sim.cur_pos[2]),
            float(getattr(sim, "speed", 0.0)),
        )

    return _read
