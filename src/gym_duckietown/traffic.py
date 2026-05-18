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
    ) -> None:
        self.base = base
        self.agent_pose_fn = agent_pose_fn
        self.stop_signs: List[StopSign] = list(stop_signs or [])
        self.traffic_lights: List[TrafficLight] = list(traffic_lights or [])
        self.dt = float(dt)
        self.t = 0.0

    def reset(self) -> None:
        self.t = 0.0
        for ss in self.stop_signs:
            ss.reset()

    def __call__(self, observation: Any, t: Optional[float] = None) -> np.ndarray:
        action = np.asarray(self.base(observation), dtype=np.float32).copy()
        if t is None:
            t = self.t

        x, z, speed = self.agent_pose_fn()
        pose = (float(x), float(z))

        mult = 1.0
        for ss in self.stop_signs:
            if ss.update(pose, float(speed), float(t)):
                mult = 0.0
        for tl in self.traffic_lights:
            f = tl.speed_multiplier(pose, float(t))
            if f is not None:
                mult = min(mult, f)

        action[0] = float(action[0]) * mult
        # Differential-drive: with v=0 and omega!=0 the wheels turn in
        # opposite directions and the bot spins in place. Force omega to 0
        # as well so a "full stop" actually holds still.
        if mult <= 0.0:
            action[1] = 0.0
        if t is None or t == self.t:
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
