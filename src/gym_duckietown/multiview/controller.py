"""Smoothing wrapper for any policy that returns ``[v, omega]`` actions."""
from __future__ import annotations

from typing import Any, Protocol

import numpy as np


class _PolicyLike(Protocol):
    def predict(self, observation: Any) -> Any: ...  # noqa: D401, E704


class SmoothedController:
    """Exponential moving average over raw policy actions.

    Wraps any object with a ``predict(obs) -> (v, omega)`` interface (the
    duckietown ``PurePursuitPolicy`` is the canonical one). On each call the
    raw policy action is multiplied by ``omega_gain`` on the steering channel
    and then blended with the previous smoothed action:

        a_t = alpha * a_{t-1} + (1 - alpha) * a_raw

    ``alpha`` in [0, 1):
      - 0.0  -> no smoothing, just gain.
      - 0.35 -> gentle smoothing, responsive on corners (default).
      - 0.7+ -> heavy smoothing; may lag and leave drivable area.

    State (``_prev``) is bootstrapped from the first call, so the first
    action is exactly ``omega_gain``-scaled raw without lag.
    """

    def __init__(self, policy: _PolicyLike, alpha: float = 0.35, omega_gain: float = 1.0):
        if not (0.0 <= alpha < 1.0):
            raise ValueError(f"alpha must be in [0, 1), got {alpha}")
        self.policy = policy
        self.alpha = float(alpha)
        self.omega_gain = float(omega_gain)
        self._prev: np.ndarray | None = None

    def reset(self) -> None:
        """Forget the running smoothed state (call between rollouts)."""
        self._prev = None

    def __call__(self, observation: Any) -> np.ndarray:
        raw = self.policy.predict(observation)
        a = np.array([float(raw[0]), float(raw[1]) * self.omega_gain], dtype=np.float32)
        if self._prev is None:
            self._prev = a
        else:
            self._prev = self.alpha * self._prev + (1.0 - self.alpha) * a
        return self._prev.copy()
