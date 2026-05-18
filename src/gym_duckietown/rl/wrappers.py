"""Thin gymnasium wrappers for RL training on JISR3.

Kept minimal — the env already emits a 5-view composite at fixed size, so
these wrappers focus on (a) optional spatial resize for downstream models,
(b) per-episode metric aggregation for W&B, and (c) reward clipping for
training stability.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces


# Names of the reward components emitted by JISR3Env.step() in
# ``info["Simulator"]["reward_parts"]``. The episode wrapper sums each one
# across the episode so a stacked-area plot in W&B can show how the policy
# is getting its return.
REWARD_PART_KEYS = (
    "r_lane", "r_heading", "r_speed", "r_alive",
    "r_stop_ok", "r_stop_bad", "r_roll", "r_collide",
)


class ResizeObservation(gym.ObservationWrapper):
    """Resize HxWx3 uint8 observations to ``(out_h, out_w)`` using cv2 INTER_AREA."""

    def __init__(self, env: gym.Env, size: Tuple[int, int]):
        super().__init__(env)
        self.out_h, self.out_w = int(size[0]), int(size[1])
        old = env.observation_space
        assert isinstance(old, spaces.Box) and old.dtype == np.uint8, (
            "ResizeObservation expects a uint8 Box observation"
        )
        c = old.shape[-1]
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(self.out_h, self.out_w, c), dtype=np.uint8
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        import cv2

        return cv2.resize(obs, (self.out_w, self.out_h), interpolation=cv2.INTER_AREA)


class RewardClip(gym.RewardWrapper):
    """Clip the per-step reward into ``[low, high]``."""

    def __init__(self, env: gym.Env, low: float = -50.0, high: float = 50.0):
        super().__init__(env)
        self.low = float(low)
        self.high = float(high)

    def reward(self, r):
        return float(np.clip(r, self.low, self.high))


class EpisodeInfoWrapper(gym.Wrapper):
    """Aggregate per-episode metrics into ``info['episode_metrics']`` on done.

    Tracked metrics (clean, presentation-ready):

    * ``return``, ``length`` — total reward and steps.
    * ``mean_lane_dist`` (m), ``in_lane_fraction`` — lane-following quality.
    * ``mean_speed``, ``max_speed`` — speed profile.
    * ``stop_approaches``, ``stop_satisfied``, ``stop_violations``,
      ``stop_compliance_rate`` — stop-sign behaviour.
    * ``collision`` (bool), ``timed_out`` (bool) — termination type.
    * ``reward/<part>`` for each component in :data:`REWARD_PART_KEYS` —
      summed across the episode so a stacked-area chart shows where the
      return came from.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._reset_state()

    def _reset_state(self):
        self._n_steps = 0
        self._lane_dist_sum = 0.0
        self._lane_dist_n = 0
        self._speed_sum = 0.0
        self._speed_max = 0.0
        self._stop_satisfied = 0
        self._stop_violations = 0
        self._stop_approaches = 0
        self._prev_triggered: Optional[list] = None
        self._return = 0.0
        self._reward_part_sums: Dict[str, float] = {k: 0.0 for k in REWARD_PART_KEYS}

    def reset(self, **kwargs):
        self._reset_state()
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        self._return += float(reward)
        self._n_steps += 1

        sim_info = info.get("Simulator", {}) if isinstance(info, dict) else {}
        lp = sim_info.get("lane_position")
        if lp is not None:
            self._lane_dist_sum += abs(float(lp["dist"]))
            self._lane_dist_n += 1
        speed = sim_info.get("robot_speed")
        if speed is not None:
            s = float(speed)
            self._speed_sum += s
            if s > self._speed_max:
                self._speed_max = s

        stop_states = sim_info.get("stop_sign_states", [])
        if self._prev_triggered is None:
            self._prev_triggered = [False] * len(stop_states)
        for i, st in enumerate(stop_states):
            if i >= len(self._prev_triggered):
                self._prev_triggered.append(False)
            triggered = bool(st.get("triggered", False))
            if triggered and not self._prev_triggered[i]:
                self._stop_approaches += 1
            self._prev_triggered[i] = triggered

        parts = sim_info.get("reward_parts", {})
        for k in REWARD_PART_KEYS:
            v = parts.get(k, 0.0)
            if v:
                self._reward_part_sums[k] += float(v)
        if parts.get("r_stop_ok", 0.0) > 0.0:
            self._stop_satisfied += 1
        if parts.get("r_stop_bad", 0.0) < 0.0:
            self._stop_violations += 1

        if done:
            mean_lane_dist = (
                self._lane_dist_sum / self._lane_dist_n if self._lane_dist_n else float("nan")
            )
            in_lane_fraction = self._lane_dist_n / max(1, self._n_steps)
            mean_speed = self._speed_sum / max(1, self._n_steps)
            compliance = (
                self._stop_satisfied / self._stop_approaches
                if self._stop_approaches > 0
                else float("nan")
            )
            done_code = sim_info.get("done_code", "unknown")
            metrics: Dict[str, float] = {
                "return": self._return,
                "length": self._n_steps,
                "mean_lane_dist": mean_lane_dist,
                "in_lane_fraction": in_lane_fraction,
                "mean_speed": mean_speed,
                "max_speed": self._speed_max,
                "stop_approaches": self._stop_approaches,
                "stop_satisfied": self._stop_satisfied,
                "stop_violations": self._stop_violations,
                "stop_compliance_rate": compliance,
                "collision": 1.0 if done_code == "invalid-pose" else 0.0,
                "timed_out": 1.0 if done_code == "max-steps-reached" else 0.0,
            }
            for k, v in self._reward_part_sums.items():
                # ``r_lane`` → ``reward/lane`` for a clean W&B panel group.
                metrics[f"reward/{k.removeprefix('r_')}"] = v
            info["episode_metrics"] = metrics
        return obs, reward, terminated, truncated, info
