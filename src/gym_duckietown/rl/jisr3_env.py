"""Gymnasium env for RL training on the ``small_loop_jisr3`` scenario.

The env wraps :class:`gym_duckietown.simulator.Simulator` and applies the
existing scenario factory (``apply_scenario(env, get_scenario("small_loop_jisr3"))``)
on each reset. It produces a multi-camera composite observation and a
custom reward that combines lane-centering, forward progress, and stop-sign
compliance — the simulator's default reward does not enforce the stop sign.

Stop-sign logic reuses :class:`gym_duckietown.traffic.StopSign`. Per step we
read the agent pose + speed and call :meth:`StopSign.update` to advance its
state machine; the rising-edge of ``satisfied`` rewards stopping, and an
exit from the trigger zone without satisfaction is penalized.

The traffic light's visual state (red/yellow/green billboard swap) is also
advanced per step via :func:`sync_traffic_light_visuals` so the side cameras
see the correct color — agents don't have to obey it, but a well-trained
policy can learn to slow down if the light is red.

The underlying Simulator returns the legacy 4-tuple step API; this wrapper
adapts to gymnasium's 5-tuple ``(obs, reward, terminated, truncated, info)``.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from gym_duckietown.multiview.cameras import render_all_views
from gym_duckietown.multiview.scenarios import (
    apply_scenario,
    get_scenario,
    sync_traffic_light_visuals,
)
from gym_duckietown.simulator import Simulator
from gym_duckietown.traffic import StopSign, TrafficLight


# ----------------------------------------------------------------------------
# Multi-view composite configuration
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class MultiViewObservation:
    """Layout for the 5-view composite observation."""

    cell_h: int = 48
    cell_w: int = 48
    cols: int = 3
    rows: int = 2

    # order of cells in row-major form; "blank" pads the empty cell.
    cell_order: Tuple[str, ...] = ("driver", "left", "right", "bev", "vantage", "blank")

    @property
    def shape(self) -> Tuple[int, int, int]:
        return (self.cell_h * self.rows, self.cell_w * self.cols, 3)


DEFAULT_MULTIVIEW = MultiViewObservation()


# ----------------------------------------------------------------------------
# Reward weights
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class RewardWeights:
    lane: float = 8.0           # multiplies |lp.dist|, sign flipped
    lane_not_in: float = 5.0    # penalty when NotInLane
    heading: float = 2.0        # multiplies max(0, lp.dot_dir)
    speed: float = 1.5          # multiplies speed * max(0, lp.dot_dir)
    alive: float = 0.02         # per-step constant cost
    collide: float = 40.0       # terminal collision penalty (added on done)
    stop_satisfied: float = 5.0     # rising-edge bonus
    stop_violation: float = 20.0    # exit zone without satisfying
    stop_roll: float = 0.5      # rolling-through pressure per step


DEFAULT_REWARDS = RewardWeights()


# ----------------------------------------------------------------------------
# Env
# ----------------------------------------------------------------------------


class JISR3Env(gym.Env):
    """RL env on the ``small_loop_jisr3`` scenario with multi-camera obs."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        *,
        max_steps: int = 1500,
        frame_rate: float = 20.0,
        seed: Optional[int] = None,
        scenario_name: str = "small_loop_jisr3",
        view_cfg: MultiViewObservation = DEFAULT_MULTIVIEW,
        rewards: RewardWeights = DEFAULT_REWARDS,
        side_cam_height: float = 0.12,
        side_cam_pitch_deg: float = -8.0,
        render_mode: Optional[str] = "rgb_array",
    ) -> None:
        super().__init__()
        self.scenario_name = scenario_name
        self.view_cfg = view_cfg
        self.rewards = rewards
        self.side_cam_height = side_cam_height
        self.side_cam_pitch_deg = side_cam_pitch_deg
        self.render_mode = render_mode

        # Cameras at least as large as a single cell; we keep them small to
        # bound GL cost (5 renders per env step).
        cam_w = max(view_cfg.cell_w, 64)
        cam_h = max(view_cfg.cell_h, 64)

        self.sim = Simulator(
            map_name="small_loop",
            max_steps=max_steps,
            domain_rand=False,
            full_transparency=True,
            accept_start_angle_deg=4,
            frame_rate=frame_rate,
            camera_width=cam_w,
            camera_height=cam_h,
            seed=seed,
        )

        # Re-create action / observation spaces in gymnasium's namespace —
        # the Simulator uses legacy `gym.spaces.Box`, which gymnasium does
        # not accept directly.
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=view_cfg.shape, dtype=np.uint8
        )

        # Filled at reset.
        self.stop_signs: List[StopSign] = []
        self.traffic_lights: List[TrafficLight] = []
        self.tl_state_billboards: List[Dict[str, Any]] = []
        self.t: float = 0.0
        self.dt: float = 1.0 / float(frame_rate)

    # ---- gymnasium API --------------------------------------------------

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self.sim.seed(seed)
        self.sim.reset()
        scenario = get_scenario(self.scenario_name)
        applied = apply_scenario(_UnwrapAdapter(self.sim), scenario)
        self.stop_signs = list(applied["stop_signs"])
        self.traffic_lights = list(applied["traffic_lights"])
        self.tl_state_billboards = list(applied["traffic_light_state_billboards"])
        for ss in self.stop_signs:
            ss.reset()
        self.t = 0.0
        obs = self._render_multi_view()
        return obs, {}

    def step(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        sim = self.sim

        prev_states = [(ss.triggered, ss.satisfied) for ss in self.stop_signs]

        for _ in range(sim.frame_skip):
            sim.update_physics(action)

        sync_traffic_light_visuals(self.traffic_lights, self.tl_state_billboards, self.t)
        obs = self._render_multi_view()

        pos_xz = (float(sim.cur_pos[0]), float(sim.cur_pos[2]))
        speed = float(sim.speed)

        stop_satisfied_events = 0
        stop_violation_events = 0
        stop_roll_pressure = 0
        per_sign_compliance: List[Dict[str, Any]] = []
        for ss, (was_triggered, was_satisfied) in zip(self.stop_signs, prev_states):
            ss.update(pos_xz, speed, self.t)
            if not was_satisfied and ss.satisfied:
                stop_satisfied_events += 1
            if was_triggered and not ss.triggered and not was_satisfied:
                stop_violation_events += 1
            if ss.triggered and not ss.satisfied and speed > ss.stop_speed_threshold:
                stop_roll_pressure += 1
            per_sign_compliance.append({
                "triggered": bool(ss.triggered),
                "satisfied": bool(ss.satisfied),
                "dist": float(hypot(pos_xz[0] - ss.position[0], pos_xz[1] - ss.position[1])),
            })

        # Use sim's own check for terminal state; split into terminated vs
        # truncated for gymnasium.
        d = sim._compute_done_reward()
        done_code = d.done_code
        terminated = bool(d.done) and done_code != "max-steps-reached"
        truncated = bool(d.done) and done_code == "max-steps-reached"

        misc = sim.get_agent_info()
        sim_info = misc["Simulator"]
        sim_info["msg"] = d.done_why
        sim_info["done_code"] = done_code

        lp = sim_info.get("lane_position")
        r = self.rewards
        parts: Dict[str, float] = {}

        if lp is None:
            parts["r_lane"] = -r.lane_not_in
            parts["r_heading"] = 0.0
            parts["r_speed"] = 0.0
            lane_dist = float("nan")
        else:
            lane_dist = float(lp["dist"])
            dot_dir = float(lp["dot_dir"])
            parts["r_lane"] = -r.lane * abs(lane_dist)
            parts["r_heading"] = r.heading * max(0.0, dot_dir)
            parts["r_speed"] = r.speed * speed * max(0.0, dot_dir)

        parts["r_alive"] = -r.alive
        parts["r_stop_ok"] = r.stop_satisfied * stop_satisfied_events
        parts["r_stop_bad"] = -r.stop_violation * stop_violation_events
        parts["r_roll"] = -r.stop_roll * stop_roll_pressure
        parts["r_collide"] = -r.collide if done_code == "invalid-pose" else 0.0

        reward = float(sum(parts.values()))

        sim_info["reward_parts"] = parts
        sim_info["stop_sign_states"] = per_sign_compliance
        sim_info["lane_dist_abs"] = float(abs(lane_dist)) if lp is not None else float("nan")
        sim_info["in_lane"] = lp is not None

        self.t += self.dt
        return obs, reward, terminated, truncated, misc

    def render(self):
        return self._render_multi_view()

    def close(self):
        try:
            self.sim.close()
        except Exception:
            pass

    # ---- internals ------------------------------------------------------

    def _render_multi_view(self) -> np.ndarray:
        """Render the 5-view composite and return an HxWx3 uint8 image."""
        views = render_all_views(
            self.sim,
            side_cam_height=self.side_cam_height,
            side_cam_pitch_deg=self.side_cam_pitch_deg,
        )
        # cv2 imported lazily to avoid breaking pyglet GL on macOS (matches
        # the pattern in simulator.render_multiview_rgb).
        import cv2

        cfg = self.view_cfg
        ch, cw = cfg.cell_h, cfg.cell_w
        cells: List[np.ndarray] = []
        for name in cfg.cell_order:
            if name == "blank" or name not in views:
                cells.append(np.zeros((ch, cw, 3), dtype=np.uint8))
                continue
            tile = views[name]
            if tile.shape[:2] != (ch, cw):
                tile = cv2.resize(tile, (cw, ch), interpolation=cv2.INTER_AREA)
            cells.append(np.ascontiguousarray(tile, dtype=np.uint8))

        rows = []
        for r in range(cfg.rows):
            row_cells = cells[r * cfg.cols : (r + 1) * cfg.cols]
            rows.append(np.concatenate(row_cells, axis=1))
        composite = np.concatenate(rows, axis=0)
        return composite


# ----------------------------------------------------------------------------
# Adapter so apply_scenario can read env.unwrapped without us being a real
# gym.Wrapper around the sim.
# ----------------------------------------------------------------------------


class _UnwrapAdapter:
    """Lets ``apply_scenario`` treat a bare Simulator like a Wrapper."""

    def __init__(self, sim: Simulator) -> None:
        self.unwrapped = sim
