"""High-level rollout pipeline: env + controller -> composite mp4.

Designed so the rendering loop, controller construction, and per-frame
composition are each replaceable. ``run_rollout`` is the convenience
one-liner; if you want to plug in a different controller or panel layout,
import the lower-level pieces and write your own loop.
"""
from __future__ import annotations

import csv
import math
import os
import time
from dataclasses import dataclass
from typing import Callable, List, Optional

import imageio.v2 as imageio

from .cameras import render_all_views
from .composer import compose_frame, default_layout
from .trajectory import TrajectoryPanel, sliding_window


@dataclass
class RolloutConfig:
    """Settings for :func:`run_rollout`.

    Attributes
    ----------
    output : str
        Destination mp4 path. Parent directory is created if missing.
    duration : float
        Clip length in seconds.
    fps : int
        Output / rendering frame rate.
    tile_width, tile_height : int
        Per-panel size in pixels. Final frame is ``tile_w * cols`` by
        ``tile_h * rows``.
    top_rear_height : float
        Meters added to ``sim.cam_height`` for the top-mounted rear camera.
    traj_window_sec : float or None
        If set, the trajectory panel shows only the last N seconds of path.
        Use this for visually loopable clips on closed circuits.
    bitrate : str
        ffmpeg bitrate (e.g. ``"4M"``).
    log_every : int
        Print progress every N frames. 0 disables.
    """

    output: str
    duration: float = 16.0
    fps: int = 20
    tile_width: int = 320
    tile_height: int = 240
    side_cam_height: float = 0.12
    side_cam_pitch_deg: float = -8.0
    cam_fov_y: float = 110.0  # vertical FOV (deg). Sim default 75 → 110 is ultra-wide.
    driver_pitch_deg: float = 8.0  # sim default 19.15 — too downward; 8 shows periphery better
    log_csv: Optional[str] = None  # path; auto-derived from output if None
    vantage_from: Optional[tuple] = None  # (x, y, z); None → SW corner default
    vantage_at: Optional[tuple] = None  # (x, y, z); None → map center
    traj_window_sec: Optional[float] = None
    bitrate: str = "4M"
    log_every: int = 0  # 0 => auto

    @property
    def num_frames(self) -> int:
        return max(1, int(round(self.duration * self.fps)))

    @property
    def traj_window_frames(self) -> Optional[int]:
        if self.traj_window_sec is None:
            return None
        return int(round(self.traj_window_sec * self.fps))


def run_rollout(
    env,
    controller: Callable,
    config: RolloutConfig,
    on_progress: Optional[Callable[[int, int, float], None]] = None,
    pre_render: Optional[Callable[[int, float, object], None]] = None,
) -> int:
    """Roll the env with ``controller``, write a composite mp4, return frames written.

    The caller is responsible for seeding and resetting ``env`` before
    calling this function (so seeding behavior is explicit and reproducible).
    The initial observation is pulled via ``sim.render_obs()``.

    ``controller`` must be callable as ``controller(obs) -> action``. Use
    :class:`SmoothedController` or any equivalent.

    ``on_progress(frame_idx, total, render_fps)`` is invoked periodically
    when set; otherwise progress is printed to stdout when
    ``config.log_every`` is positive (or auto-computed from total frames).
    """
    sim = env.unwrapped
    if config.cam_fov_y is not None:
        sim.cam_fov_y = float(config.cam_fov_y)
    if config.driver_pitch_deg is not None:
        sim.cam_angle[0] = float(config.driver_pitch_deg)
    obs = sim.render_obs()

    # Telemetry log — auto-derives from output path if not specified.
    log_path = config.log_csv
    if log_path is None:
        base, _ext = os.path.splitext(config.output)
        log_path = base + "_telemetry.csv"
    log_file = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow([
        "step", "time_s",
        "pos_x", "pos_z", "heading_rad",
        "speed_mps", "ang_vel_rad_s",
        "action_v", "action_omega",
        "effective_mult", "brake", "target_mult",
        "in_stop_zone", "stop_satisfied", "stop_clock_s",
        "in_tl_zone", "tl_color",
        "odometer_m",
    ])

    panel = TrajectoryPanel(sim, h=config.tile_height, w=config.tile_width)

    os.makedirs(os.path.dirname(config.output) or ".", exist_ok=True)
    writer = imageio.get_writer(
        config.output,
        fps=config.fps,
        macro_block_size=None,
        codec="libx264",
        bitrate=config.bitrate,
        ffmpeg_log_level="error",
    )

    total = config.num_frames
    window = config.traj_window_frames
    log_every = config.log_every or max(1, total // 20)

    xs: List[float] = []
    zs: List[float] = []
    written = 0
    last_action = (0.0, 0.0)
    odometer = 0.0
    last_pos = None
    last_angle: Optional[float] = None
    ang_vel = 0.0
    t0 = time.time()

    dt = 1.0 / float(config.fps)
    try:
        for step_idx in range(total):
            x, _y, z = sim.cur_pos
            xs.append(float(x))
            zs.append(float(z))
            if window is not None:
                sliding_window(xs, zs, window)
            # Odometer: cumulative Euclidean distance in the (x, z) plane.
            if last_pos is not None:
                odometer += float(
                    ((x - last_pos[0]) ** 2 + (z - last_pos[1]) ** 2) ** 0.5
                )
            last_pos = (float(x), float(z))

            if pre_render is not None:
                pre_render(step_idx, step_idx * dt, sim)

            views = render_all_views(
                sim,
                side_cam_height=config.side_cam_height,
                side_cam_pitch_deg=config.side_cam_pitch_deg,
                vantage_from=config.vantage_from,
                vantage_at=config.vantage_at,
            )
            # Residual actuator decay leaves sim.speed at ~0.06 m/s even when
            # the controller commanded a full stop; treat anything below
            # 0.10 m/s as effectively stationary in the dashboard.
            _raw_speed = float(getattr(sim, "speed", 0.0))
            _disp_speed = 0.0 if abs(_raw_speed) < 0.10 else _raw_speed
            # angular velocity from successive cur_angle
            cur_ang = float(getattr(sim, "cur_angle", 0.0))
            if last_angle is not None:
                # smallest signed angle delta
                da = (cur_ang - last_angle + math.pi) % (2 * math.pi) - math.pi
                ang_vel = da / dt
            # brake (0..1): set by RuleAwareController; falls back to 0 for plain controllers
            brake = float(getattr(controller, "brake", 0.0))
            metrics = {
                "speed":   f"{_disp_speed:.2f} m/s",
                "gas":     f"{max(0.0, float(last_action[0])):+.2f}",
                "brake":   f"{brake:.2f}",
                "steer":   f"{float(last_action[1]):+.2f}",
                "odo":     f"{odometer:.1f} m",
                "time":    f"{step_idx * dt:5.1f} s",
            }
            traj_img = panel.render(xs, zs, metrics=metrics)
            frame = compose_frame(
                default_layout(views, traj_img),
                tile_h=config.tile_height,
                tile_w=config.tile_width,
            )
            writer.append_data(frame)
            written += 1

            action = controller(obs)
            last_action = (float(action[0]), float(action[1]))

            # --- CSV telemetry row (one per rendered frame) -----------------
            ctrl_state = controller  # may be a RuleAwareController (exposes state)
            in_stop = getattr(ctrl_state, "in_stop_zone", []) or [False]
            stop_sat = getattr(ctrl_state, "stop_satisfied", []) or [False]
            stop_clk = getattr(ctrl_state, "stop_clock", []) or [None]
            in_tl = getattr(ctrl_state, "in_tl_zone", []) or [False]
            tl_col = getattr(ctrl_state, "tl_color", []) or [""]
            log_writer.writerow([
                step_idx, f"{step_idx * dt:.3f}",
                f"{float(x):.4f}", f"{float(z):.4f}", f"{cur_ang:.4f}",
                f"{_raw_speed:.4f}", f"{ang_vel:.4f}",
                f"{last_action[0]:.4f}", f"{last_action[1]:.4f}",
                f"{float(getattr(ctrl_state, 'effective_mult', 1.0)):.3f}",
                f"{brake:.3f}",
                f"{float(getattr(ctrl_state, 'target_mult', 1.0)):.3f}",
                int(any(in_stop)),
                int(any(stop_sat)),
                (f"{stop_clk[0]:.3f}" if stop_clk and stop_clk[0] is not None else ""),
                int(any(in_tl)),
                ";".join(tl_col),
                f"{odometer:.3f}",
            ])
            last_angle = cur_ang

            obs, _r, done, _info = env.step(action)
            if done:
                print(
                    f"[warn] environment terminated early at step {step_idx + 1}/{total}",
                    flush=True,
                )
                break

            if (step_idx + 1) % log_every == 0:
                elapsed = time.time() - t0
                rate = (step_idx + 1) / max(elapsed, 1e-6)
                if on_progress is not None:
                    on_progress(step_idx + 1, total, rate)
                else:
                    eta = (total - step_idx - 1) / max(rate, 1e-6)
                    print(
                        f"[progress] {step_idx + 1}/{total} frames "
                        f"({rate:.1f} fps render, eta {eta:.1f}s)",
                        flush=True,
                    )
    finally:
        writer.close()
        log_file.close()

    return written
