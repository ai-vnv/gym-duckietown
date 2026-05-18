#!/usr/bin/env python3
"""CLI wrapper around :mod:`gym_duckietown.multiview`.

Renders a 5-view + trajectory composite mp4 of a duckietown rollout driven
by a smoothed PurePursuit controller. All real logic lives in
``gym_duckietown.multiview``; this file just translates argparse into a
``RolloutConfig`` and a :class:`SmoothedController`.

Examples
--------
    # short demo (16 s)
    .venv/bin/python scripts/render_5view_trajectory.py \\
        --duration 16 --fps 20 \\
        -o recordings/5view_smallloop_16s.mp4

    # 5 minute loopable on small_loop
    .venv/bin/python scripts/render_5view_trajectory.py \\
        --duration 300 --fps 20 --traj-window-sec 15 \\
        -o recordings/5view_smallloop_5min.mp4
"""
from __future__ import annotations

import argparse
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "learning/imitation/iil-dagger"))

import gym  # noqa: E402
import gym_duckietown  # noqa: E402 F401
from gym_duckietown.multiview import RolloutConfig, SmoothedController, run_rollout  # noqa: E402
from teacher.pure_pursuit_policy import PurePursuitPolicy  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--env-id", default="Duckietown-small_loop-v0")
    p.add_argument(
        "-o",
        "--output",
        default=os.path.join(_REPO, "recordings", "5view_trajectory.mp4"),
    )
    p.add_argument("--duration", type=float, default=16.0, help="Clip length in seconds.")
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tile-width", type=int, default=320)
    p.add_argument("--tile-height", type=int, default=240)
    p.add_argument("--ref-velocity", type=float, default=0.45)
    p.add_argument("--following-distance", type=float, default=0.25)
    p.add_argument("--omega-gain", type=float, default=1.0)
    p.add_argument("--smooth-alpha", type=float, default=0.35)
    p.add_argument("--top-rear-height", type=float, default=0.30)
    p.add_argument(
        "--traj-window-sec",
        type=float,
        default=None,
        help="Show only the last N seconds of path (makes clips loopable).",
    )
    p.add_argument("--max-steps", type=int, default=100_000)
    p.add_argument("--bitrate", default="4M")
    args = p.parse_args()

    config = RolloutConfig(
        output=args.output,
        duration=args.duration,
        fps=args.fps,
        tile_width=args.tile_width,
        tile_height=args.tile_height,
        top_rear_height=args.top_rear_height,
        traj_window_sec=args.traj_window_sec,
        bitrate=args.bitrate,
    )

    env = gym.make(
        args.env_id,
        disable_env_checker=True,
        max_steps=max(args.max_steps, config.num_frames + 100),
    )
    env.seed(args.seed)
    env.reset()

    policy = PurePursuitPolicy(
        env.unwrapped,
        ref_velocity=args.ref_velocity,
        following_distance=args.following_distance,
    )
    controller = SmoothedController(
        policy, alpha=args.smooth_alpha, omega_gain=args.omega_gain
    )

    try:
        written = run_rollout(env, controller, config)
    finally:
        env.close()

    print(
        "wrote",
        config.output,
        "frames=",
        written,
        "target=",
        config.num_frames,
        "fps=",
        config.fps,
    )


if __name__ == "__main__":
    main()
