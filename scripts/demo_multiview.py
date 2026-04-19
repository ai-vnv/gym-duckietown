#!/usr/bin/env python3
"""MP4: 2×2 multiview (driver, map bird's-eye, top follow, rear).

Example (KFUPM-style campus + desert tint, ~30 s):

  python scripts/demo_multiview.py \\
    --env-id Duckietown-udem1_arabian-v0 \\
    --duration 30 --fps 15 \\
    -o recordings/multiview_udem1_arabian_30s.mp4
"""
from __future__ import annotations

import argparse
import os
import sys

import imageio.v2 as imageio
import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "learning/imitation/iil-dagger"))

import gym  # noqa: E402
import gym_duckietown  # noqa: E402
from teacher.pure_pursuit_policy import PurePursuitPolicy  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--env-id", default="Duckietown-small_loop-v0")
    p.add_argument(
        "-o",
        "--output",
        default=os.path.join(_REPO, "recordings", "multiview_demo.mp4"),
    )
    p.add_argument("--frames", type=int, default=90, help="Ignored if --duration is set")
    p.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Clip length in seconds; frame count = round(duration * fps)",
    )
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--max-steps",
        type=int,
        default=5000,
        help="Simulator max_steps (raise for long clips on large maps)",
    )
    p.add_argument(
        "--omega-gain",
        type=float,
        default=1.0,
        help="Multiply Pure Pursuit steering command (higher = sharper turns)",
    )
    args = p.parse_args()

    if args.duration is not None:
        num_frames = max(1, int(round(args.duration * args.fps)))
    else:
        num_frames = args.frames

    env = gym.make(
        args.env_id,
        disable_env_checker=True,
        max_steps=max(args.max_steps, num_frames + 100),
    )
    env.seed(args.seed)
    obs = env.reset()
    pol = PurePursuitPolicy(env.unwrapped, ref_velocity=0.55, following_distance=0.26)

    frames = []
    for _ in range(num_frames):
        frames.append(env.unwrapped.render_multiview_rgb(labels=True))
        raw = pol.predict(obs)
        a = np.array(
            [float(raw[0]), float(raw[1]) * args.omega_gain],
            dtype=np.float32,
        )
        obs, _r, done, _info = env.step(a)
        if done:
            break
    env.close()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    imageio.mimsave(args.output, frames, fps=args.fps, macro_block_size=None)
    print(
        "wrote",
        args.output,
        "frames=",
        len(frames),
        "target=",
        num_frames,
        "fps=",
        args.fps,
    )


if __name__ == "__main__":
    main()
