#!/usr/bin/env python3
"""Short MP4: 2×2 multiview (driver, map bird's-eye, top follow, rear)."""
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
    p.add_argument("--frames", type=int, default=90)
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    env = gym.make(args.env_id, disable_env_checker=True)
    env.seed(args.seed)
    obs = env.reset()
    pol = PurePursuitPolicy(env.unwrapped, ref_velocity=0.55, following_distance=0.26)

    frames = []
    for _ in range(args.frames):
        frames.append(env.unwrapped.render_multiview_rgb(labels=True))
        a = np.array(pol.predict(obs), dtype=np.float32)
        obs, _r, done, _info = env.step(a)
        if done:
            break
    env.close()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    imageio.mimsave(args.output, frames, fps=args.fps, macro_block_size=None)
    print("wrote", args.output, "n=", len(frames))


if __name__ == "__main__":
    main()
