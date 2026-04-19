#!/usr/bin/env python3
"""Multiview MP4 on the mining / off-road preset (winding zigzag ≈ pit switchbacks).

The simulator has no elevation mesh; “going down the pit” is approximated by the
long zigzag lane graph in ``zigzag_dists`` plus dark haul-field visuals.

  python scripts/demo_mining_pit.py --duration 25 -o recordings/mining_pit_multiview.mp4
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

DEFAULT_ENV = "Duckietown-zigzag_dists_mining-v0"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--env-id", default=DEFAULT_ENV)
    p.add_argument(
        "-o",
        "--output",
        default=os.path.join(_REPO, "recordings", "mining_pit_multiview.mp4"),
    )
    p.add_argument("--duration", type=float, default=25.0)
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=6000)
    p.add_argument("--omega-gain", type=float, default=6.5)
    p.add_argument("--ref-v", type=float, default=0.48)
    p.add_argument("--fd", type=float, default=0.26)
    p.add_argument(
        "--no-labels",
        dest="labels",
        action="store_false",
        help="Disable quadrant captions on the multiview panel",
    )
    p.set_defaults(labels=True)
    args = p.parse_args()

    num_frames = max(1, int(round(args.duration * args.fps)))

    env = gym.make(
        args.env_id,
        disable_env_checker=True,
        max_steps=max(args.max_steps, num_frames + 200),
    )
    env.seed(args.seed)
    obs = env.reset()
    pol = PurePursuitPolicy(
        env.unwrapped,
        ref_velocity=args.ref_v,
        following_distance=args.fd,
    )

    frames = []
    for _ in range(num_frames):
        frames.append(env.unwrapped.render_multiview_rgb(labels=args.labels))
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
    print("wrote", args.output, "frames=", len(frames), "target=", num_frames)


if __name__ == "__main__":
    main()
