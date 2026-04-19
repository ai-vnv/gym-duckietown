#!/usr/bin/env python3
"""
Record a fixed-length MP4: stock ``udem1`` campus map + Arabian desert sky/ground.

Duckietown has no KFUPM geodata; ``udem1`` is the largest stock *university-style*
road layout. For real buildings / satellite alignment, see README or project docs
on geospatial pipelines (OSM, photogrammetry, game engines).
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

DEFAULT_ENV = "Duckietown-udem1_arabian-v0"


def record(
    out_path: str,
    *,
    env_id: str = DEFAULT_ENV,
    duration_sec: float = 20.0,
    fps: int = 30,
    seed: int = 2,
    omega_gain: float = 5.5,
    ref_v: float = 0.55,
    fd: float = 0.28,
    max_steps: int = 5000,
) -> tuple[int, str]:
    num_frames = max(1, int(round(duration_sec * fps)))
    env = gym.make(
        env_id,
        disable_env_checker=True,
        max_steps=max(max_steps, num_frames + 50),
    )
    env.seed(seed)
    obs = env.reset()
    pol = PurePursuitPolicy(env.unwrapped, ref_velocity=ref_v, following_distance=fd)
    frames: list[np.ndarray] = []
    last_msg = ""
    for step in range(num_frames):
        frames.append(np.asarray(obs).copy())
        raw = pol.predict(obs)
        a = np.array([float(raw[0]), float(raw[1]) * omega_gain], dtype=np.float32)
        obs, _r, done, info = env.step(a)
        last_msg = info.get("Simulator", {}).get("msg", "")
        if done:
            break
    env.close()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    imageio.mimsave(out_path, frames, fps=fps, macro_block_size=None)
    return len(frames), last_msg


def main() -> None:
    p = argparse.ArgumentParser(description="Render ~KFUPM-style (udem1 + desert) Duckietown MP4.")
    p.add_argument(
        "-o",
        "--output",
        default=os.path.join(_REPO, "recordings", "kfupm_style_udem1_arabian_20s.mp4"),
        help="Output MP4 path",
    )
    p.add_argument("--env-id", default=DEFAULT_ENV)
    p.add_argument("--duration", type=float, default=20.0, help="Target clip length in seconds")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--seed", type=int, default=2)
    p.add_argument("--omega-gain", type=float, default=5.5)
    p.add_argument("--ref-v", type=float, default=0.55)
    p.add_argument("--fd", type=float, default=0.28)
    args = p.parse_args()
    n, msg = record(
        args.output,
        env_id=args.env_id,
        duration_sec=args.duration,
        fps=args.fps,
        seed=args.seed,
        omega_gain=args.omega_gain,
        ref_v=args.ref_v,
        fd=args.fd,
    )
    print("wrote", args.output, "frames", n, "/", int(round(args.duration * args.fps)))
    if msg:
        print("last simulator msg:", msg[:200])


if __name__ == "__main__":
    main()
