#!/usr/bin/env python3
"""Record short MP4s of Pure Pursuit failure modes (for pedagogy / falsification demos)."""
from __future__ import annotations

import os
import sys

import imageio.v2 as imageio
import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "learning/imitation/iil-dagger"))

import gym  # noqa: E402
import gym_duckietown  # noqa: E402
from teacher.pure_pursuit_policy import PurePursuitPolicy  # noqa: E402


def record(
    out_path: str,
    env_id: str,
    seed: int,
    omega_gain: float,
    ref_v: float = 0.7,
    fd: float = 0.24,
    max_steps: int = 400,
    fps: int = 15,
) -> tuple[int, str]:
    env = gym.make(env_id, disable_env_checker=True)
    env.seed(seed)
    obs = env.reset()
    pol = PurePursuitPolicy(env.unwrapped, ref_velocity=ref_v, following_distance=fd)
    frames: list[np.ndarray] = []
    last_msg = ""
    steps = 0
    done = False
    while steps < max_steps and not done:
        frames.append(np.asarray(obs).copy())
        raw = pol.predict(obs)
        a = np.array([float(raw[0]), float(raw[1]) * omega_gain], dtype=np.float32)
        obs, _r, done, info = env.step(a)
        last_msg = info.get("Simulator", {}).get("msg", "")
        steps += 1
    env.close()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    imageio.mimsave(out_path, frames, fps=fps, macro_block_size=None)
    return steps, last_msg


def main() -> None:
    out_dir = os.path.join(_REPO, "recordings")
    os.makedirs(out_dir, exist_ok=True)
    scenarios = [
        (
            os.path.join(out_dir, "failure_high_omega_gain.mp4"),
            "Duckietown-small_loop-v0",
            0,
            18.0,
            "High steering gain: Pure Pursuit overshoots / leaves drivable surface (invalid pose).",
        ),
        (
            os.path.join(out_dir, "failure_obstacles_map.mp4"),
            "Duckietown-loop_obstacles-v0",
            5,
            7.0,
            "Obstacle map: lane-following curve ignores off-road context; ends on floor tile (invalid pose).",
        ),
    ]
    for path, eid, seed, og, desc in scenarios:
        n, msg = record(path, eid, seed, og)
        print(desc)
        print("  wrote", path, "frames", n, "msg", (msg or "")[:80])


if __name__ == "__main__":
    main()
