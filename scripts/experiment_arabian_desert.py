#!/usr/bin/env python3
"""Run Pure Pursuit on Arabian-desert–styled envs (warm sky + sand ground tint)."""
from __future__ import annotations

import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "learning/imitation/iil-dagger"))

import numpy as np  # noqa: E402
import gym  # noqa: E402
import gym_duckietown  # noqa: E402
from teacher.pure_pursuit_policy import PurePursuitPolicy  # noqa: E402


def run_rollout(
    env_id: str,
    seed: int,
    omega_gain: float,
    ref_v: float = 0.7,
    fd: float = 0.24,
    max_steps: int = 400,
) -> tuple[int, str]:
    env = gym.make(env_id, disable_env_checker=True)
    env.seed(seed)
    obs = env.reset()
    pol = PurePursuitPolicy(env.unwrapped, ref_velocity=ref_v, following_distance=fd)
    last_msg = ""
    steps = 0
    done = False
    while steps < max_steps and not done:
        raw = pol.predict(obs)
        a = np.array(
            [float(raw[0]), float(raw[1]) * omega_gain],
            dtype=np.float32,
        )
        obs, _r, done, info = env.step(a)
        last_msg = info.get("Simulator", {}).get("msg", "")
        steps += 1
    env.close()
    return steps, last_msg


if __name__ == "__main__":
    scenarios = [
        ("Duckietown-small_loop_arabian-v0", 0, 18.0, "small_loop + desert sky/ground"),
        ("Duckietown-loop_obstacles_arabian-v0", 5, 7.0, "loop_obstacles + desert sky/ground"),
    ]
    for eid, seed, og, label in scenarios:
        n, msg = run_rollout(eid, seed, og)
        print(label)
        print(" ", eid, "steps", n, "msg", (msg or "")[:120])
