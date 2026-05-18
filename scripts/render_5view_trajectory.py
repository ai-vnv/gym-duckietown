#!/usr/bin/env python3
"""CLI wrapper around :mod:`gym_duckietown.multiview`.

Renders a 5-view + trajectory composite mp4 of a duckietown rollout.

You can either drive everything ad-hoc with flags, or pick a named
scenario from the registry::

    # list available scenarios
    .venv/bin/python scripts/render_5view_trajectory.py --list-scenarios

    # render the JISR3 scenario (60 s) — defaults come from the scenario
    .venv/bin/python scripts/render_5view_trajectory.py \\
        --scenario small_loop_jisr3 \\
        -o recordings/jisr3_60s.mp4

    # override duration / fps without leaving the scenario
    .venv/bin/python scripts/render_5view_trajectory.py \\
        --scenario small_loop_jisr3 --duration 30 --fps 20 \\
        -o recordings/jisr3_30s.mp4

    # ad-hoc: small_loop with just a custom floor logo
    .venv/bin/python scripts/render_5view_trajectory.py \\
        --duration 16 --fps 20 \\
        --logo assets/kfupm_logo.png --logo-size 0.5 \\
        -o recordings/5view_smallloop_kfupm_16s.mp4
"""
from __future__ import annotations

import argparse
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "learning/imitation/iil-dagger"))

# Candidate output roots in priority order. The first one whose parent
# exists is used as the default. Override per-run with -o.
_DEFAULT_OUTPUT_ROOTS = (
    "/Volumes/ssdvnv01/aivnv/recordings",  # external SSD when mounted
    os.path.join(_REPO, "recordings"),     # repo fallback
)


def _default_output_dir() -> str:
    """Return the first writable recordings dir from the candidate list."""
    for root in _DEFAULT_OUTPUT_ROOTS:
        parent = os.path.dirname(root)
        if os.path.isdir(parent):
            os.makedirs(root, exist_ok=True)
            return root
    # last resort: create repo/recordings
    fallback = os.path.join(_REPO, "recordings")
    os.makedirs(fallback, exist_ok=True)
    return fallback

import gym  # noqa: E402
import gym_duckietown  # noqa: E402 F401
from gym_duckietown.multiview import (  # noqa: E402
    RolloutConfig,
    SmoothedController,
    apply_scenario,
    build_rule_controller,
    get_scenario,
    list_scenarios,
    run_rollout,
    sync_traffic_light_visuals,
)
from teacher.pure_pursuit_policy import PurePursuitPolicy  # noqa: E402


def _print_scenarios() -> None:
    print("Available scenarios:")
    for name, desc in list_scenarios():
        print(f"  {name:24s}  {desc}")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--list-scenarios", action="store_true", help="Print registry and exit.")
    p.add_argument(
        "--scenario",
        default=None,
        help="Named scenario to load (see --list-scenarios). When set, scenario defaults "
        "override flag defaults; flags you pass explicitly still win.",
    )
    p.add_argument("--env-id", default=None)
    p.add_argument("-o", "--output", default=None)
    p.add_argument("--duration", type=float, default=None, help="Clip length in seconds.")
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--tile-width", type=int, default=320)
    p.add_argument("--tile-height", type=int, default=240)
    p.add_argument("--ref-velocity", type=float, default=None)
    p.add_argument("--following-distance", type=float, default=None)
    p.add_argument("--omega-gain", type=float, default=None)
    p.add_argument("--smooth-alpha", type=float, default=None)
    p.add_argument("--traj-window-sec", type=float, default=None)
    p.add_argument("--max-steps", type=int, default=100_000)
    p.add_argument("--bitrate", default="4M")
    # ad-hoc floor logo (overrides any scenario-supplied decals)
    p.add_argument("--logo", default=None)
    p.add_argument("--logo-x", type=float, default=None)
    p.add_argument("--logo-z", type=float, default=None)
    p.add_argument("--logo-size", type=float, default=0.5)
    p.add_argument("--logo-rotation", type=float, default=0.0)
    args = p.parse_args()

    if args.list_scenarios:
        _print_scenarios()
        return

    # Resolve scenario defaults (None means "fall back to scenario or library default")
    scenario = get_scenario(args.scenario) if args.scenario else None

    def _pick(flag, scen_attr, default):
        if flag is not None:
            return flag
        if scenario is not None:
            return getattr(scenario, scen_attr)
        return default

    env_id = _pick(args.env_id, "env_id", "Duckietown-small_loop-v0")
    seed = _pick(args.seed, "seed", 0)
    duration = _pick(args.duration, "duration_s", 16.0)
    ref_velocity = _pick(args.ref_velocity, "ref_velocity", 0.45)
    following_distance = _pick(args.following_distance, "following_distance", 0.25)
    omega_gain = _pick(args.omega_gain, "omega_gain", 1.0)
    smooth_alpha = _pick(args.smooth_alpha, "smooth_alpha", 0.35)
    traj_window_sec = _pick(args.traj_window_sec, "traj_window_sec", None)
    if args.output is None:
        out_dir = _default_output_dir()
        if scenario is not None:
            args.output = os.path.join(out_dir, f"{scenario.name}_{int(duration)}s.mp4")
        else:
            args.output = os.path.join(out_dir, "5view_trajectory.mp4")

    config = RolloutConfig(
        output=args.output,
        duration=duration,
        fps=args.fps,
        tile_width=args.tile_width,
        tile_height=args.tile_height,
        traj_window_sec=traj_window_sec,
        bitrate=args.bitrate,
    )

    env = gym.make(
        env_id,
        disable_env_checker=True,
        max_steps=max(args.max_steps, config.num_frames + 100),
    )
    env.seed(seed)
    env.reset()
    sim = env.unwrapped

    applied = None
    if scenario is not None:
        applied = apply_scenario(env, scenario)

    # ad-hoc floor logo: applied after scenario so it can stand alone too
    if args.logo:
        ts = float(sim.road_tile_size)
        cx = args.logo_x if args.logo_x is not None else sim.grid_width * ts / 2.0
        cz = args.logo_z if args.logo_z is not None else sim.grid_height * ts / 2.0
        sim.add_floor_decal(
            image_path=os.path.abspath(args.logo),
            position=(cx, cz),
            size=args.logo_size,
            rotation_deg=args.logo_rotation,
        )

    policy = PurePursuitPolicy(
        sim, ref_velocity=ref_velocity, following_distance=following_distance
    )
    base_controller = SmoothedController(policy, alpha=smooth_alpha, omega_gain=omega_gain)

    if applied and (applied["stop_signs"] or applied["traffic_lights"]):
        controller = build_rule_controller(
            base_controller,
            sim,
            applied["stop_signs"],
            applied["traffic_lights"],
            dt=1.0 / config.fps,
        )
    else:
        controller = base_controller

    pre_render = None
    if applied and applied["traffic_lights"]:
        traffic_lights = applied["traffic_lights"]
        state_bbs = applied["traffic_light_state_billboards"]

        def pre_render(step_idx, t, sim_):  # noqa: ARG001
            sync_traffic_light_visuals(traffic_lights, state_bbs, t)

    try:
        written = run_rollout(env, controller, config, pre_render=pre_render)
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
        "scenario=",
        scenario.name if scenario else "(none)",
    )


if __name__ == "__main__":
    main()
