"""Evaluate a trained PPO/SAC checkpoint on JISR3.

Loads one or both checkpoints, rolls N deterministic episodes per algo, logs
per-episode and aggregate metrics to W&B as a table, and writes an mp4 of
the multicamera composite obs for visual inspection.

Example:

    python scripts/rl/evaluate.py \\
        --ppo-ckpt checkpoints/ppo/<rid>/final.zip \\
        --sac-ckpt checkpoints/sac/<rid>/final.zip \\
        --episodes 20
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rl.common import build_vec_env, pick_device, setup_wandb_dir  # noqa: E402

import wandb  # noqa: E402
from stable_baselines3 import PPO, SAC  # noqa: E402


ALGOS = {"ppo": PPO, "sac": SAC}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate PPO/SAC on JISR3")
    p.add_argument("--ppo-ckpt", type=str, default=None)
    p.add_argument("--sac-ckpt", type=str, default=None)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--max-episode-steps", type=int, default=1500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="mps", choices=["mps", "cuda", "cpu", "auto"])
    p.add_argument("--wandb-project", default="duckietown-jisr3")
    p.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    p.add_argument("--video", action="store_true", help="Record an mp4 of one episode per algo")
    p.add_argument("--no-video", action="store_true", help="Disable video recording")
    p.add_argument("--video-fps", type=int, default=20)
    p.add_argument("--out-dir", type=str, default="recordings/rl_eval")
    return p.parse_args()


# ----------------------------------------------------------------------------
# Episode rollout
# ----------------------------------------------------------------------------


def _venv_obs(venv) -> np.ndarray:
    obs = venv.reset()
    return obs


def rollout_episode(
    model,
    venv,
    record: bool,
    max_steps: int,
) -> Tuple[Dict[str, float], List[np.ndarray]]:
    """Run one episode; return aggregate metrics + optional frame list."""
    obs = _venv_obs(venv)
    done = False
    ep_return = 0.0
    ep_len = 0
    lane_dist_sum = 0.0
    lane_dist_n = 0
    speed_sum = 0.0
    stop_satisfied = 0
    stop_violations = 0
    stop_approaches = 0
    prev_triggered: List[bool] = []
    collision = False
    done_code = "unknown"
    tile_visits = set()
    yaw_unwrap = 0.0
    last_angle: Optional[float] = None
    start_pos: Optional[Tuple[float, float]] = None
    last_pos: Optional[Tuple[float, float]] = None
    frames: List[np.ndarray] = []

    while not done and ep_len < max_steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, dones, infos = venv.step(action)
        done = bool(dones[0])
        info = infos[0]
        ep_return += float(reward[0])
        ep_len += 1

        if record:
            # Grab the env's render — pulling from venv requires unwrapping.
            try:
                env0 = venv.envs[0] if hasattr(venv, "envs") else venv.venv.envs[0]
                frame = env0.render(mode="rgb_array")
                if frame is not None:
                    frames.append(np.ascontiguousarray(frame))
            except Exception:
                pass

        sim_info = info.get("Simulator", {})
        lp = sim_info.get("lane_position")
        if lp is not None:
            lane_dist_sum += abs(float(lp["dist"]))
            lane_dist_n += 1
        speed_val = sim_info.get("robot_speed")
        if speed_val is not None:
            speed_sum += float(speed_val)

        states = sim_info.get("stop_sign_states", [])
        if len(prev_triggered) < len(states):
            prev_triggered += [False] * (len(states) - len(prev_triggered))
        for i, st in enumerate(states):
            tr = bool(st.get("triggered", False))
            if tr and not prev_triggered[i]:
                stop_approaches += 1
            prev_triggered[i] = tr
        parts = sim_info.get("reward_parts", {})
        if parts.get("r_stop_ok", 0.0) > 0.0:
            stop_satisfied += 1
        if parts.get("r_stop_bad", 0.0) < 0.0:
            stop_violations += 1

        tc = sim_info.get("tile_coords")
        if tc:
            tile_visits.add((int(tc[0]), int(tc[1])))

        pos = sim_info.get("cur_pos")
        ang = sim_info.get("cur_angle")
        if pos is not None and ang is not None:
            pos_xz = (float(pos[0]), float(pos[2]))
            if start_pos is None:
                start_pos = pos_xz
            last_pos = pos_xz
            if last_angle is not None:
                delta = float(ang) - last_angle
                # unwrap to nearest 2π
                while delta > math.pi:
                    delta -= 2 * math.pi
                while delta < -math.pi:
                    delta += 2 * math.pi
                yaw_unwrap += delta
            last_angle = float(ang)

        if done:
            collision = sim_info.get("done_code") == "invalid-pose"
            done_code = sim_info.get("done_code", "unknown")

    compliance = (
        stop_satisfied / stop_approaches if stop_approaches > 0 else float("nan")
    )
    mean_lane = lane_dist_sum / lane_dist_n if lane_dist_n else float("nan")
    mean_speed = speed_sum / max(1, ep_len)

    # Lap heuristic: cumulative yaw turn ≥ 2π and end close to start.
    lap_done = False
    if start_pos is not None and last_pos is not None:
        end_close = (
            math.hypot(last_pos[0] - start_pos[0], last_pos[1] - start_pos[1]) < 0.5
        )
        lap_done = abs(yaw_unwrap) > 2 * math.pi and end_close

    metrics = {
        "return": ep_return,
        "length": float(ep_len),
        "mean_lane_dist": mean_lane,
        "mean_speed": mean_speed,
        "stop_approaches": float(stop_approaches),
        "stop_satisfied": float(stop_satisfied),
        "stop_violations": float(stop_violations),
        "stop_compliance_rate": compliance,
        "collision": float(collision),
        "lap_completed": float(lap_done),
        "tiles_visited": float(len(tile_visits)),
        "yaw_unwrap_rad": float(yaw_unwrap),
        "done_code": done_code,
    }
    return metrics, frames


def evaluate(algo: str, ckpt: str, args: argparse.Namespace, out_dir: Path) -> Dict[str, float]:
    device = pick_device(args.device)
    venv = build_vec_env(seed=args.seed, n_stack=2, max_steps=args.max_episode_steps)
    Cls = ALGOS[algo]
    model = Cls.load(ckpt, env=venv, device=device)
    print(f"[eval:{algo}] loaded {ckpt} on {device}")

    per_ep: List[Dict[str, float]] = []
    table = wandb.Table(
        columns=[
            "algo", "episode", "return", "length", "mean_lane_dist",
            "mean_speed", "stop_approaches", "stop_satisfied",
            "stop_compliance_rate", "stop_violations", "collision",
            "lap_completed", "tiles_visited", "done_code",
        ]
    )

    do_video = args.video and not args.no_video
    video_path: Optional[Path] = None

    for ep in range(args.episodes):
        record = do_video and ep == 0
        metrics, frames = rollout_episode(
            model, venv, record=record, max_steps=args.max_episode_steps
        )
        per_ep.append(metrics)
        table.add_data(
            algo,
            ep,
            metrics["return"],
            metrics["length"],
            metrics["mean_lane_dist"],
            metrics["mean_speed"],
            metrics["stop_approaches"],
            metrics["stop_satisfied"],
            metrics["stop_compliance_rate"],
            metrics["stop_violations"],
            metrics["collision"],
            metrics["lap_completed"],
            metrics["tiles_visited"],
            metrics["done_code"],
        )
        print(
            f"[eval:{algo}] ep {ep:>2}: return={metrics['return']:.2f} "
            f"len={int(metrics['length'])} compliance={metrics['stop_compliance_rate']} "
            f"lane={metrics['mean_lane_dist']:.3f} speed={metrics['mean_speed']:.3f} "
            f"done={metrics['done_code']}"
        )

        if record and frames:
            try:
                import imageio

                out_dir.mkdir(parents=True, exist_ok=True)
                video_path = out_dir / f"{algo}_ep0.mp4"
                imageio.mimwrite(str(video_path), frames, fps=args.video_fps)
                print(f"[eval:{algo}] wrote video {video_path}")
            except Exception as exc:  # pragma: no cover
                print(f"[eval:{algo}] video write failed: {exc}")

    # Aggregate (ignore NaNs for compliance / lane).
    def _agg(key: str) -> float:
        vals = [m[key] for m in per_ep if isinstance(m[key], (int, float)) and not (isinstance(m[key], float) and math.isnan(m[key]))]
        return float(np.mean(vals)) if vals else float("nan")

    agg = {
        "mean_return": _agg("return"),
        "mean_length": _agg("length"),
        "mean_lane_dist": _agg("mean_lane_dist"),
        "mean_speed": _agg("mean_speed"),
        "mean_compliance": _agg("stop_compliance_rate"),
        "collision_rate": _agg("collision"),
        "lap_completion_rate": _agg("lap_completed"),
        "mean_tiles_visited": _agg("tiles_visited"),
    }
    wandb.log({f"eval/{algo}/episodes": table})
    wandb.log({f"eval/{algo}/{k}": v for k, v in agg.items()})
    if video_path is not None and video_path.exists():
        try:
            wandb.log({f"eval/{algo}/video": wandb.Video(str(video_path), fps=args.video_fps)})
        except Exception:
            pass

    venv.close()
    return agg


def main() -> int:
    args = parse_args()
    if not args.ppo_ckpt and not args.sac_ckpt:
        raise SystemExit("provide at least --ppo-ckpt or --sac-ckpt")

    out_dir = Path(args.out_dir).resolve()
    setup_wandb_dir()

    wandb.init(
        project=args.wandb_project,
        mode=args.wandb_mode,
        job_type="evaluate",
        config=vars(args),
    )

    results: Dict[str, Dict[str, float]] = {}
    if args.ppo_ckpt:
        results["ppo"] = evaluate("ppo", args.ppo_ckpt, args, out_dir)
    if args.sac_ckpt:
        results["sac"] = evaluate("sac", args.sac_ckpt, args, out_dir)

    # Side-by-side comparison table.
    if len(results) > 1:
        keys = sorted(set().union(*[r.keys() for r in results.values()]))
        comp = wandb.Table(columns=["metric"] + list(results.keys()))
        for k in keys:
            comp.add_data(k, *[results[a].get(k, float("nan")) for a in results])
        wandb.log({"eval/comparison": comp})

    print("\n=== Evaluation summary ===")
    for algo, agg in results.items():
        print(f"[{algo}]")
        for k, v in agg.items():
            print(f"  {k}: {v:.4f}")

    wandb.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
