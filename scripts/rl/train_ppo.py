"""Train PPO on the JISR3 multicamera env with W&B + checkpoint resume.

Examples:

    # Fresh run, 500k steps
    python scripts/rl/train_ppo.py --total-timesteps 500000

    # Quick smoke test (used by the plan's verification checklist)
    python scripts/rl/train_ppo.py --total-timesteps 5000 --n-steps 128 --smoke

    # Resume a previous run from a checkpoint (continues W&B run + step counter)
    python scripts/rl/train_ppo.py \\
        --resume checkpoints/ppo/<run_id>/model_175000_steps.zip \\
        --total-timesteps 500000
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Allow running as a script from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rl.common import (  # noqa: E402
    CHECKPOINT_ROOT,
    EpisodeMetricsCallback,
    build_vec_env,
    init_run_dir,
    load_run_id,
    pick_device,
    print_env_summary,
    set_global_seed,
    setup_wandb_dir,
)

import wandb  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback  # noqa: E402
from wandb.integration.sb3 import WandbCallback  # noqa: E402


ALGO = "ppo"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train PPO on JISR3 (multicamera)")
    p.add_argument("--total-timesteps", type=int, default=500_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="mps", choices=["mps", "cuda", "cpu", "auto"])
    p.add_argument("--wandb-project", default="duckietown-jisr3")
    p.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    p.add_argument("--resume", type=str, default=None, help="Path to a .zip checkpoint to resume from")
    p.add_argument("--run-id", type=str, default=None, help="W&B run id to resume (auto-detected from run.json if omitted)")
    p.add_argument("--save-freq", type=int, default=25_000)
    p.add_argument("--n-steps", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--max-episode-steps", type=int, default=1500)
    p.add_argument("--smoke", action="store_true", help="Print env+model summary before learn()")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    set_global_seed(args.seed)
    device = pick_device(args.device)
    setup_wandb_dir()

    # ---- W&B init (resume-aware) ---------------------------------------
    resume_kwargs = {}
    if args.resume:
        ckpt_path = Path(args.resume).resolve()
        if not ckpt_path.exists():
            raise SystemExit(f"resume checkpoint not found: {ckpt_path}")
        run_id = args.run_id or load_run_id(ckpt_path)
        resume_kwargs = {"id": run_id, "resume": "must"}

    run = wandb.init(
        project=args.wandb_project,
        mode=args.wandb_mode,
        sync_tensorboard=True,
        monitor_gym=False,
        save_code=False,
        config=vars(args) | {"algo": ALGO, "device": device},
        **resume_kwargs,
    )
    run_id = run.id

    state = init_run_dir(ALGO, run_id)
    tb_dir = state.checkpoint_dir / "tb"
    tb_dir.mkdir(parents=True, exist_ok=True)

    # ---- env + model ---------------------------------------------------
    venv = build_vec_env(seed=args.seed, n_stack=2, max_steps=args.max_episode_steps)

    if args.resume:
        model = PPO.load(
            args.resume,
            env=venv,
            device=device,
            tensorboard_log=str(tb_dir),
        )
        reset_num_timesteps = False
        print(f"[ppo] resumed from {args.resume} at step {model.num_timesteps}")
    else:
        # n_steps must divide evenly into the rollout the policy collects.
        # For a single env, batch_size <= n_steps.
        bs = min(args.batch_size, args.n_steps)
        model = PPO(
            policy="CnnPolicy",
            env=venv,
            device=device,
            tensorboard_log=str(tb_dir),
            n_steps=args.n_steps,
            batch_size=bs,
            n_epochs=10,
            learning_rate=2.5e-4,
            clip_range=0.2,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.01,
            vf_coef=0.5,
            seed=args.seed,
            verbose=1,
        )
        reset_num_timesteps = True

    if args.smoke:
        print_env_summary(venv, model)

    # ---- callbacks -----------------------------------------------------
    ckpt_cb = CheckpointCallback(
        save_freq=args.save_freq,
        save_path=str(state.checkpoint_dir),
        name_prefix="model",
    )
    wandb_cb = WandbCallback(verbose=2, model_save_path=None)
    metrics_cb = EpisodeMetricsCallback(ep_prefix="ep")
    callbacks = CallbackList([ckpt_cb, wandb_cb, metrics_cb])

    # ---- learn ---------------------------------------------------------
    # SB3 v2 treats `total_timesteps` as an "additional" budget when
    # reset_num_timesteps=False (it adds current num_timesteps internally).
    # We want --total-timesteps to mean the absolute target, so subtract
    # what's already been trained when resuming.
    remaining = (
        max(0, args.total_timesteps - model.num_timesteps)
        if not reset_num_timesteps
        else args.total_timesteps
    )
    if remaining == 0:
        print(f"[ppo] already at/past target ({model.num_timesteps} >= {args.total_timesteps}); skipping learn()")
    else:
        print(f"[ppo] training {remaining} more steps (target {args.total_timesteps}, current {model.num_timesteps})")
    try:
        if remaining > 0:
            model.learn(
                total_timesteps=remaining,
                callback=callbacks,
                reset_num_timesteps=reset_num_timesteps,
            )
    finally:
        final_path = state.checkpoint_dir / "final.zip"
        model.save(str(final_path))
        print(f"[ppo] saved {final_path}")
        try:
            wandb.finish()
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
