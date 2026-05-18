"""Train SAC on the JISR3 multicamera env with W&B + checkpoint resume.

Examples:

    python scripts/rl/train_sac.py --total-timesteps 500000

    python scripts/rl/train_sac.py --total-timesteps 5000 --buffer-size 5000 --learning-starts 256 --smoke

    python scripts/rl/train_sac.py \\
        --resume checkpoints/sac/<run_id>/model_175000_steps.zip \\
        --total-timesteps 500000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rl.common import (  # noqa: E402
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
from stable_baselines3 import SAC  # noqa: E402
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback  # noqa: E402
from wandb.integration.sb3 import WandbCallback  # noqa: E402


ALGO = "sac"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train SAC on JISR3 (multicamera)")
    p.add_argument("--total-timesteps", type=int, default=500_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="mps", choices=["mps", "cuda", "cpu", "auto"])
    p.add_argument("--wandb-project", default="duckietown-jisr3")
    p.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--run-id", type=str, default=None)
    p.add_argument("--save-freq", type=int, default=25_000)
    p.add_argument("--buffer-size", type=int, default=100_000)
    p.add_argument("--learning-starts", type=int, default=5_000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--max-episode-steps", type=int, default=1500)
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    set_global_seed(args.seed)
    device = pick_device(args.device)
    setup_wandb_dir()

    # ---- W&B init ------------------------------------------------------
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
        model = SAC.load(
            args.resume,
            env=venv,
            device=device,
            tensorboard_log=str(tb_dir),
        )
        reset_num_timesteps = False
        # Try to restore the replay buffer if present.
        replay_path = state.checkpoint_dir / "replay.pkl"
        if replay_path.exists():
            try:
                model.load_replay_buffer(str(replay_path))
                print(f"[sac] loaded replay buffer ({len(model.replay_buffer)} transitions)")
            except Exception as exc:  # pragma: no cover
                print(f"[sac] failed to load replay buffer: {exc}")
        print(f"[sac] resumed from {args.resume} at step {model.num_timesteps}")
    else:
        model = SAC(
            policy="CnnPolicy",
            env=venv,
            device=device,
            tensorboard_log=str(tb_dir),
            buffer_size=args.buffer_size,
            learning_starts=args.learning_starts,
            batch_size=args.batch_size,
            learning_rate=3e-4,
            train_freq=1,
            gradient_steps=1,
            tau=0.005,
            gamma=0.99,
            ent_coef="auto",
            target_entropy="auto",
            seed=args.seed,
            verbose=1,
        )
        reset_num_timesteps = True

    if args.smoke:
        print_env_summary(venv, model)

    # ---- callbacks -----------------------------------------------------
    # Intermediate checkpoints save weights only — the replay buffer (~4 GB)
    # is dumped once at exit via the finally block below. That keeps SSD
    # usage bounded for a 500k-step run (20 saves × 44 MB) instead of the
    # 80+ GB it would be with save_replay_buffer=True at every interval.
    ckpt_cb = CheckpointCallback(
        save_freq=args.save_freq,
        save_path=str(state.checkpoint_dir),
        name_prefix="model",
        save_replay_buffer=False,
    )
    wandb_cb = WandbCallback(verbose=2, model_save_path=None)
    metrics_cb = EpisodeMetricsCallback(ep_prefix="ep")
    callbacks = CallbackList([ckpt_cb, wandb_cb, metrics_cb])

    # ---- learn ---------------------------------------------------------
    # See train_ppo.py for the rationale: treat --total-timesteps as absolute.
    remaining = (
        max(0, args.total_timesteps - model.num_timesteps)
        if not reset_num_timesteps
        else args.total_timesteps
    )
    if remaining == 0:
        print(f"[sac] already at/past target ({model.num_timesteps} >= {args.total_timesteps}); skipping learn()")
    else:
        print(f"[sac] training {remaining} more steps (target {args.total_timesteps}, current {model.num_timesteps})")
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
        try:
            model.save_replay_buffer(str(state.checkpoint_dir / "replay.pkl"))
        except Exception as exc:  # pragma: no cover
            print(f"[sac] replay buffer save failed: {exc}")
        print(f"[sac] saved {final_path}")
        try:
            wandb.finish()
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
