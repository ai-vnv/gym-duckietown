"""Shared helpers for the PPO/SAC training scripts.

Builds the JISR3 env, applies the standard wrapper stack, picks a torch
device (MPS on Apple Silicon, falling back to CPU/CUDA otherwise), and
streams per-episode metrics to W&B via a small SB3 callback.
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np
import torch

import gymnasium as gym  # noqa: F401  (re-exported for callers)

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage

from gym_duckietown.rl import EpisodeInfoWrapper, JISR3Env, RewardClip


# ----------------------------------------------------------------------------
# Repo-root resolution (scripts/rl/* runs from anywhere)
# ----------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]

# Model checkpoints + tensorboard logs land on the external SSD by default so
# the system disk doesn't fill up over long training runs. Override with the
# RL_CHECKPOINT_ROOT env var (e.g. for CI or a teammate without the SSD).
_DEFAULT_SSD_CHECKPOINT_ROOT = Path("/Volumes/ssdvnv01/duckietown-jisr3/checkpoints")


def _resolve_checkpoint_root() -> Path:
    override = os.environ.get("RL_CHECKPOINT_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    if _DEFAULT_SSD_CHECKPOINT_ROOT.parent.exists():
        return _DEFAULT_SSD_CHECKPOINT_ROOT
    # SSD not mounted — fall back to a sibling of the repo so we don't
    # silently scatter writes into the repo tree.
    return REPO_ROOT / "checkpoints"


CHECKPOINT_ROOT = _resolve_checkpoint_root()


def _resolve_wandb_dir() -> Path:
    """Place wandb's local sync cache on the SSD alongside checkpoints."""
    override = os.environ.get("WANDB_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return CHECKPOINT_ROOT.parent / "wandb"


WANDB_DIR = _resolve_wandb_dir()


def setup_wandb_dir() -> None:
    """Point ``wandb.init`` at the SSD-resident cache via WANDB_DIR.

    Call before ``wandb.init``. Idempotent. Without this, wandb defaults to
    ``./wandb`` in the cwd — which lives on the system disk.
    """
    WANDB_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["WANDB_DIR"] = str(WANDB_DIR)


# ----------------------------------------------------------------------------
# Device selection
# ----------------------------------------------------------------------------


def pick_device(preferred: str = "mps") -> str:
    """Return the best available torch device string.

    Order: requested → MPS (Apple Silicon) → CUDA → CPU. Sets the MPS
    op-fallback env var so unsupported ops fall back to CPU rather than
    crashing — required for SAC's target-update path on torch 2.2.
    """
    if preferred == "cpu":
        return "cpu"
    if preferred in ("mps", "auto") and getattr(torch.backends, "mps", None) is not None:
        if torch.backends.mps.is_available():
            os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
            return "mps"
    if preferred in ("cuda", "auto") and torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ----------------------------------------------------------------------------
# Env factory + vec stack
# ----------------------------------------------------------------------------


def make_env(seed: int = 0, max_steps: int = 1500) -> Callable[[], gym.Env]:
    """Return a thunk that constructs a fully-wrapped JISR3 env.

    Wrapper order is load-bearing:
      JISR3Env → RewardClip → EpisodeInfoWrapper → Monitor

    Monitor must be the outermost wrapper so SB3 picks up
    ``info["episode"]`` with the clipped reward — that's what populates
    ``rollout/ep_rew_mean`` and ``rollout/ep_len_mean`` (100-episode rolling
    averages) in the W&B / tensorboard logs. EpisodeInfoWrapper sits inside
    Monitor so its return aggregates also reflect the clipped reward.
    """

    def _thunk() -> gym.Env:
        env = JISR3Env(max_steps=max_steps, seed=seed)
        env = RewardClip(env, low=-50.0, high=50.0)
        env = EpisodeInfoWrapper(env)
        env = Monitor(env, allow_early_resets=True)
        return env

    return _thunk


def build_vec_env(seed: int = 0, n_stack: int = 2, max_steps: int = 1500):
    """Single-env DummyVecEnv with frame-stacking + channel-first transpose.

    Pyglet's GL context does not survive macOS's spawn multiprocessing, so
    ``SubprocVecEnv`` is unsafe — we use ``DummyVecEnv`` exclusively.
    """
    venv = DummyVecEnv([make_env(seed=seed, max_steps=max_steps)])
    venv = VecFrameStack(venv, n_stack=n_stack, channels_order="last")
    venv = VecTransposeImage(venv)
    return venv


# ----------------------------------------------------------------------------
# Run state — for checkpoint resume
# ----------------------------------------------------------------------------


@dataclass
class RunState:
    algo: str
    run_id: str
    checkpoint_dir: Path
    started_at: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "algo": self.algo,
            "run_id": self.run_id,
            "checkpoint_dir": str(self.checkpoint_dir),
            "started_at": self.started_at,
        }


def init_run_dir(algo: str, run_id: str) -> RunState:
    """Create ``<CHECKPOINT_ROOT>/<algo>/<run_id>/`` and write a run.json."""
    ckpt_dir = CHECKPOINT_ROOT / algo / run_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    state = RunState(
        algo=algo, run_id=run_id, checkpoint_dir=ckpt_dir, started_at=time.time()
    )
    meta_path = ckpt_dir / "run.json"
    if not meta_path.exists():
        meta_path.write_text(json.dumps(state.to_dict(), indent=2))
    print(f"[{algo}] checkpoint dir: {ckpt_dir}")
    return state


def load_run_id(checkpoint_path: Path) -> str:
    """Read the run.json next to a checkpoint and return the W&B run id."""
    ckpt_dir = checkpoint_path.parent
    meta_path = ckpt_dir / "run.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"run.json missing in {ckpt_dir} — pass --run-id explicitly to resume."
        )
    return json.loads(meta_path.read_text())["run_id"]


# ----------------------------------------------------------------------------
# Seeding
# ----------------------------------------------------------------------------


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------------
# Episode-metrics → W&B callback
# ----------------------------------------------------------------------------


class EpisodeMetricsCallback(BaseCallback):
    """Forward ``info['episode_metrics']`` to W&B as scalar logs.

    Each metric appears under one of two groups in the dashboard:

    * ``ep/<name>``      — per-episode summary (return, length, lane, stop…)
    * ``reward/<part>``  — per-episode sum of each reward component (the
      keys ``reward/lane``, ``reward/speed``, ``reward/stop_ok`` …)

    Non-scalar values are dropped silently — no NaN noise, no NaN-only
    series cluttering the W&B side panel.
    """

    def __init__(self, ep_prefix: str = "ep", verbose: int = 0):
        super().__init__(verbose)
        self.ep_prefix = ep_prefix
        self._wandb = None
        self._summary_keys_seen: set = set()

    def _on_training_start(self) -> None:
        try:
            import wandb  # type: ignore

            self._wandb = wandb
            self._declare_metrics()
        except Exception:
            self._wandb = None

    def _declare_metrics(self) -> None:
        """Tell W&B how to group + summarise our custom scalars."""
        if self._wandb is None or self._wandb.run is None:
            return
        define = self._wandb.define_metric
        # All our custom logs index against global_step (= num_timesteps).
        define("global_step", hidden=True)
        define(f"{self.ep_prefix}/*", step_metric="global_step", summary="mean")
        define("reward/*", step_metric="global_step", summary="mean")

    def _on_step(self) -> bool:
        infos = self.locals.get("infos") or []
        for info in infos:
            metrics = info.get("episode_metrics") if isinstance(info, dict) else None
            if not metrics:
                continue
            payload: Dict[str, float] = {}
            for k, v in metrics.items():
                # Keys with a "/" stay as-is (e.g. "reward/lane"); plain keys
                # get the ep/ prefix.
                key = k if "/" in k else f"{self.ep_prefix}/{k}"
                scalar = _maybe_scalar(v)
                if scalar is None:
                    continue
                payload[key] = scalar
            if not payload:
                continue
            # Attach the env-step value so W&B's custom step metric
            # (declared in _declare_metrics) sees it.
            payload["global_step"] = float(self.num_timesteps)
            if self._wandb is not None and self._wandb.run is not None:
                # Don't pass step= here — when sync_tensorboard=True, W&B's
                # internal step counter is owned by TB; passing step= would
                # warn and be ignored. Use the custom step metric instead.
                self._wandb.log(payload)
            # Mirror to the SB3 logger so tensorboard + stdout summaries
            # surface the same numbers.
            for k, v in payload.items():
                if k != "global_step":
                    self.logger.record(k, v)
        return True


def _maybe_scalar(v: Any) -> Optional[float]:
    """Return a finite float for v, or None if v shouldn't be logged."""
    if isinstance(v, bool):
        return float(v)
    if isinstance(v, (int, float)):
        f = float(v)
        if f != f:  # NaN check
            return None
        return f
    return None


# ----------------------------------------------------------------------------
# Print helpers (used by smoke test)
# ----------------------------------------------------------------------------


def print_env_summary(venv, model) -> None:
    obs = venv.reset()
    print(f"[smoke] vec obs shape: {tuple(obs.shape)}")
    print(f"[smoke] action_space: {venv.action_space}")
    print(f"[smoke] model.device: {model.device}")
    print(f"[smoke] policy: {model.policy.__class__.__name__}")
    sys.stdout.flush()
