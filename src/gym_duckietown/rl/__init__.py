"""RL training helpers for gym-duckietown.

This subpackage is strictly additive — it wraps the existing simulator and
multiview/scenarios machinery to expose an SB3-compatible env for the JISR3
scenario, plus thin observation wrappers. Training scripts live under
``scripts/rl/``.
"""
from .jisr3_env import JISR3Env, MultiViewObservation
from .wrappers import EpisodeInfoWrapper, ResizeObservation, RewardClip

__all__ = [
    "JISR3Env",
    "MultiViewObservation",
    "EpisodeInfoWrapper",
    "ResizeObservation",
    "RewardClip",
]
