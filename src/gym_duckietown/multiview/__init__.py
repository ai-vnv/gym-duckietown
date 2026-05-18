"""Composite multi-view rendering for gym-duckietown rollouts.

Public surface is small on purpose; pick the piece you need:

    from gym_duckietown.multiview import (
        SmoothedController,
        TrajectoryPanel,
        compose_frame,
        render_all_views,
        run_rollout,
    )

Submodules can also be imported individually if you only need part of the
pipeline (for example reusing ``TrajectoryPanel`` in a different layout).
"""
from .cameras import (
    render_all_views,
    render_ego_yaw,
    render_mode,
)
from .composer import compose_frame, draw_label
from .controller import SmoothedController
from .pipeline import RolloutConfig, run_rollout
from .trajectory import TrajectoryPanel

__all__ = [
    "RolloutConfig",
    "SmoothedController",
    "TrajectoryPanel",
    "compose_frame",
    "draw_label",
    "render_all_views",
    "render_ego_yaw",
    "render_mode",
    "run_rollout",
]
