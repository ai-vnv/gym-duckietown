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
from .scenarios import (
    Scenario,
    apply_scenario,
    build_rule_controller,
    get_scenario,
    list_scenarios,
    register_scenario,
    sync_traffic_light_visuals,
)
from .trajectory import TrajectoryPanel

__all__ = [
    "RolloutConfig",
    "Scenario",
    "SmoothedController",
    "TrajectoryPanel",
    "apply_scenario",
    "build_rule_controller",
    "compose_frame",
    "draw_label",
    "get_scenario",
    "list_scenarios",
    "register_scenario",
    "render_all_views",
    "render_ego_yaw",
    "render_mode",
    "run_rollout",
    "sync_traffic_light_visuals",
]
