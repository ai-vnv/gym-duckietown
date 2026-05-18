"""Per-frame camera-rig render helpers.

These wrap ``Simulator._render_img`` for common multi-view layouts. The yaw /
height tricks (``render_ego_yaw``) re-use the ego renderer to fake side
cameras and top-mounted rear cameras without adding new modes to the
simulator.

All helpers restore ``sim.cur_angle`` and ``sim.cam_height`` even on
exceptions so they are safe to call from inside a step loop.
"""
from __future__ import annotations

import math
from typing import Dict

import numpy as np

from gym_duckietown.simulator import (
    CAMERA_MODES_EGO,
    CAMERA_MODES_MAP_TOPDOWN,
)


def render_mode(sim, mode: str) -> np.ndarray:
    """Render one frame from the named camera rig.

    Parameters
    ----------
    sim : gym_duckietown.simulator.Simulator
        The unwrapped simulator (use ``env.unwrapped``).
    mode : str
        One of the values in :data:`gym_duckietown.simulator.CAMERA_MODES`.

    Returns
    -------
    np.ndarray
        HxWx3 uint8 RGB image of size (``sim.camera_height``, ``sim.camera_width``).
    """
    img = sim._render_img(
        sim.camera_width,
        sim.camera_height,
        sim.multi_fbo,
        sim.final_fbo,
        sim.img_array,
        top_down=False,
        segment=False,
        camera_mode=mode,
    )
    return np.ascontiguousarray(img)


def render_ego_yaw(sim, yaw_offset_rad: float, extra_height: float = 0.0) -> np.ndarray:
    """Render the ego camera with the heading temporarily yawed.

    Used to fake side cameras (yaw ``+pi/2`` left, ``-pi/2`` right) and the
    top-mounted rear camera (yaw ``pi`` + ``extra_height``). ``sim.cur_angle``
    and ``sim.cam_height`` are restored on exit, including when the underlying
    render raises.

    Parameters
    ----------
    sim : gym_duckietown.simulator.Simulator
    yaw_offset_rad : float
        Radians to add to ``sim.cur_angle`` for this single render.
    extra_height : float
        Meters to add to ``sim.cam_height`` for this single render. 0 keeps
        the default eye-level cam.
    """
    saved_angle = float(sim.cur_angle)
    saved_height = float(sim.cam_height)
    try:
        sim.cur_angle = saved_angle + yaw_offset_rad
        if extra_height:
            sim.cam_height = saved_height + extra_height
        return render_mode(sim, CAMERA_MODES_EGO)
    finally:
        sim.cur_angle = saved_angle
        sim.cam_height = saved_height


def render_all_views(sim, top_rear_height: float = 0.30) -> Dict[str, np.ndarray]:
    """Render the standard 5-view set used by the composite pipeline.

    Keys: ``driver``, ``bev``, ``left``, ``right``, ``top_rear``.
    """
    return {
        "driver": render_mode(sim, CAMERA_MODES_EGO),
        "bev": render_mode(sim, CAMERA_MODES_MAP_TOPDOWN),
        "left": render_ego_yaw(sim, +math.pi / 2),
        "right": render_ego_yaw(sim, -math.pi / 2),
        "top_rear": render_ego_yaw(sim, math.pi, extra_height=top_rear_height),
    }
