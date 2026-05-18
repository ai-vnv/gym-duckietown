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
    CAMERA_MODES_EXTERNAL,
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


def render_ego_yaw(
    sim,
    yaw_offset_rad: float,
    extra_height: float = 0.0,
    pitch_deg_delta: float = 0.0,
) -> np.ndarray:
    """Render the ego camera with the heading temporarily yawed.

    Used to fake side cameras (yaw ``+pi/2`` left, ``-pi/2`` right) and the
    top-mounted rear camera (yaw ``pi`` + ``extra_height``). ``sim.cur_angle``,
    ``sim.cam_height``, and ``sim.cam_angle[0]`` (pitch) are restored on
    exit, including when the underlying render raises.

    Parameters
    ----------
    sim : gym_duckietown.simulator.Simulator
    yaw_offset_rad : float
        Radians to add to ``sim.cur_angle`` for this single render.
    extra_height : float
        Meters to add to ``sim.cam_height`` for this single render. 0 keeps
        the default eye-level cam.
    pitch_deg_delta : float
        Degrees to add to ``sim.cam_angle[0]`` (camera pitch around its own
        x axis). Positive tilts the camera *down*. Use a small negative
        value (e.g. -8) to tilt side cameras *up* so taller signage/banners
        come into frame.
    """
    saved_angle = float(sim.cur_angle)
    saved_height = float(sim.cam_height)
    saved_pitch = float(sim.cam_angle[0])
    try:
        sim.cur_angle = saved_angle + yaw_offset_rad
        if extra_height:
            sim.cam_height = saved_height + extra_height
        if pitch_deg_delta:
            sim.cam_angle[0] = saved_pitch + pitch_deg_delta
        return render_mode(sim, CAMERA_MODES_EGO)
    finally:
        sim.cur_angle = saved_angle
        sim.cam_height = saved_height
        sim.cam_angle[0] = saved_pitch


def render_vantage(sim, look_from, look_at, up=(0.0, 1.0, 0.0)) -> np.ndarray:
    """Render from a fixed world-space pose.

    The look-from / look-at / up vectors are pushed onto the sim as
    ``external_cam_from`` / ``external_cam_at`` / ``external_cam_up`` for
    the duration of the call, then restored. Lets us render a "stadium" or
    "roadside CCTV" view without any per-agent tracking.
    """
    saved = (
        getattr(sim, "external_cam_from", None),
        getattr(sim, "external_cam_at", None),
        getattr(sim, "external_cam_up", None),
    )
    try:
        sim.external_cam_from = tuple(look_from)
        sim.external_cam_at = tuple(look_at)
        sim.external_cam_up = tuple(up)
        return render_mode(sim, CAMERA_MODES_EXTERNAL)
    finally:
        sim.external_cam_from, sim.external_cam_at, sim.external_cam_up = saved


def render_all_views(
    sim,
    side_cam_height: float = 0.12,
    side_cam_pitch_deg: float = -8.0,
    vantage_from=None,
    vantage_at=None,
) -> Dict[str, np.ndarray]:
    """Render the standard 5-view set used by the composite pipeline.

    Keys: ``driver``, ``bev``, ``left``, ``right``, ``vantage``.

    The side cameras accept ``side_cam_height`` and ``side_cam_pitch_deg``
    so taller scene props (billboards, traffic lights on poles) come into
    frame. ``vantage_from`` and ``vantage_at`` set the fixed roadside
    camera; sensible defaults look at the scene from the SW corner.
    """
    if vantage_from is None:
        ts = float(sim.road_tile_size)
        # Closer, lower SW vantage: just inside the SW corner of the loop,
        # ~0.8 m above the road, looking at the map center.
        vantage_from = (0.30, 0.85, 0.30)
        vantage_at = (
            sim.grid_width * ts / 2.0,
            0.05,
            sim.grid_height * ts / 2.0,
        )
    elif vantage_at is None:
        ts = float(sim.road_tile_size)
        vantage_at = (
            sim.grid_width * ts / 2.0,
            0.05,
            sim.grid_height * ts / 2.0,
        )
    return {
        "driver": render_mode(sim, CAMERA_MODES_EGO),
        "bev": render_mode(sim, CAMERA_MODES_MAP_TOPDOWN),
        "left": render_ego_yaw(
            sim,
            +math.pi / 2,
            extra_height=side_cam_height,
            pitch_deg_delta=side_cam_pitch_deg,
        ),
        "right": render_ego_yaw(
            sim,
            -math.pi / 2,
            extra_height=side_cam_height,
            pitch_deg_delta=side_cam_pitch_deg,
        ),
        "vantage": render_vantage(sim, vantage_from, vantage_at),
    }
