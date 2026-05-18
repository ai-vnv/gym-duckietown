"""Named scenarios: bundles of map + props + rules + controller defaults.

A :class:`Scenario` is a self-contained description of what the renderer
should set up before rolling out. Scenarios are factories so they can
compute positions from the loaded map at apply time (e.g. "logo at the
geometric center of the loaded map").

The registry is just a module-level dict; add entries at the bottom of
this file or call :func:`register_scenario` from anywhere.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import math

import numpy as np

from gym_duckietown.billboards import Billboard
from gym_duckietown.decals import FloorDecal
from gym_duckietown.traffic import StopSign, TrafficLight, RuleAwareController, make_sim_pose_fn


_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def _assets_dir() -> str:
    return os.path.join(_REPO, "assets")


# ----------------------------------------------------------------------------
# Scenario dataclass
# ----------------------------------------------------------------------------


@dataclass
class Scenario:
    """A named, self-contained simulation setup.

    The lists are *factories* (functions taking the sim and returning a
    list) so positions can depend on the loaded map. Factories are evaluated
    once, right after the env is reset.
    """

    name: str
    description: str
    env_id: str = "Duckietown-small_loop-v0"
    seed: int = 0
    duration_s: float = 30.0

    # Controller params
    ref_velocity: float = 0.45
    following_distance: float = 0.25
    omega_gain: float = 1.0
    smooth_alpha: float = 0.35

    # Per-frame trajectory window. None = full history (with per-segment
    # fading so older laps dim into the background); set a finite value to
    # restrict the displayed trace to the last N seconds.
    traj_window_sec: Optional[float] = None

    # Factories — each takes the unwrapped sim and returns its list. Lets
    # entries position themselves relative to the loaded map.
    floor_decals: Callable[[object], List[FloorDecal]] = field(default=lambda sim: [])
    billboards: Callable[[object], List[Billboard]] = field(default=lambda sim: [])
    stop_signs: Callable[[object], List[StopSign]] = field(default=lambda sim: [])
    traffic_lights: Callable[[object], List[TrafficLight]] = field(default=lambda sim: [])

    # Optional traffic-light visual: when set, the renderer manages a 3-
    # billboard swap per traffic light so the on-screen color matches the
    # rule's logical state. Each entry maps a TrafficLight index to
    # ``(position_xz, y_base, width, height, rotation_deg)`` for visualisation.
    traffic_light_visuals: Callable[[object], List[Tuple[float, float, float, float, float, float]]] = field(
        default=lambda sim: []
    )

    # 3D scenery (buildings, houses, trees) — each entry is a dict with
    # ``kind`` (a mesh name from duckietown-world), ``position`` (x, z in
    # meters), ``rotation_deg`` and ``scale``.
    scenery: Callable[[object], List[dict]] = field(default=lambda sim: [])


# ----------------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------------


_REGISTRY: Dict[str, Scenario] = {}


def register_scenario(s: Scenario) -> Scenario:
    if s.name in _REGISTRY:
        raise ValueError(f"scenario {s.name!r} already registered")
    _REGISTRY[s.name] = s
    return s


def get_scenario(name: str) -> Scenario:
    if name not in _REGISTRY:
        raise KeyError(
            f"unknown scenario {name!r}; known: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name]


def list_scenarios() -> List[Tuple[str, str]]:
    return [(s.name, s.description) for s in _REGISTRY.values()]


# ----------------------------------------------------------------------------
# Application helpers
# ----------------------------------------------------------------------------


# Heights of mounted signage above the ground (meters). Tuned so the road
# user can read them from the side cameras with the new pitch.
SIGN_Y_BASE = 0.30
SIGN_HEIGHT = 0.22
TL_Y_BASE = 0.30
TL_WIDTH = 0.16
TL_HEIGHT = 0.42
POLE_WIDTH = 0.025  # meters — narrow billboard so it looks like a pipe


def _add_scene_object(sim, kind: str, position, rotation_deg: float = 0.0, scale: float = 0.3):
    """Spawn a 3D mesh object (building / house / tree / ...) at runtime.

    Uses the same code path the simulator uses to load objects from a map
    YAML, so the new object joins ``sim.objects`` and is rendered by every
    camera view automatically.
    """
    from gym_duckietown.objmesh import get_mesh
    from gym_duckietown.objects import WorldObj

    mesh = get_mesh(kind)
    obj_desc = {
        "kind": kind,
        "mesh": mesh,
        "pos": np.array([float(position[0]), 0.0, float(position[1])]),
        "angle": math.radians(float(rotation_deg)),
        "scale": float(scale),
        "optional": True,
        "static": True,
    }
    obj = WorldObj(obj_desc, getattr(sim, "domain_rand", False), 1.0)
    sim.objects.append(obj)
    return obj


def _add_pole_pair(sim, position, top_y: float, rotation_deg: float) -> None:
    """Place a cross of two thin pole quads (visible from any angle).

    Pole extends from y=0 to ``top_y``. Two quads rotated 90° apart fake an
    omnidirectional cylinder cheaply.
    """
    pole_png = os.path.join(_assets_dir(), "pole.png")
    if not os.path.isfile(pole_png):
        return  # pole asset not present; skip silently
    for offset in (0.0, 90.0):
        sim.add_billboard(
            image_path=pole_png,
            position=position,
            width=POLE_WIDTH,
            height=top_y,
            y_base=0.0,
            rotation_deg=rotation_deg + offset,
            single_sided=False,  # pole texture is symmetric; visible from any side
        )


def apply_scenario(env, scenario: Scenario):
    """Apply ``scenario`` to a *just-reset* env. Returns a dict with the
    instantiated rule objects so a caller can wire up a
    :class:`RuleAwareController` and per-frame traffic-light state swaps.
    """
    sim = env.unwrapped

    for d in scenario.floor_decals(sim):
        sim.add_floor_decal(
            image_path=d.image_path,
            position=d.position,
            size=d.size,
            rotation_deg=d.rotation_deg,
            lift=d.lift,
        )

    # Billboards (banners, advertisements) come with their own poles too.
    for b in scenario.billboards(sim):
        sim.add_billboard(
            image_path=b.image_path,
            position=b.position,
            width=b.width,
            height=b.height,
            y_base=b.y_base,
            rotation_deg=b.rotation_deg,
            enabled=b.enabled,
        )
        if b.y_base > 0.02:
            _add_pole_pair(sim, b.position, top_y=b.y_base, rotation_deg=b.rotation_deg)

    # 3D scenery (houses, buildings, trees, etc.). Loaded before signage so
    # any later add_billboard / pole calls render on top of the mesh objects.
    for spec in scenario.scenery(sim):
        _add_scene_object(
            sim,
            kind=spec["kind"],
            position=spec["position"],
            rotation_deg=spec.get("rotation_deg", 0.0),
            scale=spec.get("scale", 0.3),
        )

    stop_signs = scenario.stop_signs(sim)
    traffic_lights = scenario.traffic_lights(sim)

    # Stop signs: visible billboard + pole pair.
    stop_png = os.path.join(_assets_dir(), "stop_sign.png")
    if os.path.isfile(stop_png):
        for ss in stop_signs:
            sim.add_billboard(
                image_path=stop_png,
                position=ss.position,
                width=SIGN_HEIGHT,  # octagon is square
                height=SIGN_HEIGHT,
                y_base=SIGN_Y_BASE,
                rotation_deg=0.0,
            )
            # second sign back-to-back so the agent reads it from either approach
            sim.add_billboard(
                image_path=stop_png,
                position=ss.position,
                width=SIGN_HEIGHT,
                height=SIGN_HEIGHT,
                y_base=SIGN_Y_BASE,
                rotation_deg=180.0,
            )
            _add_pole_pair(sim, ss.position, top_y=SIGN_Y_BASE, rotation_deg=0.0)

    # Traffic lights: three swap-billboards at each visual position +
    # a pole pair beneath the housing. The pipeline toggles which colored
    # billboard is enabled each frame.
    tl_visuals = scenario.traffic_light_visuals(sim)
    tl_state_billboards = []  # list of dict(red, yellow, green) per light
    for idx, tl in enumerate(traffic_lights):
        if idx < len(tl_visuals):
            x, z, y_base, w, h, rot = tl_visuals[idx]
        else:
            x, z = tl.position
            y_base, w, h, rot = TL_Y_BASE, TL_WIDTH, TL_HEIGHT, 0.0
        states: Dict[str, Billboard] = {}
        for color in ("red", "yellow", "green"):
            path = os.path.join(_assets_dir(), f"traffic_light_{color}.png")
            if not os.path.isfile(path):
                continue
            billboard = sim.add_billboard(
                image_path=path,
                position=(x, z),
                width=w,
                height=h,
                y_base=y_base,
                rotation_deg=rot,
                enabled=False,
            )
            states[color] = billboard
        tl_state_billboards.append(states)
        # pole(s) beneath the housing
        _add_pole_pair(sim, (x, z), top_y=y_base, rotation_deg=rot)

    return {
        "stop_signs": stop_signs,
        "traffic_lights": traffic_lights,
        "traffic_light_state_billboards": tl_state_billboards,
    }


def sync_traffic_light_visuals(
    traffic_lights: List[TrafficLight],
    state_billboards: List[Dict[str, Billboard]],
    t: float,
) -> None:
    """Toggle the appropriate billboard per traffic light for time ``t``."""
    for tl, states in zip(traffic_lights, state_billboards):
        color = tl.color_at(t)
        for c, bb in states.items():
            bb.enabled = c == color


def build_rule_controller(
    base_controller: Callable,
    sim,
    stop_signs: List[StopSign],
    traffic_lights: List[TrafficLight],
    dt: float,
) -> RuleAwareController:
    return RuleAwareController(
        base=base_controller,
        agent_pose_fn=make_sim_pose_fn(sim),
        stop_signs=stop_signs,
        traffic_lights=traffic_lights,
        dt=dt,
    )


# ============================================================================
# Built-in scenarios
# ============================================================================


def _center_xz(sim) -> Tuple[float, float]:
    ts = float(sim.road_tile_size)
    return (sim.grid_width * ts / 2.0, sim.grid_height * ts / 2.0)


# --- bare ---------------------------------------------------------------------

register_scenario(
    Scenario(
        name="small_loop_bare",
        description="small_loop with no props or rules (reference rollout).",
    )
)


# --- KFUPM floor logo only ----------------------------------------------------


def _kfupm_floor_factory(sim) -> List[FloorDecal]:
    return [
        FloorDecal(
            image_path=os.path.join(_assets_dir(), "kfupm_logo.png"),
            position=_center_xz(sim),
            size=0.5,
        )
    ]


register_scenario(
    Scenario(
        name="small_loop_kfupm",
        description="KFUPM floor decal at map center; no traffic rules.",
        floor_decals=_kfupm_floor_factory,
    )
)


# --- JISR3 headline scenario --------------------------------------------------


def _jisr3_billboards(sim) -> List[Billboard]:
    """JISR3 banner on the OUTER shoulder of the south road, facing the road.

    Placed south of the map (outside the loop), facing north toward the
    south straight. The bot driving past on the south road reads it
    head-on through its right side camera; corner approaches catch it
    in the driver / fixed-vantage views.
    """
    img = os.path.join(_assets_dir(), "billboard_jisr3.png")
    ts = float(sim.road_tile_size)
    cx = sim.grid_width * ts / 2.0
    # Just outside the south map edge (z > grid_height * ts), pole on the
    # grass beyond the road, sign facing -Z (north → road).
    cz = sim.grid_height * ts + 0.20
    width = 1.40
    height = 0.55
    y_base = 0.15
    return [
        Billboard(
            image_path=img,
            position=(cx, cz),
            width=width,
            height=height,
            y_base=y_base,
            rotation_deg=180.0,  # face -Z (north) → toward the south road
        ),
    ]


def _jisr3_stop_signs(sim) -> List[StopSign]:
    """Single stop sign on the inner shoulder of the east straight.

    The bot travels CCW → heads south on the east road; ``get_right_vec``
    of that heading is -X (toward map center), so the right shoulder is
    just west of the east-road inner edge. Position is at the *middle*
    of the east-road tile (mid-straight) — earlier code mis-computed
    this for a 3×3 small_loop and placed the sign at the north entry of
    the straight, which fired the rule too early and stacked with the
    traffic light → two consecutive stops.
    """
    ts = float(sim.road_tile_size)
    east_inner_edge = (sim.grid_width - 1) * ts        # west boundary of east-road tiles
    sign_x = east_inner_edge - 0.10                    # 10 cm on the inner grass shoulder
    # East road occupies tile-row floor(grid_height/2); centre z is mid-tile.
    east_row_j = sim.grid_height // 2
    sign_z = (east_row_j + 0.5) * ts                   # middle of the east-road tile
    return [
        StopSign(
            position=(sign_x, sign_z),
            # The bot's lateral position drifts ~1-2 cm between laps; with
            # the sign 0.39 m perpendicular from the centerline, a tight
            # 0.45 m radius grazes the boundary on later laps and the bot
            # escapes before the dwell completes. 0.60 m gives a wide
            # enough zone (>2 s of bot-traversal) for every lap.
            trigger_radius=0.60,
            required_stop_time_s=3.0,
            # The duckiebot's actuator has a ~0.15 s delay, so even when the
            # rule commands action=(0,0) the wheel velocity decays over a
            # few frames. Residual sim.speed sits ~0.06 m/s. We need the
            # threshold above that residual or the dwell clock never starts.
            stop_speed_threshold=0.10,
        ),
    ]


def _jisr3_traffic_lights(sim) -> List[TrafficLight]:
    """One traffic light on the south straight (different road from the
    stop sign so the agent never stops twice back-to-back).

    Green-heavy cycle: most laps pass straight through; an occasional
    red pulse demonstrates the rule without breaking flow every lap.
    """
    ts = float(sim.road_tile_size)
    south_row_j = sim.grid_height - 1
    south_z = (south_row_j + 0.5) * ts            # south-road centerline z
    cx = sim.grid_width * ts / 2.0
    return [
        TrafficLight(
            position=(cx, south_z),                # on the bot's path
            trigger_radius=0.35,
            cycle_s=14.0,
            green_frac=0.75,
            yellow_frac=0.05,
        )
    ]


def _jisr3_tl_visuals(sim) -> List[Tuple[float, float, float, float, float, float]]:
    """Visual housing on the inside shoulder of the south straight."""
    ts = float(sim.road_tile_size)
    south_row_j = sim.grid_height - 1
    south_inner_edge_z = south_row_j * ts          # north edge of south-road tiles
    visual_z = south_inner_edge_z - 0.10           # 10 cm inside the loop (inner shoulder)
    cx = sim.grid_width * ts / 2.0
    # (x, z, y_base, w, h, rotation_deg) — face +Z (south) toward the road
    return [(cx, visual_z, 0.45, 0.16, 0.42, 0.0)]


def _jisr3_scenery(sim) -> List[dict]:
    """Urban/suburban props placed along the outer roadside.

    Buildings cluster at the corners (taller silhouettes), houses fill the
    long outer edges, and trees scatter between them. Everything sits
    outside the road perimeter so the bot's cameras can see them but the
    bot can never collide with them.
    """
    ts = float(sim.road_tile_size)
    W = sim.grid_width * ts
    H = sim.grid_height * ts
    items: List[dict] = []

    def add(kind, x, z, rot=0.0, scale=0.18):
        items.append({"kind": kind, "position": (x, z), "rotation_deg": rot, "scale": scale})

    # All offsets are "beyond the road edge". Stacking: trees nearest the
    # road (closest to the bot), houses behind them, buildings on the far
    # outside at the corners. Gives a visible "shoulder + lawn + houses +
    # taller buildings" depth instead of one wall of meshes.

    building_off = 1.80
    house_off = 1.10
    tree_off = 0.45

    # ---- corner buildings (far back, taller) -----------------------------
    add("building", -building_off,     -building_off,     rot=45.0,    scale=0.28)
    add("building",  W + building_off, -building_off,     rot=-45.0,   scale=0.28)
    add("building",  W + building_off,  H + building_off, rot=-135.0,  scale=0.28)
    add("building", -building_off,      H + building_off, rot=135.0,   scale=0.28)

    # ---- houses along the outer edges ------------------------------------
    add("house",  W * 0.30,  -house_off,         rot=180.0,  scale=0.20)
    add("house",  W * 0.70,  -house_off,         rot=180.0,  scale=0.20)
    # South edge: leave the billboard at x=cx clear; houses well behind it.
    add("house",  W * 0.18,   H + house_off + 0.40, rot=0.0,  scale=0.20)
    add("house",  W * 0.82,   H + house_off + 0.40, rot=0.0,  scale=0.20)
    add("house", -house_off,  H * 0.30,           rot=90.0,   scale=0.20)
    add("house", -house_off,  H * 0.70,           rot=90.0,   scale=0.20)
    add("house",  W + house_off, H * 0.30,        rot=-90.0,  scale=0.20)
    add("house",  W + house_off, H * 0.70,        rot=-90.0,  scale=0.20)

    # ---- trees near the road (act as shoulder greenery) ------------------
    tree_scale = 0.38
    tree_spots = [
        # north shoulder
        (W * 0.15, -tree_off),
        (W * 0.40, -tree_off),
        (W * 0.60, -tree_off),
        (W * 0.85, -tree_off),
        # west shoulder
        (-tree_off, H * 0.25),
        (-tree_off, H * 0.50),
        (-tree_off, H * 0.75),
        # east shoulder — leave a gap around the stop sign (z ≈ 0.55*H)
        (W + tree_off, H * 0.20),
        (W + tree_off, H * 0.80),
        # south shoulder — flanking the billboard, not in front of it
        (W * 0.10, H + tree_off),
        (W * 0.90, H + tree_off),
        # deeper "park" behind the billboard
        (W * 0.30, H + tree_off + 1.6),
        (W * 0.70, H + tree_off + 1.6),
    ]
    for tx, tz in tree_spots:
        add("tree", tx, tz, rot=0.0, scale=tree_scale)

    return items


register_scenario(
    Scenario(
        name="small_loop_jisr3",
        description=(
            "KFUPM floor decal + 'Happy JISR3-ing!' billboard, one stop sign, "
            "one traffic light, plus urban/suburban scenery (buildings, "
            "houses, trees) around the outer perimeter. Agent obeys all rules."
        ),
        duration_s=60.0,
        floor_decals=_kfupm_floor_factory,
        billboards=_jisr3_billboards,
        stop_signs=_jisr3_stop_signs,
        traffic_lights=_jisr3_traffic_lights,
        traffic_light_visuals=_jisr3_tl_visuals,
        scenery=_jisr3_scenery,
    )
)
