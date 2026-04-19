# coding=utf-8
__version__ = "6.2.0"

import platform

from zuper_commons.logs import ZLogger

from duckietown_world.resources import list_maps2

logger = ZLogger("gym-duckietown")
import os

import pyglet

on_mac = "Darwin" in platform.system()
_force_headless = os.environ.get("DUCKIETOWN_HEADLESS", "").lower() in ("1", "true", "yes")
print(pyglet.options)
if _force_headless:
    pyglet.options["headless"] = True
elif on_mac:
    pyglet.options["headless"] = False
else:
    pyglet.options["headless"] = True

path = os.path.dirname(os.path.dirname(__file__))
logger.debug(f"gym-duckietown version {__version__} path {path}\n")

from gym.envs.registration import register

from .utils import get_subdir_path


def reg_map_env(map_name0: str, map_file: str):
    gym_id = f"Duckietown-{map_name0}-v0"

    # logger.info('Registering gym environment id: %s' % gym_id)

    register(
        id=gym_id,
        entry_point="gym_duckietown.envs:DuckietownEnv",
        reward_threshold=400.0,
        kwargs={"map_name": map_file},
    )


for map_name, filename in list_maps2().items():
    # Register a gym environment for each map file available
    if "regress" not in filename:
        reg_map_env(map_name, filename)

register(id="MultiMap-v0", entry_point="gym_duckietown.envs:MultiMapEnv", reward_threshold=400.0)

register(id="Duckiebot-v0", entry_point="gym_duckietown.envs:DuckiebotEnv", reward_threshold=400.0)

# Experimental “Arabian desert outdoor” look: warm sky + sand ground (same maps, tinted horizon/ground).
from .scene_presets import ARABIAN_DESERT_OUTDOOR, MINING_PIT_OUTDOOR

register(
    id="Duckietown-small_loop_arabian-v0",
    entry_point="gym_duckietown.envs:DuckietownEnv",
    reward_threshold=400.0,
    kwargs={**ARABIAN_DESERT_OUTDOOR, "map_name": "small_loop"},
)
register(
    id="Duckietown-loop_obstacles_arabian-v0",
    entry_point="gym_duckietown.envs:DuckietownEnv",
    reward_threshold=400.0,
    kwargs={**ARABIAN_DESERT_OUTDOOR, "map_name": "loop_obstacles"},
)
# Campus-scale road network (duckietown-world “udem1”) + desert outdoor tint — closest stock proxy for a large university site (e.g. KFUPM-style) in this simulator.
register(
    id="Duckietown-udem1_arabian-v0",
    entry_point="gym_duckietown.envs:DuckietownEnv",
    reward_threshold=400.0,
    kwargs={**ARABIAN_DESERT_OUTDOOR, "map_name": "udem1"},
)
# Winding torture-track map + mining/off-road look (switchback “pit path” metaphor; terrain is still flat).
register(
    id="Duckietown-zigzag_dists_mining-v0",
    entry_point="gym_duckietown.envs:DuckietownEnv",
    reward_threshold=400.0,
    kwargs={**MINING_PIT_OUTDOOR, "map_name": "zigzag_dists"},
)

from .pwm_numpy_compat import apply_patch as _apply_pwm_numpy_compat

_apply_pwm_numpy_compat()
