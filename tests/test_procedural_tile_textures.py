# coding=utf-8
"""Tests for programmatic tile textures (see ``vnv/procedural_tiles_vnvspec.json``)."""

import numpy as np
import pytest
from numpy.random import Generator, RandomState

from gym_duckietown.graphics import _ensure_rgba_uint8, _resize_rgba_for_gl, load_texture_from_rgba
from gym_duckietown.scene_presets import (
    _mining_floor_rgba,
    _mining_grass_rgba,
    _rng_integers,
    _rng_normal,
    mining_procedural_tile_textures,
)


@pytest.mark.parametrize("rng_factory", [lambda: RandomState(42), lambda: Generator(np.random.PCG64(42))])
def test_rng_helpers_work_with_randomstate_and_generator(rng_factory):
    rng = rng_factory()
    u = _rng_integers(rng, 0, 10, (4, 5))
    assert u.shape == (4, 5)
    assert u.dtype == np.int32
    assert np.all((u >= 0) & (u < 10))
    n = _rng_normal(rng, (2, 3, 4))
    assert n.shape == (2, 3, 4)


@pytest.mark.parametrize("rng_factory", [lambda: RandomState(0), lambda: Generator(np.random.PCG64(0))])
def test_mining_rgb_shapes_and_dtype(rng_factory):
    rng = rng_factory()
    for fn in (_mining_floor_rgba, _mining_grass_rgba):
        a = fn(rng, 64)
        assert a.shape == (64, 64, 3)
        assert a.dtype == np.uint8
        assert int(a.min()) >= 0 and int(a.max()) <= 255


def test_mining_procedural_dict_callables():
    d = mining_procedural_tile_textures(32)
    rng = RandomState(1)
    for k in ("floor", "grass"):
        out = d[k](rng)
        assert out.shape == (32, 32, 3) and out.dtype == np.uint8


def test_pot_resize_and_rgba_expand():
    x = np.zeros((10, 20, 3), dtype=np.uint8)
    r = _resize_rgba_for_gl(_ensure_rgba_uint8(x), enforce_pot=True, max_side=4096)
    assert r.shape == (16, 32, 4)
    assert r.dtype == np.uint8


@pytest.mark.skip(reason="Needs active Pyglet GL context (run manually or in integration job)")
def test_load_texture_from_rgba_gl_upload():
    arr = np.zeros((32, 32, 4), dtype=np.uint8)
    arr[:, :, 0] = 200
    arr[:, :, 3] = 255
    tex = load_texture_from_rgba(arr, enforce_pot=False, max_side=4096)
    assert tex.id > 0


def test_mining_prog_env_reset_and_render():
    """End-to-end: Cocoa/EGL context from Simulator; may skip in broken headless CI."""
    import gym

    import gym_duckietown  # noqa: F401 — registers envs

    try:
        env = gym.make("Duckietown-zigzag_dists_mining_prog-v0")
        env.reset()
        img = env.render("rgb_array")
        assert img.ndim == 3 and img.shape[2] == 3
        assert img.dtype == np.uint8
        assert np.isfinite(img.mean())
        env.close()
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"GL/display not available: {exc}")
