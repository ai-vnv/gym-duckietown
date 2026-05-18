"""Stubs for testing the multiview package without booting the GL simulator.

The pieces under test (controller, trajectory math, composer) only need a
handful of attributes off ``sim``; we expose those via lightweight dataclass
stubs so tests stay fast and headless.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pytest


@dataclass
class StubTile:
    drivable: bool = True
    kind: str = "straight"

    def get(self, key: str, default=None):
        return getattr(self, key, default)


@dataclass
class StubSim:
    """Minimal stand-in for ``Simulator`` for non-GL tests."""

    grid_width: int = 4
    grid_height: int = 4
    road_tile_size: float = 0.585
    cur_angle: float = 0.0
    cam_height: float = 0.108
    cam_angle: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    cur_pos: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    camera_width: int = 160
    camera_height: int = 120
    multi_fbo: int = 0
    final_fbo: int = 0
    img_array: Optional[np.ndarray] = None
    _tiles: Dict[Tuple[int, int], StubTile] = field(default_factory=dict)
    render_calls: List[Tuple[float, float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Default: all tiles in a ring around the border are drivable; center
        # tiles are non-drivable. Gives the trajectory panel something to draw.
        if not self._tiles:
            for i in range(self.grid_width):
                for j in range(self.grid_height):
                    border = (
                        i == 0 or j == 0 or i == self.grid_width - 1 or j == self.grid_height - 1
                    )
                    self._tiles[(i, j)] = StubTile(drivable=border)

    def _get_tile(self, i: int, j: int):
        return self._tiles.get((i, j))

    def _render_img(self, width, height, *_, **_kw):
        """Stub renderer: records (cur_angle, cam_height, cam_pitch) at call time."""
        self.render_calls.append(
            (float(self.cur_angle), float(self.cam_height), float(self.cam_angle[0]))
        )
        return np.full((int(height), int(width), 3), 128, dtype=np.uint8)

    def render_obs(self):
        return self._render_img(self.camera_width, self.camera_height)


@pytest.fixture
def stub_sim() -> StubSim:
    return StubSim()


@pytest.fixture
def constant_policy():
    """Policy that always returns the same action; usable with SmoothedController."""

    class _ConstPolicy:
        def __init__(self, v: float = 0.5, omega: float = 0.2):
            self.v = v
            self.omega = omega
            self.n_calls = 0

        def predict(self, obs):  # noqa: ARG002
            self.n_calls += 1
            return [self.v, self.omega]

    return _ConstPolicy
