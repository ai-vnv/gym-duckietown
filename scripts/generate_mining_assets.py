#!/usr/bin/env python3
"""
Procedurally generate mining-pit style assets (no external art tools).

Outputs (under ``--out-dir``):

- **Textures** (tiling-friendly RGB PNG, default 512×512):

  - ``sand_white.png`` — high-key warm silica
  - ``sand_brown.png`` — iron / wet sand bias
  - ``gravel.png`` — coarse speckle, low saturation
  - ``spoil_brown.png`` — compacted haul-road brown (for large pads)

- **Mesh** (Wavefront OBJ + MTL):

  - ``mining_pit_bowl.obj`` — smooth axisymmetric crater (depression in +Y up frame)
  - ``mining_pit_bowl.mtl`` — references ``spoil_brown.png`` as diffuse map

Coordinate note: Duckietown uses Y up with roads in the XZ plane; this bowl is
authored the same way (pit goes **down** along **-Y**). You can import the OBJ
into Blender / MeshLab, or adapt loaders to merge with other pipelines.

Integration with ``gym-duckietown``'s built-in tile loader is **not** automatic:
those paths still read duckietown-world atlases. Use this pack as offline art,
or wire custom OBJ/texture loading in a fork / separate renderer.

Examples::

    python scripts/generate_mining_assets.py
    python scripts/generate_mining_assets.py --out-dir ./assets/mining_generated --size 1024 --seed 42
"""
from __future__ import annotations

import argparse
import math
import os
from typing import Tuple

import numpy as np
from PIL import Image


def _upscale_noise(h: int, w: int, sh: int, sw: int, rng: np.random.Generator) -> np.ndarray:
    small = (rng.random((sh, sw)) * 255.0).astype(np.uint8)
    im = Image.fromarray(small, mode="L")
    up = np.asarray(im.resize((w, h), Image.Resampling.BICUBIC), dtype=np.float64) / 255.0
    return up


def _fbm2d(h: int, w: int, rng: np.random.Generator, octaves: int = 6) -> np.ndarray:
    """fBm in [0,1] via summed bicubic upscaled octave noise."""
    acc = np.zeros((h, w), dtype=np.float64)
    for o in range(octaves):
        scale = 2 ** (o + 2)
        sh = max(4, h // scale)
        sw = max(4, w // scale)
        layer = _upscale_noise(h, w, sh, sw, rng)
        acc += (0.5**o) * layer
    acc -= acc.min()
    acc /= acc.max() + 1e-9
    return acc


def _to_uint8_rgb(arr: np.ndarray) -> np.ndarray:
    return np.clip(np.round(arr * 255.0), 0, 255).astype(np.uint8)


def texture_white_sand(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    n = _fbm2d(h, w, rng, octaves=6)
    r = 0.92 + 0.06 * n
    g = 0.88 + 0.05 * n + 0.02 * rng.standard_normal((h, w)) * 0.05
    b = 0.78 + 0.06 * n
    speck = rng.random((h, w))
    sparkle = np.where(speck > 0.985, 0.08, 0.0)
    rgb = np.stack([r, g, b], axis=-1) + sparkle[..., None]
    return _to_uint8_rgb(rgb)


def texture_brown_sand(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    n = _fbm2d(h, w, rng, octaves=5)
    r = 0.55 + 0.18 * n
    g = 0.38 + 0.12 * n
    b = 0.22 + 0.08 * n
    rgb = np.stack([r, g, b], axis=-1)
    return _to_uint8_rgb(rgb)


def texture_gravel(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    n = _fbm2d(h, w, rng, octaves=4)
    speck = rng.random((h, w))
    chips = np.where(speck > 0.65, rng.random((h, w)) * 0.25, 0.0)
    base = 0.35 + 0.25 * n + chips
    rgb = np.stack([base, base * 0.98, base * 0.95], axis=-1)
    # crush saturation slightly per channel jitter
    jitter = rng.standard_normal((h, w, 3)) * 0.04
    rgb = np.clip(rgb + jitter, 0, 1)
    return _to_uint8_rgb(rgb)


def texture_spoil_brown(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    n = _fbm2d(h, w, rng, octaves=5)
    r = 0.42 + 0.12 * n
    g = 0.34 + 0.10 * n
    b = 0.26 + 0.08 * n
    tire = _fbm2d(h, w, rng, octaves=3)
    r -= 0.03 * tire
    g -= 0.02 * tire
    rgb = np.stack([r, g, b], axis=-1)
    return _to_uint8_rgb(rgb)


def write_png(path: str, rgb: np.ndarray) -> None:
    Image.fromarray(rgb, mode="RGB").save(path, optimize=True)


def generate_pit_mesh(
    path_obj: str,
    path_mtl: str,
    mtl_name: str,
    *,
    half_extent: float = 12.0,
    grid: int = 64,
    pit_radius: float = 7.5,
    max_depth: float = 2.2,
    rim_lift: float = 0.15,
) -> None:
    """
    Axisymmetric bowl + slight outer berm lift. Y is up; pit is negative Y.
    """
    step = (2.0 * half_extent) / (grid - 1)
    verts: list[Tuple[float, float, float]] = []
    uvs: list[Tuple[float, float]] = []
    normals: list[Tuple[float, float, float]] = []

    def height(x: float, z: float) -> float:
        r = math.hypot(x, z)
        if r >= pit_radius * 1.35:
            return 0.0
        # Core crater
        u = min(1.0, r / pit_radius)
        bowl = -max_depth * (1.0 - u * u) ** 2
        # Soft rim mound outside pit
        if pit_radius * 0.85 < r < pit_radius * 1.25:
            t = (r - pit_radius * 0.85) / (pit_radius * 0.4)
            bowl += rim_lift * math.sin(math.pi * t) * max(0.0, 1.0 - u)
        return bowl

    # First pass: heights
    hmap = np.zeros((grid, grid), dtype=np.float64)
    for j in range(grid):
        for i in range(grid):
            x = -half_extent + i * step
            z = -half_extent + j * step
            hmap[j, i] = height(x, z)

    # Vertices + UV
    for j in range(grid):
        for i in range(grid):
            x = -half_extent + i * step
            z = -half_extent + j * step
            y = hmap[j, i]
            verts.append((x, y, z))
            uvs.append((i / (grid - 1), j / (grid - 1)))

    # Normals via central differences on height field
    ny = np.ones_like(hmap)
    hx = np.zeros_like(hmap)
    hz = np.zeros_like(hmap)
    hx[:, 1:-1] = (hmap[:, 2:] - hmap[:, :-2]) / (2 * step)
    hz[1:-1, :] = (hmap[2:, :] - hmap[:-2, :]) / (2 * step)
    nx = -hx
    nz = -hz
    nlen = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-12
    nx /= nlen
    ny /= nlen
    nz /= nlen

    for j in range(grid):
        for i in range(grid):
            normals.append((float(nx[j, i]), float(ny[j, i]), float(nz[j, i])))

    faces: list[Tuple[int, int, int]] = []
    vi = lambda i, j: j * grid + i + 1  # 1-based OBJ

    for j in range(grid - 1):
        for i in range(grid - 1):
            a = vi(i, j)
            b = vi(i + 1, j)
            c = vi(i + 1, j + 1)
            d = vi(i, j + 1)
            faces.append((a, b, c))
            faces.append((a, c, d))

    mtl_base = os.path.basename(path_mtl)
    with open(path_obj, "w") as f:
        f.write("# mining_pit_bowl — generated by scripts/generate_mining_assets.py\n")
        f.write(f"mtllib {mtl_base}\n")
        f.write(f"usemtl {mtl_name}\n")
        for x, y, z in verts:
            f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
        for u, v in uvs:
            f.write(f"vt {u:.6f} {v:.6f}\n")
        for nx, ny, nz in normals:
            f.write(f"vn {nx:.6f} {ny:.6f} {nz:.6f}\n")
        for a, b, c in faces:
            f.write(f"f {a}/{a}/{a} {b}/{b}/{b} {c}/{c}/{c}\n")

    tex_name = "spoil_brown.png"
    with open(path_mtl, "w") as f:
        f.write(f"newmtl {mtl_name}\n")
        f.write("Ka 0.2 0.2 0.2\n")
        f.write("Kd 0.85 0.8 0.75\n")
        f.write("Ks 0.05 0.05 0.05\n")
        f.write(f"map_Kd {tex_name}\n")


def main() -> None:
    p = argparse.ArgumentParser(description="Generate sand textures + mining pit OBJ/MTL.")
    p.add_argument(
        "--out-dir",
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "assets",
            "mining_generated",
        ),
        help="Output directory (created if missing)",
    )
    p.add_argument("--size", type=int, default=512, help="Texture width/height in pixels")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--grid", type=int, default=64, help="Pit mesh grid resolution per axis")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    h = w = int(args.size)

    paths = {
        "sand_white.png": texture_white_sand(h, w, rng),
        "sand_brown.png": texture_brown_sand(h, w, rng),
        "gravel.png": texture_gravel(h, w, rng),
        "spoil_brown.png": texture_spoil_brown(h, w, rng),
    }
    for name, arr in paths.items():
        out = os.path.join(args.out_dir, name)
        write_png(out, arr)
        print("wrote", out)

    obj_path = os.path.join(args.out_dir, "mining_pit_bowl.obj")
    mtl_path = os.path.join(args.out_dir, "mining_pit_bowl.mtl")
    generate_pit_mesh(obj_path, mtl_path, "SpoilPit", grid=args.grid)
    print("wrote", obj_path)
    print("wrote", mtl_path)
    print("done — open OBJ in Blender with textures in the same folder.")


if __name__ == "__main__":
    main()
