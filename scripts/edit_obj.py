#!/usr/bin/env python3
"""
Programmatically edit Wavefront OBJ geometry.

Vertices are lines of the form ``v x y z`` (must be the three floats after ``v``).
Lines ``vn``, ``vt``, ``f``, ``mtllib``, ``usemtl``, ``o``, ``g``, ``#``, etc. are
passed through unchanged, so face indices stay valid.

Transforms are applied in order: **scale** (per axis), then **translate**.

Examples::

    # Deepen the pit (more negative Y in a Y-up file)
    python scripts/edit_obj.py assets/mining_generated/mining_pit_bowl.obj \\
        assets/mining_generated/mining_pit_deep.obj --scale 1 1.4 1

    # Move bowl origin in world space
    python scripts/edit_obj.py in.obj out.obj --translate 2 0 -3

    # Uniform shrink + lift
    python scripts/edit_obj.py in.obj out.obj --scale 0.5 0.5 0.5 --translate 0 0.2 0
"""
from __future__ import annotations

import argparse
import os


def transform_obj(
    src: str,
    dst: str,
    *,
    sx: float,
    sy: float,
    sz: float,
    tx: float,
    ty: float,
    tz: float,
) -> tuple[int, int]:
    n_v = 0
    n_lines = 0
    with open(src, "r") as inf, open(dst, "w") as outf:
        for line in inf:
            n_lines += 1
            if line.startswith("v ") and not line.startswith("vn ") and not line.startswith("vt "):
                parts = line.split()
                if len(parts) >= 4:
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    x = x * sx + tx
                    y = y * sy + ty
                    z = z * sz + tz
                    outf.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
                    n_v += 1
                    continue
            outf.write(line)
    return n_v, n_lines


def main() -> None:
    p = argparse.ArgumentParser(description="Edit OBJ vertex positions (v lines only).")
    p.add_argument("input_obj")
    p.add_argument("output_obj")
    p.add_argument("--scale", nargs=3, type=float, default=[1.0, 1.0, 1.0], metavar=("SX", "SY", "SZ"))
    p.add_argument(
        "--translate",
        nargs=3,
        type=float,
        default=[0.0, 0.0, 0.0],
        metavar=("TX", "TY", "TZ"),
    )
    args = p.parse_args()

    src = os.path.abspath(args.input_obj)
    dst = os.path.abspath(args.output_obj)
    if not os.path.isfile(src):
        raise SystemExit(f"not found: {src}")

    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    nv, nl = transform_obj(
        src,
        dst,
        sx=args.scale[0],
        sy=args.scale[1],
        sz=args.scale[2],
        tx=args.translate[0],
        ty=args.translate[1],
        tz=args.translate[2],
    )
    print(f"wrote {dst}  (transformed {nv} vertices, {nl} input lines)")


if __name__ == "__main__":
    main()
