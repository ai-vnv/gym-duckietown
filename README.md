# Gym-Duckietown (ai-vnv fork)

[![CI](https://github.com/ai-vnv/gym-duckietown/actions/workflows/ci.yml/badge.svg?branch=daffy)](https://github.com/ai-vnv/gym-duckietown/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Docs source](https://img.shields.io/badge/docs-Sphinx-2980B9.svg)](https://github.com/ai-vnv/gym-duckietown/tree/daffy/docs)
[![Upstream](https://img.shields.io/badge/upstream-duckietown%2Fgym--duckietown-lightgrey)](https://github.com/duckietown/gym-duckietown)

**Repository:** [`ai-vnv/gym-duckietown`](https://github.com/ai-vnv/gym-duckietown) · **branch:** `daffy`  
**Upstream:** [duckietown/gym-duckietown](https://github.com/duckietown/gym-duckietown) (`daffy`)

[Duckietown](https://duckietown.org/) self-driving car **Gym** environments in Python + OpenGL (Pyglet). This README is for **this fork** (modern NumPy 1.x, Pyglet 1.x, macOS Cocoa, lab presets). For legacy install/troubleshooting text, see the [upstream README](https://github.com/duckietown/gym-duckietown/blob/daffy/README.md).

Please cite the original work:

```bibtex
@misc{gym_duckietown,
  author = {Chevalier-Boisvert, Maxime and Golemo, Florian and Cao, Yanjun and Mehta, Bhairav and Paull, Liam},
  title = {Duckietown Environments for OpenAI Gym},
  year = {2018},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/duckietown/gym-duckietown}},
}
```

---

## Documentation & CI

| Resource | Link / command |
|----------|----------------|
| **Sphinx docs** (API + fork guide) | Sources in [`docs/`](docs/); local build: `pip install -e ".[dev]" && sphinx-build -b html docs docs/_build/html`. **Read the Docs:** import this repo (`.readthedocs.yaml` at root) to publish `latest`. |
| **CI** | [GitHub Actions](https://github.com/ai-vnv/gym-duckietown/actions) — `pytest` under **Xvfb** (Linux) + Sphinx HTML **artifact** |
| **V&V register** | [`vnv/procedural_tiles_vnvspec.json`](vnv/procedural_tiles_vnvspec.json) |
| **Pedagogy notebook** | [`notebooks/gym_duckietown_pedagogy.ipynb`](notebooks/gym_duckietown_pedagogy.ipynb) |

---

## Install (this repository)

```bash
git clone -b daffy https://github.com/ai-vnv/gym-duckietown.git
cd gym-duckietown
python3.10 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"    # includes pytest + Sphinx for docs/tests
```

Optional: `jupyter`, `imageio`, `imageio-ffmpeg` for notebooks and MP4 scripts.

**Merge upstream when needed:**

```bash
git remote add upstream https://github.com/duckietown/gym-duckietown.git   # once
git fetch upstream && git merge upstream/daffy && git push origin daffy
```

---

## Fork vs upstream (summary)

| Area | This fork |
|------|-----------|
| **Dependencies** | `numpy>=1.21,<2`, `pyglet>=1.5,<2` in `setup.py` |
| **macOS** | Cocoa GL path in `check_hw.py`; no forced EGL-only |
| **Headless Linux** | `DUCKIETOWN_HEADLESS=1` in `__init__.py` |
| **PWM / NumPy** | `pwm_numpy_compat.py` loaded from `gym_duckietown/__init__.py` |

**Fork-only features** (see [`docs/fork_features.rst`](docs/fork_features.rst) and [`scene_presets.py`](src/gym_duckietown/scene_presets.py)):

- **Desert outdoor** — `Duckietown-*_arabian-v0`, warm sky / sand ground tint.
- **Multiview** — `env.unwrapped.render_multiview_rgb()`; `scripts/demo_multiview.py` → `recordings/multiview_demo.mp4`.
- **Mining preset** — `Duckietown-zigzag_dists_mining-v0`; `scripts/demo_mining_pit.py` → `recordings/mining_pit_multiview.mp4`.
- **Mining + file textures** — `Duckietown-zigzag_dists_mining_gl-v0` (run `scripts/generate_mining_assets.py` first).
- **Mining + procedural GL tiles** — `Duckietown-zigzag_dists_mining_prog-v0` (`custom_tile_textures`, no PNGs required).
- **Procedural assets (offline)** — `scripts/generate_mining_assets.py`, `scripts/edit_obj.py`.

---

## Media (regenerate locally)

MP4s are **gitignored** under `recordings/`. Run the scripts below, then open the files or embed them in slides / the pedagogy notebook.

| Output (default path) | Command |
|------------------------|---------|
| `recordings/kfupm_style_udem1_arabian_20s.mp4` | `python scripts/render_kfupm_style_video.py` |
| `recordings/multiview_demo.mp4` | `python scripts/demo_multiview.py` |
| `recordings/mining_pit_multiview.mp4` | `python scripts/demo_mining_pit.py` |
| `recordings/mining_prog_multiview.mp4` | `python scripts/demo_mining_pit.py --env-id Duckietown-zigzag_dists_mining_prog-v0 -o recordings/mining_prog_multiview.mp4` |
| `recordings/failure_*.mp4` | `python scripts/record_pp_failures.py` |

---

## Registered environments (excerpt)

Stock-style: `Duckietown-udem1-v0`, `Duckietown-small_loop-v0`, `Duckietown-zigzag_dists-v0`, …

**Fork IDs:** `Duckietown-small_loop_arabian-v0`, `Duckietown-udem1_arabian-v0`, `Duckietown-zigzag_dists_mining-v0`, `Duckietown-zigzag_dists_mining_gl-v0`, `Duckietown-zigzag_dists_mining_prog-v0`, `MultiMap-v0`, `Duckiebot-v0`.

---

## Usage (quick)

```bash
python manual_control.py --env-name Duckietown-udem1-v0
```

RL / imitation / Docker: see upstream README and `pytorch_rl/`, `experiments/`.

---

## Design (short)

- **Maps:** YAML in duckietown-world data; [Duckietown specs](https://docs.duckietown.org/daffy/opmanual_duckietown/out/duckietown_specs.html).
- **Observations:** RGB camera (`Simulator` defaults, often 640×480).
- **Actions:** continuous `[velocity, steering]` in \([-1,1]\).

---

## Troubleshooting

- **No display (SSH/Linux):** Xvfb or EGL; install `freeglut3-dev` where needed. CI uses `xvfb-run -a pytest …`.
- **Fork bugs:** [ai-vnv/gym-duckietown issues](https://github.com/ai-vnv/gym-duckietown/issues).

---

<p align="center">
<img src="media/simplesim_free.png" width="280" alt="simulator screenshot">
<br>
<img src="media/finalmain.gif" width="400" alt="duckietown demo gif">
</p>
