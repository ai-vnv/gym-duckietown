# Gym-Duckietown

**Repository:** [`ai-vnv/gym-duckietown`](https://github.com/ai-vnv/gym-duckietown) · **branch:** `daffy`

> **This is a fork**, not the official Duckietown README verbatim.  
> **Upstream project:** [duckietown/gym-duckietown](https://github.com/duckietown/gym-duckietown) (`daffy`).  
> Sections labeled **Fork note** describe **this fork only**—maintainers, install from this repo, and patches for modern Python/tooling. The science, simulator design, and citation below follow the **original** Duckietown Gym-Duckietown work.

[Duckietown](https://duckietown.org/) self-driving car simulator environments for OpenAI Gym, written in Python/OpenGL (Pyglet).

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

Simulator origins: [Mila](https://mila.quebec/).

<p align="center">
<img src="media/simplesim_free.png" width="300px"><br>
</p>

---

## Fork note — maintainers & scope

This fork is maintained for **ai-vnv** lab use: keep compatibility with upstream **`daffy`**, while making `pip install -e .` and the simulator run on **current Python (3.10+)**, **NumPy 1.x**, **Pyglet 1.x**, **macOS** (Cocoa GL), and **Linux/Colab** (headless/EGL or Xvfb).  
**These fixes are not from the original Duckietown maintainers**; they live only in this fork unless upstream merges them.

| Area | Change (this fork only) |
|------|-------------------------|
| **`setup.py`** | `numpy>=1.21,<2`; `pyglet>=1.5,<2` (Pyglet 2 drops GLU used here). |
| **`pwm_numpy_compat.py`** | Patches `duckietown-world` PWM integration so accelerations stay scalar with current NumPy; loaded from `gym_duckietown/__init__.py`. |
| **`check_hw.py`** | On **macOS**, do not force EGL-only headless mode (use Cocoa). Linux unchanged for servers/Colab. |
| **`__init__.py`** | Optional `DUCKIETOWN_HEADLESS=1` to force headless on Linux. |

**Extras in this fork (not upstream):**

- [`notebooks/gym_duckietown_pedagogy.ipynb`](notebooks/gym_duckietown_pedagogy.ipynb) — pedagogy / falsification demo (uses this repo’s clone URL in Colab).
- [`scripts/record_pp_failures.py`](scripts/record_pp_failures.py) — short MP4 recordings of Pure Pursuit failure modes.
- **Arabian desert outdoor** — warm sky + sand ground tint (same maps; `domain_rand=False` for stable colors). Env IDs: `Duckietown-small_loop_arabian-v0`, `Duckietown-loop_obstacles_arabian-v0`, `Duckietown-udem1_arabian-v0` (stock **udem1** campus road layout + desert look — closest built-in proxy for a large university site; not georeferenced to any real campus). Preset: [`scene_presets.py`](src/gym_duckietown/scene_presets.py); demo rollouts: [`scripts/experiment_arabian_desert.py`](scripts/experiment_arabian_desert.py); 20s clip: `python scripts/render_kfupm_style_video.py` → `recordings/kfupm_style_udem1_arabian_20s.mp4`.
- **Multiview (2×2)** — driver, whole-map bird’s-eye, top-follow, and rear cameras in one RGB panel: `env.unwrapped.render_multiview_rgb()` on any `DuckietownEnv`, or wrap with `gym_duckietown.wrappers.MultiViewObservationWrapper` (four extra full renders per step — for video/demos). Example: `python scripts/demo_multiview.py` → `recordings/multiview_demo.mp4`.

---

## Fork note — install **this** repository

```bash
git clone -b daffy https://github.com/ai-vnv/gym-duckietown.git
cd gym-duckietown
python3.10 -m venv .venv && source .venv/bin/activate   # 3.10+ recommended
pip install -e .
```

Optional: `pip install jupyter imageio imageio-ffmpeg` for the notebook and recordings.

---

## Fork note — stay aligned with upstream Duckietown

```bash
git remote add upstream https://github.com/duckietown/gym-duckietown.git   # once, if missing
git fetch upstream
git merge upstream/daffy
git push origin daffy
```

---

## Fork note — Google Colab

Use [`notebooks/gym_duckietown_pedagogy.ipynb`](notebooks/gym_duckietown_pedagogy.ipynb): the bootstrap cell clones **`https://github.com/ai-vnv/gym-duckietown.git`** (`daffy`), runs `pip install -e .`, and sets up a display (e.g. Xvfb) when needed. **No manual patching** of upstream `setup.py` is required when installing **this fork**.

---

## Introduction (original project)

Gym-Duckietown places your agent (a Duckiebot) in a Duckietown map: roads, turns, intersections, obstacles, pedestrians, and other agents. It supports RL, imitation learning, classical control, domain randomization, and tools aimed at sim-to-real transfer.

<p align="center">
<img src="media/finalmain.gif"><br>
</p>

**Registered environments** (maps under `src/gym_duckietown` / packaged data) include:

- `Duckietown-straight_road-v0`, `Duckietown-4way-v0`, `Duckietown-udem1-v0`
- `Duckietown-small_loop-v0`, `Duckietown-small_loop_cw-v0`, `Duckietown-zigzag_dists-v0`
- `Duckietown-loop_obstacles-v0`, `Duckietown-loop_pedestrians-v0`
- `Duckietown-small_loop_arabian-v0`, `Duckietown-loop_obstacles_arabian-v0`, `Duckietown-udem1_arabian-v0` (**fork:** desert-style sky/ground; same geometry as the maps above)
- `MultiMap-v0` (cycles maps), `Duckiebot-v0`

Hardware and AIDO templates: see [Duckietown embodied docs](https://docs-old.duckietown.org/daffy/AIDO/out/embodied.html).

---

## Usage (summary)

**Manual control:**

```bash
python manual_control.py --env-name Duckietown-udem1-v0
```

**RL (code under `pytorch_rl/`):** install PyTorch, then e.g.:

```bash
python3 pytorch_rl/main.py --no-vis --env-name Duckietown-small_loop-v0 --algo a2c --lr 0.0002 --max-grad-norm 0.5 --num-steps 20
```

**Imitation learning:** see `experiments/` (e.g. `gen_demos.py`, `train_imitation.py`).  
**Docker:** image `duckietown/gym-duckietown` on Docker Hub (upstream-maintained image; may differ from this fork’s dependency pins).

---

## Design (summary)

- **Maps:** YAML tiles and objects; see packaged maps and [Duckietown specs](https://docs.duckietown.org/daffy/opmanual_duckietown/out/duckietown_specs.html).
- **Observations:** camera images (shape depends on build; often 640×480 RGB in current `daffy`).
- **Actions:** continuous `[velocity, steering]` in \([-1,1]\); see `DiscreteWrapper` for discrete actions.
- **Reward / done:** lane following vs. centerline; episode ends on invalid pose or `max_steps`. Details in simulator / env code.

---

## Troubleshooting

- **No display (SSH/Docker):** use Xvfb or EGL headless; see upstream README sections on headless training.
- **GLU / GL:** on Linux, `freeglut3-dev` etc.; **this fork** pins Pyglet 1.x to keep GLU bindings used by the code.
- **More detail:** full legacy troubleshooting and conda/docker blocks live in the [upstream `README`](https://github.com/duckietown/gym-duckietown/blob/daffy/README.md) from Duckietown; **report fork-specific bugs** on [ai-vnv/gym-duckietown issues](https://github.com/ai-vnv/gym-duckietown/issues).

---

<p align="center">
<img src="media/duckiebot_1.png" width="300px"><br>
<em>Duckiebot-v0 (concept)</em>
</p>
