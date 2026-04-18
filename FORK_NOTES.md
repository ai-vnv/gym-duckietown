# Fork maintenance notes

This branch keeps **upstream `daffy`-compatible** behavior while fixing modern Python, NumPy, Pyglet, and platform issues so `pip install -e .` and the simulator run on current macOS and Linux (including Colab-style headless EGL).

## Changes vs upstream `duckietown/gym-duckietown` (`daffy`)

| Area | Change |
|------|--------|
| **Dependencies** (`setup.py`) | Relax `numpy` to `>=1.21,<2` (wheels on Python 3.10+ and ARM macOS). Pin `pyglet>=1.5,<2` (Pyglet 2 removes GLU symbols this codebase uses). |
| **PWM dynamics** (`src/gym_duckietown/pwm_numpy_compat.py`) | Monkeypatch `DynamicModel.integrate` in `duckietown-world`: acceleration vectors are `(2,1)`; recent NumPy makes `longitudinal` a 0-d array and breaks `se2_from_linear_angular`. Coerce to floats before building the velocity vector. Applied from `gym_duckietown/__init__.py` after env registration. |
| **macOS OpenGL** (`check_hw.py`) | Do not set `pyglet.options["headless"] = True` on **Darwin** (no system EGL); keep Cocoa GL. Still use headless on Linux for servers/Colab. |
| **Headless override** (`__init__.py`) | If `DUCKIETOWN_HEADLESS=1` is set, force headless mode (Linux servers). Default remains windowed on Mac, headless on other platforms. |

## Optional extras in this fork

- `notebooks/gym_duckietown_pedagogy.ipynb` — teaching / falsification demo.
- `scripts/record_pp_failures.py` — records short MP4s of two Pure Pursuit failure modes.

## Install

```bash
git clone -b daffy --depth 1 https://github.com/<your-username>/gym-duckietown.git
cd gym-duckietown
python3.10 -m venv .venv && source .venv/bin/activate   # 3.10+ recommended
pip install -e .
```

## Publish your fork on GitHub

1. Fork **duckietown/gym-duckietown** in the GitHub UI (use `daffy` as default branch if prompted).
2. Add your fork as `origin` and keep upstream for merges:

```bash
git remote rename origin upstream
git remote add origin git@github.com:<your-username>/gym-duckietown.git
git push -u origin daffy
```

3. Optionally set the default branch on GitHub to `daffy` or merge these commits into `main` on your fork.

## Colab

Use a virtual framebuffer (`xvfb`) or rely on EGL headless on Linux; install with `pip install -e .` from your fork URL. A Colab-oriented notebook can apply the same `setup.py` edits if installing from unpatched upstream; installing **this fork** avoids manual patching.
