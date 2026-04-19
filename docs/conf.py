# Sphinx configuration for gym-duckietown (ai-vnv fork).
# Build locally: ``sphinx-build -b html docs docs/_build/html``

import os
import sys

# Quiet ``gym_duckietown`` package import side effects during autodoc.
os.environ.setdefault("GYM_DUCKIETOWN_SPHINX", "1")

sys.path.insert(0, os.path.abspath("../src"))

project = "gym-duckietown"
copyright = "2018 Duckietown authors; fork patches ai-vnv"
author = "Duckietown / ai-vnv"

def _read_version():
    init = os.path.join(os.path.dirname(__file__), "..", "src", "gym_duckietown", "__init__.py")
    with open(init, encoding="utf-8") as f:
        for line in f:
            if line.startswith("__version__"):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return "0.0.0"


version = release = _read_version()

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}

napoleon_google_docstring = True
napoleon_numpy_docstring = True

# Mock heavy / optional deps. Do **not** mock the top-level ``pyglet`` package: ``gym_duckietown``
# assigns ``pyglet.options[...]`` at import time and a mocked ``pyglet`` breaks that.
autodoc_mock_imports = [
    "cv2",
    "gym",
    "gym.utils",
    "gym.spaces",
    "geometry",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}
