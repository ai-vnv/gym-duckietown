Developer guide
================

Install from source
-------------------

.. code-block:: bash

   git clone -b daffy https://github.com/ai-vnv/gym-duckietown.git
   cd gym-duckietown
   python3.10 -m venv .venv && source .venv/bin/activate
   pip install -e ".[dev]"

Tests
-----

Linux CI uses **Xvfb** for a virtual framebuffer. Locally on macOS, Cocoa is used by default.

.. code-block:: bash

   # Linux headless example
   xvfb-run -a python -m pytest tests/ -v

   # macOS
   python -m pytest tests/ -v

Documentation build
---------------------

.. code-block:: bash

   sphinx-build -b html docs docs/_build/html

Open ``docs/_build/html/index.html``. **Read the Docs** can build from ``.readthedocs.yaml`` in the repository root.

Continuous integration
------------------------

GitHub Actions workflow ``.github/workflows/ci.yml`` runs pytest and uploads **Sphinx HTML** as a workflow artifact (``sphinx-html``).

Read the Docs
--------------

The repo root contains ``.readthedocs.yaml``. After you **import the project** on Read the Docs and point it at this repository, builds install the package plus ``docs/requirements-rtd.txt`` (Sphinx only) and run ``sphinx-build`` from ``docs/conf.py``.
