Fork features
==============

Visual presets (``gym_duckietown.scene_presets``)
--------------------------------------------------

* **``ARABIAN_DESERT_OUTDOOR``** — warm sky and sand-tinted horizon/ground; stable colors with ``domain_rand=False``.
* **``MINING_PIT_OUTDOOR``** — remaps large ``asphalt`` pads to ``floor`` textures, brown lane tints, pale horizon, sand/gravel floor scatter triangles.
* **``MINING_PIT_OUTDOOR_GL``** — same as mining plus **file-based** custom tile textures under ``assets/mining_generated/`` (run ``scripts/generate_mining_assets.py`` first). Environment ``Duckietown-zigzag_dists_mining_gl-v0``.
* **``MINING_PIT_OUTDOOR_PROG``** — same as mining plus **fully programmatic** RGB tile textures via ``custom_tile_textures`` / ``load_texture_from_rgba``. Environment ``Duckietown-zigzag_dists_mining_prog-v0``.

Simulator kwargs (see :class:`gym_duckietown.simulator.Simulator`)
-------------------------------------------------------------------

* ``texture_kind_remap`` — map YAML tile kind → atlas kind for texture lookup.
* ``custom_tile_texture_paths`` — per-kind paths to PNG/JPG on disk.
* ``custom_tile_textures`` — per-kind ``uint8`` array or ``callable(rng) -> array`` (takes precedence over paths).
* ``procedural_tile_texture_enforce_pot``, ``procedural_tile_texture_max_side`` — control in-memory upload sizing.

Multiview
---------

``env.unwrapped.render_multiview_rgb()`` returns a single RGB image with ego, map, top-follow, and rear panels. See ``scripts/demo_multiview.py`` and ``scripts/demo_mining_pit.py``.

Mining assets (offline)
-----------------------

``scripts/generate_mining_assets.py`` writes procedural PNGs and ``mining_pit_bowl.obj`` for Blender or external tools; pit mesh is **not** imported into the live tile renderer.

V&V
----

Requirement register (JSON): ``vnv/procedural_tiles_vnvspec.json``. Tests: ``tests/test_procedural_tile_textures.py``.
