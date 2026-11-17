"""Vendored TextCraft-Synth world (platoon-textcraft 0.4.0, MIT).

- ``game.py``: the benchmark's standard-library world and episode rules
  (``RecipeBook`` + ``Game``), from ``harbor/game.py``.
- ``synth_recipe_generator.py``: the deterministic synthetic recipe world
  generator (seed 42, 25 items per domain/tier, 1,452 items), used only by
  ``scripts/make_textcraft_synth_split.py`` to build ``recipes.json``.

The frozen split manifests and ``recipes.json`` live under
``data/textcraft_synth/``. See ``LICENSE`` for the upstream MIT license.
"""
from jevmaster.envs.textcraft_synth.game import Game, RecipeBook  # noqa: F401

__all__ = ["Game", "RecipeBook"]
