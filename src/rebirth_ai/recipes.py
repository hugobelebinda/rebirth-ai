"""YAML training recipes (Phase 13): ``configs/recipes/<name>.yaml`` → ``train_lora.py`` defaults.

A recipe is a flat mapping of ``train_lora.py`` options (``lora_r``, ``quant``, ``epochs``, …;
dashes or underscores) plus optional ``name`` / ``description``. Explicit CLI flags always win:
the recipe only replaces argparse *defaults*. Unknown keys are an error (typos must not silently
train the wrong thing).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rebirth_ai.lora import ROOT

RECIPES_DIR = ROOT / "configs" / "recipes"
META_KEYS = {"name", "description"}


def resolve_recipe_path(recipe: str | Path) -> Path:
    p = Path(recipe)
    if p.is_file():
        return p
    cand = RECIPES_DIR / f"{recipe}.yaml"
    if cand.is_file():
        return cand
    raise FileNotFoundError(f"recipe not found: {recipe} (looked in {RECIPES_DIR})")


def load_recipe(recipe: str | Path, parser: argparse.ArgumentParser) -> tuple[dict, dict]:
    """Return (argparse defaults, metadata) for a recipe, validated against ``parser``."""
    import yaml

    path = resolve_recipe_path(recipe)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: recipe must be a mapping")
    actions = {a.dest: a for a in parser._actions}
    defaults: dict = {}
    meta = {"recipe_file": str(path), "recipe_name": raw.get("name") or path.stem}
    for key, val in raw.items():
        if key in META_KEYS:
            continue
        dest = key.replace("-", "_")
        if dest not in actions or dest in {"help", "recipe"}:
            raise ValueError(f"{path}: unknown option '{key}'")
        act = actions[dest]
        if act.type is Path and val is not None:
            val = Path(val)
        defaults[dest] = val
    return defaults, meta
