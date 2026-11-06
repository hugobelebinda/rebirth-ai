"""Model registry (Phase 13): one registered model per language, adapters as versions.

    rebirth-<lang>            registered model (e.g. rebirth-go)
      version N               one trained adapter (7B or 14B), source = runs:/<train run>/adapter
        tags                  base, adapter path, trained_at, recipe, eval.<set>.<metric> scores
      alias "champion"        the adapter ``rebirth migrate --backend peft`` should serve
      alias "previous"        the champion before the last promotion (rollback target)
      alias "challenger"      newest candidate not yet promoted

Promotion is a gate, not a judgement call: a challenger replaces the champion only if, on the
**same** eval set, soft-exact >= champion + ``min_gain`` (default 1 — a tie keeps the incumbent:
no churn to a bigger/slower adapter for zero gain) and parses >= champion - 1
(``training/lora_gate.py:verdict``). Decisions are written as version tags.
"""

from __future__ import annotations

from pathlib import Path

from rebirth_ai.lora import ROOT, normalize_lang, read_meta, rel_to_root
from rebirth_ai.tracking import base_label, client

CHAMPION = "champion"
PREVIOUS = "previous"
CHALLENGER = "challenger"


def model_name(lang: str) -> str:
    return f"rebirth-{normalize_lang(lang)}"


def _ensure_model(name: str) -> None:
    c = client()
    try:
        c.get_registered_model(name)
    except Exception:  # noqa: BLE001 — RESOURCE_DOES_NOT_EXIST
        c.create_registered_model(
            name, description=f"Rebirth AI LoRA adapters for {name.removeprefix('rebirth-')}"
        )


def _versions(lang: str) -> list:
    c = client()
    return list(c.search_model_versions(f"name = '{model_name(lang)}'"))


def find_version(lang: str, out_dir: Path) -> str | None:
    """Registered version for this adapter dir (latest training of it), or None."""
    meta = read_meta(Path(out_dir)) or {}
    ident = f"{rel_to_root(Path(out_dir).resolve())}@{meta.get('trained_at')}"
    for v in _versions(lang):
        if (v.tags or {}).get("adapter_id") == ident:
            return str(v.version)
    return None


def register_adapter(out_dir: Path, train_run_id: str, *, meta: dict | None = None) -> str:
    """Register an adapter as a new version (idempotent) and mark it challenger."""
    out_dir = Path(out_dir).resolve()
    meta = meta or read_meta(out_dir) or {}
    lang = normalize_lang(meta.get("language") or "")
    if not lang:
        raise ValueError(f"{out_dir}: rebirth_meta.json has no language")
    existing = find_version(lang, out_dir)
    if existing:
        return existing
    name = model_name(lang)
    _ensure_model(name)
    hp = meta.get("hyperparams") or {}
    recipe = meta.get("recipe") or {}
    tags = {
        "base": base_label(meta.get("base_model")),
        "base_model": str(meta.get("base_model")),
        "adapter": rel_to_root(out_dir),
        "adapter_id": f"{rel_to_root(out_dir)}@{meta.get('trained_at')}",
        "trained_at": str(meta.get("trained_at")),
        "train_loss": str(meta.get("train_loss")),
        "recipe": (
            f"quant={recipe.get('quant', 'none')} r={hp.get('lora_r')} "
            f"targets={hp.get('lora_targets', 'attn')} seq={hp.get('max_length')} "
            f"epochs={hp.get('epochs')} rows={meta.get('train_rows')}"
        ),
        "data": str(meta.get("data")),
    }
    weights = out_dir / "adapter" / "adapter_model.safetensors"
    if weights.is_file():
        from rebirth_ai.tracking import file_sha

        tags["adapter_sha256"] = file_sha(weights)
    c = client()
    v = c.create_model_version(
        name, f"runs:/{train_run_id}/adapter", run_id=train_run_id, tags=tags,
        description=f"{tags['adapter']} — {tags['recipe']}",
    )
    c.set_registered_model_alias(name, CHALLENGER, str(v.version))
    return str(v.version)


def record_eval(lang: str, version: str, set_id: str, metrics: dict, *, eval_run_id: str | None = None) -> None:
    c = client()
    name = model_name(lang)
    for k in ("n", "soft_exact", "parses", "mean_symbol_jaccard", "mean_token_overlap"):
        if k in metrics:
            c.set_model_version_tag(name, version, f"eval.{set_id}.{k}", f"{metrics[k]:g}")
    if eval_run_id:
        c.set_model_version_tag(name, version, f"eval.{set_id}.run", eval_run_id)


def version_scores(lang: str, version: str, set_id: str) -> dict | None:
    v = client().get_model_version(model_name(lang), version)
    prefix = f"eval.{set_id}."
    scores = {k[len(prefix):]: v.tags[k] for k in (v.tags or {}) if k.startswith(prefix)}
    if "soft_exact" not in scores or "parses" not in scores:
        return None
    out = {}
    for k, val in scores.items():
        if k == "run":
            continue
        try:
            out[k] = float(val)
        except ValueError:
            pass
    return out


def alias_version(lang: str, alias: str = CHAMPION) -> str | None:
    try:
        return str(client().get_model_version_by_alias(model_name(lang), alias).version)
    except Exception:  # noqa: BLE001 — no such alias / model
        return None


def champion_adapter(lang: str) -> Path | None:
    """Adapter dir of the current champion (for routing), or None."""
    ver = alias_version(lang, CHAMPION)
    if ver is None:
        return None
    adapter = (client().get_model_version(model_name(lang), ver).tags or {}).get("adapter")
    return (ROOT / adapter / "adapter") if adapter else None


def promote(lang: str, version: str, *, reason: str) -> None:
    """Make ``version`` the champion (old champion → ``previous``)."""
    c = client()
    name = model_name(lang)
    version = str(version)
    old = alias_version(lang, CHAMPION)
    if old == version:
        return
    if old is not None:
        c.set_registered_model_alias(name, PREVIOUS, old)
        c.set_model_version_tag(name, old, "retired_by", version)
    c.set_registered_model_alias(name, CHAMPION, version)
    c.set_model_version_tag(name, version, "promoted_reason", reason[:500])
    if alias_version(lang, CHALLENGER) == version:
        c.delete_registered_model_alias(name, CHALLENGER)


def rollback(lang: str) -> str:
    """Swap champion and previous. Returns the new champion version."""
    prev = alias_version(lang, PREVIOUS)
    cur = alias_version(lang, CHAMPION)
    if prev is None:
        raise ValueError(f"{model_name(lang)} has no previous champion to roll back to")
    c = client()
    name = model_name(lang)
    c.set_registered_model_alias(name, CHAMPION, prev)
    if cur is not None:
        c.set_registered_model_alias(name, PREVIOUS, cur)
    c.set_model_version_tag(name, prev, "promoted_reason", f"rollback from v{cur}")
    return prev


def gate(
    lang: str,
    challenger: str,
    set_id: str,
    *,
    parse_slack: int = 1,
    min_gain: int = 1,
    apply: bool = True,
) -> dict:
    """Challenger vs champion on ``set_id``. Promotes on pass when ``apply``.

    With no champion yet, the challenger is promoted only via ``promote`` (first champion is a
    deliberate decision, e.g. "passed its gate vs base 7B").
    """
    import sys

    sys.path.insert(0, str(ROOT / "training"))
    from lora_gate import verdict

    challenger = str(challenger)
    champ = alias_version(lang, CHAMPION)
    result: dict = {"lang": lang, "set": set_id, "challenger": challenger, "champion": champ}
    if champ is None:
        result.update(decision="no_champion", passed=None)
        return result
    if champ == challenger:
        result.update(decision="already_champion", passed=True)
        return result
    cs, hs = version_scores(lang, challenger, set_id), version_scores(lang, champ, set_id)
    if cs is None or hs is None:
        missing = [f"v{v}" for v, s in ((challenger, cs), (champ, hs)) if s is None]
        result.update(decision="needs_eval", passed=None, missing=missing)
        return result
    v = verdict(cs, hs, parse_slack=parse_slack)
    gain_ok = cs["soft_exact"] >= hs["soft_exact"] + min_gain
    v["checks"][f"soft_exact>=champion+{min_gain}"] = gain_ok
    v["checks"].pop("soft_exact>=base", None)
    v["checks"] = {k.replace("base", "champion"): ok for k, ok in v["checks"].items()}
    v["pass"] = bool(gain_ok and all(ok for k, ok in v["checks"].items() if k.startswith("parses")))
    result.update(passed=v["pass"], checks=v["checks"], challenger_scores=v["adapter"], champion_scores=v["base"])
    c = client()
    name = model_name(lang)
    c.set_model_version_tag(name, challenger, f"gate.{set_id}", "pass" if v["pass"] else "fail")
    if v["pass"] and apply:
        promote(lang, challenger, reason=f"gate pass vs v{champ} on {set_id}")
        result["decision"] = "promoted"
    else:
        result["decision"] = "kept_champion" if not v["pass"] else "pass_not_applied"
    return result
