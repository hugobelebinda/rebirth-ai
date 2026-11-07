"""Backfill MLflow from what is already on disk (Phase 13).

Imports, idempotently:
  1. every trained adapter with a rebirth_meta.json (data/models/lora_*_7b, data/models/exp/*,
     v1 backups) → training run + registered model version;
  2. eval reports → eval runs, linked to model versions where the scored adapter is known:
       data/reports/eval_heldout_<lang>_{peft7b_v2,base7b_chat,ollama}.json   (24 shortest held-out)
       data/reports/eval_heldout_<lang>_peft7b_v2_cap64.json                   (64-token-cap bug run)
       data/reports/exp/<lang>_<name>.json                                     (frozen 100-row sets)
  3. first champions: each language's v2 adapter, if its gate file says it passed vs base 7B.
Experiment adapters are gated against the champion in dry-run mode only (reported, not promoted).
"""

from __future__ import annotations

import json
from pathlib import Path

from rich import print

from rebirth_ai.lora import LORA_LANGS, ROOT, lora_target, normalize_lang

MODELS = ROOT / "data" / "models"
REPORTS = ROOT / "data" / "reports"
EVALSETS = ROOT / "data" / "datasets" / "evalsets"


def _adapter_dirs() -> list[Path]:
    dirs = [d for d in sorted(MODELS.glob("lora_*")) if (d / "rebirth_meta.json").is_file()]
    dirs += [d for d in sorted((MODELS / "exp").glob("*")) if (d / "rebirth_meta.json").is_file()]
    return dirs


def _exp_report_model(lang: str, name: str) -> tuple[str, Path | None]:
    """data/reports/exp/<lang>_<name>.json → (model label, adapter dir or None)."""
    if name in {"v2", "A_v2"}:
        return "adapter", lora_target(lang).out
    if name.startswith("B_base"):
        return "base-7b-chat", None
    if name.startswith("O_ollama"):
        return "ollama-14b", None
    cand = MODELS / "exp" / f"{lang}_{name}"
    if (cand / "rebirth_meta.json").is_file():
        return "adapter", cand
    # e.g. D_all_r16_exp_e2_ep1ckpt370 → checkpoint of an exp adapter
    return f"exp:{name}", None


def _eval_jobs() -> list[dict]:
    jobs: list[dict] = []
    for lang in LORA_LANGS:
        out = lora_target(lang).out
        for suffix, model, adapter, tags in (
            ("peft7b_v2", "adapter", out, {}),
            ("base7b_chat", "base-7b-chat", None, {}),
            ("ollama", "ollama-14b", None, {}),
            ("peft7b_v2_cap64", "adapter-cap64-bug", None, {"rebirth.note": "SFT token floor 64 truncated outputs (fixed)"}),
        ):
            rep = REPORTS / f"eval_heldout_{lang}_{suffix}.json"
            if rep.is_file():
                jobs.append(dict(report=rep, model=model, lang=lang, adapter=adapter, eval_file=None, tags=tags))
    # Phase 11 history: the first (recipe v1, seq 128) Python LoRA and the base-7B SFT-prompt reference.
    v1 = MODELS / "lora_python_7b_v1"
    for name, model, adapter in (
        ("eval_heldout_python_peft7b.json", "adapter", v1 if (v1 / "rebirth_meta.json").is_file() else None),
        ("eval_heldout_python_base7b_sft.json", "base-7b-sft", None),
    ):
        if (REPORTS / name).is_file():
            jobs.append(dict(report=REPORTS / name, model=model, lang="python", adapter=adapter,
                             eval_file=None, tags={"rebirth.note": "recipe v1 era (seq 128, before decode fixes)"}))
    for rep in sorted((REPORTS / "exp").glob("*.json")):
        stem = rep.stem
        lang = next((lg for lg in sorted(LORA_LANGS, key=len, reverse=True) if stem.startswith(lg + "_")), None)
        if lang is None:
            continue
        model, adapter = _exp_report_model(lang, stem[len(lang) + 1 :])
        jobs.append(dict(report=rep, model=model, lang=lang, adapter=adapter,
                         eval_file=EVALSETS / f"{lang}.jsonl", tags={"rebirth.exp": stem}))
    return jobs


def run(*, dry_run: bool = False) -> dict:
    from rebirth_ai import registry, tracking

    summary = {"trained": [], "evals": [], "champions": [], "gates": []}
    for d in _adapter_dirs():
        rel = d.relative_to(ROOT)
        if dry_run:
            print(f"train  {rel}")
            continue
        run_id = tracking.log_training(d)
        summary["trained"].append((str(rel), run_id))
        print(f"[green]train[/green]  {rel} → {run_id[:8]}")
    for job in _eval_jobs():
        rel = job["report"].relative_to(ROOT)
        if dry_run:
            print(f"eval   {rel}  model={job['model']}  adapter={job['adapter']}")
            continue
        run_id = tracking.log_eval(job["report"], model_label=job["model"], lang=job["lang"],
                                   adapter=job["adapter"], eval_file=job["eval_file"], tags=job["tags"])
        summary["evals"].append((str(rel), run_id))
        print(f"[green]eval[/green]   {rel} → {run_id[:8]}")
    if dry_run:
        return summary

    # First champions: the v2 adapters that passed their Phase 11 gate vs base 7B.
    for lang in LORA_LANGS:
        gate_file = REPORTS / f"gate_{lang}_v2.json"
        ver = registry.find_version(lang, lora_target(lang).out)
        if ver is None or not gate_file.is_file():
            continue
        g = json.loads(gate_file.read_text(encoding="utf-8"))
        if g.get("pass") and registry.alias_version(lang, registry.CHAMPION) is None:
            registry.promote(lang, ver, reason="Phase 11 v2: passed gate vs base 7B (heldout 24, SFT floor 256)")
            summary["champions"].append((lang, ver))
            print(f"[bold green]champion[/bold green] {registry.model_name(lang)} → v{ver}")

    # Experiments vs champion on the frozen set — reported only (promotion is a deliberate step).
    for d in sorted((MODELS / "exp").glob("*")):
        meta_file = d / "rebirth_meta.json"
        if not meta_file.is_file():
            continue
        lang = normalize_lang(json.loads(meta_file.read_text(encoding="utf-8")).get("language") or "")
        ver = registry.find_version(lang, d) if lang else None
        if ver is None:
            continue
        set_ids = sorted({k[len("eval."):].rsplit(".", 1)[0]
                          for k in (tracking.client().get_model_version(registry.model_name(lang), ver).tags or {})
                          if k.startswith("eval.evalset:")})
        for set_id in set_ids:
            res = registry.gate(lang, ver, set_id, apply=False)
            summary["gates"].append(res)
            print(f"gate   {d.name} v{ver} vs champion v{res.get('champion')} on {set_id}: "
                  f"{res.get('decision')} {res.get('checks', '')}")
    return summary
