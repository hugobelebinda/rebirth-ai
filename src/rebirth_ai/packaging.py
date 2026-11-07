"""Package registry champions as versioned Ollama models (Phase 16).

    rebirth package --lang go            # champion → GGUF → rebirth-go:7b-v<N> → serving gate
                                         #   → rebirth-go:7b alias + configs/routing.yaml + model card

Steps per language:

1. **Version**: the registry champion (or ``--version``); its adapter dir comes from the version's
   ``adapter`` tag.
2. **GGUF**: ``llama.cpp/convert_lora_to_gguf.py`` (pinned checkout in ``.tools/llama.cpp``),
   f16, ~10 MB at r8. Cached per version + adapter sha256.
3. **Ollama model**: ``FROM <quantized base>`` + ``ADAPTER``; raw ``{{ .Prompt }}`` template and the
   SFT stop strings (the adapters never saw a chat template). Tag ``rebirth-<lang>:<size>-v<N>``.
4. **Serving gate**: the *served* model is scored on the frozen eval set through
   ``OllamaAdapterClient`` (the same client ``rebirth migrate`` uses). It passes if
   - fidelity: soft-exact ≥ PEFT score − ``FIDELITY_SLACK`` and parses ≥ PEFT parses − ``FIDELITY_SLACK``
     (quantized base + GGUF must reproduce what was validated), and
   - it still beats the base model: soft-exact ≥ base 7B soft-exact (when a base reference exists).
   Parse regressions vs base are reported as warnings.
5. **Publish** (only on pass): ``ollama cp`` the tag to the serving alias ``rebirth-<lang>:<size>``,
   write the route to ``configs/routing.yaml``, write ``docs/models/rebirth-<lang>-<size>.md``.

Scores go to MLflow (``rebirth-eval``) and onto the registry version under
``served:<label>:<set>`` — never over the PEFT score the promotion gate uses.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from rebirth_ai.lora import ROOT, normalize_lang, read_meta, rel_to_root

LLAMA_CPP = ROOT / ".tools" / "llama.cpp"
LLAMA_CPP_REPO = "https://github.com/ggml-org/llama.cpp.git"
GGUF_DIR = ROOT / "data" / "models" / "gguf"
OLLAMA_DIR = ROOT / "data" / "models" / "ollama"
REPORT_DIR = ROOT / "data" / "reports" / "served"
CARD_DIR = ROOT / "docs" / "models"
FIDELITY_SLACK = 2

# (base size, quant) → Ollama base tag. q4 = Q4_K_M, the size an offline install ships.
OLLAMA_BASES = {
    ("7b", "q4"): "qwen2.5-coder:7b-instruct",
    ("7b", "q8"): "qwen2.5-coder:7b-instruct-q8_0",
    ("14b", "q4"): "qwen2.5-coder:14b-instruct-q4_K_M",
    ("14b", "q6"): "qwen2.5-coder:14b-instruct-q6_K",
}
SFT_STOPS = ("\n### Objective", "\n### Legacy", "\n### Modern", "\n###", "\n```")


def ollama_name(lang: str) -> str:
    """Ollama model name: ``rebirth-go``, ``rebirth-js-ts`` (no underscores in names)."""
    return f"rebirth-{normalize_lang(lang).replace('_', '-')}"


def serving_alias(lang: str, size: str) -> str:
    return f"{ollama_name(lang)}:{size}"


def version_tag(lang: str, size: str, version: str) -> str:
    return f"{ollama_name(lang)}:{size}-v{version}"


def modelfile(base_tag: str, gguf: Path) -> str:
    stops = "".join(f'PARAMETER stop "{s}"\n' for s in SFT_STOPS)
    return (
        f"FROM {base_tag}\n"
        f"ADAPTER {gguf}\n"
        'TEMPLATE """{{ .Prompt }}"""\n'
        "PARAMETER temperature 0\n"
        "PARAMETER num_ctx 4096\n"
        f"{stops}"
    )


@dataclass
class ServeGate:
    passed: bool
    checks: dict[str, bool]
    warnings: list[str] = field(default_factory=list)


def serve_gate(served: dict, peft: dict | None, base: dict | None, *, slack: int = FIDELITY_SLACK) -> ServeGate:
    checks: dict[str, bool] = {}
    warnings: list[str] = []
    if peft is not None:
        checks[f"soft_exact>=peft-{slack}"] = served["soft_exact"] >= peft["soft_exact"] - slack
        checks[f"parses>=peft-{slack}"] = served["parses"] >= peft["parses"] - slack
    else:
        warnings.append("no PEFT score on this eval set — fidelity not checked")
    if base is not None:
        checks["soft_exact>=base"] = served["soft_exact"] >= base["soft_exact"]
        if served["parses"] < base["parses"] - 1:
            warnings.append(f"parses {served['parses']:g} < base {base['parses']:g} − 1")
    else:
        warnings.append("no base-model reference on this eval set")
    return ServeGate(passed=bool(checks) and all(checks.values()), checks=checks, warnings=warnings)


# --------------------------------------------------------------------------- steps


def _run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, text=True, capture_output=True, **kw)


def llama_cpp_commit() -> str | None:
    if not (LLAMA_CPP / ".git").exists():
        return None
    return _run(["git", "-C", str(LLAMA_CPP), "rev-parse", "--short", "HEAD"]).stdout.strip()


def hf_snapshot(base_model: str) -> Path:
    """Local HF snapshot dir of the base (only config + tokenizer are read by the converter)."""
    from huggingface_hub import try_to_load_from_cache

    cfg = try_to_load_from_cache(base_model, "config.json")
    if not isinstance(cfg, str):
        raise FileNotFoundError(f"{base_model} is not in the local HF cache (config.json missing)")
    return Path(cfg).parent


def convert_to_gguf(adapter_dir: Path, base_model: str, out: Path) -> Path:
    """PEFT safetensors adapter → f16 GGUF LoRA (CPU only; cached next to a sha sidecar)."""
    from rebirth_ai.tracking import file_sha

    script = LLAMA_CPP / "convert_lora_to_gguf.py"
    if not script.is_file():
        raise FileNotFoundError(
            f"{script} missing — clone it once: git clone --depth 1 {LLAMA_CPP_REPO} {LLAMA_CPP}"
        )
    sha = file_sha(adapter_dir / "adapter_model.safetensors")
    side = out.with_suffix(".json")
    if out.is_file() and side.is_file() and json.loads(side.read_text()).get("adapter_sha256") == sha:
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONPATH": str(LLAMA_CPP / "gguf-py"), "HF_HUB_OFFLINE": "1",
           "CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": ""}
    _run([sys.executable, str(script), "--base", str(hf_snapshot(base_model)), "--outtype", "f16",
          "--outfile", str(out), str(adapter_dir)], env=env)
    side.write_text(json.dumps({"adapter": rel_to_root(adapter_dir), "adapter_sha256": sha,
                                "base_model": base_model, "llama_cpp": llama_cpp_commit()}, indent=2) + "\n")
    return out


def ollama_create(tag: str, base_tag: str, gguf: Path) -> Path:
    d = OLLAMA_DIR / tag.replace(":", "_")
    d.mkdir(parents=True, exist_ok=True)
    mf = d / "Modelfile"
    mf.write_text(modelfile(base_tag, gguf), encoding="utf-8")
    _run(["ollama", "create", tag, "-f", str(mf)])
    return mf


def ollama_has(tag: str) -> bool:
    return subprocess.run(["ollama", "show", tag], capture_output=True, check=False).returncode == 0


def ollama_alias(src: str, dst: str) -> None:
    _run(["ollama", "cp", src, dst])


def ollama_unload(tag: str) -> None:
    subprocess.run(["ollama", "stop", tag], capture_output=True, check=False)


def serve_eval(tag: str, lang: str, eval_file: Path, out: Path, *, n: int = 100) -> dict:
    """Score the served Ollama model on a frozen eval set with the production client."""
    sys.path.insert(0, str(ROOT / "training"))
    from eval_behavioral import score_row, summarize

    from rebirth_ai.generate.ollama_adapter_client import OllamaAdapterClient
    from rebirth_ai.lora import load_rows

    rows = load_rows(eval_file)[:n]
    client = OllamaAdapterClient(model=tag, language=lang)
    results = []
    for i, row in enumerate(rows, 1):
        pred = client.migrate(objective=row["objective"], legacy_code=row["legacy"],
                              file_path=row.get("path") or "snippet.js",
                              language=row.get("language", "javascript"))
        results.append(score_row(i, row, pred.code, total=len(rows)))
    suite = summarize(results, suite=f"evalset:{eval_file.stem}", backend="ollama-adapter",
                      lang=lang, model=f"ollama:{tag}")
    report = {"backend": "ollama-adapter", "suites": {"synthetic": suite}}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def _scores(suite: dict) -> dict:
    return {k: suite[k] for k in ("n", "soft_exact", "parses", "mean_symbol_jaccard", "mean_token_overlap")}


def _base_reference(lang: str) -> dict | None:
    p = ROOT / "data" / "reports" / "exp" / f"{lang}_B_base_chat.json"
    if not p.is_file():
        return None
    return _scores(json.loads(p.read_text())["suites"]["synthetic"])


# --------------------------------------------------------------------------- model card


def write_model_card(lang: str, size: str, version: str, *, entry: dict, meta: dict) -> Path:
    hp = meta.get("hyperparams") or {}
    peft, served, base = entry.get("peft") or {}, entry.get("served") or {}, entry.get("base") or {}

    def row(name: str, s: dict) -> str:
        if not s:
            return f"| {name} | — | — | — | — |\n"
        return (f"| {name} | {s['soft_exact']:g} / {s['n']:g} | {s['parses']:g} / {s['n']:g} | "
                f"{s.get('mean_symbol_jaccard', 0):.3f} | {s.get('mean_token_overlap', 0):.3f} |\n")

    text = (
        f"# {serving_alias(lang, size)} (registry `rebirth-{lang}` v{version})\n\n"
        f"> Generated by `rebirth package` on {entry['packaged_at']}. Do not edit by hand.\n\n"
        "## What it does\n\n"
        f"Rewrites legacy **{lang}** code into modern idioms of the same language (one file or snippet "
        "at a time, inside Rebirth's generate → test → retry loop).\n\n"
        "## Artifact\n\n"
        f"| | |\n|---|---|\n| Ollama tag | `{entry['tag']}` (alias `{entry['model']}`) |\n"
        f"| Base | `{entry['base_tag']}` (from `{meta.get('base_model')}`) |\n"
        f"| Adapter | LoRA r{hp.get('lora_r')} α{hp.get('lora_alpha')} on `{hp.get('lora_targets', 'attn')}`, "
        f"GGUF f16 `{entry['gguf']}` |\n"
        f"| Adapter sha256 | `{entry.get('adapter_sha256', '?')}` |\n"
        f"| Converter | llama.cpp `{entry.get('llama_cpp')}` |\n\n"
        "## Training\n\n"
        f"| | |\n|---|---|\n| Data | `{meta.get('data')}` ({meta.get('train_rows')} rows) |\n"
        f"| Recipe | {meta.get('recipe_name') or 'v2'}: seq {hp.get('max_length')}, {hp.get('epochs')} epoch(s), "
        f"lr {hp.get('learning_rate')}, completion-only loss |\n"
        f"| Trained | {meta.get('trained_at')} (train loss {meta.get('train_loss')}) |\n"
        "| Data license | CommitPackFT rows from permissively licensed repos only (filter in "
        "`training/mine_migration_pairs.py`) |\n\n"
        f"## Evaluation — frozen set `{entry.get('eval_set')}`\n\n"
        "| Model | Soft-exact | Parses | Symbol J. | Token overlap |\n|---|---:|---:|---:|---:|\n"
        + row("Base 7B (chat, no adapter)", base)
        + row("Adapter, PEFT bf16 (validated)", peft)
        + row(f"**Served** (`{entry['tag']}`)", served)
        + "\nServing gate: " + ("**pass**" if entry["gate"]["passed"] else "**fail**")
        + " — " + ", ".join(f"{k} {'✓' if v else '✗'}" for k, v in entry["gate"]["checks"].items())
        + ("".join(f"\n- ⚠️ {w}" for w in entry["gate"]["warnings"]))
        + "\n\n## Limitations\n\n"
        "- Trained on single-file commits ≤ 512 tokens; long files go through the chat prompt path, "
        "which the frozen set does not measure.\n"
        "- Soft-exact compares against one human commit; other valid rewrites score as misses.\n"
        "- Same-language modernization only (cross-language is Phase 17).\n"
    )
    CARD_DIR.mkdir(parents=True, exist_ok=True)
    p = CARD_DIR / f"{ollama_name(lang)}-{size}.md"
    p.write_text(text, encoding="utf-8")
    return p


# --------------------------------------------------------------------------- orchestration


def package(lang: str, *, version: str | None = None, quant: str = "q4", gate: bool = True,
            publish: bool = True, n: int = 100, log=print) -> dict:
    from rebirth_ai import registry
    from rebirth_ai.tracking import client, eval_set_id, file_sha, try_log_eval

    lang = normalize_lang(lang)
    if lang is None:
        raise ValueError("unknown language")
    version = str(version or registry.alias_version(lang, registry.CHAMPION) or "")
    if not version:
        raise ValueError(f"rebirth-{lang} has no champion; pass --version")
    tags = client().get_model_version(registry.model_name(lang), version).tags or {}
    out_dir = ROOT / tags["adapter"]
    adapter_dir = out_dir / "adapter"
    meta = read_meta(out_dir) or {}
    size = tags.get("base", "7b")
    base_tag = OLLAMA_BASES[(size, quant)]
    tag = version_tag(lang, size, version)

    log(f"[package] {lang} v{version} ({rel_to_root(out_dir)}) → {tag} on {base_tag}")
    gguf = convert_to_gguf(adapter_dir, meta.get("base_model") or tags["base_model"],
                           GGUF_DIR / f"{lang}_{size}_v{version}.f16.gguf")
    if subprocess.run(["ollama", "show", base_tag], capture_output=True, check=False).returncode != 0:
        log(f"[package] pulling {base_tag}")
        _run(["ollama", "pull", base_tag])
    ollama_create(tag, base_tag, gguf)
    entry = {
        "model": serving_alias(lang, size), "tag": tag, "version": version, "base_tag": base_tag,
        "quant": quant, "gguf": rel_to_root(gguf), "adapter": rel_to_root(adapter_dir),
        "adapter_sha256": file_sha(adapter_dir / "adapter_model.safetensors"),
        "llama_cpp": llama_cpp_commit(),
        "packaged_at": _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
    }
    if not gate:
        log(f"[package] {tag} created (no serving gate, not published)")
        return entry

    eval_file = ROOT / "data" / "datasets" / "evalsets" / f"{lang}.jsonl"
    rp = REPORT_DIR / f"{lang}_{size}_v{version}_{quant}.json"
    log(f"[package] serving gate: scoring {tag} on {rel_to_root(eval_file)}")
    report = serve_eval(tag, lang, eval_file, rp, n=n)
    ollama_unload(tag)
    served = _scores(report["suites"]["synthetic"])
    set_id = eval_set_id(report, eval_file)
    label = f"served-{size}-{quant}"
    try_log_eval(rp, model_label=label, lang=lang, adapter=adapter_dir, eval_file=eval_file,
                 tags={"rebirth.engine": "ollama", "rebirth.ollama_tag": tag})
    peft = registry.version_scores(lang, version, set_id)
    base = _base_reference(lang)
    g = serve_gate(served, peft, base)
    entry.update({"eval_set": set_id, "served": served, "peft": peft, "base": base,
                  "gate": {"passed": g.passed, "checks": g.checks, "warnings": g.warnings}})
    c = client()
    c.set_model_version_tag(registry.model_name(lang), version, f"served.{quant}.tag", tag)
    c.set_model_version_tag(registry.model_name(lang), version, f"served.{quant}.gate",
                            "pass" if g.passed else "fail")
    log(f"[package] served {served['soft_exact']:g}/{served['n']:g} parses {served['parses']:g} | "
        f"peft {peft and peft['soft_exact']} | base {base and base['soft_exact']} → "
        f"{'PASS' if g.passed else 'FAIL'} {g.checks} {g.warnings or ''}")
    if g.passed and publish:
        publish_route(lang, entry, meta=meta)
    return entry


def publish_route(lang: str, entry: dict, *, meta: dict | None = None) -> None:
    """Point the serving alias at ``entry['tag']``, record the route and the model card."""
    from rebirth_ai import routing

    cur = routing.route(lang)
    hist = dict((cur or {}).get("history") or {})
    hist.update(entry.get("history") or {})
    if cur and cur.get("version") != entry["version"]:
        hist[cur["version"]] = _strip_history(cur)       # rollback target
    hist.pop(entry["version"], None)
    entry = {**_strip_history(entry), **({"history": hist} if hist else {})}
    ollama_alias(entry["tag"], entry["model"])
    routing.set_route(lang, entry)
    size = entry["model"].split(":")[-1]
    if meta is None:
        meta = read_meta(ROOT / Path(entry["adapter"]).parent) or {}
    write_model_card(lang, size, entry["version"], entry=entry, meta=meta)


def serve_rollback(lang: str, log=print) -> dict:
    """Registry rollback (champion ↔ previous) + re-point the serving alias to that version."""
    from rebirth_ai import registry, routing

    lang = normalize_lang(lang)
    cur = routing.route(lang)
    target = registry.alias_version(lang, registry.PREVIOUS)
    if target is None:
        raise RuntimeError(f"{registry.model_name(lang)} has no previous champion to roll back to")
    entry = ((cur or {}).get("history") or {}).get(target)
    if entry is None or not ollama_has(entry["tag"]):  # check first: registry and serving stay in sync
        raise RuntimeError(
            f"previous champion v{target} has no passing packaged build — "
            f"run `rebirth package --lang {lang} --version {target}` first"
        )
    new_ver = registry.rollback(lang)
    publish_route(lang, dict(entry))
    log(f"[serve] {lang}: {entry['model']} → {entry['tag']} (v{new_ver})")
    return entry


def _strip_history(entry: dict) -> dict:
    return {k: v for k, v in entry.items() if k != "history"}
