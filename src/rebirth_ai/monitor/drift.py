"""Drift: is the code people feed the tool still like the code the adapters were trained on?

A **reference profile** per language is built once from the training inputs — the ``legacy``
side of the same-language corpus (``data/datasets/<lang>_migration.jsonl``) and the source
programs of the cross-language pool (``data/datasets/xlang/stage3/pool.jsonl``) — and saved to
``data/monitor/reference/<lang>.json``: length quantiles + bucket shares, and import frequencies.

Live inputs (the ``input`` profile in each run's telemetry, last ``window`` runs per language)
are compared with it:

- **length PSI** (population stability index over the reference's length buckets):
  ≥ 0.25 → ``warn``, 0.10–0.25 → ``info`` (the usual PSI reading);
- **very long inputs**: share above the reference's p99 length ≥ 10 % → ``warn``;
- **unseen imports**: share of live import mentions the reference never saw ≥ 30 % → ``warn``
  (the most common unseen ones are listed — candidates for new training data).

Nothing is checked below ``min_runs`` live inputs.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from rebirth_ai.monitor.imports import extract_imports, family
from rebirth_ai.monitor.store import Record
from rebirth_ai.paths import DATA

REFERENCE_DIR = DATA / "monitor" / "reference"
LANGS = ("js_ts", "python", "go", "java", "csharp", "rust", "cpp", "swift")
POOL_LANG = {"python": "python", "cpp": "cpp", "java": "java"}  # stage-3 pool reference solutions
# log-spaced character-length bucket edges: 0, 200, 400, 800, … 51,200, ∞
EDGES = [0] + [200 * 2**i for i in range(9)]


def _bucket_shares(lengths: list[int]) -> list[float]:
    counts = [0] * len(EDGES)
    for n in lengths:
        i = max(j for j, e in enumerate(EDGES) if n >= e)
        counts[i] += 1
    total = max(1, len(lengths))
    return [c / total for c in counts]


def _quantile(sorted_vals: list[int], q: float) -> int:
    if not sorted_vals:
        return 0
    return sorted_vals[min(len(sorted_vals) - 1, int(q * (len(sorted_vals) - 1) + 0.5))]


def profile(codes: list[str], lang: str, *, top_imports: int = 3000) -> dict[str, Any]:
    lengths = sorted(len(c) for c in codes)
    imports = Counter(i for c in codes for i in extract_imports(c, lang))
    return {
        "lang": lang,
        "rows": len(codes),
        "length_quantiles": {str(q): _quantile(lengths, q) for q in (0.05, 0.25, 0.5, 0.75, 0.95, 0.99)},
        "bucket_edges": EDGES,
        "bucket_shares": [round(x, 5) for x in _bucket_shares(lengths)],
        "imports": dict(imports.most_common(top_imports)),
        "rows_with_imports": sum(1 for c in codes if extract_imports(c, lang)),
    }


def _read_jsonl(path: Path):
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def reference_inputs(lang: str, root: Path = Path(".")) -> list[str]:
    """Training inputs for ``lang``: corpus ``legacy`` sides + stage-3 pool sources."""
    codes: list[str] = []
    corpus = root / "data" / "datasets" / f"{lang}_migration.jsonl"
    if corpus.is_file():
        codes.extend(r["legacy"] for r in _read_jsonl(corpus) if r.get("legacy"))
    pool = root / "data" / "datasets" / "xlang" / "stage3" / "pool.jsonl"
    if lang in POOL_LANG and pool.is_file():
        for r in _read_jsonl(pool):
            if r.get("src_lang") == POOL_LANG[lang] and r.get("source"):
                codes.append(r["source"])
    return codes


def build_reference(lang: str, root: Path = Path("."), out_dir: Path | None = None) -> Path:
    out = (out_dir or root / REFERENCE_DIR) / f"{lang}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    prof = profile(reference_inputs(lang, root), lang)
    out.write_text(json.dumps(prof, indent=1) + "\n", encoding="utf-8")
    return out


def load_reference(lang: str, ref_dir: Path = REFERENCE_DIR) -> dict[str, Any] | None:
    p = Path(ref_dir) / f"{lang}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


def psi(expected: list[float], actual: list[float], eps: float = 1e-4) -> float:
    return sum((a - e) * math.log((a + eps) / (e + eps)) for e, a in zip(expected, actual, strict=True))


def check_lang(lang: str, live: list[dict[str, Any]], ref: dict[str, Any], *, min_runs: int = 20) -> list[dict[str, Any]]:
    """``live`` = input profiles (``chars``, ``imports``) of recent runs whose source is ``lang``."""
    if len(live) < min_runs:
        return [{"level": "info", "kind": "too_few_runs", "lang": lang,
                 "message": f"{lang}: {len(live)} live input(s) (< {min_runs}); drift not checked yet"}]
    out: list[dict[str, Any]] = []
    shares = _bucket_shares([p["chars"] for p in live])
    v = psi(ref["bucket_shares"], shares)
    if v >= 0.10:
        out.append({"level": "warn" if v >= 0.25 else "info", "kind": "length_psi", "lang": lang, "psi": round(v, 3),
                    "message": f"{lang}: input length distribution shifted vs training (PSI {v:.2f}; median live "
                               f"{sorted(p['chars'] for p in live)[len(live) // 2]} chars vs {ref['length_quantiles']['0.5']} in training)"})
    p99 = ref["length_quantiles"]["0.99"]
    long_share = sum(1 for p in live if p["chars"] > p99) / len(live)
    if long_share >= 0.10:
        out.append({"level": "warn", "kind": "long_inputs", "lang": lang, "share": round(long_share, 3),
                    "message": f"{lang}: {100 * long_share:.0f} % of live inputs are longer than 99 % of training inputs (> {p99} chars)"})
    mentions = [i for p in live for i in p.get("imports") or []]
    if mentions:
        known = ref["imports"]
        unseen = Counter(i for i in mentions if i not in known)
        share = sum(unseen.values()) / len(mentions)
        if share >= 0.30:
            top = ", ".join(f"{i} ×{n}" for i, n in unseen.most_common(5))
            out.append({"level": "warn", "kind": "unseen_imports", "lang": lang, "share": round(share, 3),
                        "top_unseen": unseen.most_common(10),
                        "message": f"{lang}: {100 * share:.0f} % of imports in live inputs never appear in training ({top})"})
    return out


def check(records: list[Record], *, ref_dir: Path = REFERENCE_DIR, window: int = 50, min_runs: int = 20) -> list[dict[str, Any]]:
    by_lang: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        prof = r.t.get("input")
        lang = family(r.t.get("source_lang") or r.t.get("lang"))
        if isinstance(prof, dict) and lang:
            by_lang.setdefault(lang, []).append(prof)
    out: list[dict[str, Any]] = []
    for lang, profs in sorted(by_lang.items()):
        ref = load_reference(lang, ref_dir)
        if ref is None:
            out.append({"level": "info", "kind": "no_reference", "lang": lang,
                        "message": f"{lang}: no reference profile — run `rebirth monitor reference`"})
            continue
        out.extend(check_lang(lang, profs[-window:], ref, min_runs=min_runs))
    return out
