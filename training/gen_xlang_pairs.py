#!/usr/bin/env python3
"""Phase 17 stage 3: verified cross-language training pairs from CodeContests *train* programs.

  pool      download the train split (39 parquet files, ~7.5 GB → data/cache/code_contests/train/),
            pick one reference solution per problem (Python 3 / C++ / Java, rotating) that passes 10
            of the problem's tests in our sandbox → data/datasets/xlang/stage3/pool.jsonl (CPU only;
            resumable per shard)
  generate  Ollama 14B translates each pool program into the 7 other targets with the product prompt
            and the product retry loop (``migrate --to``: compiler output / first failing test fed
            back, 2 retries, temperature 0.1 → 0.4 → 0.7). Generations → stage3/gens.jsonl, verdicts
            → stage3/verdicts.jsonl (GPU; resumable; runs until the pool is used up or it is stopped)
  pairs     verified versions of one program → rows for the shared adapter (stage3/pairs.jsonl +
            .stats.json): forward (reference → translation), reverse (translation → reference) and
            cross (translation → translation of the same reference). JavaScript sources come from
            passing TypeScript translations with the types stripped, re-checked on the tests.

Verification = the problem's own judge tests (same judge as the program-level eval). Leak guard:
problems whose title or statement matches a CodeContests valid / test problem (the eval suite) are
skipped, and so are repeated statements inside train.

  python training/gen_xlang_pairs.py pool                 # CPU, ~2–3 h for all 39 shards
  python training/gen_xlang_pairs.py generate             # GPU (systemd unit rebirth-gpu-job)
  python training/gen_xlang_pairs.py pairs                # CPU, any time
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "training"))

import build_xlang_programs as bxp  # noqa: E402

from rebirth_ai.xlang.fixups import apply_fixups  # noqa: E402
from rebirth_ai.xlang.languages import get_lang  # noqa: E402
from rebirth_ai.xlang.prompts import build_translation_prompt, build_xlang_sft_prompt, extract_code  # noqa: E402
from rebirth_ai.xlang.translator import get_translator, max_new_tokens_for, retry_temperature  # noqa: E402
from rebirth_ai.xlang.verify import run_io_programs, ts_to_js  # noqa: E402

TRAIN_DIR = bxp.CACHE / "train"
HF_TREE = "https://huggingface.co/api/datasets/deepmind/code_contests/tree/main/data"
HF_FILE = "https://huggingface.co/datasets/deepmind/code_contests/resolve/main/"
STAGE3 = ROOT / "data" / "datasets" / "xlang" / "stage3"
POOL = STAGE3 / "pool.jsonl"
POOL_DONE = STAGE3 / "pool.shards_done"
POOL_COMPLETE = STAGE3 / "pool.complete"
POOL_STATS = STAGE3 / "pool.stats.json"
GENS = STAGE3 / "gens.jsonl"
VERDICTS = STAGE3 / "verdicts.jsonl"
PAIRS = STAGE3 / "pairs.jsonl"
JS_CACHE = STAGE3 / "js_from_ts.jsonl"

SRC_LANGS = ["python", "cpp", "java"]
TARGETS = ["python", "typescript", "go", "java", "csharp", "rust", "cpp", "swift"]
PLATFORM = {0: "unknown", 1: "codechef", 2: "codeforces", 3: "hackerearth", 4: "codejam", 5: "atcoder", 6: "aizu"}
COLS = ["name", "description", "source", "cf_contest_id", "cf_index", "cf_rating", "cf_tags",
        "public_tests", "private_tests", "generated_tests", "solutions", "input_file", "output_file"]
VAL_PERCENT = 3  # problems held out of training (by id hash) to watch the adapter's val loss


def _h(text: str) -> int:
    return int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:12], 16)


def _jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    # split on "\n" only: str.splitlines() also breaks on U+2028, \x1c, \x85 … which appear unescaped
    # inside test inputs (rows are written with ensure_ascii=False)
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").split("\n") if ln.strip()]


def _append(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


# ----------------------------------------------------------------------------------------- pool

_CF_PREFIX = re.compile(r"^\s*\d+_[A-Z]\d*\.\s*")


def title_key(name: str) -> str:
    """'1566_A. Median Maximization' → 'median maximization' (Div. 1 / Div. 2 copies share it)."""
    return re.sub(r"\s+", " ", _CF_PREFIX.sub("", str(name))).strip().lower()


def statement_key(desc: str) -> int:
    return _h(re.sub(r"\s+", " ", str(desc)).strip().lower()[:400])


def eval_keys() -> tuple[set[str], set[int]]:
    import pandas as pd

    titles, statements = set(), set()
    for f in bxp.FILES:
        df = pd.read_parquet(bxp.CACHE / f, columns=["name", "description"])
        titles |= {title_key(n) for n in df["name"]}
        statements |= {statement_key(d) for d in df["description"]}
    return titles, statements


def train_shards() -> list[str]:
    with urllib.request.urlopen(HF_TREE, timeout=60) as r:
        tree = json.load(r)
    return sorted(x["path"] for x in tree if "/train-" in x["path"] and x["path"].endswith(".parquet"))


def download(path: str) -> Path:
    dst = TRAIN_DIR / Path(path).name
    if not dst.is_file():
        TRAIN_DIR.mkdir(parents=True, exist_ok=True)
        part = dst.with_suffix(".part")
        urllib.request.urlretrieve(HF_FILE + path, part)
        part.rename(dst)
    return dst


def shard_candidates(df, seen_titles: set[str], seen_statements: set[int], seed: int, drop: Counter) -> dict[str, dict]:
    """Problems of one shard that pass the filters, each with its reference candidates in order."""
    out: dict[str, dict] = {}
    for _, row in df.iterrows():
        name = str(row["name"])
        tk, sk = title_key(name), statement_key(row["description"])
        if tk in seen_titles or sk in seen_statements:
            drop["eval_or_duplicate"] += 1
            continue
        seen_titles.add(tk)
        seen_statements.add(sk)
        if str(row["input_file"]).strip() or str(row["output_file"]).strip():
            drop["file_io"] += 1
            continue
        if re.search(r"\binteract(ive|ion)\b", str(row["description"]), re.IGNORECASE):
            drop["interactive"] += 1
            continue
        rng = random.Random(f"{seed}:{name}")
        tests = bxp.pick_tests(row, rng)
        if tests is None:
            drop["float_or_few_tests"] += 1
            continue
        start = _h(f"{seed}:lang:{name}") % len(SRC_LANGS)
        order = SRC_LANGS[start:] + SRC_LANGS[:start]  # rotate the preferred source language
        cands: list[tuple[str, str]] = []
        for k, lang in enumerate(order):
            got = bxp.candidates(row, lang, rng)
            cands += [(lang, c) for c in (got if k == 0 else got[:1])]
        cands = cands[:4]
        if not cands:
            drop["no_short_solution"] += 1
            continue
        pid = "CCT_" + hashlib.sha1(name.encode("utf-8")).hexdigest()[:10]
        rating = row["cf_rating"]
        out[pid] = {"id": pid, "name": name, "platform": PLATFORM.get(int(row["source"]), "unknown"),
                    "rating": int(rating) if rating == rating and rating else None, "tags": list(row["cf_tags"]),
                    "tests": tests, "cands": cands}
    return out


def validate(pending: dict[str, dict], workers: int, timeout: int) -> dict[str, tuple[str, str]]:
    """First candidate per problem that passes all its tests → (lang, code)."""
    chosen: dict[str, tuple[str, str]] = {}
    for k in range(4):
        by_lang: dict[str, dict] = defaultdict(dict)
        for pid, p in pending.items():
            if pid not in chosen and len(p["cands"]) > k:
                lang, code = p["cands"][k]
                by_lang[lang][pid] = (code, p["tests"])
        for lang, progs in by_lang.items():
            for pid, r in run_io_programs(get_lang(lang), progs, timeout_sec=timeout, workers=workers).items():
                if r.status == "pass":
                    chosen[pid] = (lang, pending[pid]["cands"][k][1])
    return chosen


def pool(args) -> int:
    import pandas as pd

    done = set(POOL_DONE.read_text().split()) if POOL_DONE.is_file() else set()
    stats = json.loads(POOL_STATS.read_text()) if POOL_STATS.is_file() else {"dropped": {}, "kept": 0, "by_lang": {}}
    drop, by_lang = Counter(stats["dropped"]), Counter(stats["by_lang"])
    titles, statements = eval_keys()
    for r in _jsonl(POOL):  # keep dedupe across restarts
        titles.add(title_key(r["name"]))
    shards = train_shards()[: args.max_shards or None]
    todo = [s for s in shards if Path(s).name not in done]
    print(f"{len(shards)} train shards, {len(todo)} to do; eval titles {len(titles)}", flush=True)
    with ThreadPoolExecutor(max_workers=1) as dl:
        nxt = dl.submit(download, todo[0]) if todo else None
        for i, shard in enumerate(todo):
            t0 = time.time()
            path = nxt.result()
            nxt = dl.submit(download, todo[i + 1]) if i + 1 < len(todo) else None  # prefetch
            df = pd.read_parquet(path, columns=COLS)
            pending = shard_candidates(df, titles, statements, args.seed, drop)
            del df
            chosen = validate(pending, args.workers, args.timeout)
            rows = []
            for pid in sorted(chosen):
                lang, code = chosen[pid]
                p = pending[pid]
                rows.append({"id": pid, "name": p["name"], "platform": p["platform"], "rating": p["rating"],
                             "tags": p["tags"], "src_lang": lang, "source": code, "lines": bxp.nonblank_lines(code),
                             "tests": [list(t) for t in p["tests"]]})
                by_lang[lang] += 1
            drop["reference_failed"] += len(pending) - len(chosen)
            _append(POOL, rows)
            with POOL_DONE.open("a") as fh:
                fh.write(Path(shard).name + "\n")
            stats = {"source": "DeepMind CodeContests train split (CC BY 4.0)", "kept": sum(by_lang.values()),
                     "by_lang": dict(by_lang), "dropped": dict(drop), "seed": args.seed}
            POOL_STATS.write_text(json.dumps(stats, indent=1) + "\n")
            print(f"[{i + 1}/{len(todo)}] {Path(shard).name}: {len(pending)} candidates → {len(rows)} kept "
                  f"({time.time() - t0:.0f}s); pool {stats['kept']} {dict(by_lang)}", flush=True)
    if len(done) + len(todo) >= len(train_shards()) and not args.max_shards:
        POOL_COMPLETE.write_text(time.strftime("%Y-%m-%dT%H:%M:%S") + "\n")
    print(json.dumps(stats, indent=1))
    return 0


# ------------------------------------------------------------------------------------- generate


def program_path(lang: str) -> str:
    return {"java": "Main.java", "csharp": "Program.cs"}.get(lang, f"main{get_lang(lang).ext}")


def error_class_for(status: str, log: str) -> str:
    if status == "compile_error":
        return "CompileError"
    if status == "timeout":
        return "Timeout"
    return "WrongAnswer" if "wrong answer" in log else "RuntimeError"


def _translator(args):
    if args.backend == "vllm":
        from rebirth_ai.xlang.vllm_backend import VllmTranslator

        return VllmTranslator(args.server, args.model, max_model_len=args.max_model_len)
    return get_translator("ollama")


def summarize(verdicts: dict, gens: dict, retries: int) -> dict:
    """First-try / verified counts per target over every (problem, target) with a round-0 verdict."""
    tried, first, verified = Counter(), Counter(), Counter()
    for (pid, t, r), v in verdicts.items():
        if r == 0:
            tried[t] += 1
            first[t] += v["status"] == "pass"
            verified[t] += any((verdicts.get((pid, t, k)) or {}).get("status") == "pass" for k in range(retries + 1))
    n = sum(tried.values())
    return {"translations": n, "first_try": sum(first.values()), "verified": sum(verified.values()),
            "first_try_pct": round(100 * sum(first.values()) / n, 1) if n else 0.0,
            "verified_pct": round(100 * sum(verified.values()) / n, 1) if n else 0.0,
            "generations": len(gens), "generation_seconds": round(sum(g.get("seconds", 0) for g in gens.values()), 1),
            "by_target": {t: {"n": tried[t], "first_try": first[t], "verified": verified[t]} for t in sorted(tried)}}


def generate(args) -> int:
    """Translate pool programs into the 7 other targets with the retry loop; cache every generation
    (gens.jsonl) and judge verdict (verdicts.jsonl) so any run resumes where the last one stopped.

    Ollama: one request at a time (``--concurrency 1 --pipelines 1``, the defaults). vLLM: up to
    ``--concurrency`` requests in flight and ``--pipelines`` chunks at once, so the GPU generates one
    chunk while the CPU judges another. ``--pilot N``: N problems picked independently of ``--seed``
    (the same N for every model) into ``--cache-dir``, then a summary report (``--report``)."""
    import threading

    cache = Path(args.cache_dir) if args.cache_dir else STAGE3
    gens_path, verdicts_path = cache / "gens.jsonl", cache / "verdicts.jsonl"
    translator = _translator(args)
    gens = {(g["problem"], g["tgt"], g["round"]): g for g in _jsonl(gens_path)}
    verdicts = {(v["problem"], v["tgt"], v["round"]): v for v in _jsonl(verdicts_path)}
    print(f"model {translator.model_name}; cache {cache}; cached: {len(gens)} generations, {len(verdicts)} verdicts",
          flush=True)
    lock = threading.Lock()
    gave_up: set[str] = set()  # problems whose generation errored this run (not retried forever)
    t_start, processed = time.time(), [0]
    deadline = t_start + args.deadline_hours * 3600 if args.deadline_hours else None
    gen_pool = ThreadPoolExecutor(max_workers=max(1, args.concurrency))

    def finished(pid: str, tgt: str) -> bool:
        hist = [verdicts.get((pid, tgt, r)) for r in range(args.retries + 1)]
        return any(v and v["status"] == "pass" for v in hist) or all(hist)

    def one(p: dict, t: str, rnd: int) -> dict | None:
        prev = verdicts.get((p["id"], t, rnd - 1)) if rnd else None
        t0 = time.time()
        try:
            if args.prompt_style == "sft":  # shared adapter: SFT prompt, retries resample (no feedback in the format)
                prompt = build_xlang_sft_prompt(source=get_lang(p["src_lang"]), target=get_lang(t), source_code=p["source"])
                out = translator.complete_sft(prompt, max_new_tokens=max_new_tokens_for(p["source"]),
                                              temperature=retry_temperature(rnd) if rnd else 0.0)
            else:
                prompt = build_translation_prompt(
                    source=get_lang(p["src_lang"]), target=get_lang(t), source_code=p["source"],
                    source_path=program_path(p["src_lang"]), target_path=program_path(t), is_program=True,
                    previous=prev["code"] if prev else None,
                    error_class=error_class_for(prev["status"], prev["log"]) if prev else None,
                    logs=prev["log"] if prev else None,
                )
                out = translator.complete(prompt, max_new_tokens=max_new_tokens_for(p["source"]), temperature=retry_temperature(rnd))
        except Exception as exc:
            print(f"  generation failed {p['id']} -> {t} round {rnd}: {str(exc)[:200]}", flush=True)
            return None
        return {"problem": p["id"], "src": p["src_lang"], "tgt": t, "round": rnd, "raw": out.raw,
                "code": extract_code(out.raw, get_lang(t)), "model": out.model, "seconds": round(time.time() - t0, 1)}

    def process(chunk: list[dict]) -> None:
        if deadline and time.time() > deadline:
            return
        for rnd in range(args.retries + 1):
            with lock:
                items = [(p, t) for p in chunk for t in TARGETS
                         if t != p["src_lang"] and p["id"] not in gave_up and not finished(p["id"], t)
                         and (p["id"], t, rnd) not in verdicts and (not rnd or (p["id"], t, rnd - 1) in verdicts)]
            if not items:
                continue
            todo = [(p, t) for p, t in items if (p["id"], t, rnd) not in gens]  # GPU
            for (p, t), g in zip(todo, gen_pool.map(lambda pt, rnd=rnd: one(pt[0], pt[1], rnd), todo), strict=True):
                with lock:
                    if g is None:
                        gave_up.add(p["id"])
                        continue
                    _append(gens_path, [g])
                    gens[(p["id"], t, rnd)] = g
            by_target: dict[str, dict] = defaultdict(dict)  # CPU: judge
            for p, t in items:
                g = gens.get((p["id"], t, rnd))
                if g is None:
                    continue
                code = apply_fixups(get_lang(t), g["code"])
                by_target[t][f"{p['id']}__{t}"] = (code, [tuple(x) for x in p["tests"]], p, t)
            new = []
            for t, progs in by_target.items():
                res = run_io_programs(get_lang(t), {k: (v[0], v[1]) for k, v in progs.items()},
                                      timeout_sec=args.timeout, workers=args.workers)
                for k, (code, _tests, p, _t) in progs.items():
                    r = res[k]
                    new.append({"problem": p["id"], "src": p["src_lang"], "tgt": t, "round": rnd, "status": r.status,
                                "exit": r.exit_code, "code": code, "log": r.log_tail[-1500:]})
            with lock:
                _append(verdicts_path, new)
                verdicts.update({(v["problem"], v["tgt"], v["round"]): v for v in new})
        with lock:
            processed[0] += len(chunk)
            s = summarize(verdicts, gens, args.retries)
            rate = s["generation_seconds"] / max(1, s["generations"])
            hours = (time.time() - t_start) / 3600
        print(f"[{time.strftime('%H:%M')}] +{len(chunk)} problems ({processed[0]} this run, {hours:.2f} h); first try "
              f"{s['first_try']}/{s['translations']}, verified {s['verified']} "
              f"{ {t: v['verified'] for t, v in s['by_target'].items()} }; {rate:.1f}s/generation", flush=True)

    pool_path = Path(args.pool_file) if args.pool_file else POOL
    while True:
        pool_rows = _jsonl(pool_path)
        if args.ids:
            keep = set(args.ids.split(","))
            pool_rows = [p for p in pool_rows if p["id"] in keep]
        if args.pilot:
            pool_rows = sorted(pool_rows, key=lambda p: _h(f"pilot:{p['id']}"))[: args.pilot]
        pending = [p for p in pool_rows if p["id"] not in gave_up
                   and not all(finished(p["id"], t) for t in TARGETS if t != p["src_lang"])]
        pending.sort(key=lambda p: _h(f"{args.seed}:{p['id']}"))
        if args.max_problems:
            started = len(pool_rows) - len(pending)
            pending = pending[: max(0, args.max_problems - started)]
        if deadline and time.time() > deadline:
            print(f"deadline reached ({args.deadline_hours} h)", flush=True)
            break
        if not pending:
            if POOL_COMPLETE.is_file() or args.max_problems or args.pilot or args.ids or args.pool_file:
                break
            print("pool used up — waiting for `pool` to add problems", flush=True)
            time.sleep(300)
            continue
        # one pass over the pending problems: --pipelines chunks in flight (1 = the old sequential loop)
        chunks = [pending[i : i + args.chunk] for i in range(0, len(pending), args.chunk)]
        with ThreadPoolExecutor(max_workers=max(1, args.pipelines)) as pipes:
            list(pipes.map(process, chunks))
    gen_pool.shutdown()
    s = summarize(verdicts, gens, args.retries)
    if args.pilot:
        ids = {p["id"] for p in sorted(_jsonl(pool_path), key=lambda p: _h(f"pilot:{p['id']}"))[: args.pilot]}
        s = summarize({k: v for k, v in verdicts.items() if k[0] in ids}, {k: g for k, g in gens.items() if k[0] in ids},
                      args.retries)
    s.update({"model": translator.model_name, "problems": args.pilot or None, "wall_hours": round((time.time() - t_start) / 3600, 2),
              "retries": args.retries, "cache": str(cache)})
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(s, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in s.items() if k != "by_target"}), flush=True)
    print("done: deadline" if deadline and time.time() > deadline else "done")
    return 0


# ---------------------------------------------------------------------------------------- pairs


def model_tag(model: str) -> str:
    """Short generator name for a pair's origin: 14b (Ollama or vLLM Qwen2.5-Coder 14B) or the repo name."""
    return "14b" if not model or "14b" in model.lower() else model.split("/")[-1].split(":")[-1].lower()


def pairs(args) -> int:
    pool_rows = {p["id"]: p for p in _jsonl(POOL)}
    verified: dict[str, dict[str, tuple[str, str]]] = defaultdict(dict)  # pid → lang → (code, origin)
    models = {(g["problem"], g["tgt"], g["round"]): g.get("model", "") for g in _jsonl(GENS)}
    for v in _jsonl(VERDICTS):
        if v["status"] == "pass" and v["problem"] in pool_rows and v["tgt"] not in verified[v["problem"]]:
            verified[v["problem"]][v["tgt"]] = (v["code"], f"{model_tag(models.get((v['problem'], v['tgt'], v['round']), ''))}_round{v['round']}")
    # JavaScript (source-only language): passing TypeScript with its types stripped, re-judged
    js_done = {(r["problem"], r["ts_hash"]): r for r in _jsonl(JS_CACHE)}
    ts = {pid: langs["typescript"][0] for pid, langs in verified.items() if "typescript" in langs}
    todo = {pid: code for pid, code in ts.items() if (pid, _h(code)) not in js_done}
    if todo:
        js = ts_to_js(todo)
        res = run_io_programs(get_lang("javascript"), {pid: (c, [tuple(t) for t in pool_rows[pid]["tests"]]) for pid, c in js.items()},
                              timeout_sec=args.timeout, workers=args.workers)
        new = [{"problem": pid, "ts_hash": _h(todo[pid]), "pass": pid in js and res[pid].status == "pass", "code": js.get(pid, "")}
               for pid in todo]
        _append(JS_CACHE, new)
        js_done.update({(r["problem"], r["ts_hash"]): r for r in new})
    for pid, code in ts.items():
        r = js_done.get((pid, _h(code)))
        if r and r["pass"]:
            verified[pid]["javascript"] = (r["code"], "typescript_stripped")

    rows, kinds, dirs = [], Counter(), Counter()
    for pid, langs in sorted(verified.items()):
        p = pool_rows[pid]
        ref = p["src_lang"]
        versions = {ref: (p["source"], "reference"), **{k: v for k, v in langs.items() if k != ref}}
        split = "val" if _h(pid) % 100 < VAL_PERCENT else "train"
        for s, (s_code, s_origin) in versions.items():
            for t, (t_code, t_origin) in versions.items():
                if s == t or t == "javascript" or (s == "javascript" and t == "typescript"):
                    continue
                kind = "forward" if s_origin == "reference" else "reverse" if t_origin == "reference" else "cross"
                prompt = build_xlang_sft_prompt(source=get_lang(s), target=get_lang(t), source_code=s_code)
                rows.append({"id": f"{pid}__{s}__{t}", "problem": pid, "split": split, "kind": kind,
                             "source_lang": s, "target_lang": t, "source_origin": s_origin, "target_origin": t_origin,
                             "prompt": prompt, "completion": t_code.rstrip() + "\n",
                             "approx_tokens": round((len(prompt) + len(t_code)) / 3.3)})
                kinds[kind] += 1
                dirs[f"{s}->{t}"] += 1
    PAIRS.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    toks = sorted(r["approx_tokens"] for r in rows)
    stats = {"problems_with_pairs": len(verified), "rows": len(rows), "kinds": dict(kinds),
             "val_rows": sum(r["split"] == "val" for r in rows), "directions": len(dirs),
             "rows_per_direction": {"min": min(dirs.values(), default=0), "max": max(dirs.values(), default=0)},
             "approx_tokens": {"median": toks[len(toks) // 2] if toks else 0,
                               "le_1024": sum(t <= 1024 for t in toks), "le_2048": sum(t <= 2048 for t in toks)},
             "by_direction": dict(sorted(dirs.items()))}
    PAIRS.with_suffix(".stats.json").write_text(json.dumps(stats, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in stats.items() if k != "by_direction"}, indent=1))
    print(f"wrote {PAIRS.relative_to(ROOT)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["pool", "generate", "pairs"])
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--workers", type=int, default=4, help="parallel sandbox containers")
    ap.add_argument("--timeout", type=int, default=10, help="seconds per test run")
    ap.add_argument("--max-shards", type=int, default=0, help="pool: only the first N train shards (0 = all)")
    ap.add_argument("--retries", type=int, default=2, help="generate: retry rounds after a failed translation")
    ap.add_argument("--chunk", type=int, default=10, help="generate: problems per batch (×7 targets)")
    ap.add_argument("--max-problems", type=int, default=0, help="generate: stop after N problems (0 = whole pool)")
    ap.add_argument("--backend", choices=["ollama", "vllm"], default="ollama", help="generate: model server")
    ap.add_argument("--server", default="http://127.0.0.1:8000/v1", help="generate: vLLM OpenAI endpoint")
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-14B-Instruct", help="generate: vLLM served model")
    ap.add_argument("--max-model-len", type=int, default=8192, help="generate: the vLLM server's --max-model-len")
    ap.add_argument("--concurrency", type=int, default=1, help="generate: requests in flight (vLLM: 48)")
    ap.add_argument("--pipelines", type=int, default=1, help="generate: chunks in flight (vLLM: 2 — GPU and judge overlap)")
    ap.add_argument("--deadline-hours", type=float, default=0, help="generate: start no new chunk after N hours (0 = none)")
    ap.add_argument("--cache-dir", default="", help="generate: gens/verdicts directory (default stage3/)")
    ap.add_argument("--pilot", type=int, default=0, help="generate: only N problems in a model-independent order")
    ap.add_argument("--ids", default="", help="generate: only these comma-separated pool ids")
    ap.add_argument("--report", default="", help="generate: write the run summary JSON here")
    ap.add_argument("--pool-file", default="", help="generate: pool rows from this file (e.g. an eval pool)")
    ap.add_argument("--prompt-style", choices=["product", "sft"], default="product",
                    help="generate: product chat prompt + retry feedback, or the shared adapter's SFT prompt (vLLM)")
    args = ap.parse_args()
    return {"pool": pool, "generate": generate, "pairs": pairs}[args.step](args)


if __name__ == "__main__":
    raise SystemExit(main())
