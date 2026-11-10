#!/usr/bin/env python3
"""Batched eval against a vLLM OpenAI-compatible server (Phase 14; used on the cloud GPU).

Same rows, prompts, token budgets, decode cleanup, metrics and report schema as
``eval_behavioral.py --suite synthetic --eval-file`` — only the generator differs:

  * prompts / budgets: ``rebirth_ai.generate.peft_client.plan_generation`` (shared with PEFT)
  * cleanup:           ``_clean_sft_completion`` + ``extract_code_block`` (shared)
  * scoring + schema:  ``eval_behavioral.score_row`` / ``summarize`` (shared)

All rows are sent concurrently, so a 100-row eval takes minutes instead of 30–45 min.
vLLM greedy decoding is not bit-identical to HF ``generate``: compare models **within one
engine** (re-score baselines on vLLM too).

Server (on the instance, see scripts/cloud/vllm_serve.sh):
  vllm serve Qwen/Qwen2.5-Coder-14B-Instruct --enable-lora \\
      --lora-modules go14b=/data/models/lora_go_14b/adapter --max-lora-rank 16

  python training/eval_vllm.py --server http://127.0.0.1:8000/v1 --model go14b --style sft \\
      --lang go --eval-file data/datasets/evalsets/go.jsonl --out data/reports/exp/go_14b_v1.json
  python training/eval_vllm.py ... --model Qwen/Qwen2.5-Coder-14B-Instruct --style chat \\
      --model-label base-14b-chat --out data/reports/exp/go_base14b_chat.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "training"))

from rebirth_ai.generate.peft_client import _clean_sft_completion, plan_generation  # noqa: E402
from rebirth_ai.generate.prompts import extract_code_block  # noqa: E402


def _post(url: str, payload: dict, *, timeout: float, retries: int = 3) -> dict:
    body = json.dumps(payload).encode("utf-8")
    last: Exception | None = None
    for attempt in range(retries):
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:  # server answered: a 4xx will not change on retry
            detail = exc.read().decode("utf-8", "replace")[:500]
            if exc.code < 500:
                raise RuntimeError(f"vLLM rejected the request (HTTP {exc.code}): {detail}") from exc
            last = RuntimeError(f"HTTP {exc.code}: {detail}")
            time.sleep(2 * (attempt + 1))
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            last = exc
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"vLLM request failed after {retries} tries: {last}")


def minimal_stops(stops: tuple[str, ...] | list[str]) -> list[str]:
    """Drop stop strings that start with another stop (that one always fires first) — same output,
    and the OpenAI API vLLM implements accepts at most 4 (the SFT plan has 5, 3 of them redundant)."""
    keep = [s for s in dict.fromkeys(stops) if not any(t != s and s.startswith(t) for t in stops)]
    if len(keep) > 4:
        raise ValueError(f"more than 4 independent stop strings: {keep}")
    return keep


def generate(server: str, model: str, row: dict, *, style: str, max_new_tokens: int, timeout: float) -> str:
    """One row → cleaned code, using the exact plan the PEFT client would use."""
    plan = plan_generation(
        prompt_style=style,
        objective=row["objective"],
        legacy_code=row["legacy"],
        file_path=row.get("path") or "snippet.js",
        language=row.get("language", "javascript"),
        max_new_tokens=max_new_tokens,
    )
    common = {"model": model, "max_tokens": plan.max_new_tokens, "temperature": 0.0, "seed": 0}
    if plan.style == "sft":
        out = _post(f"{server}/completions", {**common, "prompt": plan.prompt, "stop": minimal_stops(plan.stop)}, timeout=timeout)
        text = out["choices"][0]["text"]
    else:
        messages = [{"role": "system", "content": plan.system}, {"role": "user", "content": plan.prompt}]
        out = _post(f"{server}/chat/completions", {**common, "messages": messages}, timeout=timeout)
        text = out["choices"][0]["message"]["content"]
    return extract_code_block(_clean_sft_completion(text))


def run(args: argparse.Namespace) -> dict:
    from eval_behavioral import score_row, summarize

    from rebirth_ai.lora import load_rows

    rows = load_rows(args.eval_file)[: args.n]
    server = args.server.rstrip("/")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        codes = list(pool.map(
            lambda r: generate(server, args.model, r, style=args.style,
                               max_new_tokens=args.max_new_tokens, timeout=args.timeout),
            rows,
        ))
    gen_secs = time.time() - t0
    results = [score_row(i, r, c, total=len(rows)) for i, (r, c) in enumerate(zip(rows, codes, strict=True), 1)]
    suite = summarize(results, suite=f"evalset:{args.eval_file.stem}", backend="vllm",
                      lang=args.lang, model=f"vllm:{args.model}")
    suite["generation_seconds"] = round(gen_secs, 1)
    return {"backend": "vllm", "suites": {"synthetic": suite}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", required=True, help="Served model id: base name or a --lora-modules name")
    ap.add_argument("--style", choices=("sft", "chat"), default="sft", help="sft for adapters, chat for base models")
    ap.add_argument("--lang", required=True)
    ap.add_argument("--eval-file", type=Path, required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--max-new-tokens", type=int, default=1024, help="Same default as PeftClient")
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--track", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--model-label", default=None, help="MLflow label (default: adapter for sft, base-<model>-chat)")
    ap.add_argument("--adapter", type=Path, default=None, help="Adapter dir scored (links scores to its registry version)")
    ap.add_argument("--engine", choices=("vllm", "ollama"), default="vllm",
                    help="Server type. ollama = a served GGUF copy: scores go to the registry under "
                         "served:<label>:<set>, never over the version's primary score")
    args = ap.parse_args()

    report = run(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    s = report["suites"]["synthetic"]
    print(f"\n=== {s['suite']} via {args.engine} ({args.model}) — soft-exact {s['soft_exact']}/{s['n']}, "
          f"parses {s['parses']}/{s['n']}, {s['generation_seconds']}s ===")
    if args.track:
        from rebirth_ai.tracking import try_log_eval

        label = args.model_label or ("adapter" if args.style == "sft" else f"base-{args.model.split('/')[-1]}-chat")
        run_id = try_log_eval(args.out, model_label=label, lang=args.lang, adapter=args.adapter,
                              eval_file=args.eval_file, tags={"rebirth.engine": args.engine})
        if run_id:
            print(f"[tracking] eval → MLflow run {run_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
