"""MLflow experiment tracking (Phase 13) — post-hoc, never inside the training loop.

Training and eval already write everything to disk (``logs/train/<run>/`` step logs + GPU
telemetry, ``rebirth_meta.json``, eval report JSON). This module *reads those files after the
fact* and records them in MLflow, so tracking can never slow down or crash a GPU job, and the
same code backfills history.

Store: SQLite ``mlruns/mlflow.db`` + artifacts under ``mlruns/artifacts/`` (gitignored).
Override with ``REBIRTH_MLFLOW_URI``; disable with ``REBIRTH_TRACKING=0``.
Client: ``pip install -e ".[mlops]"`` (mlflow-skinny + SQLAlchemy). UI: ``rebirth mlflow-ui``.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

from rebirth_ai.lora import ROOT, normalize_lang, read_meta, rel_to_root

TRACKING_DIR = ROOT / "mlruns"
DB_PATH = TRACKING_DIR / "mlflow.db"
ARTIFACT_ROOT = TRACKING_DIR / "artifacts"
LOGS_TRAIN = ROOT / "logs" / "train"
EXP_TRAIN = "rebirth-train"
EXP_EVAL = "rebirth-eval"
EVAL_METRICS = (
    "n", "soft_exact", "soft_exact_rate", "parses", "parse_rate",
    "mean_symbol_jaccard", "mean_token_overlap",
)


class TrackingUnavailable(RuntimeError):
    """MLflow is not installed or tracking is disabled."""


def tracking_uri() -> str:
    return os.environ.get("REBIRTH_MLFLOW_URI") or f"sqlite:///{DB_PATH}"


def enabled() -> bool:
    if os.environ.get("REBIRTH_TRACKING", "1").lower() in {"0", "false", "no", "off"}:
        return False
    try:
        import mlflow  # noqa: F401
    except ImportError:
        return False
    return True


def mlflow_mod():
    """Configured ``mlflow`` module (tracking + registry URIs set)."""
    if not enabled():
        raise TrackingUnavailable(
            'MLflow tracking unavailable (pip install -e ".[mlops]", or REBIRTH_TRACKING=0 set)'
        )
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    import mlflow

    TRACKING_DIR.mkdir(parents=True, exist_ok=True)
    uri = tracking_uri()
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    return mlflow


def client():
    mlflow = mlflow_mod()
    return mlflow.MlflowClient()


def experiment_id(name: str) -> str:
    c = client()
    exp = c.get_experiment_by_name(name)
    if exp is not None:
        return exp.experiment_id
    # Explicit artifact location: otherwise MLflow uses ./mlruns relative to the cwd.
    return c.create_experiment(name, artifact_location=str(ARTIFACT_ROOT / name))


# ----------------------------------------------------------------------------- helpers


def flatten(d: dict, prefix: str = "") -> dict[str, str]:
    out: dict[str, str] = {}
    for k, v in (d or {}).items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        elif v is not None:
            out[key] = str(v)[:500]
    return out


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8", errors="replace").split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # torn last line after a hard GPU reset
    return rows


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def base_label(base_model: str | None) -> str:
    s = (base_model or "").lower()
    for size in ("32b", "14b", "7b", "3b", "1.5b", "0.5b"):
        if size in s:
            return size
    return "unknown"


def _cmd_value(cmd: list[str], flag: str) -> str | None:
    """Last value of ``flag`` in an argv list (argparse: last occurrence wins)."""
    val = None
    for i, tok in enumerate(cmd[:-1]):
        if tok == flag:
            val = cmd[i + 1]
    return val


def run_out_dir(run_json: dict) -> Path | None:
    """Adapter output dir a ``train_logged.sh`` run wrote to (from its recorded argv)."""
    cmd = [str(c) for c in run_json.get("cmd") or []]
    out = _cmd_value(cmd, "--out")
    if out:
        return (ROOT / out).resolve()
    lang = _cmd_value(cmd, "--lang")
    if lang:
        from rebirth_ai.lora import lora_target

        return lora_target(lang).out.resolve()
    return None


def find_run_dirs(out_dir: Path, meta: dict) -> list[Path]:
    """``logs/train`` dirs that produced this adapter (oldest first; >1 when resumed).

    Picks the newest matching run that started before ``trained_at``; if its first step is not
    1 (a ``--resume`` continuation), walks back through earlier matching runs.
    """
    from datetime import datetime

    out_dir = out_dir.resolve()
    want_len = str((meta.get("hyperparams") or {}).get("max_length", ""))
    trained_at = meta.get("trained_at")
    t_end = datetime.fromisoformat(trained_at).timestamp() if trained_at else float("inf")
    cands: list[tuple[float, Path]] = []
    for run_json in LOGS_TRAIN.glob("*/run.json"):
        try:
            rj = json.loads(run_json.read_text(encoding="utf-8"))
            start = datetime.fromisoformat(rj["start"]).timestamp()
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            continue
        if run_out_dir(rj) != out_dir or start > t_end:
            continue
        if want_len and _cmd_value([str(c) for c in rj.get("cmd", [])], "--max-length") != want_len:
            continue
        cands.append((start, run_json.parent))
    cands.sort()
    chain: list[Path] = []
    for _, d in reversed(cands):
        chain.insert(0, d)
        firsts = [r.get("step") for r in read_jsonl(d / "steps.jsonl") if r.get("event") == "step_begin"]
        if not firsts or firsts[0] <= 1:
            break
    return chain


def _existing_run(exp_name: str, key: str, value: str) -> str | None:
    c = client()
    hits = c.search_runs(
        [experiment_id(exp_name)], filter_string=f"tags.`{key}` = '{value}'", max_results=1
    )
    return hits[0].info.run_id if hits else None


# ----------------------------------------------------------------------------- training


def log_training(
    out_dir: Path,
    *,
    run_dirs: list[Path] | None = None,
    log_adapter: str = "config",
    tags: dict[str, str] | None = None,
    register: bool = True,
) -> str:
    """Record a finished training (adapter dir with ``rebirth_meta.json``) as an MLflow run.

    Idempotent: a run already recorded for this adapter + ``trained_at`` is returned as-is.
    Returns the MLflow run id. Optionally registers the adapter as a model version.

    ``log_adapter``: ``"config"`` (default) logs adapter_config.json + README and tags the
    weights' sha256 — the weights stay in ``data/models`` (versioned by DVC in Phase 14) instead
    of being duplicated (20–155 MB each); ``"full"`` copies the whole adapter dir; ``"none"``.
    """
    out_dir = Path(out_dir).resolve()
    meta = read_meta(out_dir)
    if not meta:
        raise FileNotFoundError(f"no rebirth_meta.json in {out_dir}")
    lang = normalize_lang(meta.get("language") or "") if meta.get("language") else "unknown"
    ident = f"{rel_to_root(out_dir)}@{meta.get('trained_at')}"
    mlflow = mlflow_mod()
    existing = _existing_run(EXP_TRAIN, "rebirth.adapter_id", ident)
    if existing:
        run_id = existing
    else:
        run_dirs = run_dirs if run_dirs is not None else find_run_dirs(out_dir, meta)
        hp = dict(meta.get("hyperparams") or {})
        hp.setdefault("lora_targets", "attn")  # pre-Phase-12 adapters were attention-only
        recipe = dict(meta.get("recipe") or {})
        features = recipe.pop("features", {}) or {}
        params = {
            "lang": lang,
            "base_model": meta.get("base_model"),
            "lr": meta.get("lr"),
            "data": meta.get("data"),
            "corpus_sha256": (meta.get("corpus_sha256") or "")[:16],
            "split_seed": meta.get("split_seed"),
            "eval_frac": meta.get("eval_frac"),
            "split_index": meta.get("split_index"),
            **hp,
            **{f"recipe.{k}": v for k, v in recipe.items()},
        }
        run_tags = {
            "rebirth.kind": "train",
            "rebirth.lang": lang,
            "rebirth.base": base_label(meta.get("base_model")),
            "rebirth.adapter": rel_to_root(out_dir),
            "rebirth.adapter_id": ident,
            "rebirth.trained_at": str(meta.get("trained_at")),
            "rebirth.run_dirs": ",".join(rel_to_root(d) for d in run_dirs),
            "rebirth.smoke": str(bool(meta.get("smoke"))),
        }
        weights = out_dir / "adapter" / "adapter_model.safetensors"
        if weights.is_file():
            run_tags["rebirth.adapter_sha256"] = file_sha(weights)
            run_tags["rebirth.adapter_mb"] = f"{weights.stat().st_size / 2**20:.1f}"
        if run_dirs:
            try:
                rj = json.loads((run_dirs[-1] / "run.json").read_text(encoding="utf-8"))
                run_tags["git.sha"] = str(rj.get("git"))
                run_tags["git.dirty"] = str(rj.get("git_dirty"))
                run_tags["host.kernel"] = str(rj.get("kernel"))
            except (OSError, json.JSONDecodeError):
                pass
        run_tags.update(tags or {})
        name = f"{lang}/{out_dir.name}"
        with mlflow.start_run(experiment_id=experiment_id(EXP_TRAIN), run_name=name, tags=run_tags) as run:
            mlflow.log_params({k: str(v)[:500] for k, v in params.items() if v is not None})
            summary = {
                "train_loss": meta.get("train_loss"),
                "global_steps": meta.get("global_steps"),
                "train_rows": meta.get("train_rows"),
                "eval_rows": meta.get("eval_rows"),
                "features_kept": features.get("kept"),
                "features_dropped_overlong": features.get("dropped_overlong"),
                "label_tokens": features.get("label_tokens"),
            }
            mlflow.log_metrics({k: float(v) for k, v in summary.items() if isinstance(v, (int, float))})
            _log_step_series(mlflow, run_dirs)
            for f in ("rebirth_meta.json", "train_meta.json", "split.json"):
                if (out_dir / f).is_file():
                    mlflow.log_artifact(str(out_dir / f), "meta")
            for d in run_dirs:
                for f in ("run.json", "exit.json", "kernel.log"):
                    if (d / f).is_file():
                        mlflow.log_artifact(str(d / f), f"logs/{d.name}")
            adapter_dir = out_dir / "adapter"
            if log_adapter == "full" and adapter_dir.is_dir():
                mlflow.log_artifacts(str(adapter_dir), "adapter")
            elif log_adapter == "config":
                for f in ("adapter_config.json", "README.md"):
                    if (adapter_dir / f).is_file():
                        mlflow.log_artifact(str(adapter_dir / f), "adapter")
            run_id = run.info.run_id
    if register and lang != "unknown":
        from rebirth_ai import registry

        registry.register_adapter(out_dir, run_id, meta=meta)
    return run_id


def _log_step_series(mlflow, run_dirs: list[Path]) -> None:
    """Per-step loss / lr / grad-norm / VRAM from steps.jsonl + a telemetry summary."""
    batch = []
    from mlflow.entities import Metric

    t_first = t_last = None
    step_times: list[float] = []
    peak_vram = 0
    for d in run_dirs:
        prev_t = None
        for r in read_jsonl(d / "steps.jsonl"):
            ev, step, t = r.get("event"), r.get("step"), r.get("t")
            if t is not None:
                t_first = t if t_first is None else min(t_first, t)
                t_last = t if t_last is None else max(t_last, t)
            if ev == "log" and step is not None:
                for k in ("loss", "grad_norm", "learning_rate"):
                    if isinstance(r.get(k), (int, float)):
                        batch.append(Metric(k, float(r[k]), int((t or 0) * 1000), int(step)))
            elif ev == "step_end" and step is not None:
                if isinstance(r.get("peak_mb"), (int, float)):
                    peak_vram = max(peak_vram, r["peak_mb"])
                    if step % 10 == 0:
                        batch.append(Metric("torch_peak_mb", float(r["peak_mb"]), int((t or 0) * 1000), int(step)))
                if prev_t is not None and t is not None:
                    step_times.append(t - prev_t)
                prev_t = t
        tele = [r for r in read_jsonl(d / "telemetry.jsonl") if "vram_used_mb" in r]
        if tele:
            mlflow.log_metrics({
                f"gpu.vram_used_mb_max": max(r["vram_used_mb"] for r in tele),
                f"gpu.temp_junction_c_max": max(r.get("temp_junction_c") or 0 for r in tele),
                f"gpu.power_w_mean": sum(r.get("power_w") or 0 for r in tele) / len(tele),
            })
    client_ = mlflow.MlflowClient()
    run_id = mlflow.active_run().info.run_id
    for i in range(0, len(batch), 900):
        client_.log_batch(run_id, metrics=batch[i : i + 900])
    extra = {"torch_peak_mb_max": peak_vram} if peak_vram else {}
    if step_times:
        extra["sec_per_step_mean"] = sum(step_times) / len(step_times)
    if t_first is not None and t_last is not None:
        extra["wall_minutes"] = (t_last - t_first) / 60
    if extra:
        mlflow.log_metrics(extra)


# ----------------------------------------------------------------------------- eval


def eval_set_id(report: dict, eval_file: Path | None = None) -> str:
    """Stable id for 'which rows were scored' — eval runs are only comparable within one id."""
    suite = (report.get("suites") or {}).get("synthetic") or {}
    if eval_file is not None and Path(eval_file).is_file():
        lang = normalize_lang(suite.get("lang") or Path(eval_file).stem)
        return f"evalset:{lang}:{file_sha(Path(eval_file))[:10]}:n{suite.get('n')}"
    return f"{suite.get('suite', 'synthetic')}:n{suite.get('n')}"


def log_eval(
    report_path: Path,
    *,
    model_label: str,
    lang: str | None = None,
    adapter: Path | None = None,
    eval_file: Path | None = None,
    tags: dict[str, str] | None = None,
) -> str:
    """Record an eval report (``eval_behavioral.py`` JSON) as an MLflow run.

    ``model_label``: what was scored, e.g. ``adapter``, ``base-7b-chat``, ``ollama-14b``.
    With ``adapter``: also writes the scores onto the matching registered model version.
    Idempotent per report content (sha256).
    """
    report_path = Path(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    suite = (report.get("suites") or {}).get("synthetic")
    if not suite:
        raise ValueError(f"{report_path}: no synthetic suite")
    lang = normalize_lang(lang or suite.get("lang") or "unknown")
    set_id = eval_set_id(report, eval_file)
    sha = file_sha(report_path)
    mlflow = mlflow_mod()
    metrics = {k: float(suite[k]) for k in EVAL_METRICS if isinstance(suite.get(k), (int, float))}
    run_id = _existing_run(EXP_EVAL, "rebirth.report_sha256", sha)
    if not run_id:
        run_tags = {
            "rebirth.kind": "eval",
            "rebirth.lang": lang,
            "rebirth.model": model_label,
            "rebirth.eval_set": set_id,
            "rebirth.report": rel_to_root(report_path.resolve()),
            "rebirth.report_sha256": sha,
            "rebirth.backend": str(suite.get("backend") or report.get("backend")),
            "rebirth.generator": str(suite.get("model")),
        }
        if adapter is not None:
            run_tags["rebirth.adapter"] = rel_to_root(Path(adapter).resolve())
        run_tags.update(tags or {})
        name = f"{lang}/{model_label}@{set_id.split(':')[0]}"
        with mlflow.start_run(experiment_id=experiment_id(EXP_EVAL), run_name=name, tags=run_tags) as run:
            mlflow.log_params({"eval_set": set_id, "n": suite.get("n"), "model": model_label})
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(str(report_path), "report")
            run_id = run.info.run_id
    if adapter is not None:
        from rebirth_ai import registry

        out_dir = _adapter_out_dir(Path(adapter))
        version = registry.find_version(lang, out_dir) if out_dir else None
        if version is not None:
            registry.record_eval(lang, version, set_id, metrics, eval_run_id=run_id)
    return run_id


def _adapter_out_dir(adapter: Path) -> Path | None:
    """``…/lora_go_7b/adapter`` or ``…/runs/checkpoint-N`` → the dir holding rebirth_meta.json."""
    p = adapter.resolve()
    for cand in (p, p.parent, p.parent.parent):
        if (cand / "rebirth_meta.json").is_file():
            return cand
    return None


def warn(msg: str) -> None:
    print(f"[tracking] {msg}", file=sys.stderr)


def try_log_training(out_dir: Path, **kw: Any) -> str | None:
    """Best-effort wrapper for pipelines: never raises."""
    if not enabled():
        return None
    try:
        return log_training(out_dir, **kw)
    except Exception as exc:  # noqa: BLE001 — tracking must never break a training pipeline
        warn(f"training not tracked ({type(exc).__name__}: {exc})")
        return None


def try_log_eval(report_path: Path, **kw: Any) -> str | None:
    if not enabled():
        return None
    try:
        return log_eval(report_path, **kw)
    except Exception as exc:  # noqa: BLE001
        warn(f"eval not tracked ({type(exc).__name__}: {exc})")
        return None
