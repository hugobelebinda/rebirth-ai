"""``rebirth track …`` / ``rebirth registry …`` / ``rebirth mlflow-ui`` (Phase 13)."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import typer
from rich import print
from rich.table import Table

from rebirth_ai.lora import LORA_LANGS, ROOT, normalize_lang

track_app = typer.Typer(help="Record training/eval runs in MLflow (post-hoc, from files on disk)")
registry_app = typer.Typer(help="Model registry: champion / challenger adapters per language")


def _need_tracking() -> None:
    from rebirth_ai.tracking import enabled

    if not enabled():
        print('[red]MLflow client missing — pip install -e ".[mlops]" (or REBIRTH_TRACKING=0 is set)[/red]')
        raise typer.Exit(2)


@track_app.command("train")
def track_train(
    target: Path = typer.Argument(..., help="Adapter out dir (has rebirth_meta.json) or a logs/train run dir"),
    adapter_artifacts: str = typer.Option(
        "config", "--adapter-artifacts", help="config (default: config + weights sha256) | full | none"
    ),
) -> None:
    """Record a finished training run (and register its adapter)."""
    _need_tracking()
    from rebirth_ai.tracking import log_training, run_out_dir

    target = target.resolve()
    run_dirs = None
    if (target / "run.json").is_file():  # a logs/train/<run> dir
        out = run_out_dir(json.loads((target / "run.json").read_text(encoding="utf-8")))
        if out is None:
            raise typer.BadParameter(f"{target}/run.json does not name an output dir")
        target = out
    run_id = log_training(target, run_dirs=run_dirs, log_adapter=adapter_artifacts)
    print(f"[green]tracked[/green] {target.relative_to(ROOT) if target.is_relative_to(ROOT) else target} → run {run_id}")


@track_app.command("eval")
def track_eval(
    report: Path = typer.Argument(..., exists=True, dir_okay=False),
    model: str = typer.Option(..., "--model", help="Label: adapter | base-7b-chat | ollama-14b | …"),
    lang: str | None = typer.Option(None, "--lang"),
    adapter: Path | None = typer.Option(None, "--adapter", help="Adapter dir that was scored"),
    eval_file: Path | None = typer.Option(None, "--eval-file", help="Frozen eval set used"),
) -> None:
    """Record an eval report; with --adapter also writes scores onto its model version."""
    _need_tracking()
    from rebirth_ai.tracking import log_eval

    run_id = log_eval(report, model_label=model, lang=lang, adapter=adapter, eval_file=eval_file)
    print(f"[green]tracked[/green] {report} → run {run_id}")


@track_app.command("backfill")
def track_backfill(
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Import existing adapters + eval reports (v2 for all languages, Go experiments, references)."""
    _need_tracking()
    from rebirth_ai import backfill

    backfill.run(dry_run=dry_run)


@registry_app.command("list")
def registry_list(lang: str | None = typer.Option(None, "--lang")) -> None:
    """Versions per language with aliases and scores."""
    _need_tracking()
    from rebirth_ai import registry
    from rebirth_ai.tracking import client

    c = client()
    langs = [normalize_lang(lang)] if lang else list(LORA_LANGS)
    for lg in langs:
        try:
            versions = sorted(registry._versions(lg), key=lambda v: int(v.version))
        except Exception:
            versions = []
        if not versions:
            continue
        aliases = {}
        try:
            for a, ver in (c.get_registered_model(registry.model_name(lg)).aliases or {}).items():
                aliases.setdefault(str(ver), []).append(a)
        except Exception:
            pass
        t = Table(title=registry.model_name(lg))
        for col in ("v", "aliases", "base", "adapter", "recipe", "scores (soft-exact/parses)"):
            t.add_column(col)
        for v in versions:
            tags = v.tags or {}
            scores = []
            for k, val in sorted(tags.items()):
                if k.startswith("eval.") and k.endswith(".soft_exact"):
                    set_id = k[len("eval."):-len(".soft_exact")]
                    scores.append(f"{set_id.split(':')[0]} (n={tags.get(f'eval.{set_id}.n', '?')}): "
                                  f"{val}/{tags.get(f'eval.{set_id}.parses', '?')}")
            t.add_row(str(v.version), ",".join(aliases.get(str(v.version), [])), tags.get("base", ""),
                      tags.get("adapter", ""), tags.get("recipe", ""), "\n".join(scores))
        print(t)


@registry_app.command("gate")
def registry_gate(
    lang: str = typer.Option(..., "--lang"),
    version: str | None = typer.Option(None, "--version", help="Challenger version (or use --adapter)"),
    adapter: Path | None = typer.Option(None, "--adapter", help="Challenger adapter dir (resolves its version)"),
    eval_set: str | None = typer.Option(None, "--eval-set", help="Eval set id (see `registry list`)"),
    eval_file: Path | None = typer.Option(None, "--eval-file", help="Frozen eval set file (resolves the id)"),
    n: int = typer.Option(100, "--n", help="Rows scored, with --eval-file"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Evaluate the gate without promoting"),
    min_gain: int = typer.Option(1, "--min-gain", help="Soft-exact rows the challenger must gain (0 = ties promote)"),
) -> None:
    """Challenger vs champion on one eval set; promotes on pass."""
    _need_tracking()
    from rebirth_ai import registry
    from rebirth_ai.tracking import eval_set_id

    lg = normalize_lang(lang)
    if version is None:
        if adapter is None:
            raise typer.BadParameter("give --version or --adapter")
        version = registry.find_version(lg, adapter.resolve())
        if version is None:
            print(json.dumps({"lang": lg, "decision": "not_registered", "adapter": str(adapter), "passed": None}))
            raise typer.Exit(1)
    if eval_set is None:
        if eval_file is None:
            raise typer.BadParameter("give --eval-set or --eval-file")
        eval_set = eval_set_id({"suites": {"synthetic": {"n": n, "lang": lg}}}, eval_file)
    res = registry.gate(lg, version, eval_set, min_gain=min_gain, apply=not dry_run)
    print(json.dumps(res, indent=2))
    raise typer.Exit(0 if res.get("passed") else 1)


@registry_app.command("promote")
def registry_promote(
    lang: str = typer.Option(..., "--lang"),
    version: str = typer.Option(..., "--version"),
    reason: str = typer.Option(..., "--reason"),
) -> None:
    """Force a champion (e.g. the first one). Prefer `registry gate`."""
    _need_tracking()
    from rebirth_ai import registry

    registry.promote(normalize_lang(lang), version, reason=reason)
    print(f"[green]{registry.model_name(lang)} champion → v{version}[/green]")


@registry_app.command("rollback")
def registry_rollback(lang: str = typer.Option(..., "--lang")) -> None:
    """Swap champion and previous."""
    _need_tracking()
    from rebirth_ai import registry

    v = registry.rollback(normalize_lang(lang))
    print(f"[yellow]{registry.model_name(lang)} champion → v{v} (rollback)[/yellow]")


def mlflow_ui(
    port: int = typer.Option(5000, "--port"),
    host: str = typer.Option("127.0.0.1", "--host", help="Keep 127.0.0.1 (local only)"),
) -> None:
    """Open the MLflow UI on the local store (uses .venv-mlflow, the full MLflow install)."""
    from rebirth_ai.tracking import DB_PATH, tracking_uri

    exe = ROOT / ".venv-mlflow" / "bin" / "mlflow"
    if not exe.exists():
        print("[red]Missing .venv-mlflow — python3 -m venv .venv-mlflow && .venv-mlflow/bin/pip install mlflow[/red]")
        raise typer.Exit(2)
    if not DB_PATH.exists():
        print("[yellow]No runs yet — run `rebirth track backfill` first[/yellow]")
    env = {**os.environ, "MLFLOW_DISABLE_AGENT_HINT": "1", "MLFLOW_DISABLE_TELEMETRY": "true", "DO_NOT_TRACK": "1"}
    print(f"MLflow UI → http://{host}:{port}  (store: {tracking_uri()})  Ctrl-C to stop")
    subprocess.run(
        [str(exe), "ui", "--backend-store-uri", tracking_uri(), "--host", host, "--port", str(port)],
        env=env, check=False,
    )


# --------------------------------------------------------------------------- Phase 16: serving

serve_app = typer.Typer(help="Packaged Ollama models: routes (configs/routing.yaml) and rollback")


def package_cmd(
    lang: str = typer.Option(..., "--lang", "-l", help="Language id, or 'all'"),
    version: str | None = typer.Option(None, "--version", help="Registry version (default: champion)"),
    quant: str = typer.Option("q4", "--quant", help="Base quantization: q4 (Q4_K_M, default) | q8 | q6 (14B)"),
    gate: bool = typer.Option(True, "--gate/--no-gate", help="Score the served model on the frozen set"),
    publish: bool = typer.Option(True, "--publish/--no-publish", help="On pass: alias + route + model card"),
    n: int = typer.Option(100, "--n", help="Eval rows for the serving gate"),
) -> None:
    """Champion adapter → GGUF → versioned Ollama model → serving gate → route."""
    _need_tracking()
    from rebirth_ai.packaging import package

    langs = list(LORA_LANGS) if lang == "all" else [lang]
    failed = []
    for one in langs:
        entry = package(one, version=version if lang != "all" else None, quant=quant,
                        gate=gate, publish=publish, n=n)
        if gate and not entry["gate"]["passed"]:
            failed.append(one)
    if failed:
        print(f"[red]serving gate failed: {', '.join(failed)} (not routed)[/red]")
        raise typer.Exit(1)


@serve_app.command("status")
def serve_status() -> None:
    """Show which packaged model serves each language."""
    from rebirth_ai.routing import load_routing, routing_path

    data = load_routing()
    t = Table(title=f"routes ({routing_path()}) — served/peft = soft-exact/parses(/n) on the frozen sets")
    for col in ("lang", "served tag", "v", "served", "peft", "base", "gate", "rollback"):
        t.add_column(col)
    for lang in LORA_LANGS:
        e = (data.get("languages") or {}).get(lang)
        if not e:
            t.add_row(lang, "[dim]plain Ollama (fallback)[/dim]", "", "", "", "", "", "")
            continue
        s, p, b = e.get("served") or {}, e.get("peft") or {}, e.get("base") or {}
        t.add_row(
            lang, e["tag"], str(e["version"]),
            f"{s.get('soft_exact', '?'):g}/{s.get('parses', '?'):g}/{s.get('n', '?'):g}" if s else "",
            f"{p.get('soft_exact'):g}/{p.get('parses'):g}" if p else "",
            f"{b.get('soft_exact'):g}" if b else "",
            "pass" if (e.get("gate") or {}).get("passed") else "?",
            ", ".join(f"v{v}" for v in (e.get("history") or {})),
        )
    x = data.get("translation")
    if x:
        ev = x.get("eval") or {}
        score = " · ".join(f"{k} {v['pass']}/{v['n']}" for k, v in ev.items())
        t.add_row("--to (xlang)", x["tag"], str(x["version"]), score, "", "",
                  "pass" if (x.get("gate") or {}).get("passed") else "?",
                  ", ".join(f"v{v}" for v in (x.get("history") or {})))
    else:
        t.add_row("--to (xlang)", "[dim]rebirth-xlang:14b if installed, else plain 14B (no route)[/dim]", "", "", "", "", "", "")
    print(t)


@serve_app.command("rollback")
def serve_rollback_cmd(lang: str = typer.Option(..., "--lang", "-l", help="language id, or xlang (--to model)")) -> None:
    """Roll back champion ↔ previous in the registry AND re-point the served Ollama alias.
    ``--lang xlang``: swap the cross-language model with its previous version (again = roll forward)."""
    if lang.strip().lower() in {"xlang", "translation", "to"}:
        from rebirth_ai.packaging import rollback_translation

        try:
            rollback_translation()
        except RuntimeError as exc:
            print(f"[red]{exc}[/red]")
            raise typer.Exit(1) from exc
        return
    _need_tracking()
    from rebirth_ai.packaging import serve_rollback

    try:
        serve_rollback(lang)
    except (RuntimeError, ValueError) as exc:
        print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc


@serve_app.command("publish-xlang")
def serve_publish_xlang(
    tag: str = typer.Option(..., "--tag", help="candidate Ollama tag, e.g. rebirth-xlang:14b-v2"),
    version: str = typer.Option(..., "--version"),
    reports: list[Path] = typer.Option(..., "--report", exists=True, help="candidate eval_translate reports"),
    baseline: list[Path] = typer.Option(..., "--baseline", exists=True,
                                        help="reports of the model it replaces, same translations"),
    note: str = typer.Option("", "--note", help="what changed (goes into the route)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="show the gate only"),
) -> None:
    """Gate a cross-language model against the one it replaces, then serve it as rebirth-xlang:14b."""
    from rebirth_ai.packaging import publish_translation, xlang_gate, xlang_scores

    new, old = xlang_scores(reports), xlang_scores(baseline)
    g = xlang_gate(new, old)
    for k, ok in g.checks.items():
        print(f"  {'[green]✓[/green]' if ok else '[red]✗[/red]'} {k}")
    for w in g.warnings:
        print(f"  [yellow]note[/yellow] {w}")
    if not g.passed:
        print("[red]gate failed — not published[/red]")
        raise typer.Exit(1)
    if dry_run:
        print("[green]gate passed[/green] (dry run, nothing changed)")
        return
    from rebirth_ai.lora import rel_to_root

    entry = {"tag": tag, "version": str(version), "eval": new, "baseline": old,
             "gate": {"passed": g.passed, "checks": g.checks, "warnings": g.warnings},
             "reports": [rel_to_root(Path(r).resolve()) for r in reports], **({"note": note} if note else {})}
    try:
        publish_translation(entry)
    except RuntimeError as exc:
        print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
