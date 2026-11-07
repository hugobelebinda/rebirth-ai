"""CLI entrypoint for Rebirth AI."""

from __future__ import annotations

from pathlib import Path

import typer
from rich import print
from rich.table import Table

from rebirth_ai import __version__

app = typer.Typer(help="Rebirth AI — autonomous migration agent", no_args_is_help=True)

# Phase 13 MLOps: experiment tracking + model registry (MLflow; optional ".[mlops]" extra)
from rebirth_ai.mlops_cli import mlflow_ui, package_cmd, registry_app, serve_app, track_app  # noqa: E402

app.add_typer(track_app, name="track")
app.add_typer(registry_app, name="registry")
app.add_typer(serve_app, name="serve")  # Phase 16
app.command("mlflow-ui")(mlflow_ui)
app.command("package")(package_cmd)  # Phase 16


@app.command()
def version() -> None:
    """Print package version."""
    print(f"rebirth-ai {__version__}")


@app.command()
def doctor() -> None:
    """Check local prerequisites (Docker, Ollama, sandbox images, Python)."""
    import shutil
    import subprocess
    import sys

    print(f"[bold]Python[/bold]: {sys.version.split()[0]}")
    print(f"[bold]docker[/bold]: {'ok' if shutil.which('docker') else 'MISSING'}")
    print(f"[bold]ollama[/bold]: {'ok' if shutil.which('ollama') else 'MISSING'}")
    print(f"[bold]node[/bold]: {'ok' if shutil.which('node') else 'MISSING'}")

    try:
        r = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        print(f"[bold]Docker daemon[/bold]: {'ok' if r.returncode == 0 else 'NOT RUNNING'}")
    except Exception as exc:
        print(f"[bold]Docker daemon[/bold]: error ({exc})")

    try:
        from docker.errors import ImageNotFound

        import docker
        from rebirth_ai.adapters.registry import adapter_summaries

        client = docker.from_env()
        for row in adapter_summaries():
            image = str(row["image"])
            label = f"Sandbox ({row['id']})"
            try:
                client.images.get(image)
                print(f"[bold]{label}[/bold]: ok ({image})")
            except ImageNotFound:
                hint = row.get("dockerfile_dir") or "docker/<lang>_sandbox"
                print(f"[bold]{label}[/bold]: MISSING ({image}) — docker build -t {image} {hint}")
    except Exception as exc:
        print(f"[bold]Sandbox images[/bold]: error ({exc})")

    try:
        import httpx

        resp = httpx.get("http://127.0.0.1:11434/api/tags", timeout=5.0)
        if resp.status_code == 200:
            models = [m.get("name") for m in resp.json().get("models", [])]
            print(f"[bold]Ollama[/bold]: ok ({len(models)} models)")
            for name in models[:8]:
                print(f"  - {name}")
        else:
            print(f"[bold]Ollama[/bold]: HTTP {resp.status_code}")

        ps = httpx.get("http://127.0.0.1:11434/api/ps", timeout=5.0)
        if ps.status_code == 200:
            loaded = ps.json().get("models", [])
            if not loaded:
                print("[bold]Ollama loaded[/bold]: (none)")
            resident_vram = 0.0
            for m in loaded:
                size = m.get("size") or 0
                vram = m.get("size_vram") or 0
                resident_vram += float(vram or 0)
                pct = (100.0 * vram / size) if size else 0.0
                where = "GPU" if vram and pct >= 99 else ("mixed" if vram else "CPU")
                print(
                    f"[bold]Ollama loaded[/bold]: {m.get('name')} "
                    f"{vram/1e9:.1f}/{size/1e9:.1f}GB → {where} ({pct:.0f}% VRAM)"
                )
            # Phase 6: warn before stacking Ollama resident weights + PEFT 7B + ST
            peft_requested = (
                __import__("os").environ.get("REBIRTH_GENERATOR", "").strip().lower()
                in {"peft", "lora", "local"}
            )
            if resident_vram >= 4e9 or (loaded and peft_requested):
                why = (
                    "REBIRTH_GENERATOR=peft while Ollama is loaded"
                    if peft_requested and loaded
                    else "Ollama has a large model resident"
                )
                print(
                    f"[yellow]VRAM warning[/yellow]: {why}. "
                    "Do not also run `--backend peft` (7B) or SentenceTransformers on the "
                    "same GPU — unload first: "
                    "`ollama stop qwen2.5-coder:14b-instruct-q6_K` "
                    "(or set REBIRTH_EMBEDDER=hash / REBIRTH_PEFT_DEVICE=cpu)."
                )
            elif loaded:
                print(
                    "[dim]tip: unload Ollama before PEFT (`ollama stop <model>`) "
                    "to avoid VRAM stacking[/dim]"
                )
            else:
                print(
                    "[dim]PEFT policy: use hash RAG (`REBIRTH_EMBEDDER=hash`); "
                    "never stack Ollama 14B + PEFT 7B + SentenceTransformers[/dim]"
                )
    except Exception as exc:
        print(f"[bold]Ollama[/bold]: not reachable ({exc})")


@app.command("adapters")
def list_adapters() -> None:
    """List registered language adapters (id, extensions, sandbox image)."""
    from rebirth_ai.adapters.registry import adapter_summaries

    table = Table(title="Language adapters")
    table.add_column("id")
    table.add_column("extensions")
    table.add_column("image")
    table.add_column("test_command")
    for row in adapter_summaries():
        table.add_row(
            str(row["id"]),
            " ".join(row["extensions"]),  # type: ignore[arg-type]
            str(row["image"]),
            str(row["test_command"]),
        )
    print(table)


@app.command("parse")
def parse_file(
    path: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    lang: str | None = typer.Option(
        None,
        "--lang",
        "-l",
        help="Adapter id override (js_ts | python); default = detect from suffix",
    ),
) -> None:
    """AST-chunk a source file and print structured chunks."""
    from rebirth_ai.adapters.registry import get_adapter

    if lang:
        adapter = get_adapter(lang)
    else:
        adapter = get_adapter("python" if path.suffix.lower() == ".py" else "js_ts")
    if path.suffix.lower() not in set(getattr(adapter, "SOURCE_SUFFIXES", set())):
        raise typer.BadParameter(f"Unsupported source suffix for {adapter.name}: {path.suffix}")
    chunks = adapter.parse(path)
    table = Table(title=f"Chunks: {path.name}")
    table.add_column("kind")
    table.add_column("name")
    table.add_column("lines")
    for c in chunks:
        table.add_row(c.kind, c.name, f"{c.start_line}-{c.end_line}")
    print(table)
    print(f"[dim]{len(chunks)} chunk(s)[/dim]")


@app.command("sandbox")
def sandbox_run(
    project: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=False,
        dir_okay=True,
        help="Project root to mount into the sandbox",
    ),
    test: str | None = typer.Option(
        None,
        "--test",
        "-t",
        help="Container-relative test path (e.g. test/math.test.js)",
    ),
    lang: str | None = typer.Option(
        None,
        "--lang",
        "-l",
        help="Adapter id override (js_ts | python); default = auto-detect",
    ),
) -> None:
    """Run fixture/project tests inside the ephemeral Docker sandbox."""
    from rebirth_ai.adapters.registry import get_adapter_for_project
    from rebirth_ai.sandbox.runner import SandboxRunner

    adapter = get_adapter_for_project(project, lang=lang)
    runner = SandboxRunner(adapter)
    result = runner.run_project(project, relative_test=test)
    status = "[green]PASS[/green]" if result.ok else "[red]FAIL[/red]"
    print(f"{status} exit={result.exit_code} in {result.duration_sec:.2f}s adapter={adapter.name}")
    print(f"[dim]cmd:[/dim] {result.command}")
    if result.stdout:
        print("[bold]stdout[/bold]")
        # Escape Rich markup — test logs often contain [brackets]
        print(result.stdout.replace("[", "\\["))
    if result.stderr:
        print("[bold]stderr[/bold]")
        print(result.stderr.replace("[", "\\["))
    raise typer.Exit(code=0 if result.ok else 1)


@app.command("index")
def index_project(
    project: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=False,
        help="Project root to index into the vector DB",
    ),
    db: Path = typer.Option(
        Path("data/embeddings.db"),
        "--db",
        help="SQLite vector DB path",
    ),
) -> None:
    """AST-chunk + embed a project into SQLite for RAG."""
    from rebirth_ai.rag.retrieve import Retriever

    retriever = Retriever(db_path=db)
    stats = retriever.index_project(project)
    print(
        f"[green]indexed[/green] files={stats.files} chunks={stats.chunks} db={stats.db_path}"
    )


@app.command("query")
def query_index(
    text: str = typer.Argument(..., help="Free-text retrieval query"),
    db: Path = typer.Option(Path("data/embeddings.db"), "--db"),
    top_k: int = typer.Option(5, "--top-k", "-k"),
) -> None:
    """Query the vector index for relevant code chunks."""
    from rebirth_ai.rag.retrieve import Retriever

    retriever = Retriever(db_path=db)
    hits = retriever.query(text, top_k=top_k)
    table = Table(title="RAG hits")
    table.add_column("score")
    table.add_column("kind")
    table.add_column("name")
    table.add_column("file")
    for h in hits:
        table.add_row(f"{h.score:.3f}", h.kind, h.name, h.file_path)
    print(table)


@app.command("train-classifier")
def train_classifier(
    dataset: Path = typer.Option(
        Path("data/datasets/error_logs.jsonl"),
        "--dataset",
        exists=True,
        dir_okay=False,
    ),
    model_dir: Path = typer.Option(
        Path("data/models/error_classifier"),
        "--model-dir",
    ),
) -> None:
    """Train the TF-IDF + RandomForest error classifier."""
    from rebirth_ai.classify.model import ErrorClassifier

    _, report = ErrorClassifier.train_from_jsonl(dataset, model_dir=model_dir)
    print(f"[green]trained[/green] accuracy={report.accuracy:.3f} → {report.model_dir}")
    print(report.report)


@app.command("generate")
def generate_migration(
    path: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    objective: str = typer.Option(
        "Modernize legacy JavaScript/TypeScript to current idioms while preserving behavior",
        "--objective",
        "-o",
    ),
    use_rag: bool = typer.Option(True, "--rag/--no-rag", help="Attach retrieved context"),
    db: Path = typer.Option(Path("data/embeddings.db"), "--db"),
    out: Path | None = typer.Option(None, "--out", help="Write draft code to this path"),
    backend: str = typer.Option(
        "auto",
        "--backend",
        "-b",
        help="auto (default: packaged adapter from configs/routing.yaml, else plain Ollama) | ollama | peft",
    ),
) -> None:
    """Generate a modernized draft via Ollama or local PEFT LoRA (optional RAG)."""
    from rebirth_ai.generate.factory import get_generator

    source = path.read_text(encoding="utf-8")
    language = "typescript" if path.suffix.lower() in {".ts", ".tsx"} else "javascript"
    context: list[str] = []
    if use_rag and db.exists():
        from rebirth_ai.rag.retrieve import Retriever

        retriever = Retriever(db_path=db)
        hits = retriever.query_for_file(path, top_k=4)
        context = [f"// {h.file_path} :: {h.kind} {h.name}\n{h.text}" for h in hits]

    client = get_generator(backend, language=language)
    result = client.migrate(
        objective=objective,
        legacy_code=source,
        file_path=str(path),
        language=language,
        retrieved_context=context or None,
    )
    print(f"[dim]model={result.model} prompt_chars={result.prompt_chars}[/dim]")
    print(result.code)
    if out is not None:
        out.write_text(result.code, encoding="utf-8")
        print(f"[green]wrote[/green] {out}")


@app.command("mine-dataset")
def mine_dataset(
    max_pairs: int = typer.Option(1500, "--max-pairs"),
    require_parse: bool = typer.Option(
        False,
        "--require-parse/--no-require-parse",
        help="Drop pairs that fail Tree-sitter parse",
    ),
    dry_run: bool = typer.Option(False, "--dry-run"),
    langs: list[str] = typer.Option(
        [],
        "--lang",
        help="CommitPackFT language(s): javascript, typescript, python (repeatable)",
    ),
) -> None:
    """Mine migration pairs from CommitPackFT (JS/TS and/or Python)."""
    import subprocess
    import sys
    from pathlib import Path

    # src/rebirth_ai/cli.py → repo root is parents[2]
    script = Path(__file__).resolve().parents[2] / "training" / "mine_migration_pairs.py"
    cmd = [sys.executable, str(script), "--max-pairs", str(max_pairs)]
    if require_parse:
        cmd.append("--require-parse")
    if dry_run:
        cmd.append("--dry-run")
    if langs:
        cmd.extend(["--langs", *langs])
    raise typer.Exit(subprocess.call(cmd))


@app.command("migrate")
def migrate_file(
    project: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=False,
        help="Project root (sandbox + relative paths)",
    ),
    file: str = typer.Option(
        "src/math.js",
        "--file",
        "-f",
        help="Project-relative source file to migrate",
    ),
    test: str = typer.Option(
        "test/math.test.js",
        "--test",
        "-t",
        help="Project-relative test file",
    ),
    objective: str = typer.Option(
        "Fix the file so project tests pass; modernize idioms without changing public API",
        "--objective",
        "-o",
    ),
    seed_draft: Path | None = typer.Option(
        None,
        "--seed-draft",
        exists=True,
        dir_okay=False,
        help="Broken draft to start from (enables --test-first)",
    ),
    test_first: bool = typer.Option(
        False,
        "--test-first/--generate-first",
        help="Run tests on current draft before first generate",
    ),
    write: bool = typer.Option(
        False,
        "--write/--no-write",
        help="Write successful draft back onto the project file",
    ),
    db: Path = typer.Option(Path("data/embeddings.db"), "--db"),
    max_iterations: int = typer.Option(5, "--max-iterations"),
    out: Path | None = typer.Option(None, "--out", help="Also write final draft here"),
    backend: str = typer.Option(
        "auto",
        "--backend",
        "-b",
        help="auto (default: packaged adapter from configs/routing.yaml, else plain Ollama) | ollama | peft",
    ),
    lang: str | None = typer.Option(
        None,
        "--lang",
        "-l",
        help="Adapter id override (js_ts | python); default = auto-detect",
    ),
) -> None:
    """Closed-loop LangGraph migrate: retrieve → generate → test → classify → route."""
    import os

    from rebirth_ai.graph.app import build_migration_graph, initial_state

    os.environ["REBIRTH_GENERATOR"] = backend
    # PEFT 7B already fills ~14GB VRAM — do not also load SentenceTransformers on GPU
    if backend.strip().lower() in {"peft", "lora", "local"}:
        os.environ.setdefault("REBIRTH_EMBEDDER", "hash")
        # Phase 6: agent/migrate path prefers chat (SFT truncates whole-file repairs)
        os.environ.setdefault("REBIRTH_PEFT_PROMPT_STYLE", "chat")
        print(
            "[dim]peft backend: hash RAG + chat prompts "
            "(avoids ST+7B VRAM stack / SFT truncation)[/dim]"
        )
    seed_text = seed_draft.read_text(encoding="utf-8") if seed_draft else None
    use_test_first = test_first or seed_draft is not None
    state = initial_state(
        project_dir=project,
        relative_file=file,
        relative_test=test,
        objective=objective,
        seed_draft=seed_text,
        test_first=use_test_first,
        db_path=db,
        max_iterations=max_iterations,
        adapter_id=lang,
    )
    from rebirth_ai.runs import write_run_log

    graph = build_migration_graph()
    print("[dim]running migration graph…[/dim]")
    final = graph.invoke(state)

    status = final.get("status", "unknown")
    color = {
        "success": "green",
        "human_review": "yellow",
        "failed": "red",
    }.get(status, "white")
    print(f"[{color}]status={status}[/{color}] iterations={final.get('iteration_count')}")
    print(f"[dim]error_class={final.get('error_class')}[/dim]")
    for step in final.get("trace") or []:
        print(f"  • {step}")

    draft = final.get("current_draft") or ""
    if out is not None:
        out.write_text(draft, encoding="utf-8")
        print(f"[green]wrote[/green] {out}")
    if write and status == "success":
        target = Path(final["file_path"])
        target.write_text(draft, encoding="utf-8")
        print(f"[green]wrote[/green] {target}")

    run_path = write_run_log(
        kind="migrate",
        payload={
            "project": str(project),
            "file": file,
            "test": test,
            "lang": final.get("adapter_id") or lang,
            "language": final.get("language"),
            "backend": backend,
            "status": status,
            "iterations": final.get("iteration_count"),
            "error_class": final.get("error_class"),
            "trace": final.get("trace") or [],
            "test_logs": (final.get("test_logs") or "")[:4000],
        },
    )
    print(f"[dim]run log[/dim] {run_path}")

    raise typer.Exit(code=0 if status == "success" else 1)


@app.command("migrate-repo")
def migrate_repo(
    project: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=False,
        help="Project root to walk and migrate",
    ),
    objective: str = typer.Option(
        "Fix files so project tests pass; modernize idioms without changing public API",
        "--objective",
        "-o",
    ),
    under: str = typer.Option(
        "src",
        "--under",
        help="Comma-separated roots to discover sources (e.g. src,lib,app)",
    ),
    file: list[str] = typer.Option(
        [],
        "--file",
        "-f",
        help="Limit to these project-relative files (repeatable)",
    ),
    seed: list[str] = typer.Option(
        [],
        "--seed",
        help="Broken draft map: relative/file.js=path/to/seed.js (repeatable)",
    ),
    write: bool = typer.Option(
        False,
        "--write/--no-write",
        help="Write successful drafts back onto project files",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Plan only: discover + topo-order + test map; no LLM / writes",
    ),
    db: Path = typer.Option(Path("data/embeddings.db"), "--db"),
    max_iterations: int = typer.Option(5, "--max-iterations"),
    backend: str = typer.Option(
        "auto",
        "--backend",
        "-b",
        help="auto (default: packaged adapter from configs/routing.yaml, else plain Ollama) | ollama | peft",
    ),
    report: Path = typer.Option(
        Path("data/reports/migrate_repo.json"),
        "--report",
        help="Write JSON batch report here",
    ),
    open_pr: bool = typer.Option(
        False,
        "--open-pr/--no-open-pr",
        help="After migrate, branch/commit/push and open a GitHub PR",
    ),
    pr_branch: str = typer.Option("rebirth/migrate", "--pr-branch"),
    pr_base: str | None = typer.Option(
        None,
        "--pr-base",
        help="PR base branch (default: auto-detect main/master)",
    ),
    draft: bool = typer.Option(
        False,
        "--draft/--no-draft",
        help="When --open-pr, create a draft PR",
    ),
    push: bool = typer.Option(
        True,
        "--push/--no-push",
        help="When --open-pr, push the branch before creating the PR",
    ),
    lang: str | None = typer.Option(
        None,
        "--lang",
        "-l",
        help="Adapter id override (js_ts | python); default = auto-detect",
    ),
) -> None:
    """Phase 5–6: migrate many files (topo-ordered); dry-run; optional GitHub PR."""
    import os

    from rebirth_ai.github import GitHubError, open_migration_pr
    from rebirth_ai.graph.batch import migrate_repository, render_pr_body

    os.environ["REBIRTH_GENERATOR"] = backend
    if backend.strip().lower() in {"peft", "lora", "local"}:
        os.environ.setdefault("REBIRTH_EMBEDDER", "hash")
        os.environ.setdefault("REBIRTH_PEFT_PROMPT_STYLE", "chat")
        print(
            "[dim]peft backend: hash RAG + chat prompts "
            "(avoids ST+7B VRAM stack / SFT truncation)[/dim]"
        )
    mode = "dry-run" if dry_run else ("write" if write else "no-write")
    print(
        f"[dim]migrate-repo under={under!r} backend={backend} mode={mode} lang={lang or 'auto'}[/dim]"
    )

    batch = migrate_repository(
        project,
        objective=objective,
        files=list(file) or None,
        under=under,
        seeds=list(seed) or None,
        write=write and not dry_run,
        dry_run=dry_run,
        db_path=db,
        max_iterations=max_iterations,
        lang=lang,
        on_file=lambda p: print(f"[cyan]→[/cyan] {p}"),
    )
    batch.write_json(report)
    print(render_pr_body(batch))
    # Show patch previews when not writing
    if not dry_run and not write:
        for r in batch.results:
            if r.draft_preview:
                print(f"[bold]diff[/bold] {r.relative_file}")
                print(r.draft_preview.replace("[", "\\["))
    print(f"[green]report[/green] {report}")

    from rebirth_ai.runs import write_run_log

    run_path = write_run_log(
        kind="migrate-repo",
        payload={
            "project": str(project),
            "lang": lang,
            "backend": backend,
            "dry_run": dry_run,
            "report": str(report),
            "summary": batch.to_dict().get("summary"),
            "results": [
                {
                    "file": r.relative_file,
                    "status": r.status,
                    "iterations": r.iterations,
                    "error_class": r.error_class,
                }
                for r in batch.results
            ],
        },
    )
    print(f"[dim]run log[/dim] {run_path}")

    if open_pr and not dry_run:
        try:
            result = open_migration_pr(
                Path.cwd(),
                batch,
                branch=pr_branch,
                base=pr_base,
                push=push,
                create_pr=True,
                draft=draft,
                report_path=report if report.exists() else None,
            )
            print(f"[green]{result.message}[/green]")
            if result.url:
                print(result.url)
        except GitHubError as exc:
            print(f"[red]PR failed:[/red] {exc}")
            raise typer.Exit(code=2) from exc
    elif open_pr and dry_run:
        print("[yellow]skipping --open-pr during --dry-run[/yellow]")

    if dry_run:
        planned = sum(1 for r in batch.results if r.status == "planned")
        raise typer.Exit(code=0 if planned or batch.results else 1)

    # Exit 0 if at least one success and no hard failures, else 1 when any failed
    raise typer.Exit(code=0 if batch.succeeded and not batch.failed else 1)


@app.command("eval")
def eval_cmd(
    suite: str = typer.Option(
        "heldout",
        "--suite",
        help="sandbox | heldout | synthetic | both | all | fixtures",
    ),
    backend: str = typer.Option("ollama", "--backend", "-b"),
    lang: str = typer.Option(
        "js_ts",
        "--lang",
        "-l",
        help="Adapter id or 'all' (fixtures suite supports all 8 langs)",
    ),
    out: Path = typer.Option(
        Path("data/reports/eval_behavioral.json"),
        "--out",
    ),
    freeze_baseline: Path | None = typer.Option(
        None,
        "--freeze-baseline",
        help="Write baseline snapshot (e.g. data/reports/baselines/ollama_js_ts.json)",
    ),
    manifest: Path = typer.Option(
        Path("fixtures/eval_suite/manifest.json"),
        "--manifest",
    ),
    target: float = typer.Option(0.90, "--target", help="Pass-rate gate for fixtures suite"),
) -> None:
    """Run behavioral / held-out / multi-lang fixture sandbox evals."""
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[2]
    if suite == "fixtures" or lang == "all":
        script = root / "training" / "eval_fixture_sandboxes.py"
        cmd = [
            sys.executable,
            str(script),
            "--lang",
            "all" if lang == "all" else lang,
            "--target",
            str(target),
            "--out",
            str(out if out != Path("data/reports/eval_behavioral.json") else Path("data/reports/eval_fixture_sandboxes.json")),
        ]
        raise typer.Exit(subprocess.call(cmd))

    script = root / "training" / "eval_behavioral.py"
    cmd = [
        sys.executable,
        str(script),
        "--backend",
        backend,
        "--suite",
        suite,
        "--out",
        str(out),
        "--manifest",
        str(manifest),
    ]
    if freeze_baseline is not None:
        cmd.extend(["--freeze-baseline", str(freeze_baseline)])
    raise typer.Exit(subprocess.call(cmd))


@app.command("report")
def report_cmd(
    path: Path = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        readable=True,
        help="Batch / eval / run summary JSON",
    ),
) -> None:
    """Pretty-print a migrate-repo / eval / run JSON report (secrets redacted)."""
    from rebirth_ai.report_view import load_report, render_report

    render_report(load_report(path))


@app.command("classify")
def classify_log(
    log_file: Path | None = typer.Option(
        None,
        "--file",
        "-f",
        exists=True,
        dir_okay=False,
        help="Path to a log file (otherwise read stdin text arg)",
    ),
    text: str | None = typer.Argument(None, help="Raw log text if --file not set"),
    model_dir: Path = typer.Option(
        Path("data/models/error_classifier"),
        "--model-dir",
    ),
) -> None:
    """Classify a test/CI failure log into a discrete error class."""
    from rebirth_ai.classify.model import ErrorClassifier

    if log_file is not None:
        payload = log_file.read_text(encoding="utf-8")
    elif text:
        payload = text
    else:
        raise typer.BadParameter("Provide TEXT or --file")

    clf = ErrorClassifier.load(model_dir)
    label = clf.predict(payload)
    probs = clf.predict_proba(payload)
    print(f"[bold]{label}[/bold]")
    for cls_name, p in sorted(probs.items(), key=lambda kv: kv[1], reverse=True):
        print(f"  {cls_name}: {p:.3f}")


if __name__ == "__main__":
    app()

