"""CLI entrypoint for Rebirth AI."""

from __future__ import annotations

from pathlib import Path
from typing import NoReturn

import typer
from rich import print
from rich.table import Table
from typer.core import TyperGroup

from rebirth_ai import __version__
from rebirth_ai.paths import CLASSIFIER_DIR, DEFAULT_DB, DEFAULT_REPORT, ERROR_LOGS, tilde

# `rebirth --help`: commands grouped by who needs them, in this order (assigned at the end of the file).
PANELS = {
    "Migrate": ["migrate", "migrate-repo"],
    "Setup": ["doctor", "install", "version"],
    "Tools": ["index", "query", "sandbox", "report", "dashboard", "parse", "classify", "adapters", "generate"],
    "Maintainers: models, data and monitoring": [
        "eval", "monitor", "review", "registry", "serve", "package", "track", "mlflow-ui", "mine-dataset",
        "train-classifier",
    ],
}
_ORDER = {name: i for i, name in enumerate(n for names in PANELS.values() for n in names)}


class _OrderedGroup(TyperGroup):
    def list_commands(self, ctx):  # click.s signature
        return sorted(super().list_commands(ctx), key=lambda n: _ORDER.get(n, len(_ORDER)))


app = typer.Typer(
    cls=_OrderedGroup,
    help="Rebirth AI: modernize legacy code, or translate it to another language, checked by your own tests. "
    "Run `rebirth` with no command for the guided start screen.",
)


@app.callback(invoke_without_command=True)
def _main(
    ctx: typer.Context,
    show_version: bool = typer.Option(False, "--version", "-V", is_eager=True, help="Print the version and exit."),
) -> None:
    if show_version:
        print(f"rebirth-ai {__version__}")
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        from rebirth_ai.start import start

        raise typer.Exit(start())


# Phase 13 MLOps: experiment tracking + model registry (MLflow; optional ".[mlops]" extra)
from rebirth_ai.mlops_cli import mlflow_ui, package_cmd, registry_app, serve_app, track_app  # noqa: E402

app.add_typer(track_app, name="track")
app.add_typer(registry_app, name="registry")
app.add_typer(serve_app, name="serve")  # Phase 16
app.command("mlflow-ui")(mlflow_ui)
app.command("package")(package_cmd)  # Phase 16

from rebirth_ai.monitor_cli import dashboard_cmd, monitor_app, review_app  # noqa: E402

app.command("dashboard")(dashboard_cmd)  # Phase 18
app.add_typer(monitor_app, name="monitor")
app.add_typer(review_app, name="review")


def _fail(message: str, code: int = 2) -> NoReturn:
    """Print an error without a traceback and exit (2 = bad input / environment)."""
    print(f"[red]error:[/red] {message}")
    raise typer.Exit(code=code)


def _preflight(*, ollama: bool = False, docker: bool = False) -> None:
    from rebirth_ai import preflight

    for problem in ((preflight.ollama_error() if ollama else None), (preflight.docker_error() if docker else None)):
        if problem:
            _fail(problem)


@app.command()
def version() -> None:
    """Print the version."""
    print(f"rebirth-ai {__version__}")


@app.command()
def install(
    pack: str = typer.Option("both", "--pack", help="both | modernize (7B, 8 languages) | migrate (14B, --to)"),
    from_dir: Path | None = typer.Option(None, "--from", help="folder with the release files (mirror / USB)"),
    from_local: bool = typer.Option(False, "--from-local", help="use this repo's own build outputs"),
    force: bool = typer.Option(False, "--force", help="re-create models that are already installed"),
    keep_files: bool = typer.Option(False, "--keep-files", help="keep downloaded files after ollama create"),
    dry_run: bool = typer.Option(False, "--dry-run", help="show the plan only"),
) -> None:
    """Download the model packs, check their sha256 and register them in Ollama."""
    from rebirth_ai.install import InstallError
    from rebirth_ai.install import install as run_install

    try:
        steps = run_install(pack, from_dir=from_dir, from_local=from_local, force=force,
                            keep_files=keep_files, dry_run=dry_run)
    except InstallError as exc:
        print(f"[red]install failed:[/red] {exc}")
        raise typer.Exit(1) from exc
    if dry_run:
        for s in steps:
            print(f"{s.action:7s} {s.pack:9s} {s.tag:24s} ← {s.source}")
    else:
        print(f"[green]{sum(s.action == 'install' for s in steps)} installed, "
              f"{sum(s.action == 'skip' for s in steps)} already present[/green] — run `rebirth doctor`")


def _gpu_vram_gb() -> tuple[str, float] | None:
    """(name, total VRAM GB) of the first GPU from nvidia-smi or rocm-smi, else None."""
    import json
    import shutil
    import subprocess

    if shutil.which("nvidia-smi"):
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10, check=False)
        if r.returncode == 0 and r.stdout.strip():
            name, mib = r.stdout.splitlines()[0].rsplit(",", 1)
            return name.strip(), float(mib) / 1024
    if shutil.which("rocm-smi"):
        r = subprocess.run(["rocm-smi", "--showmeminfo", "vram", "--json"], capture_output=True, text=True,
                           timeout=10, check=False)
        try:
            card = next(iter(json.loads(r.stdout).values()))
            total = next(v for k, v in card.items() if "Total Memory" in k)
            return "AMD GPU", float(total) / 1e9
        except Exception:
            pass
    # amdgpu kernel driver, no ROCm tools needed (ROCm may come from pip wheels only)
    sizes = []
    for f in Path("/sys/class/drm").glob("card*/device/mem_info_vram_total"):
        try:
            sizes.append(int(f.read_text().strip()))
        except (OSError, ValueError):
            continue
    if sizes:
        return "AMD GPU (amdgpu)", max(sizes) / 2**30
    return None


@app.command()
def doctor(
    strict: bool = typer.Option(False, "--strict", help="exit 1 if a requirement for the installed packs is missing"),
) -> None:
    """Check that everything Rebirth needs is installed and running (Docker, Ollama, models, GPU, disk)."""
    import shutil
    import subprocess
    import sys

    problems: list[str] = []
    py_ok = sys.version_info >= (3, 11)
    print(f"[bold]Python[/bold]: {sys.version.split()[0]}" + ("" if py_ok else " — needs ≥ 3.11"))
    if not py_ok:
        problems.append("Python < 3.11")
    for tool in ("docker", "ollama", "git"):
        ok = bool(shutil.which(tool))
        print(f"[bold]{tool}[/bold]: {'ok' if ok else 'MISSING'}")
        if not ok:
            problems.append(f"{tool} missing")
    print(f"[bold]node[/bold]: {'ok' if shutil.which('node') else 'missing (only needed for host-side JS tools)'}")

    free = shutil.disk_usage(Path.cwd()).free / 1e9
    print(f"[bold]Disk free[/bold]: {free:.0f} GB" + (" — low (packs need ~17 GB)" if free < 20 else ""))
    gpu = _gpu_vram_gb()
    if gpu:
        name, vram = gpu
        note = "" if vram >= 15 else (" — migrate pack (14B q6_K) needs ~16 GB; modernize (7B) fits"
                                      if vram >= 7 else " — below 8 GB: models run partly on CPU (slow)")
        print(f"[bold]GPU[/bold]: {name}, {vram:.0f} GB VRAM{note}")
    else:
        print("[bold]GPU[/bold]: not detected (nvidia-smi / rocm-smi missing) — Ollama may still use one")

    try:
        from rebirth_ai.install import status as pack_status

        packs = pack_status()
        for name, st in packs.items():
            ok = st["installed"] == st["total"]
            line = f"{st['installed']}/{st['total']} models"
            if not ok:
                line += f" — missing {', '.join(st['missing'][:3])}{' …' if len(st['missing']) > 3 else ''}"
                line += f" (rebirth install --pack {name})"
            print(f"[bold]Pack {name}[/bold]: {'ok, ' if ok else ''}{line}")
            if 0 < st["installed"] < st["total"]:
                problems.append(f"pack {name} incomplete")
        if not any(st["installed"] for st in packs.values()):
            problems.append("no model pack installed (rebirth install)")
    except Exception as exc:
        print(f"[bold]Model packs[/bold]: error ({exc})")
        problems.append("model packs not checked (Ollama not reachable?)")

    try:
        r = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        print(f"[bold]Docker daemon[/bold]: {'ok' if r.returncode == 0 else 'NOT RUNNING'}")
        if r.returncode != 0:
            problems.append("Docker daemon not reachable (running? user in the docker group?)")
    except Exception as exc:
        print(f"[bold]Docker daemon[/bold]: error ({exc})")
        problems.append("Docker daemon not reachable")

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
                problems.append(f"sandbox image {image} missing (bash scripts/bootstrap.sh builds all 8)")
    except Exception as exc:
        print(f"[bold]Sandbox images[/bold]: error ({exc})")
        problems.append("sandbox images not checked (Docker not reachable?)")

    try:
        import httpx

        from rebirth_ai.generate.ollama_client import ollama_host

        resp = httpx.get(f"{ollama_host()}/api/tags", timeout=5.0)
        if resp.status_code == 200:
            models = [m.get("name") for m in resp.json().get("models", [])]
            print(f"[bold]Ollama[/bold]: ok ({len(models)} models)")
            for name in models[:8]:
                print(f"  - {name}")
        else:
            print(f"[bold]Ollama[/bold]: HTTP {resp.status_code}")
            problems.append(f"Ollama server answered HTTP {resp.status_code}")

        ps = httpx.get(f"{ollama_host()}/api/ps", timeout=5.0)
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
        problems.append("Ollama server not reachable (ollama serve / systemctl start ollama)")

    if problems:
        print(f"[yellow]{len(problems)} problem(s):[/yellow]")
        for p in dict.fromkeys(problems):
            print(f"  - {p}")
    else:
        print("[green]all checks passed[/green]")
    if strict and problems:
        raise typer.Exit(1)


@app.command("adapters")
def list_adapters() -> None:
    """List the 8 languages: id, file types, test command and sandbox image."""
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
        help="Language (js_ts | python | go | java | csharp | rust | cpp | swift); default: from the file suffix",
    ),
) -> None:
    """Split a source file into functions and classes, as code search sees it."""
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
        help="Language (js_ts | python | go | java | csharp | rust | cpp | swift); default: detected from the project",
    ),
) -> None:
    """Run a project's tests in the Docker sandbox, as Rebirth does."""
    from rebirth_ai.adapters.registry import detect_adapter_id, get_adapter_for_project
    from rebirth_ai.sandbox.runner import SandboxRunner

    _preflight(docker=True)
    if lang is None and detect_adapter_id(project) is None:
        _fail(f"could not tell the language of {project} (no package.json, pyproject.toml, go.mod, … and "
              "no source files); pass --lang")
    adapter = get_adapter_for_project(project, lang=lang)
    runner = SandboxRunner(adapter)
    from rebirth_ai.sandbox.runner import SandboxError

    try:
        result = runner.run_project(project, relative_test=test)
    except SandboxError as exc:
        _fail(str(exc))
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
        DEFAULT_DB,
        "--db",
        help="SQLite vector DB path (one project at a time: indexing another replaces it)",
        show_default="data/embeddings.db in the Rebirth folder",
    ),
) -> None:
    """Build a project's code-search index (optional: gives the model the code a file uses)."""
    from rebirth_ai.rag.retrieve import Retriever

    retriever = Retriever(db_path=db)
    stats = retriever.index_project(project)
    print(
        f"[green]indexed[/green] files={stats.files} chunks={stats.chunks} db={stats.db_path}"
    )


@app.command("query")
def query_index(
    text: str = typer.Argument(..., help="Free-text retrieval query"),
    db: Path = typer.Option(DEFAULT_DB, "--db", help="Code-search index (`rebirth index`)", show_default="data/embeddings.db in the Rebirth folder"),
    top_k: int = typer.Option(5, "--top-k", "-k"),
) -> None:
    """Search the code-search index."""
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
        ERROR_LOGS,
        "--dataset",
        exists=True,
        dir_okay=False,
        show_default="data/datasets/error_logs.jsonl in the Rebirth folder",
    ),
    model_dir: Path = typer.Option(
        CLASSIFIER_DIR,
        "--model-dir",
        show_default="data/models/error_classifier in the Rebirth folder",
    ),
) -> None:
    """Train the error classifier (TF-IDF + logistic regression; runs on first use if missing)."""
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
    db: Path = typer.Option(DEFAULT_DB, "--db", help="Code-search index (`rebirth index`)", show_default="data/embeddings.db in the Rebirth folder"),
    out: Path | None = typer.Option(None, "--out", help="Write draft code to this path"),
    backend: str = typer.Option(
        "auto",
        "--backend",
        "-b",
        help="auto (default: packaged adapter from configs/routing.yaml, else plain Ollama) | ollama | peft",
    ),
) -> None:
    """One draft from the model, without running the tests (for debugging)."""
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
        print(f"[green]wrote[/green] {tilde(out)}")


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
    """Mine before/after training pairs from CommitPackFT."""
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
        ...,
        "--file",
        "-f",
        help="Project-relative source file to migrate",
    ),
    test: str | None = typer.Option(
        None,
        "--test",
        "-t",
        help="Project-relative test file (default: the test that imports --file, else the whole suite)",
    ),
    objective: str = typer.Option(
        "Fix the file so project tests pass; modernize idioms without changing public API; keep comments and docstrings",
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
    db: Path = typer.Option(DEFAULT_DB, "--db", help="Code-search index (`rebirth index`)", show_default="data/embeddings.db in the Rebirth folder"),
    max_iterations: int = typer.Option(5, "--max-iterations"),
    out: Path | None = typer.Option(None, "--out", help="Also write final draft here"),
    backend: str = typer.Option(
        "auto",
        "--backend",
        "-b",
        help="auto (default: the packaged model — rebirth-<lang>:7b from configs/routing.yaml, or with --to the served rebirth-xlang:14b — else plain Ollama) | ollama | peft",
    ),
    lang: str | None = typer.Option(
        None,
        "--lang",
        "-l",
        help="Language (js_ts | python | go | java | csharp | rust | cpp | swift); default: detected from the project",
    ),
    to: str | None = typer.Option(
        None,
        "--to",
        help="Translate to another language (python | typescript | javascript | go | java | csharp | "
        "rust | cpp | swift). Writes a new file; the source is never modified.",
    ),
    out_dir: Path | None = typer.Option(
        None, "--out-dir", help="--to: output root (default <project>/rebirth_out/<to>, mirrors the source tree)"
    ),
    target_project: Path | None = typer.Option(
        None,
        "--target-project",
        exists=True,
        file_okay=False,
        help="--to: a project in the target language whose tests the translation must pass",
    ),
    target_file: str | None = typer.Option(
        None, "--target-file", help="--to: where the translation goes inside --target-project"
    ),
    target_test: str | None = typer.Option(
        None, "--target-test", help="--to: target-project test file (also shown to the model)"
    ),
    fallback: bool | None = typer.Option(
        None,
        "--fallback/--no-fallback",
        help="When the served 7B adapter passes the tests but changes nothing, run once more with a 14B "
        "in Ollama (default on; REBIRTH_FALLBACK=0 turns it off, REBIRTH_FALLBACK_MODEL picks the model)",
    ),
) -> None:
    """Modernize one file, or translate it with --to. Every draft is checked by the project's tests.

    Without --write nothing in the project changes: the result and its diff are printed. With --to, the
    translation is a new file under <project>/rebirth_out/<language>/ and the source is not changed.
    """
    import os

    from rebirth_ai.safety import InputError, check_source_file, inside

    try:
        check_source_file(project, file)
        if to:
            from rebirth_ai.xlang.languages import get_lang

            get_lang(to)
            if target_project is not None and target_file:
                inside(target_project, target_file, what="--target-project")
    except (InputError, ValueError) as exc:
        _fail(str(exc))
    _preflight(ollama=backend.strip().lower() not in {"peft", "lora", "local"} or bool(to), docker=True)

    if to:
        _translate_file(
            project=project,
            file=file,
            to=to,
            source_lang=lang,
            backend=backend,
            max_iterations=max_iterations,
            out_dir=out_dir,
            out=out,
            target_project=target_project,
            target_file=target_file,
            target_test=target_test,
            write=write,
        )
        return

    if test is None:
        from rebirth_ai.graph.discovery import infer_test_path

        test = infer_test_path(project, file) or ""
        print(f"[dim]test: {test}[/dim]" if test else
              "[yellow]no test file imports this file: running the whole test suite, which may not "
              "exercise it (pass --test)[/yellow]")
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
    import time as _time

    from rebirth_ai.runs import new_run_id, runs_dir, write_run_log

    source_path = project / file
    source_text = source_path.read_text(encoding="utf-8", errors="replace") if source_path.is_file() else None
    graph = build_migration_graph()
    print("[dim]running migration graph…[/dim]")
    t0 = _time.time()
    from rebirth_ai.graph.fallback import invoke as fallback_run
    from rebirth_ai.sandbox.runner import SandboxError

    try:
        final = fallback_run(graph, state, original=source_text, seed=seed_text, backend=backend, use=fallback)
    except SandboxError as exc:
        _fail(str(exc))
    wall = _time.time() - t0

    status = final.get("status", "unknown")
    from rebirth_ai.graph.batch import UNCHANGED_MESSAGE, no_change

    if status == "success" and final.get("current_draft") and source_text is not None and no_change(
            final["current_draft"], source_text, seed_text):
        status = "unchanged"
    color = {
        "success": "green",
        "unchanged": "yellow",
        "human_review": "yellow",
        "failed": "red",
    }.get(status, "white")
    print(f"[{color}]status={status}[/{color}] iterations={final.get('iteration_count')}")
    if status == "unchanged":
        print(f"[yellow]{UNCHANGED_MESSAGE}[/yellow]")
    print(f"[dim]error_class={final.get('error_class')}[/dim]")
    for step in final.get("trace") or []:
        print(f"  • {step}")

    draft = final.get("current_draft") or ""
    if status == "success" and draft and source_text is not None:
        _print_diff(file, source_text, draft)
    if out is not None:
        out.write_text(draft, encoding="utf-8")
        print(f"[green]wrote[/green] {tilde(out)}")
    if write and status == "success":
        from rebirth_ai.textfile import write_like

        target = Path(final["file_path"])
        write_like(target, draft)
        print(f"[green]wrote[/green] {tilde(target)}")
    if status == "success":
        print("[dim]Passing tests are evidence, not proof: review the diff before you keep it.[/dim]")

    from rebirth_ai.monitor import telemetry
    from rebirth_ai.monitor.flywheel import save_artifacts

    rid = new_run_id()
    run_lang = final.get("adapter_id") or lang
    run_path = write_run_log(
        kind="migrate",
        run_id=rid,
        payload={
            "project": str(project),
            "file": file,
            "test": test,
            "objective": objective,
            "lang": run_lang,
            "language": final.get("language"),
            "backend": backend,
            "status": status,
            "iterations": final.get("iteration_count"),
            "error_class": final.get("error_class"),
            "fallback_model": final.get("fallback_model"),
            "trace": final.get("trace") or [],
            "test_logs": (final.get("test_logs") or "")[:4000],
            "artifacts": save_artifacts(runs_dir() / rid, source_name=file, source=source_text,
                                        draft_name=file, draft=draft or None),
            "telemetry": telemetry.build(
                task="modernize", source_lang=run_lang, target_lang=None, backend=backend, model=None,
                status=status, iterations=final.get("iteration_count"), trace=final.get("trace"),
                wall_seconds=wall, source_code=source_text, draft=draft or None),
        },
    )
    print(f"[dim]run log[/dim] {tilde(run_path)}")
    if status not in ("success", "unchanged"):
        print(f"[dim]last draft and test log:[/dim] {tilde(runs_dir() / rid)}")

    raise typer.Exit(code=0 if status in ("success", "unchanged") else 1)


def _translate_file(
    *,
    project: Path,
    file: str,
    to: str,
    source_lang: str | None,
    backend: str,
    max_iterations: int,
    out_dir: Path | None,
    out: Path | None,
    target_project: Path | None,
    target_file: str | None,
    target_test: str | None,
    write: bool,
) -> None:
    """``rebirth migrate --to``: translate one file; never touches the source file."""
    from rebirth_ai.runs import write_run_log
    from rebirth_ai.xlang.graph import build_translation_graph, initial_translation_state
    from rebirth_ai.xlang.languages import get_lang

    target = get_lang(to)
    if target_project is not None and not target_file:
        print("[red]--target-project needs --target-file (where the translation goes in that project)[/red]")
        raise typer.Exit(2)
    project = project.resolve()
    try:
        state = initial_translation_state(
            project_dir=project,
            source_rel=file,
            target_lang=target.id,
            source_lang=source_lang,
            target_project=target_project,
            target_file=target_file,
            target_test=target_test,
            backend=backend,
            max_iterations=max_iterations,
        )
    except ValueError as exc:
        print(f"[red]{exc}[/red]")
        raise typer.Exit(2) from exc
    import time as _time

    print(f"[dim]translating {file} ({state['source_lang']} → {target.id}) …[/dim]")
    source_path = project / file
    source_text = source_path.read_text(encoding="utf-8", errors="replace") if source_path.is_file() else None
    t0 = _time.time()
    from rebirth_ai.sandbox.runner import SandboxError

    try:
        final = build_translation_graph().invoke(state)
    except SandboxError as exc:
        _fail(str(exc))
    wall = _time.time() - t0

    status = final.get("status", "unknown")
    color = {"success": "green", "human_review": "yellow"}.get(status, "red")
    print(f"[{color}]status={status}[/{color}] iterations={final.get('iteration_count')} verify={final.get('verify_mode')}")
    for step in final.get("trace") or []:
        print(f"  • {step}")

    draft = final.get("current_draft") or ""
    root = (out_dir or (project / "rebirth_out" / target.id)).resolve()
    dest = root / (final.get("target_rel") or state["target_rel"])
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(draft, encoding="utf-8")
    print(f"[green]wrote[/green] {tilde(dest)}" + ("" if status == "success" else " [yellow](not verified — review it)[/yellow]"))
    if out is not None:
        out.write_text(draft, encoding="utf-8")
        print(f"[green]wrote[/green] {tilde(out)}")
    if write and status == "success" and target_project is not None and target_file:
        from rebirth_ai.textfile import write_like

        write_like(target_project / target_file, draft)
        print(f"[green]wrote[/green] {tilde(target_project / target_file)}")

    from rebirth_ai.monitor import telemetry
    from rebirth_ai.monitor.flywheel import save_artifacts
    from rebirth_ai.runs import new_run_id, runs_dir

    rid = new_run_id()
    run_path = write_run_log(
        kind="translate",
        run_id=rid,
        payload={
            "project": str(project),
            "file": file,
            "source_lang": final.get("source_lang"),
            "target_lang": target.id,
            "target_rel": final.get("target_rel"),
            "target_project": final.get("target_project"),
            "target_test": target_test,
            "output": str(dest),
            "backend": backend,
            "model": final.get("model"),
            "status": status,
            "verify_mode": final.get("verify_mode"),
            "iterations": final.get("iteration_count"),
            "error_class": final.get("error_class"),
            "trace": final.get("trace") or [],
            "test_logs": (final.get("test_logs") or "")[:4000],
            "artifacts": save_artifacts(runs_dir() / rid, source_name=file, source=source_text,
                                        draft_name=str(dest.name), draft=draft or None),
            "telemetry": telemetry.build(
                task="translate", source_lang=final.get("source_lang"), target_lang=target.id, backend=backend,
                model=final.get("model"), status=status, iterations=final.get("iteration_count"),
                trace=final.get("trace"), wall_seconds=wall, source_code=source_text, draft=draft or None,
                verify_mode=final.get("verify_mode")),
        },
    )
    print(f"[dim]run log[/dim] {tilde(run_path)}")
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
        "Fix files so project tests pass; modernize idioms without changing public API; keep comments and docstrings",
        "--objective",
        "-o",
    ),
    under: str | None = typer.Option(
        None,
        "--under",
        help="Comma-separated folders to look in, e.g. src,lib (default: src if the project has one, "
        "else the whole project)",
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
    db: Path = typer.Option(DEFAULT_DB, "--db", help="Code-search index (`rebirth index`)", show_default="data/embeddings.db in the Rebirth folder"),
    max_iterations: int = typer.Option(5, "--max-iterations"),
    backend: str = typer.Option(
        "auto",
        "--backend",
        "-b",
        help="auto (default: packaged adapter from configs/routing.yaml, else plain Ollama) | ollama | peft",
    ),
    report: Path = typer.Option(
        DEFAULT_REPORT,
        "--report",
        help="Write JSON batch report here",
        show_default="data/reports/migrate_repo.json in the Rebirth folder",
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
        help="Language (js_ts | python | go | java | csharp | rust | cpp | swift); default: detected from the project",
    ),
    fallback: bool | None = typer.Option(
        None,
        "--fallback/--no-fallback",
        help="When the served 7B adapter passes the tests but changes nothing, run once more with a 14B "
        "in Ollama (default on; REBIRTH_FALLBACK=0 turns it off, REBIRTH_FALLBACK_MODEL picks the model)",
    ),
) -> None:
    """Modernize every source file of a project, in dependency order; optionally open a pull request.

    --dry-run shows the plan only. Without --write nothing in the project changes. --open-pr commits the
    files whose tests passed to a new branch and opens a GitHub pull request.
    """
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
    under = under or ("src" if (project / "src").is_dir() else ".")
    mode = "dry-run" if dry_run else ("write" if write else "no-write")
    print(
        f"[dim]migrate-repo under={under!r} backend={backend} mode={mode} lang={lang or 'auto'}[/dim]"
    )
    if not dry_run:
        _preflight(ollama=backend.strip().lower() not in {"peft", "lora", "local"}, docker=True)
        from rebirth_ai.adapters.registry import get_adapter_for_project
        from rebirth_ai.sandbox.runner import SandboxError, SandboxRunner

        try:
            SandboxRunner(get_adapter_for_project(project, lang=lang)).ensure_image()
        except SandboxError as exc:
            _fail(str(exc))
    if open_pr and not dry_run:
        from rebirth_ai.github import client as gh

        try:
            gh.git_root(project)
            gh.repo_slug_from_remote(gh.git_root(project))
            gh.require_auth()
            if gh.branch_exists(gh.git_root(project), pr_branch):
                raise GitHubError(f"branch {pr_branch!r} already exists: pass --pr-branch <new name>, or delete it")
        except GitHubError as exc:
            _fail(f"--open-pr: {exc}")
        if not write:
            print("[dim]--open-pr writes the files whose tests pass (implies --write)[/dim]")
            write = True

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
        fallback=fallback,
    )
    batch.write_json(report)
    if not batch.results:
        print(f"[yellow]No source files found under {under!r} in {project}[/yellow] "
              "(--under picks the folders, --file the files).")
    print(render_pr_body(batch))
    # Show patch previews when not writing
    if not dry_run and not write:
        for r in batch.results:
            if r.draft_preview:
                print(f"[bold]diff[/bold] {r.relative_file}")
                print(r.draft_preview.replace("[", "\\["))
    print(f"[green]report[/green] {tilde(report)}")

    from rebirth_ai.monitor import telemetry as _telemetry
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
            "telemetry_files": [
                _telemetry.build(task="modernize", source_lang=lang, target_lang=None, backend=backend, model=None,
                                 status=r.status, iterations=r.iterations, trace=r.trace)
                for r in batch.results
                if not r.status.startswith("skipped") and r.status != "planned"
            ],
        },
    )
    print(f"[dim]run log[/dim] {tilde(run_path)}")

    if open_pr and not dry_run:
        try:
            result = open_migration_pr(
                project,
                batch,
                branch=pr_branch,
                base=pr_base,
                push=push,
                create_pr=True,
                draft=draft,
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
    raise typer.Exit(code=0 if (batch.succeeded or batch.unchanged) and not batch.failed else 1)


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
    """Run the evaluation suites."""
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
    """Print a run, migrate-repo or eval report, with secrets redacted."""
    from rebirth_ai.report_view import load_report, render_report

    try:
        data = load_report(path)
    except ValueError as exc:  # json.JSONDecodeError included
        _fail(f"{path} is not a Rebirth JSON report ({exc})")
    render_report(data)


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
        CLASSIFIER_DIR,
        "--model-dir",
        show_default="data/models/error_classifier in the Rebirth folder",
    ),
) -> None:
    """Classify a test failure log (the class the retry loop acts on)."""
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


def _print_diff(name: str, before: str, after: str, *, limit: int = 400) -> None:
    """The change as a coloured unified diff (the first ``limit`` lines)."""
    import difflib

    from rich.console import Console
    from rich.text import Text

    lines = list(difflib.unified_diff(before.splitlines(), after.splitlines(), f"a/{name}", f"b/{name}",
                                      lineterm=""))
    out = Text()
    for line in lines[:limit]:
        style = ("bold" if line.startswith(("---", "+++")) else "green" if line.startswith("+")
                 else "red" if line.startswith("-") else "cyan" if line.startswith("@@") else "")
        out.append(line + "\n", style=style)
    if len(lines) > limit:
        out.append(f"… {len(lines) - limit} more diff lines (--out <file> saves the whole draft)\n", style="dim")
    Console(highlight=False).print(out, end="")


def _assign_panels() -> None:
    panel_of = {name: panel for panel, names in PANELS.items() for name in names}
    for info in app.registered_commands:
        name = info.name or info.callback.__name__.replace("_", "-")
        info.rich_help_panel = panel_of.get(name, "Other")
    for info in app.registered_groups:
        info.rich_help_panel = panel_of.get(info.name or "", "Other")


_assign_panels()
