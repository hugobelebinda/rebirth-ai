"""Install the model packs (Phase 19): ``rebirth install --pack both|modernize|migrate``.

Two packs, matching the two-model decision (ADR-001):

- **modernize** — ``ollama pull`` of the Qwen2.5-Coder-7B base (Q4_K_M) + 8 GGUF LoRA adapters
  (~10 MB each) → ``rebirth-<lang>:7b-v1`` with the serving alias ``rebirth-<lang>:7b``, exactly the
  Modelfile ``rebirth package`` writes (raw prompt template, SFT stop strings).
- **migrate** — one merged 14B GGUF (Q6_K, ~12 GB) → ``rebirth-xlang:14b-v2`` + alias
  ``rebirth-xlang:14b``, with the chat template it was evaluated with (``configs/ollama/``).

Files and checksums are listed in ``configs/models.json``. Sources, first match wins:
``--from-local`` (this repo's own build outputs), ``--from DIR`` (a folder holding the release file
names: a mirror, a USB stick), else Hugging Face (``https://huggingface.co/<ns>/<repo>/resolve/main/``).
Every file is checked against its sha256 before Ollama sees it. Downloads resume after an interrupt
and are deleted once Ollama has its own copy (``--keep-files`` keeps them). Models that are already
installed are skipped (``--force`` re-creates them). Nothing else leaves or enters the machine.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rebirth_ai.lora import ROOT

MANIFEST = ROOT / "configs" / "models.json"
DOWNLOAD_DIR = ROOT / "data" / "models" / "downloads"
INSTALL_DIR = ROOT / "data" / "models" / "ollama"
PACKS = ("modernize", "migrate")
HF = "https://huggingface.co"
CHUNK = 1 << 20


class InstallError(RuntimeError):
    pass


def load_manifest(path: Path = MANIFEST) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def packs_for(choice: str) -> tuple[str, ...]:
    choice = choice.strip().lower()
    if choice in ("both", "all"):
        return PACKS
    if choice not in PACKS:
        raise InstallError(f"unknown pack {choice!r}: both, {', '.join(PACKS)}")
    return (choice,)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


@dataclass
class Step:
    pack: str
    tag: str
    alias: str | None
    file: str
    source: str  # "local:<path>" | "dir:<path>" | URL
    action: str  # "skip" | "install"


def _url(manifest: dict, pack: dict, model: dict) -> str:
    ns = (manifest.get("hf_namespace") or "").strip()
    if not ns:
        raise InstallError("models are not published yet (configs/models.json: hf_namespace is empty) — "
                           "install with --from <dir> or --from-local")
    return f"{HF}/{ns}/{pack['hf_repo']}/resolve/main/{model['file']}"


def _source(manifest: dict, pack: dict, model: dict, *, from_dir: Path | None, from_local: bool) -> str:
    if from_local:
        return f"local:{ROOT / model['local']}"
    if from_dir is not None:
        return f"dir:{Path(from_dir) / model['file']}"
    return _url(manifest, pack, model)


def plan(choice: str, *, manifest: dict | None = None, from_dir: Path | None = None,
         from_local: bool = False, force: bool = False,
         has: Callable[[str], bool] | None = None) -> list[Step]:
    """What ``install`` would do, without touching anything."""
    from rebirth_ai.packaging import ollama_has

    manifest = manifest or load_manifest()
    has = has or ollama_has
    steps = []
    for name in packs_for(choice):
        pack = manifest["packs"][name]
        for m in pack["models"]:
            done = not force and has(m["tag"]) and (not m.get("alias") or has(m["alias"]))
            steps.append(Step(name, m["tag"], m.get("alias"), m["file"],
                              _source(manifest, pack, m, from_dir=from_dir, from_local=from_local),
                              "skip" if done else "install"))
    return steps


def download(url: str, dest: Path, *, size: int | None = None,
             progress: Callable[[str], None] = print) -> Path:
    """Stream ``url`` to ``dest``, resuming a partial ``.part`` file (HTTP Range)."""
    import httpx

    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    if size and have > size:  # longer than the release file: corrupt, start over
        part.unlink()
        have = 0
    headers = {"Range": f"bytes={have}-"} if have else {}
    try:
        return _stream(httpx, url, dest, part, have, headers, size, progress)
    except httpx.HTTPError as exc:  # connection lost, timeout: the .part file stays for the next run
        raise InstallError(f"download of {dest.name} interrupted ({type(exc).__name__}): run the same "
                           "command again to resume") from exc


def _stream(httpx, url, dest, part, have, headers, size, progress) -> Path:
    with httpx.stream("GET", url, headers=headers, follow_redirects=True, timeout=60.0) as r:
        if r.status_code == 416:
            if size and have == size:  # already complete
                part.rename(dest)
                return dest
            part.unlink(missing_ok=True)  # the server cannot continue this .part: start over next run
            raise InstallError(f"download of {dest.name} could not resume (HTTP 416): run the command again")
        if r.status_code in (401, 404):
            raise InstallError(f"{url} not found (HTTP {r.status_code}) — release files not published yet? "
                               "Install from a folder with --from <dir>")
        if r.status_code not in (200, 206):
            raise InstallError(f"download failed: HTTP {r.status_code} for {url}")
        mode = "ab" if r.status_code == 206 else "wb"
        got = have if r.status_code == 206 else 0
        total = size or (got + int(r.headers.get("content-length") or 0)) or None
        last = -1
        with part.open(mode) as fh:
            for block in r.iter_bytes(CHUNK):
                fh.write(block)
                got += len(block)
                pct = int(100 * got / total) if total else -1
                if pct != last and pct % 5 == 0:
                    progress(f"  {dest.name}: {got / 1e9:.2f} GB" + (f" ({pct} %)" if total else ""))
                    last = pct
    part.rename(dest)
    return dest


def _fetch(step: Step, model: dict, *, progress: Callable[[str], None]) -> tuple[Path, bool]:
    """Path of a verified copy of the model file, and whether it is a temporary download."""
    kind, _, where = step.source.partition(":")
    if kind in ("local", "dir"):
        path, temp = Path(where), False
        if not path.is_file():
            raise InstallError(f"{path} not found")
    else:
        path, temp = DOWNLOAD_DIR / model["file"], True
        if not path.is_file():
            download(step.source, path, size=model.get("size"), progress=progress)
    if model.get("size") and path.stat().st_size != model["size"]:
        have = path.stat().st_size
        if temp:  # else every later run would find the bad file and skip the download
            path.unlink()
        raise InstallError(f"{path.name}: size {have} ≠ {model['size']} (truncated download?)"
                           + (" — deleted, run again to download it" if temp else ""))
    progress(f"  checking {path.name} …")
    got = sha256_file(path)
    if got != model["sha256"]:
        if temp:
            path.unlink()
        raise InstallError(f"{path.name}: sha256 {got[:12]}… ≠ expected {model['sha256'][:12]}…"
                           + (" — deleted, run again to download it" if temp else ""))
    return path, temp


def _modelfile(pack_name: str, pack: dict, model: dict, gguf: Path) -> str:
    if pack_name == "modernize":
        from rebirth_ai.packaging import modelfile

        return modelfile(pack["ollama_base"], gguf)
    template = (ROOT / model["modelfile"]).read_text(encoding="utf-8")
    if "{gguf}" not in template:
        raise InstallError(f"{model['modelfile']} has no FROM {{gguf}} placeholder")
    return template.replace("{gguf}", str(gguf))


def _ollama(*args: str) -> None:
    r = subprocess.run(["ollama", *args], capture_output=True, text=True, check=False)
    if r.returncode != 0:
        raise InstallError(f"ollama {' '.join(args[:2])} failed: {(r.stderr or r.stdout).strip()[-400:]}")


def install(choice: str = "both", *, from_dir: Path | None = None, from_local: bool = False,
            force: bool = False, keep_files: bool = False, dry_run: bool = False,
            manifest: dict | None = None, progress: Callable[[str], None] = print) -> list[Step]:
    from rebirth_ai.packaging import ollama_has

    if from_dir is not None and not Path(from_dir).is_dir():
        raise InstallError(f"--from {from_dir}: no such folder")
    manifest = manifest or load_manifest()
    steps = plan(choice, manifest=manifest, from_dir=from_dir, from_local=from_local, force=force)
    if dry_run:
        return steps
    by_tag = {m["tag"]: (name, manifest["packs"][name], m)
              for name in packs_for(choice) for m in manifest["packs"][name]["models"]}
    pulled: set[str] = set()
    for step in steps:
        if step.action == "skip":
            progress(f"{step.tag}: already installed")
            continue
        name, pack, model = by_tag[step.tag]
        base = pack.get("ollama_base")
        if base and base not in pulled and not ollama_has(base):
            progress(f"pulling base model {base} (Ollama library) …")
            r = subprocess.run(["ollama", "pull", base], check=False)
            if r.returncode != 0:
                raise InstallError(f"ollama pull {base} failed")
        if base:
            pulled.add(base)
        progress(f"{step.tag}: {step.source}")
        path, temp = _fetch(step, model, progress=progress)
        d = INSTALL_DIR / step.tag.replace(":", "_")
        d.mkdir(parents=True, exist_ok=True)
        (d / "Modelfile").write_text(_modelfile(name, pack, model, path.resolve()), encoding="utf-8")
        _ollama("create", step.tag, "-f", str(d / "Modelfile"))
        if step.alias:
            _ollama("cp", step.tag, step.alias)
        if temp and not keep_files:
            path.unlink()
        progress(f"{step.tag}: installed" + (f" (alias {step.alias})" if step.alias else ""))
    return steps


def status(manifest: dict | None = None, has: Callable[[str], bool] | None = None) -> dict[str, dict]:
    """Per pack: installed / total, and the missing tags (for ``rebirth doctor``)."""
    from rebirth_ai.packaging import ollama_has

    manifest = manifest or load_manifest()
    has = has or ollama_has
    out = {}
    for name in PACKS:
        models = manifest["packs"][name]["models"]
        missing = [m.get("alias") or m["tag"] for m in models if not has(m.get("alias") or m["tag"])]
        out[name] = {"installed": len(models) - len(missing), "total": len(models), "missing": missing}
    return out
