"""Run translated code in the *target* language's sandbox.

- :func:`verify_translation` — one translated file: the target project's tests when there are
  any (``--target-project``), else a compile-only check in a scratch project
- :func:`run_programs`       — many self-contained test programs (eval harness), batched: one
  container runs a whole chunk, chunks run in parallel
"""

from __future__ import annotations

import base64
import re
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from rebirth_ai.adapters.registry import get_adapter
from rebirth_ai.xlang.languages import COMPILE_FAILED, XLang, check_spec, program_spec


@dataclass(frozen=True)
class VerifyResult:
    ok: bool
    exit_code: int
    logs: str
    mode: str  # target_tests | compile
    command: str
    duration_sec: float
    tested_code: str = ""  # the file as tested (Rust: with the project's own test module)


def _runner(lang: XLang):
    from rebirth_ai.sandbox.runner import SandboxRunner

    return SandboxRunner(get_adapter(lang.adapter_id))


_RUST_TEST_MOD = re.compile(r"^#\[cfg\(test\)\]\s*\n\s*mod\s+\w+\s*\{", re.MULTILINE)


def _rust_test_mod(text: str) -> tuple[int, int] | None:
    m = _RUST_TEST_MOD.search(text)
    if not m:
        return None
    depth = 0
    for j in range(text.index("{", m.start()), len(text)):
        depth += {"{": 1, "}": -1}.get(text[j], 0)
        if depth == 0:
            return m.start(), j + 1
    return None


def keep_inline_tests(target: XLang, original: str | None, draft: str) -> str:
    """Rust keeps unit tests in the source file (``#[cfg(test)] mod tests``). When the translation
    replaces that file, the project's own test module is carried over and any test module the
    model wrote is dropped — otherwise ``cargo test`` runs 0 tests, or the model's own tests."""
    if target.id != "rust" or not original:
        return draft
    span = _rust_test_mod(original)
    if span is None:
        return draft
    own = _rust_test_mod(draft)
    if own is not None:
        draft = draft[: own[0]] + draft[own[1] :]
    body = draft.strip("\n")
    return (body + "\n\n" if body else "") + original[span[0] : span[1]] + "\n"


def tests_ran(target: XLang, logs: str) -> bool | None:
    """False when the run passed only because no test executed; None = cannot tell."""
    if target.adapter_id == "rust":
        return any(int(n) > 0 for n in re.findall(r"test result: ok\. (\d+) passed", logs))
    if target.adapter_id == "go":
        return "[no test files]" not in logs or bool(re.search(r"^ok\s", logs, re.MULTILINE))
    if target.adapter_id == "js_ts":
        m = re.findall(r"^# (?:tests|pass) (\d+)", logs, re.MULTILINE)
        return any(int(n) > 0 for n in m) if m else None
    return None


def verify_translation(
    *,
    target: XLang,
    code: str,
    target_rel: str,
    target_project: Path | None = None,
    target_test: str | None = None,
) -> VerifyResult:
    """Target tests if a target project is given (translation overlaid at ``target_rel``), else compile."""
    runner = _runner(target)
    if target_project is not None:
        existing = target_project / target_rel
        code = keep_inline_tests(target, existing.read_text(encoding="utf-8") if existing.is_file() else None, code)
        res = runner.run_project(target_project, relative_test=target_test, overlay_text=code, overlay_dest=target_rel)
        mode = "target_tests"
        logs = (res.stdout or "") + ("\n" + res.stderr if res.stderr else "")
        if res.ok and tests_ran(target, logs) is False:
            return VerifyResult(False, 1, logs + "\n[rebirth] no target tests ran — not verified", mode, res.command,
                                res.duration_sec, code)
    else:
        scaffold, cmd = check_spec(target, target_rel, code)
        with tempfile.TemporaryDirectory(prefix="rebirth-xlang-") as tmp:
            root = Path(tmp)
            for rel, text in {**scaffold, target_rel: code}.items():
                p = root / rel
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(text, encoding="utf-8")
            res = runner.run_project(root, command=cmd)
        mode = "compile"
    logs = (res.stdout or "") + ("\n" + res.stderr if res.stderr else "")
    return VerifyResult(res.ok, res.exit_code, logs, mode, res.command, res.duration_sec, code)


# --------------------------------------------------------------------------- batched programs


@dataclass(frozen=True)
class ProgramResult:
    id: str
    exit_code: int
    status: str  # pass | compile_error | fail | timeout | missing
    log_tail: str


_SAFE_ID = re.compile(r"[^A-Za-z0-9_.-]")
_RESULT_RE = re.compile(r"^@@R (\S+) (-?\d+) ?(\S*)$", re.MULTILINE)
_DRIVER = """for d in progs/*/; do
  id=$(basename "$d")
  (cd "$d" && sh run.sh > log.txt 2>&1)
  rc=$?
  echo "@@R $id $rc $(tail -c {tail} "$d/log.txt" | base64 | tr -d '\\n')"
done
"""


def status_for(exit_code: int) -> str:
    if exit_code == 0:
        return "pass"
    if exit_code == COMPILE_FAILED:
        return "compile_error"
    if exit_code in (124, 137):
        return "timeout"
    return "fail"


def _run_chunk(
    lang: XLang, chunk: dict[str, str], timeout_sec: int, tail: int, extra_files: dict[str, str]
) -> dict[str, ProgramResult]:
    spec = program_spec(lang)
    with tempfile.TemporaryDirectory(prefix="rebirth-progs-") as tmp:
        root = Path(tmp)
        for pid, source in chunk.items():
            d = root / "progs" / pid
            d.mkdir(parents=True)
            for rel, text in {**spec.scaffold, **extra_files}.items():
                (d / rel).parent.mkdir(parents=True, exist_ok=True)
                (d / rel).write_text(text, encoding="utf-8")
            (d / spec.filename).write_text(source, encoding="utf-8")
            (d / "run.sh").write_text(spec.script(timeout_sec), encoding="utf-8")
        (root / "driver.sh").write_text(_DRIVER.replace("{tail}", str(tail)), encoding="utf-8")
        res = _runner(lang).run_project(root, command="sh driver.sh")
    out: dict[str, ProgramResult] = {}
    for pid, rc, b64 in _RESULT_RE.findall(res.stdout or ""):
        try:
            log = base64.b64decode(b64).decode("utf-8", errors="replace") if b64 else ""
        except ValueError:
            log = ""
        out[pid] = ProgramResult(pid, int(rc), status_for(int(rc)), log)
    for pid in chunk:  # container died / driver output lost
        out.setdefault(pid, ProgramResult(pid, -1, "missing", ((res.stderr or "") + (res.stdout or ""))[-tail:]))
    return out


def run_programs(
    lang: XLang,
    programs: dict[str, str],
    *,
    timeout_sec: int = 15,
    workers: int = 4,
    chunk_size: int = 40,
    tail: int = 1500,
    extra_files: dict[str, str] | None = None,
) -> dict[str, ProgramResult]:
    """Compile + run each program (id → full source) in ``lang``'s sandbox; exit 0 = pass.

    ``extra_files`` (relative path → text) are written next to every program.
    """
    ids = {pid: _SAFE_ID.sub("_", pid) for pid in programs}
    if len(set(ids.values())) != len(ids):
        raise ValueError("program ids collide after sanitizing")
    items = [(ids[pid], src) for pid, src in programs.items()]
    chunks = [dict(items[i : i + chunk_size]) for i in range(0, len(items), chunk_size)]
    merged: dict[str, ProgramResult] = {}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for part in pool.map(lambda c: _run_chunk(lang, c, timeout_sec, tail, extra_files or {}), chunks):
            merged.update(part)
    back = {v: k for k, v in ids.items()}
    return {back[k]: ProgramResult(back[k], r.exit_code, r.status, r.log_tail) for k, r in merged.items()}
