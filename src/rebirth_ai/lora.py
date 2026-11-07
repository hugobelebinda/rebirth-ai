"""Per-language LoRA registry: dataset paths, adapter dirs, train/eval split, metadata.

Phase 11 harness. Paths are relative to the repo root (resolved via ``ROOT``) so the
training script, the PEFT client and tests all agree on one table.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATASETS_DIR = ROOT / "data" / "datasets"
MODELS_DIR = ROOT / "data" / "models"
SPLITS_DIR = DATASETS_DIR / "splits"

DEFAULT_BASE_7B = "Qwen/Qwen2.5-Coder-7B-Instruct"
DEFAULT_EVAL_FRAC = 0.05
DEFAULT_SEED = 1337

LORA_LANGS: tuple[str, ...] = ("js_ts", "python", "go", "java", "csharp", "rust", "cpp", "swift")

# Prompt-language names (as used in MigrationState.language / CLI) → adapter id.
_LANG_ALIASES = {
    "javascript": "js_ts",
    "typescript": "js_ts",
    "js": "js_ts",
    "ts": "js_ts",
    "py": "python",
    "golang": "go",
    "c#": "csharp",
    "cs": "csharp",
    "c++": "cpp",
}


def normalize_lang(language: str | None) -> str | None:
    """Map ``javascript`` / ``py`` / ``c++`` … to a registry id; ``None`` if unknown."""
    if not language:
        return None
    key = language.strip().lower()
    key = _LANG_ALIASES.get(key, key)
    return key if key in LORA_LANGS else None


@dataclass(frozen=True)
class LoraTarget:
    lang: str
    data: Path
    out: Path
    split_index: Path

    @property
    def adapter(self) -> Path:
        return self.out / "adapter"


def lora_target(lang: str, *, models_dir: Path | None = None) -> LoraTarget:
    lang_id = normalize_lang(lang)
    if lang_id is None:
        raise ValueError(f"Unknown LoRA language {lang!r}; expected one of {', '.join(LORA_LANGS)}")
    base = models_dir or MODELS_DIR
    return LoraTarget(
        lang=lang_id,
        data=DATASETS_DIR / f"{lang_id}_migration.jsonl",
        out=base / f"lora_{lang_id}_7b",
        split_index=SPLITS_DIR / f"{lang_id}.split.json",
    )


def _row_key(row: dict) -> str:
    return f"{row.get('objective', '')}\x00{row.get('legacy', '')}\x00{row.get('modern', '')}"


def is_eval_row(row: dict, *, eval_frac: float, seed: int) -> bool:
    """Content-hash split: stable across appends, identical rows never straddle train/eval."""
    if eval_frac <= 0:
        return False
    digest = hashlib.sha256(f"{seed}\x00{_row_key(row)}".encode()).digest()
    bucket = int.from_bytes(digest[:8], "big") / 2**64
    return bucket < eval_frac


def load_rows(path: Path) -> list[dict]:
    """Load SFT rows; skip blank / unparsable / incomplete lines (same rule as corpus manifest)."""
    rows: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and all(k in row for k in ("objective", "legacy", "modern")):
                rows.append(row)
    return rows


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class Split:
    train: list[dict]
    eval: list[dict]
    eval_indices: list[int]


def split_rows(rows: list[dict], *, eval_frac: float = DEFAULT_EVAL_FRAC, seed: int = DEFAULT_SEED) -> Split:
    train: list[dict] = []
    held: list[dict] = []
    idx: list[int] = []
    for i, row in enumerate(rows):
        if is_eval_row(row, eval_frac=eval_frac, seed=seed):
            held.append(row)
            idx.append(i)
        else:
            train.append(row)
    return Split(train=train, eval=held, eval_indices=idx)


def write_split_index(
    path: Path,
    *,
    lang: str | None,
    data: Path,
    corpus_sha256: str,
    split: Split,
    eval_frac: float,
    seed: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "language": lang,
        "data": rel_to_root(data),
        "corpus_sha256": corpus_sha256,
        "seed": seed,
        "eval_frac": eval_frac,
        "method": "sha256(seed|objective|legacy|modern) bucket < eval_frac",
        "n_total": len(split.train) + len(split.eval),
        "n_train": len(split.train),
        "n_eval": len(split.eval),
        "eval_indices": split.eval_indices,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def read_meta(adapter_or_out: Path) -> dict | None:
    """Read ``rebirth_meta.json`` from an adapter dir or its parent out dir."""
    for cand in (adapter_or_out / "rebirth_meta.json", adapter_or_out.parent / "rebirth_meta.json"):
        if cand.is_file():
            return json.loads(cand.read_text(encoding="utf-8"))
    return None


def rel_to_root(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)
