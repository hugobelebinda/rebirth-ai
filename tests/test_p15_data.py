"""Phase 15 data prep: full-CommitPack filters, tool-pair guards, DPO pair labels, corpus capping."""

from __future__ import annotations

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))

from build_dpo_pairs import classify as dpo_reason  # noqa: E402
from build_p15_corpus import _cap_tool_rows  # noqa: E402
from make_tool_pairs import _strip_rtry, cpp_candidate, rust_candidate, typedefs_preserved  # noqa: E402
from mine_commitpack_full import classify  # noqa: E402

LEGACY = "fn add(a: i32, b: i32) -> i32 {\n    return a + b;\n}\n\nfn main() {\n    println!(\"{}\", add(1, 2));\n}\n"
MODERN = LEGACY.replace("    return a + b;", "    a + b")


def _row(subject: str, old: str = LEGACY, new: str = MODERN, **kw) -> dict:
    return {"subject": subject, "message": subject, "license": "mit", "old_file": "src/main.rs",
            "new_file": "src/main.rs", "old_contents": old, "new_contents": new, **kw}


# --------------------------------------------------------------------------- miner


def test_modern_tier_needs_keyword_generic_needs_three_words() -> None:
    assert classify(_row("Remove needless return, simplify add"), "rust") == ("modern", "kept")
    assert classify(_row("Return the sum directly from add"), "rust") == ("generic", "kept")
    assert classify(_row("Tidy add"), "rust")[0] is None  # two words, no keyword


def test_miner_rejects_noise() -> None:
    assert classify(_row("Merge branch 'master' of github"), "rust")[1] == "subject_deny"
    assert classify(_row("Fix typo in add"), "rust")[1] == "subject_deny"
    assert classify(_row("Return the sum directly", license="gpl-3.0"), "rust")[1] == "license"
    ws = LEGACY.replace("    return", "  return")
    assert classify(_row("Return the sum directly", new=ws), "rust")[1] == "whitespace_only"
    renamed = {**_row("Return the sum directly"), "new_file": "src/lib.rs"}
    assert classify(renamed, "rust")[1] == "rename"


# --------------------------------------------------------------------------- tool pairs


def test_strip_rtry_balances_parens_and_wraps_binary_exprs() -> None:
    src = "let a = r#try!(f(g(1), 2));\nlet b = r#try!(x as u8 + 1);\n"
    out, n = _strip_rtry(src)
    assert n == 2 and out == "let a = f(g(1), 2)?;\nlet b = (x as u8 + 1)?;\n"


def test_typedef_guard_catches_dropped_const() -> None:
    assert not typedefs_preserved("typedef const Vec3& InParam;\n", "using InParam = Vec3&;\n")
    assert typedefs_preserved("typedef long long ll;\n", "using ll = long long;\n")
    assert not typedefs_preserved("typedef void (*fn)(int);\n", "using fn = void (*)(int);\n")


def test_standalone_candidates() -> None:
    assert cpp_candidate("#include <vector>\n#include <stdio.h>\nint main(){}\n")
    assert not cpp_candidate('#include "foo.h"\nint main(){}\n')
    assert rust_candidate("extern crate std;\nfn main() {}\n")
    assert not rust_candidate("extern crate serde;\nfn main() {}\n")
    assert not rust_candidate("mod util;\nfn main() {}\n")


# --------------------------------------------------------------------------- dpo + corpus


def test_dpo_reasons() -> None:
    row = {"legacy": "var x = 1;", "modern": "const x = 1;"}
    ok = {"pred": "const x = 1;", "ast": {"soft_exact": True, "parses": True, "token_overlap": 1.0}}
    assert dpo_reason(row, ok, 0.6) is None
    assert dpo_reason(row, {"pred": "var x =  1;", "ast": {"parses": True}}, 0.6) == "noop"
    assert dpo_reason(row, {"pred": "const x = (", "ast": {"parses": False}}, 0.6) == "unparsable"
    assert dpo_reason(row, {"pred": "let y = 2;", "ast": {"parses": True, "token_overlap": 0.2}}, 0.6) == "low_overlap"
    assert dpo_reason(row, {"pred": "let x = 1;", "ast": {"parses": True, "token_overlap": 0.8}}, 0.6) == "partial"


def test_tool_rows_capped_per_check() -> None:
    rows = [{"checks": ["a"]}] * 50 + [{"checks": ["b"]}] * 5 + [{"checks": ["c"]}] * 5
    picked = _cap_tool_rows([dict(r) for r in rows], 20, 0.5, random.Random(0))
    counts = {k: sum(r["checks"][0] == k for r in picked) for k in "abc"}
    assert len(picked) == 20 and counts["a"] == 10 and counts["b"] == 5 and counts["c"] == 5
