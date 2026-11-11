# Phase 17 — Cross-language migration (all 8 languages)

> Status: **COMPLETE (2026-11-10) — stages 1–5 done, stage 6 not run.** Stage 1 (pipeline + eval harness) and stage 2 (zero-shot baseline, 14B **91.1 %** on HumanEval over 56 directions) done 2026-11-07. Stages 3–4 ran on AWS on 2026-11-09: the shared adapter `lora_xlang_14b` lifts the 14B from **208 → 265 / 420** verified translations on the program suite (q6_K + GGUF in Ollama, the shipped runtime; cloud bf16: 215 → 274). Since 2026-11-10 it is the default engine for `rebirth migrate --to` (`rebirth-xlang:14b-v1`). Stage 5 (2026-11-10, local): on the other 35 directions the adapter lifts the program suite **343 → 407 / 511**, and HumanEval over all 56 directions goes **91.1 → 92.6 %** first try (no regression). Stage 6 (a second, cloud-teacher data round) **not run**: the owner decided on 2026-11-10 to spend no more cloud credit without a guaranteed jump, and the 30B teacher is only on par with the adapter (282 vs 274), so v1 stays. Decided 2026-11-05; was "Phase 15" before the renumbering.
> Previous: [PHASE_16_MLOPS_CICD_PACKAGING.md](PHASE_16_MLOPS_CICD_PACKAGING.md) · Next: [PHASE_18_MLOPS_MONITORING.md](PHASE_18_MLOPS_MONITORING.md)
> Owner decision: do it, local first, AWS GPU for the heavy stages.
> Cloud setup, access model and budget split: [../CLOUD_GPU_PLAN.md](../CLOUD_GPU_PLAN.md) — stages 3–4 ran on one g6e.xlarge (L40S 48 GB) session, 2026-11-08 23:32 → 2026-11-09 16:32 EST (~$32); the 14B generated its own training data (no teacher model got through the pilot, see Stage 3).

## Goal

Translate a file from one supported language to another — e.g. Python → Go — through the same
closed loop (generate → sandbox test → classify → retry) Rebirth already uses for same-language
migration. Today the tool **cannot** do this:

| Blocker today | Why |
|---------------|-----|
| Training data | CommitPackFT rows are same-language commits (file before → file after). No adapter has seen Python-in / Go-out. |
| SFT prompt | `### Objective / ### Legacy / ### Modern` — no target-language field. |
| Pipeline | `rebirth migrate --lang` picks adapter, sandbox and test runner from the **source** language; output overwrites the same file/extension; the source-language tests judge the result. No `--to` anywhere in `src/`. |

The base Qwen2.5-Coder 7B and Ollama 14B already translate reasonably in plain chat — it is just
not wired in.

## Stage 1 — pipeline + eval harness (done 2026-11-07)

Decisions (user, 2026-11-07, "go with your picks"): judge translations with **existing per-language
benchmark tests** (MultiPL-E) instead of a custom I/O harness; **files / functions up to ~100 lines**
first; output is a **new file** in an output tree that mirrors the source tree; **zero-shot** models
first (Ollama 14B default, base 7B chat via `--backend peft`).

### `rebirth migrate --to <lang>`

```bash
rebirth migrate <project> --file src/math_ops.py --to go \
  [--target-project fixtures/go_sample --target-file src/mathops.go --target-test src/mathops_test.go] \
  [--out-dir DIR] [--backend ollama|peft] [--max-iterations N]
```

- Closed loop (`src/rebirth_ai/xlang/graph.py`, LangGraph): translate → verify in the **target**
  sandbox → classify → retry with the target compiler / test output. The source file is never
  modified; the translation goes to `<project>/rebirth_out/<to>/<same folders>/<name>.<ext>`
  (target naming convention; Java = the generated public class name).
- Verification: with `--target-project`, that project's tests with the translation overlaid at
  `--target-file` (the model sees the tests, never the target implementation; C++ also gets the
  headers the tests include). Without it, a compile check in a scratch project (`py_compile`,
  `node --check`, TS strip + check, `go build`, `javac`, `dotnet build`, `rustc --emit=metadata`,
  `g++ -fsyntax-only`, `swiftc -typecheck`).
- Guards against false passes: Rust's inline `#[cfg(test)]` module of the target project always
  replaces any test module the model writes; a run where **no test executed** (`cargo test` 0
  passed, Go `[no test files]`, `node --test` 0 tests) is a failure.
- Languages: 9 ids — JS/TS is `javascript` and `typescript` (same sandbox; Node 22 runs `.ts` by
  stripping types, **no type check** yet — no `tsc` in the image).
- Error classes: target compiler output → `CompileError`; else the Phase 7 classifier.

### Eval harness — `training/eval_translate.py`

- Problems `data/datasets/xlang/humaneval.jsonl` (`training/build_xlang_eval.py`, frozen):
  164 HumanEval problems; **tests** in 9 languages (MultiPL-E, MIT: C++ 161, C# 158, Go 154,
  Java 158, JS 161, Rust 156, Swift 158, TS 159; Python = OpenAI HumanEval, MIT); **sources** with
  reference solutions in Python (HumanEval) and C++ / Go / Java / JS (HumanEval-X, Apache-2.0),
  comments / docstrings removed so the model translates instead of re-solving from the statement.
  Rust / C# / Swift / TS get sources once verified translations exist.
- A translation passes when target header + translated function + target tests compile and every
  assert holds in the target sandbox (exit 0). Programs run batched: one container per chunk of 40,
  4 in parallel; compile errors, failed asserts and timeouts are counted apart.
- Java: the MultiPL-E prompts import `org.javatuples` (8 problems use `Pair`) — a small shim is
  compiled next to every program instead of dropping them.
- C#: MultiPL-E's tests compare results with `.Equals`, which is *reference* equality for `List` /
  `Dictionary` — every list-returning problem (45 of 158) failed even with a correct answer (found in
  stage 2: 12 problems failed from all 7 sources). `assemble()` rewrites `Debug.Assert(A.Equals(B))`
  to a structural comparison (`RebirthEq.Same`, same result for strings / tuples / nullables).
- Known upstream test bugs are listed in `BROKEN_TESTS` (`multipl_e.py`), reported as
  `broken_test` and left out of pass rates — so far Go `HumanEval_87` (the test passes
  `[]interface{}` where the signature takes `[][]int`; nothing compiles).
- **Self-check** (no model): `--oracle fixtures/xlang_oracle/humaneval.json` — hand-written correct
  and deliberately wrong solutions for 3–4 problems in all 9 languages (incl. a list-returning C#
  one), some wrapped the way models wrap them (class wrapper, `package main` + imports, extra
  `main`): **39 / 39 as expected** (also `tests/test_xlang_sandbox.py`).
- Resumable: generations go to `<out>.gen.jsonl` as they are made; `--rescore` re-runs the tests on
  those cached generations only (no model call; refuses if any is missing).

### Fixture matrix (stage 1 exit check)

`training/eval_translate.py --fixtures fixtures/xlang_fixtures.json`: each of the 8 sample projects
translated into each other one (56 directions; JS/TS = JavaScript here because the JS sample's tests
import `.js`), full loop with Ollama 14B, ≤ 3 rounds, judged by the **target project's own tests**
(`pytest`, `node --test`, `go test`, Maven, `dotnet test`, `cargo test`, CTest, `swift test`).
Report: `data/reports/xlang/fixtures_ollama14b.json` (earlier runs `_r1`–`_r3`).

| Run | Change | Pass | First try | Minutes |
|-----|--------|-----:|----------:|--------:|
| r1 | first version | 52 / 56 | 43 | 7.9 |
| r2 | retry shows the file *as tested* (Rust: with the project's test module); "tests are fixed" | 52 / 56 | 41 | 8.2 |
| r3 | retries at temperature 0.4 / 0.7; "don't wrap in a namespace" (too blunt: dropped C#'s `namespace MathOps`) | 48 / 56 | 31 | 14.2 |
| **r4** | placement rule taken from the tests ("declare the namespace the tests use, nothing that only copies the source") | **53 / 56** | 40 | 10.1 |

```
               python javascript    go  java csharp  rust   cpp swift   (rounds to pass; x = no)
python              -          1     2     2      1     1     2     2
javascript          1          -     1     1      1     1     2     2
go                  x          1     -     1      1     x     1     1
java                1          1     2     -      2     2     1     1
csharp              1          1     2     1      -     1     1     1
rust                1          1     2     1      1     -     1     1
cpp                 1          1     1     1      1     1     -     1
swift               x          2     3     1      1     1     1     -
```

- The 3 failures are real translation bugs: Go's `byte('0' + n%10)` copied literally into Python /
  Rust, and a `Calculator` class the Go / Swift samples do not have, which the model has to invent
  from the Python tests (it got the accumulate semantics wrong).
- Retries matter: 13 of 53 passes needed a 2nd or 3rd round (compiler / test output fed back).
- Run-to-run noise is a few directions (r1 vs r2: same score, different failures), so the fixture
  matrix is a smoke test of the pipeline, not a benchmark — that is what the HumanEval harness is
  for. No more prompt tuning against these fixtures.
- Infra flake seen once (r3): Maven failed to download a plugin (the Java sandbox needs network).

HumanEval harness, real-model check (Ollama 14B, Python / Java → Rust / Java / Swift / C#, 3
problems each): **20 / 21** pass, ~3.6 s per translation. Showing the model the target file's header
(imports, `class Problem {`), as MultiPL-E itself does, fixed a `javafx.util.Pair` import.

## Stage 2 — zero-shot baseline (done 2026-11-07)

`bash scripts/xlang_stage2.sh` (LIMIT=40, seed 1337, Ollama `qwen2.5-coder:14b-instruct-q6_K`,
temperature 0.1, **one try per translation** — no retry loop, unlike `migrate --to`):

1. 35 directions with reference sources (Python, JS, Go, Java, C++ → 8 targets) × 40 problems =
   1,400 translations, 72 min (~3.1 s each) → `data/reports/xlang/zeroshot_ollama14b.json`.
2. 21 directions from Rust / C# / Swift: their sources are pass-1 translations *into* those
   languages that passed the tests (Python-sourced preferred; all 40 sampled problems got one in each
   language), 836 translations, 38 + 4 min → `zeroshot_ollama14b_bootstrap.json`.

Every direction uses the same seeded problem order, so cells are comparable. Pass = every MultiPL-E
(or HumanEval) assert holds. 7 rows of Go `HumanEval_87` are excluded (broken test). Numbers below
include the Go import fix-up added 2026-11-08 (`src/rebirth_ai/xlang/fixups.py`, also applied by
`migrate --to`); raw model output (`--no-fixups`) scored Go 73.5 % and 89.3 % overall.

| src \ tgt | py | ts | go | java | c# | rust | c++ | swift | all |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **py** | – | 39/40 | 32/39 | 36/40 | 38/40 | 36/40 | 40/40 | 34/40 | 91 % |
| **js** | 35/40 | – | 35/39 | 37/40 | 37/40 | 34/40 | 38/40 | 35/40 | 90 % |
| **go** | 38/40 | 36/40 | – | 37/40 | 38/40 | 37/40 | 39/40 | 34/40 | 92 % |
| **java** | 36/40 | 37/40 | 34/39 | – | 38/40 | 36/40 | 40/40 | 32/40 | 91 % |
| **c#** | 40/40 | 37/40 | 36/39 | 36/40 | – | 35/40 | 37/40 | 33/40 | 91 % |
| **rust** | 39/40 | 39/40 | 34/39 | 35/40 | 36/40 | – | 38/40 | 33/40 | 91 % |
| **c++** | 35/40 | 37/40 | 35/39 | 39/40 | 40/40 | 35/40 | – | 35/40 | 92 % |
| **swift** | 39/40 | 38/40 | 34/38 | 34/39 | 36/39 | 30/39 | 37/40 | – | 90 % |

(js → ts is skipped: TypeScript accepts the JavaScript as is.)

| Target | 14B pass | Compile error | Wrong answer | Base 7B pass (2026-11-08) |
|--------|-----:|--------------:|-------------:|------:|
| C++ | **96.1 %** (269 / 280) | 5 | 6 | 90.0 % |
| C# | 94.3 % (263 / 279) | 7 | 9 | 91.4 % |
| TypeScript | 93.9 % (263 / 280) | 1 | 16 | 90.0 % |
| Python | 93.6 % (262 / 280) | 0 | 18 | 93.2 % |
| Java | 91.0 % (254 / 279) | 12 | 13 | 87.1 % |
| Go | 88.2 % (240 / 272) — raw 73.5 % | 13 | 19 | 84.9 % |
| Rust | 87.1 % (243 / 279) | 17 | 19 | 80.6 % |
| Swift | 84.3 % (236 / 280) | 15 | 29 | **63.9 %** |
| **All** | **91.1 % (2,030 / 2,229)** | 70 | 129 | **85.2 % (1,898 / 2,229)** |

No timeouts. By source the spread is small (Go 92.5 % … Java 87.5 %): the target language decides.

What it says:

- **The 14B is already good at short functions in every direction.** Every direction is ≥ 77 %,
  four are 40 / 40; the weakest are now into Swift / Rust.
- **Go: most raw failures were missing imports.** 42 of 55 Go compile errors were `undefined: sort /
  strings / math / strconv` — the model calls `sort.Ints` but does not write `import "sort"` (the
  prompt asks it to). The fix-up (add missing / drop unused standard-library imports, what
  `goimports` does) turned 40 of them into passes and 2 into wrong answers: Go 73.5 % → 88.2 %,
  all 89.3 % → 91.1 %. No other row changed.
- **Swift and Rust** fail on type / API details (`String.Index` subscripts, `Character` vs
  `CharacterSet`, integer casts, a `regex` crate that is not in the standard library).
- C# looked like the weakest target (66 %) until the harness bug above was fixed: 56 rows moved
  from fail to pass on re-scoring, no other row changed.

**Base 7B (2026-11-08)** — Ollama `qwen2.5-coder:7b-instruct` (q4_K_M, the base the per-language
`rebirth-<lang>:7b` adapters run on), same 2,229 items, same Rust / C# / Swift sources as the 14B
(`--add-sources-from` the 14B pass-1 report), 1.6 s per translation, one run
(`data/reports/xlang/zeroshot_ollama7b.json`): **85.2 %** vs the 14B's 91.1 %. Paired: 197 items
only the 14B passes, 65 only the 7B — a clear, not noisy, gap. Close to the 14B on Python / C# /
TypeScript (−0.4 to −3.9), behind on Rust / C++ (−6), far behind on **Swift (63.9 %, −20)**: 49 of
its 73 Swift compile errors are the function renamed to camelCase (`isMultiplyPrime`) against the
required `is_multiply_prime` signature. **Verdict: the 7B works for most directions but the 14B is
the translator** (Swift especially); the 7B is a low-memory fallback. Quantization differs (q4_K_M
vs q6_K) — both as they run locally.

Caveats:

- **HumanEval functions are short (~10 lines) and famous** — the model may have seen solutions in
  each language. Treat this as an upper bound for real file migration (several functions, classes,
  project idioms), which the fixture matrix (53 / 56 with retries) only smoke-tests.
- 40 of ~158 problems per direction: ± ~5 points per cell (binomial SE at 90 %), ± ~2 points per
  target row. Extending to all 164 is incremental (same seeded order, cached generations): ≈ 4–5 h.
- Rust / C# / Swift sources are model-written (verified) translations, so those 21 directions
  start from code in the model's own style — slightly easier than human-written sources.

Also fixed on 2026-11-08: Ollama runs with a **4,096-token window by default and silently drops the
start of a longer prompt**; neither the translator nor the same-language `migrate` set `num_ctx`.
`OllamaClient.generate_raw` now sizes it to prompt + output (4K / 8K / 16K / 32K buckets, shared with
the Phase 16 adapter client; `REBIRTH_OLLAMA_NUM_CTX` pins one value for a whole run). HumanEval
prompts fit in 4K, so stage-2 numbers are unaffected; long files and retry prompts were not.

### Program-level suite — harder eval (built and baselined 2026-11-08)

At 91 % HumanEval leaves little room to show what a trained adapter adds, and its ~10-line famous
functions are not what file migration looks like. The program suite translates **whole programs**
(imports, stdin parsing, helpers, types, 20–150 lines) and judges them like an online judge.

- **Problems** (`training/build_xlang_programs.py` → `data/datasets/xlang/programs.jsonl`, frozen):
  Codeforces problems from the DeepMind CodeContests **valid + test** splits (CC BY 4.0; attribution
  in [DATASET.md](../DATASET.md)). Kept when not interactive, stdin → stdout, no floating-point
  output, and a Python 3, a C++ and a Java reference solution each pass all 10 chosen tests in our
  sandboxes **and** a known-incorrect solution from the dataset fails them. 282 → **163 problems**,
  rating 800 / 1500 / 2600 (min / median / max), median 41 / 47 / 76 non-blank lines (py / cpp / java).
- **Judge** (`verify.run_io_programs`): optimized builds (`-O2`, `-O`, Release), 10 s per test,
  token-wise case-insensitive output comparison (like Codeforces checkers). One test set judges a
  translation into any of the 8 targets. Oracle fixture `fixtures/xlang_oracle/programs.json`: a
  correct program in all 9 languages passes, three wrong ones (`int` overflow, i32, wrong parity)
  are rejected.
- **Eval:** `training/eval_translate.py --suite programs` — same directions / cache / `--rescore` as
  HumanEval; the prompt adds that the target must be a complete program with the same output and
  the same algorithm.

**Baseline (pass 1):** Ollama 14B q6_K, the three sources with human-written solutions (Python,
C++, Java) → all 8 targets = 21 directions × the first 20 problems of the seeded order (median
rating 1700), with Go fix-ups; 13.5 s / translation, ~95 min
(`data/reports/xlang/programs_ollama14b.json`):

| source ↓ / target → | py | ts | go | java | c# | rust | c++ | swift | all |
|---|---|---|---|---|---|---|---|---|---|
| python | – | 2/20 | 6/20 | 9/20 | 8/20 | 5/20 | 8/20 | 4/20 | 30 % |
| cpp | 9/20 | 3/20 | 6/20 | 13/20 | 5/20 | 2/20 | – | 4/20 | 30 % |
| java | 13/20 | 5/20 | 5/20 | – | 6/20 | 4/20 | 10/20 | 4/20 | 34 % |

**Overall 31.2 % (131 / 420)** — vs 91.1 % on HumanEval with the same model, prompts and fix-ups.

| Target | Pass | HumanEval | Compile error | Runtime error | Wrong / empty output |
|---|---|---|---|---|---|
| Python | 55 % (22/40) | 93.6 % | 0 | 15 | 3 |
| Java | 55 % (22/40) | 91.0 % | 9 | 5 | 4 |
| C++ | 45 % (18/40) | 96.1 % | 16 | 2 | 4 |
| C# | 32 % (19/60) | 94.3 % | 22 | 10 | 9 |
| Go | 28 % (17/60) | 88.2 % | 24 | 8 | 11 |
| Swift | 20 % (12/60) | 84.3 % | 29 | 11 + 1 timeout | 7 |
| Rust | 18 % (11/60) | 87.1 % | 32 | 11 + 1 timeout | 5 |
| TypeScript | 17 % (10/60) | 93.9 % | 6 | 20 | 24 |

What fails (all checked to be model errors, not harness ones — no CR line endings in the tests, the
references pass the same judge):

- **Input parsing is the #1 failure.** C++ `cin >>` / Java `StringTokenizer` read tokens across
  lines; the translations read lines (`int(input())` on `"0 2 1"` → `ValueError`, `EOFError`;
  C# `FormatException`). TypeScript is the worst: no standard way to read stdin, so the model
  invents one — a global `readline()` (Codeforces' V8 runtime), `process.stdin.webidlConverter()`,
  a function named `process` shadowing Node's, `static` inside an arrow function.
- **Compile errors are 138 of the 289 failures** — strictness the short functions never hit: Go
  unused variables and `strconv.Atoi`'s two results; Rust `use std::io::Read` missing, ambiguous
  numeric types; Swift `let` vs `var`, Optional binding on non-optionals; C# instance members in a
  `static class`; Java `_` as a name (preview in 21), `long → int` narrowing.
- **Integer width** (`long long` → `int`/i32 overflow) and index-out-of-range panics in the rest.
- Problems range from 1 / 21 to 12 / 21 directions passing — none trivial, none impossible; rating
  is a weak predictor (CC_1548D1 at 2300 passes 12, CC_1613D at 1900 passes 1).

Caveats: 20 problems per cell → ± ~10 points per cell, ± ~7 per target row, ± ~2.3 overall; every
direction uses the same 20 problems, so paired comparisons (adapter vs baseline) are much tighter
than the cell SE suggests. Pass@1 without the compile-feedback retry that `rebirth migrate --to`
does, so this is a floor for the product, not its number. CodeContests problems and solutions are
public too: the suite is harder, not contamination-free.

**Implications for stage 3:**

- **Data source: the CodeContests *train* split** (~13K problems, 39 parquet files ≈ 7.5 GB,
  disjoint from the valid + test problems used here). Translate its Python / C++ / Java solutions
  into the other targets and keep only translations that pass the problem's own tests — verification
  with real judge tests instead of generated inputs, on exactly the failure modes above (I/O idioms,
  integer widths, strict compilers). Mix with corpus snippets so file-level `migrate` (classes,
  project idioms) is not forgotten.
- **Report both suites**: HumanEval guards against regressions, the program suite shows the gain.
- Follow-up runs (user's go 2026-11-08, queue `scripts/xlang_programs_queue.sh`): the 14B with the
  product retry loop (`eval_translate.py --retries 2`: the compiler output or first failing test is
  fed back, temperature 0.1 → 0.4 → 0.7, like `migrate --to`), the 35 directions from JS / Go / C# /
  Rust / Swift sources (bootstrapped from passing translations; JS = passing TypeScript with its
  types stripped by Node, re-judged), and the base 7B. Results (2026-11-08, Ollama, local):

  | Run | Directions | First try | After ≤ 2 retries |
  |-----|-----------:|----------:|------------------:|
  | 14B, Python / C++ / Java sources | 21 | 131 / 420 (31.2 %) | **208 / 420 (49.5 %)** |
  | 14B, JS / Go / C# / Rust / Swift sources (bootstrapped) | 35 | 279 / 511 (54.6 %) | 343 / 511 (67.1 %) |
  | base 7B, Python / C++ / Java sources | 21 | 78 / 420 (18.6 %) | 122 / 420 (29.0 %) |

  The retry loop is worth +18 points; the bootstrapped directions score higher because their
  sources are programs the 14B already translated correctly once (an easier subset). The 7B is far
  behind (Go 5 / 60, TypeScript 2 / 60) → the 14B stays the translator.

## Stage 3 — data generation (done 2026-11-09, AWS)

`training/gen_xlang_pairs.py`, three resumable steps (outputs in `data/datasets/xlang/stage3/`,
git-ignored until final):

1. **`pool`** (CPU, ~150 s per shard, ~1.6 h for all 39): CodeContests **train** split, same
   filters as the program suite, one reference solution per problem (preferred language rotates
   Python 3 / C++ / Java), kept only if it passes the problem's 10 tests. Leak guard: titles and
   statements of every valid / test problem are skipped (Div. 1 / Div. 2 copies share a title).
   Shard 1: 251 candidates → 238 kept (C++ 110, Java 70, Python 58) → ~9K problems expected.
2. **`generate`** (GPU, `rebirth-gpu-job`): the 14B translates each program into the 7 other
   targets with the product prompt and retry loop; every generation and judge verdict is cached.
   1-problem smoke: 4/7 first try, 7/7 after retries, ~9 s per generation.
3. **`pairs`** (CPU): every two verified versions of one program form a pair — **forward**
   (reference → translation), **reverse** (translation → human-written reference), **cross**
   (translation → translation). JavaScript joins as a source (stripped TypeScript). 3 % of problems
   held out as `val`. A fully verified problem gives 63 rows; cross pairs dominate (48 / 63), so
   stage 4 subsamples them.

Why this data: it targets exactly what the program suite shows failing (input parsing, strict
compilers, integer widths), verification uses real judge tests, and reverse pairs give human-written
targets for Python / C++ / Java. Programs only teach competitive-programming style — mix with
corpus snippets in stage 4 so file-level `migrate` (classes, project idioms) is not forgotten.

**Run (AWS g6e.xlarge, vLLM, `scripts/cloud/session_p17.sh`, 2026-11-08 23:32 → 2026-11-09 09:27 EST):**

- **Teacher pilot** (40 pool problems × 7 targets, product prompt + 2 retries): the 14B verified
  **61.1 %** (47.9 % first try). Neither teacher got measured: Qwen3-Coder-30B-A3B-FP8 hit the
  10-minute vLLM readiness timeout (a cold ~30 GB load runs at ~45 MB/s from gp3, ~12 min) and
  Qwen3.6-27B-FP8 found no disk (the HF hub's shared blob store kept the "deleted" 30B). Both fixed
  (`READY_MIN=30` in `vllm_serve.sh`; `drop()` in `scripts/cloud/hub.sh` also deletes unreferenced
  blobs) → stage 3 ran with the 14B, as planned when no teacher clearly wins.
- **Generation:** 7,870 pool problems; 9.3 h cap → **9,387 translations** (20,553 generations with
  retries): **34.7 %** first try, **50.1 %** verified. Verified per target: C# 821, Go 672,
  Java 628, Python 605, C++ 508, Rust 500, TypeScript 500, Swift 470.
- **Pairs:** 1,194 problems with ≥ 2 verified versions → **30,211 rows** over 63 directions (forward
  4,704 · reverse 5,204 · cross 20,303; 927 val rows); 256–867 rows per direction, median ~680
  tokens, 97 % ≤ 2,048. Data in `data/datasets/xlang/stage3/` (git-ignored, also in S3).

## Stage 4 — shared adapter + eval (done 2026-11-09, AWS + local check)

**Training** (`training/train_xlang.py`, `scripts/cloud/session_p17_stage4.sh`): mix = 12,000
translation rows (≤ 190 per direction, 63 directions) + 2,000 same-language corpus rows → 13,616
rows ≤ 2,048 tokens; 14B bf16 LoRA r16 / α32, all linear layers, 2 epochs, 3,404 steps, **4 h 38 min**
on the L40S. Train loss 0.10 → 0.02 in epoch 2 (memorizing), but held-out problems still improve a lot.
Adapter: `data/models/lora_xlang_14b/` (DVC).

**Program suite** (same 20 problems × 21 directions as above, product prompt + ≤ 2 retries; CodeContests
valid / test problems, disjoint from the train-split pool; the 3 verbatim overlaps found were all a
shared fast-I/O header, not solutions):

| Model | Runtime | First try | After ≤ 2 retries |
|-------|---------|----------:|------------------:|
| base 14B | vLLM bf16 (cloud) | 147 (35.0 %) | 215 (51.2 %) |
| 14B + `lora_xlang_14b`, product prompt | vLLM bf16 (cloud) | 208 (49.5 %) | 274 (65.2 %) |
| 14B + `lora_xlang_14b`, SFT prompt | vLLM bf16 (cloud) | **237 (56.4 %)** | 268 (63.8 %) |
| Qwen3-Coder-30B-A3B-FP8, zero-shot | vLLM (cloud) | 209 (49.8 %) | **282 (67.1 %)** |
| Qwen3.6-27B-FP8 | — | vLLM engine failed to start (root cause cut from the log); not rerun | |
| base 14B | **Ollama q6_K (local)** | 131 (31.2 %) | 208 (49.5 %) |
| 14B + `lora_xlang_14b` | **Ollama q6_K + f16 GGUF (local)** | **207 (49.3 %)** | **265 (63.1 %)** |

Verified per target, local (base → adapter): TypeScript 14 → 27, Rust 21 → 32, Swift 22 → 32,
C# 36 → 49, Go 29 → 36, Java 29 → 32, Python 28 → 28, C++ 29 → 29. Per direction: 17 of 21 better,
3 equal, 1 worse (Java → Python 16 → 15).

- **The gain survives the shipped runtime**: q6_K + GGUF matches the cloud's first try (207 vs 208);
  after retries it is within sampling noise (265 vs 274).
- **Qwen3-Coder-30B zero-shot ≈ our adapter** (282 vs 274, within noise; it leads on Rust 41 vs 31).
  It does not fit the 16 GB target GPU (~19 GB at 4-bit), so it is a reference point, and the best
  candidate teacher for a stage-6 round, not the product model.
- **Product**: `scripts/ollama_xlang.sh` packages the adapter as `rebirth-xlang:14b-v1` (FROM the q6_K
  14B + ADAPTER, the base chat template kept: the adapter is served with the product prompt it was
  evaluated with). `rebirth migrate --to` (`--backend auto`) uses it when installed, else the plain
  14B; `--backend ollama` stays the plain 14B; `REBIRTH_XLANG_MODEL` overrides / turns it off.
  Reports: `data/reports/cloud/p17/eval_programs_*.json`, `data/reports/xlang/programs_xlang14b_q6k.json`.

## Stage 5 — the other 35 directions + HumanEval regression check (done 2026-11-10, local)

`scripts/xlang_stage5.sh`, on the shipped runtime (`rebirth-xlang:14b-v1`, Ollama q6_K + f16 GGUF
adapter), same settings as the plain-14B baselines (product prompt, ≤ 2 retries on programs). Run time
5 h 37 min on the 7800 XT.

**Program suite, the 35 directions with a bootstrapped source** (JavaScript, Go, C#, Rust, Swift as
sources; 511 translations, same problems as `programs_ollama14b_bootstrap.json`):

| Model | First try | After ≤ 2 retries |
|-------|----------:|------------------:|
| base 14B (q6_K) | 279 (54.6 %) | 343 (67.1 %) |
| 14B + `lora_xlang_14b` (q6_K + GGUF) | **371 (72.6 %)** | **407 (79.6 %)** |

Verified per target (base → adapter): TypeScript 21 → 41, Swift 25 → 39, Java 57 → 68, C# 45 → 50,
Go 40 → 46, Rust 38 → 44, C++ 55 → 57, Python 62 → 62. Per direction: 23 of 35 better, 5 equal, 7 worse
(C# → C++ −3, C# → Swift −2, Rust → Go −2, four more −1). **All 56 directions together: 551 → 672 / 931
verified.**

**HumanEval, all 56 directions** (2,229 short functions, first try, no retries):

| Model | Pass |
|-------|-----:|
| base 14B (q6_K) | 2,030 (91.1 %) |
| 14B + `lora_xlang_14b` | **2,064 (92.6 %)** |

28 directions better, 13 equal, 15 worse, none by more than 2 of ~40 → **no regression** on short
functions (the adapter was trained on whole programs). Per target (base → adapter, of ~280): Go 240 → 250,
Rust 243 → 253, the rest within ±4 (C# 263 → 260).

**One defect found — a Go input idiom.** 17 of the adapter's 115 first-try Go drafts on the program
suite build a `bufio.Scanner` and pass it to `fmt.Fscan*` (does not compile: a Scanner is not an
`io.Reader`); the plain q6_K 14B did it 0 times on the same 55 drafts. It costs Rust → Go most (6 fewer
first-try passes; retries recover most, 11 → 9 verified). The training data never does this (3,586 Go
targets: 2,000 use `bufio.NewReader` + `fmt.Fscan`, 1,197 use a Scanner alone, 0 mix them), and in the
cloud bf16 eval it was rare for both models (adapter 3 / 123, base 4 / 134). So it is mostly the local
runtime: an f16 LoRA applied on top of a q6_K base. The planned fix is the Phase 19 step **merge the LoRA
into the bf16 weights, then quantize** (ADR-001 item 4); re-check this count after merging. If it
persists, a one-line hint in the Go target notes or a stage-6 data round would cover it.

**Re-check after merging (Phase 19, 2026-11-11):** the merged v2 (`rebirth-xlang:14b-v2`, same scores as v1:
programs 667 vs 672 / 931, HumanEval 2,065 vs 2,064 / 2,229) does it in **13 / 115** first-try Go drafts
(v1 17). Merging reduced it but did not remove it, so it is not only the run-time LoRA. Retries recover most
of these; a Go target hint stays the cheap follow-up ([PHASE_19 § Step 2](PHASE_19_LAUNCH.md#step-2--merged-cross-language-model-rebirth-xlang14b-v2)).

Reports: `data/reports/xlang/programs_xlang14b_q6k_bootstrap.json`, `zeroshot_xlang14b_q6k.json`,
`zeroshot_xlang14b_q6k_bootstrap.json` (+ `.gen.jsonl` drafts); logs `logs/exp/xlang/stage5_*.log`.

## Design decisions

- **One shared cross-language adapter**, not 56. 8 languages = 56 directed pairs; the prompt names
  both sides so one run covers every direction and directions share what they learn:

  ```
  ### Source language
  python
  ### Target language
  go
  ### Source
  <code>
  ### Target
  ```
- **Zero-shot first.** Measure Ollama 14B / base 7B chat on all 56 directions before training
  anything. If it is good enough for most directions, train only for the weak ones.
- **Verification = reward.** Generated training pairs are kept only if the translation compiles in
  the target sandbox **and** behaves the same as the source (same inputs → same outputs). This is
  the rejection-sampling / best-of-N idea from Phase 11 applied to translation.
- **Sequence length ~1024.** A translation row holds the code twice (source + target), so 512
  tokens only fits small functions. Local 7800 XT: all-linear r16 QLoRA peaked at 9.2 GB @ 512 —
  1024 *may* fit but must be probed first (`training/qlora_probe.py --lengths 1024`); bf16 > 128
  crashes this GPU (see Phase 11 GPU incident). Safer on a 24 GB NVIDIA card.

## Stages and time estimate

Estimates use measured throughput (2026-11-05): ~9 s / optimizer step (8 rows @ 512, 4-bit) on the
7800 XT, ~25 min per 100-row 7B eval, a few seconds per short snippet on Ollama 14B.

| # | Stage | What | Time | Where | Needs the user? |
|---|-------|------|------|-------|-----------------|
| 1 | Pipeline | ✅ **Done 2026-11-07** — `rebirth migrate --to <lang>`, target-sandbox verification, MultiPL-E eval harness (see Stage 1 above) | 1 day | Local | Done |
| 2 | Zero-shot baseline | ✅ **Done 2026-11-07** — Ollama 14B, 56 directions × 40 HumanEval problems: **89.3 %** pass@1 (see Stage 2 above) | ~2 h | Local | Done |
| 3a | Teacher pilot | ✅ **Done 2026-11-09** — 14B 61.1 % verified; both teachers failed to load (fixed), 14B kept. Planned: **first cloud session of Phase 17 (~$2–4):** ~200 identical source snippets translated by the 14B and by the 32B teacher (vLLM), both verified the same way → measured verified-pass rate per model. Use the 32B for stage 3 only if it clearly wins (my estimate: +10–20 points, unmeasured) | ~1–2 h | **AWS** | Go to launch |
| 3 | Data generation | ✅ **Done 2026-11-09 (AWS, 9.3 h)** — 9,387 translations, 50.1 % verified → 30,211 pairs. Plan: CodeContests train programs translated by the 14B with the retry loop, judged by their own tests (see Stage 3 above); target ~1–2K fully processed problems → ~8–10K+ pairs | ~1.5–2 days local (~9–14 s / generation), or ~10–20 h on AWS with vLLM | Local (generation + verification) | No |
| 4 | Train | ✅ **Done 2026-11-09 (AWS, 4.6 h)** — `lora_xlang_14b`, 14K rows, 2 epochs, seq 2048; program suite 208 → 265 / 420 locally | ~5–8 h | **AWS** (or local if the 1024 probe passes) | No |
| 5 | Eval | ✅ **Done 2026-11-10 (local, 5.6 h)** — 35 bootstrapped directions 343 → 407 / 511; HumanEval 91.1 → 92.6 % over 56 directions, no regression; one Go idiom defect (see Stage 5) | ~6–8 h | Either | No |
| 6 | Fix-up round | ⏭ **Not run (owner, 2026-11-10)** — no further cloud spend without a guaranteed jump; Qwen3-Coder-30B as teacher is only on par with the adapter (282 vs 274 / 420). Scripts kept (`scripts/cloud/session_p17*.sh`) if a much stronger teacher appears. Planned: more data for weak directions (likely Swift, Rust), retrain | ~1 day | Either | Mostly no |

**Total ≈ 1 week calendar, ~2–3 days of hands-on work, ~60–80 GPU-hours** (fewer on AWS with batched vLLM generation).

### 32B teacher: what it is for (decided 2026-11-06)

- **Main model stays 14B.** A 32B student would cost ~$100–170 to train for 8 languages and does not
  fit a 16 GB GPU at 4-bit (~19–20 GB), so it cannot be the offline product. The user dropped
  "32B as the main model" on 2026-11-06.
- **Teacher only, on the cloud.** Expected value: **large for cross-language** (no parallel corpus
  covers 56 directions; a better teacher means more verified pairs per GPU hour), **small for
  same-language** (≈ +0–3 soft-exact / 100, i.e. noise. Our metric scores against human commits,
  so a better-but-different rewrite does not score). Side uses in the same session: good rewrites
  for the **no-op cases** (DPO "chosen" vs the adapter's unchanged output) and cleaning weak rows.
- **Not for C++/Rust same-language data.** `clang-tidy` / `cargo fix` pairs are correct by
  construction ([../research/RUST_CPP_DATA_SOURCES.md](../research/RUST_CPP_DATA_SOURCES.md)).
- **Every teacher output is verified** (target compiles + behaves like the source) before it is
  trained on. Unverified teacher data teaches the teacher's mistakes.
- License: Qwen2.5-Coder-32B-Instruct is Apache 2.0, so its outputs can be used as training data.
  Re-check newer coder models (e.g. Qwen3-Coder) at the time of the pilot.

Biggest risks: the cross-language equivalence harness for 8 languages (largest engineering piece);
sequence length on the local GPU; public parallel datasets (e.g. Rosetta Code, which has the same
tasks solved in all 8 languages) could replace part of stage 3, but licenses and quality must be
checked first.

## AWS plan (original estimate — superseded by [CLOUD_GPU_PLAN.md](../CLOUD_GPU_PLAN.md))

The user has an AWS account with **$120 credit** for all phases (G/VT quota: 8 vCPUs requested, case open 2026-11-05).

| Instance | GPU | On-demand (us-east-1, 3rd-party price lists — verify on AWS) | Hours for $100 |
|----------|-----|-------------------------------:|---------------:|
| **g6.xlarge** (default choice) | NVIDIA L4 24 GB | ~$0.80/h | ~125 h |
| g5.xlarge | NVIDIA A10G 24 GB | ~$1.01/h | ~100 h |
| g6e.xlarge | NVIDIA L40S 48 GB | ~$1.86/h | ~54 h |

Spot is ~half price but can be interrupted (fine for stage 3 with checkpointed output). Add a few
dollars/month for a ~100 GB EBS volume.

**Split (estimated ~$20–40 total, leaves room for a second round):**

| Stage | Where | Est. cost |
|-------|-------|----------:|
| 1–2 pipeline + zero-shot baseline | Local | $0 |
| 3 generation (vLLM, batched — ~10× Ollama one-at-a-time) | AWS g6.xlarge | ~$10–16 |
| 3b compile + equivalence verification (CPU) | Local (more cores than the 4 vCPU g6.xlarge) | $0 |
| 4 train shared adapter @ ~1024 | AWS g6.xlarge | ~$7–12 |
| 5 eval | Either | $0–8 |

Why NVIDIA helps here: 24 GB + mature CUDA make seq 1024 safe, **4-bit generation works** (it is
broken on ROCm), and vLLM batches generation. The same-language adapters stay local — they fit the
7800 XT.

### Before starting (account checklist)

1. **Free plan probably blocks GPU instances** (not confirmed in AWS docs; user reports say an
   upgrade to the **Paid plan** is needed). After upgrading, spend beyond the credit bills the card.
2. Request a **Service Quotas** increase now — *Running On-Demand G and VT instances* → 4–8 vCPUs
   (new accounts are usually 0; approval can take hours to days).
3. Set **AWS Budgets** alerts at ~$50 and ~$90.
4. Always stop/terminate the instance when a job finishes (scripts should self-shutdown).

## Exit criteria

- ✅ (stage 1, 2026-11-07: 53 / 56 fixture directions, failures documented above) `rebirth migrate
  <repo> --lang python --to go` produces a Go file that passes the target sandbox on the fixture
  samples, for all 56 directions or a documented subset.
- ✅ Per-direction eval table (compile rate, equivalence rate) vs the zero-shot baseline. (Baseline
  2026-11-07: Stage 2 above; with the adapter: Stages 4–5, per-direction rows in the reports.)
- Adapter (if trained) beats zero-shot on the directions it covers; otherwise ship zero-shot.
  ✅ (2026-11-09, program suite, 21 directions: 17 better, 3 equal, 1 worse by one problem; 208 → 265
  verified. Shipped as the `migrate --to` default. 2026-11-10, stage 5: the other 35 directions 343 →
  407, all 56 together 551 → 672 / 931; HumanEval 91.1 → 92.6 %, no direction down by more than 2 / 40.)

## Sources (AWS facts, checked 2026-11-05)

- [AWS Free Tier explained (Rackspace)](https://spot.rackspace.com/blog/aws-free-tier)
- [Reddit thread on Free plan instance access](https://reddit.sentinel-team.org/posts/1rpn9gm/snapshots/2026-03-11T04%3A58%3A06.070402Z)
- [g6.xlarge pricing (DoiT)](https://www.doit.com/compute/spot/us-east-1/g6.xlarge) · [g5.xlarge (DoiT)](https://compute.doit.com/spot/us-east-1/g5.xlarge) · [g6e.xlarge (DoiT)](https://compute.doit.com/spot/us-east-1/g6e.xlarge)
