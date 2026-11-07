# Phase 17 — Cross-language migration (all 8 languages)

> Status: **IN PROGRESS** — stage 1 (pipeline + eval harness) and stage 2 (zero-shot baseline: Ollama 14B **89.3 %** pass@1 over all 56 directions on HumanEval) done 2026-11-07, started early while Phase 15 waits on the AWS quota. Next: stage 3 decisions (see "Next" below). Decided 2026-11-05; was "Phase 15" before the renumbering.
> Previous: [PHASE_16_MLOPS_CICD_PACKAGING.md](PHASE_16_MLOPS_CICD_PACKAGING.md) · Next: [PHASE_18_MLOPS_MONITORING.md](PHASE_18_MLOPS_MONITORING.md)
> Owner decision: do it, local first, AWS GPU for the heavy stages.
> Cloud setup, access model and budget split: [../CLOUD_GPU_PLAN.md](../CLOUD_GPU_PLAN.md) — Phase 17 now uses the 14B base, a 32B teacher on g6e.xlarge (L40S 48 GB) and ~$60 of the $120 credit.

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
(or HumanEval) assert holds. 7 rows of Go `HumanEval_87` are excluded (broken test).

| src \ tgt | py | ts | go | java | c# | rust | c++ | swift | all |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **py** | – | 39/40 | 27/39 | 36/40 | 38/40 | 36/40 | 40/40 | 34/40 | 90 % |
| **js** | 35/40 | – | 33/39 | 37/40 | 37/40 | 34/40 | 38/40 | 35/40 | 89 % |
| **go** | 38/40 | 36/40 | – | 37/40 | 38/40 | 37/40 | 39/40 | 34/40 | 92 % |
| **java** | 36/40 | 37/40 | 25/39 | – | 38/40 | 36/40 | 40/40 | 32/40 | 87 % |
| **c#** | 40/40 | 37/40 | 29/39 | 36/40 | – | 35/40 | 37/40 | 33/40 | 89 % |
| **rust** | 39/40 | 39/40 | 29/39 | 35/40 | 36/40 | – | 38/40 | 33/40 | 89 % |
| **c++** | 35/40 | 37/40 | 30/39 | 39/40 | 40/40 | 35/40 | – | 35/40 | 90 % |
| **swift** | 39/40 | 38/40 | 27/38 | 34/39 | 36/39 | 30/39 | 37/40 | – | 88 % |

(js → ts is skipped: TypeScript accepts the JavaScript as is.)

| Target | Pass | Compile error | Wrong answer |
|--------|-----:|--------------:|-------------:|
| C++ | **96.1 %** (269 / 280) | 5 | 6 |
| C# | 94.3 % (263 / 279) | 7 | 9 |
| TypeScript | 93.9 % (263 / 280) | 1 | 16 |
| Python | 93.6 % (262 / 280) | 0 | 18 |
| Java | 91.0 % (254 / 279) | 12 | 13 |
| Rust | 87.1 % (243 / 279) | 17 | 19 |
| Swift | 84.3 % (236 / 280) | 15 | 29 |
| Go | **73.5 %** (200 / 272) | 55 | 17 |
| **All** | **89.3 % (1,990 / 2,229)** | 112 | 127 |

No timeouts. By source the spread is small (Go 92.5 % … Java 87.5 %): the target language decides.

What it says:

- **The 14B is already good at short functions in every direction.** The 6 weakest directions are
  all *into Go* (64–77 %); every other direction is ≥ 75 %, four are 40 / 40.
- **Go: most failures are missing imports.** 42 of 55 Go compile errors are `undefined: sort /
  strings / math / strconv` — the model calls `sort.Ints` but does not write `import "sort"`
  (the prompt asks it to). A deterministic goimports-style fix-up of standard-library imports would
  lift Go to roughly 89 % without any training; `migrate --to` already recovers these through its
  retry loop (the compiler error is fed back).
- **Swift and Rust** fail on type / API details (`String.Index` subscripts, `Character` vs
  `CharacterSet`, integer casts, a `regex` crate that is not in the standard library).
- C# looked like the weakest target (66 %) until the harness bug above was fixed: 56 rows moved
  from fail to pass on re-scoring, no other row changed.

Caveats:

- **HumanEval functions are short (~10 lines) and famous** — the model may have seen solutions in
  each language. Treat this as an upper bound for real file migration (several functions, classes,
  project idioms), which the fixture matrix (53 / 56 with retries) only smoke-tests.
- 40 of ~158 problems per direction: ± ~5 points per cell (binomial SE at 90 %), ± ~2 points per
  target row. Extending to all 164 is incremental (same seeded order, cached generations): ≈ 4–5 h.
- Rust / C# / Swift sources are model-written (verified) translations, so those 21 directions
  start from code in the model's own style — slightly easier than human-written sources.

### Next — stage 3 decisions (the user's call)

- **Go import fix-up first** (CPU, no training): add missing standard-library imports for Go before
  verification, in both `migrate --to` and the eval, then `--rescore`.
- **Base 7B baseline** (`--backend peft`, ~2–3 h GPU): tells whether the local 7B is usable for
  translation or the product needs the 14B for it.
- **A harder eval before generating training data:** HumanEval leaves little headroom
  (89 % zero-shot). Stage 3 data should target file-level translation (multi-function files,
  classes), and needs an eval that can show a gain there.

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
| 3a | Teacher pilot | **First cloud session of Phase 17 (~$2–4):** ~200 identical source snippets translated by the 14B and by the 32B teacher (vLLM), both verified the same way → measured verified-pass rate per model. Use the 32B for stage 3 only if it clearly wins (my estimate: +10–20 points, unmeasured) | ~1–2 h | **AWS** | Go to launch |
| 3 | Data generation | Translate snippets from the existing per-language corpus into each target; ~20K candidates → expect 40–60 % verified → **~8–10K pairs** | ~1.5–2 days local, or ~10–20 h on AWS with vLLM | **AWS** (generation) + local (verification) | No |
| 4 | Train | One shared adapter, ~8–10K pairs, 2 epochs, seq ~1024 | ~5–8 h | **AWS** (or local if the 1024 probe passes) | No |
| 5 | Eval | 56 directions vs the stage-2 baseline | ~6–8 h | Either | No |
| 6 | Fix-up round | More data for weak directions (likely Swift, Rust), retrain | ~1 day | Either | Mostly no |

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
- Per-direction eval table (compile rate, equivalence rate) vs the zero-shot baseline. (Baseline
  ✅ 2026-11-07: Stage 2 above.)
- Adapter (if trained) beats zero-shot on the directions it covers; otherwise ship zero-shot.

## Sources (AWS facts, checked 2026-11-05)

- [AWS Free Tier explained (Rackspace)](https://spot.rackspace.com/blog/aws-free-tier)
- [Reddit thread on Free plan instance access](https://reddit.sentinel-team.org/posts/1rpn9gm/snapshots/2026-03-11T04%3A58%3A06.070402Z)
- [g6.xlarge pricing (DoiT)](https://www.doit.com/compute/spot/us-east-1/g6.xlarge) · [g5.xlarge (DoiT)](https://compute.doit.com/spot/us-east-1/g5.xlarge) · [g6e.xlarge (DoiT)](https://compute.doit.com/spot/us-east-1/g6e.xlarge)
