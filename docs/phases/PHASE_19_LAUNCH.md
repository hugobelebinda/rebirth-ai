# Phase 19 — Packaging, UX & Official v1.0 Launch

> Status: **RELEASE READY (2026-11-11)** — last phase (was "Phase 11" before the 2026-11-05 renumbering). Models published on Hugging Face 2026-11-11; version 1.0.0 on `rebirth/v1.0.0`; the `v1.0.0` tag and the GitHub release wait for the owner's go.  
> Previous: [PHASE_18_MLOPS_MONITORING.md](PHASE_18_MLOPS_MONITORING.md)  
> Depends on: Phase 10 complete **and** LoRA specialization waves ([PHASE_11_LORA_PER_LANGUAGE.md](PHASE_11_LORA_PER_LANGUAGE.md)) — or explicit user skip of LoRA  
> Full plan: [PRODUCTION_ROADMAP.md](../PRODUCTION_ROADMAP.md) § Phase 19  
> Last docs refresh: 2026-11-11 (night: pre-launch review)  
> Handoff: [SESSION_HANDOFF.md](../SESSION_HANDOFF.md)

## Why deferred

Phases 0–10 delivered **8 language verticals + ≥15K corpus + quality CLI**.  
Next priority is **train a LoRA adapter per language** (Phase 11, done 2026-11-05; 14B upgrade in Phase 15). Official **`v1.0.0`** packaging waits until those adapters exist (or you choose an Ollama-only launch).

## Objectives (when resumed)

- Bootstrap script installs tooling and **all** sandbox images.  
- README / docs cover every target language (+ PEFT adapter map if shipped).  
- Tag **`v1.0.0`** official launch (not preview).

## Locked launch surface

Languages: **JS/TS, Python, Go, Java, C++, C#, Rust, Swift**  
Corpus: **≥15K** mined multi-language migration pairs (**24 640** already)  
Quality: fixture sandbox target **≥90%** per language; truthful multi-file PR reports  
Preferred brain: Ollama 14B default + **per-lang PEFT** when adapters exist

## Exit criteria

- [x] Phases 11–18 complete **or** written skip decision — 11–17 complete; 18's first flywheel retrain deferred to after launch (needs real use, [PHASE_18](PHASE_18_MLOPS_MONITORING.md))  
- [x] `scripts/bootstrap.sh` + doctor green path documented for second machine — [INSTALL.md](../INSTALL.md), clean-copy test 2026-11-10  
- [x] Fixture E2E for all 8 languages — 8 / 8 closed-loop with the served models, hardened sandbox (2026-11-11, step 4)  
- [x] ≥1 real JS/TS repo + ≥1 real non-JS repo migrate report — is-number (JS), go-homedir (Go), inflection (Python) + 2 `--to` (2026-11-11, step 4). The served 7B adapters often return files unchanged: the 14B fallback (shipped) handles them  
- [x] `migrate-repo --open-pr` on a throwaway repo — private `rebirth-openpr-test`, PRs #1 (JS) and #2 (Python), 2026-11-11 (step 4)  
- [x] Security checklist (secrets, sandbox network default `none`) — [SECURITY.md](../SECURITY.md); model-written code never has network (step 4)  
- [x] Corpus ≥15K documented in `DATASET.md` (24,640)  
- [ ] Tag `v1.0.0`  

## Plan (agreed 2026-11-10)

Two model packs (ADR-001), models hosted on **Hugging Face Hub** (owner's choice 2026-11-10). Steps 1–5 run
now; step 6 (publish + tag) only after a review with the owner.

| # | Step | State |
|---|---|---|
| 1 | Merge PR #5 (Phase 17 + 18) | done (owner merged it 2026-11-10) |
| 2 | Merge the cross-language LoRA into the 14B before quantizing → `rebirth-xlang:14b-v2`, gate vs v1 | done 2026-11-11: gate passed, v2 served, rollback demo done |
| 3 | `scripts/bootstrap.sh` + `rebirth install` + `rebirth doctor --strict` | done, clean-copy test passed |
| 4 | Release checks: fixtures 8/8, real-repo reports, `--open-pr`, security checklist | done (2026-11-11), plus fallback, bug hunt and the [pre-launch review](#pre-launch-review-and-fixes-2026-11-11-evening) |
| 5 | Docs: INSTALL, README (now a user guide), SECURITY, CHANGELOG, Hugging Face model cards | done; cards are drafts until step 6 |
| 6 | Hugging Face upload, `configs/models.json` `hf_namespace`, version 1.0.0, tag | models **published 2026-11-11** (checked anonymously: sha256 of every file); 1.0.0, README, demo on `rebirth/v1.0.0`; tag + release on the owner's go |

## Built (2026-11-10)

### Step 2 — merged cross-language model `rebirth-xlang:14b-v2`

v1 is the q6_K 14B with the f16 LoRA applied at run time (`ADAPTER`). v2 merges the LoRA into the **bf16**
weights first, then quantizes once:

1. `training/merge_lora.py`: merges shard by shard (`W += (α / r) · B·A` in fp32, back to bf16); RAM peak ≈ one
   shard (~5 GB), so it runs on the 30 GB machine where `merge_and_unload()` (whole model in RAM) cannot.
   `tests/test_merge_lora.py` checks it against PEFT's own `merge_and_unload()` on a tiny Qwen2 (weights and
   logits). On the 14B: 336 / 336 LoRA weights merged, spot checks within bf16 rounding (≤ 0.0016).
2. llama.cpp (pinned `6c01da6`): `convert_hf_to_gguf.py --outtype bf16` → `llama-quantize Q6_K` (12.12 GB,
   within 160 bytes of Ollama's official q6_K).
3. Ollama model with the base's chat template, system prompt and license verbatim
   (`configs/ollama/xlang-14b.Modelfile`). Whole pipeline: `scripts/xlang_v2_merge.sh` (~60 GB disk at peak).

Speed on the 7800 XT (same prompt, greedy): **40.5 vs 32.7 tokens/s** generating (+24 %), **466 vs 321
tokens/s** reading the prompt (+45 %).

Gate (`scripts/xlang_v2_eval.sh`, then `rebirth serve publish-xlang`): the same 931 program translations and
2,229 HumanEval translations as v1; v2 must not drop by more than 2 % (programs) / 1 % (HumanEval) of n.
Also watched: the Go `bufio.Scanner` + `fmt.Fscan` defect (v1: 17 / 115 first-try Go drafts).

**Result (2026-11-10 → 11, ~13 h on the 7800 XT, paused overnight after A1 and resumed):**

| Suite | Translations | v1 (adapter at run time) | **v2 (merged)** | Allowed drop |
|---|---:|---:|---:|---:|
| Programs, 21 directions with human sources (A1) | 420 | 265 | **265** | |
| Programs, 35 bootstrapped directions (A2) | 511 | 407 | **402** | |
| **Programs, verified after ≤ 2 retries** | **931** | **672** | **667** | 18 |
| HumanEval, 35 directions (B1) | 1,396 | 1,296 | **1,295** | |
| HumanEval, 21 bootstrapped directions (B2) | 833 | 768 | **770** | |
| **HumanEval, first try** | **2,229** | **2,064** | **2,065** | 22 |

Gate passed: v2 is v1 within noise (per target language, programs move by at most 4 translations either way), and it is
faster. Go defect: **13 / 115** first-try Go drafts (v1 17), 22 / 206 over all rounds (v1 30 / 211).
`xlang_v2_eval.sh` now skips a stage whose report exists, so a stopped run restarts where it left off.

Served (2026-11-11): `rebirth serve publish-xlang` put v1 on the translation route (against the plain 14B:
672 vs 551, 2,064 vs 2,030), then v2 (against v1). Rollback demo: `rebirth serve rollback --lang xlang`
→ `rebirth-xlang:14b` = v1 (Ollama digest `c5f020f8d613`), again → v2 (`9611346cead7`), history keeps v1.
`migrate --to` (`--backend auto`) now follows the route's alias `rebirth-xlang:14b` instead of a fixed
`rebirth-xlang:14b-v1` tag (`REBIRTH_XLANG_MODEL` still overrides).

### Step 3 — install

- `scripts/bootstrap.sh [--pack both|modernize|migrate] [--from DIR | --from-local]`: checks tools (no sudo),
  `.venv` with CPU torch, embedding model cache, the 8 sandbox images (also tagged with the app version),
  `rebirth install`, `rebirth doctor --strict`. Re-runnable.
- `rebirth install` (`src/rebirth_ai/install.py`, `configs/models.json`): per pack, downloads from Hugging Face
  (resumable), checks size + sha256, writes the Modelfile, `ollama create` + alias; skips what is installed.
  `--from DIR` installs from a folder (mirror / USB). `tests/test_install.py` (6).
- `rebirth doctor --strict`: adds GPU + VRAM (nvidia-smi, rocm-smi or the amdgpu driver), disk, per-pack
  model counts, and exits 1 on a missing requirement.
- Translation route: `translation:` in `configs/routing.yaml`, `rebirth serve publish-xlang` (gate on eval
  reports) and `rebirth serve rollback --lang xlang` (swap with the previous version; again = roll forward).
  `tests/test_xlang_serving.py` (4). Gate for v1 vs the plain 14B: programs 672 vs 551, HumanEval 2,064 vs 2,030.
- Fixed: PyYAML is now a core dependency (`rebirth migrate` reads `configs/routing.yaml`; it only arrived
  through `transformers` before).
- Clean-copy test (2026-11-10): fresh copy of the repo without `.venv`, `bootstrap.sh --from <release
  folder>` → 3 min 56 s, `doctor --strict`: all checks passed.

### Step 4 — release checks (2026-11-11)

**Sandbox hardening** (security checklist; [SECURITY.md](../SECURITY.md), `src/rebirth_ai/sandbox/`). Before,
Maven / NuGet downloads ran during the tests, so Java and C# drafts (and their translation checks) ran
with network, and packages a draft asked for were installed in the same networked container. Now:

- **Setup, then test, in one container.** The project's declared dependencies are downloaded first,
  online, before the draft is written in. The container is then disconnected from every network
  (checked; the run fails closed), the draft is written in, and the tests run offline. Per language:
  `npm ci` / `npm install`, `pip install --user`, `mvn test` (Maven only fetches its test provider when
  tests run), `dotnet restore`, `go mod download`, `cargo fetch`, `swift package resolve`.
- **Limits:** 4 GB memory with no swap, 1,024 processes, all capabilities dropped, `no-new-privileges`,
  `timeout` inside the container plus a host kill 30 s later.
- **Package policy:** a missing package is installed only if the project declares it or the original
  file imports it. Names are validated and shell-quoted, and Python import names are mapped to PyPI names.
  A refused package is explained to the model in the next prompt.
- **Also:** the code-search embedder loads offline, and bootstrap no longer writes a `.env` the CLI never
  reads.
- **Tests:** `tests/test_sandbox_hardening.py` (15), `tests/test_sandbox_confinement_docker.py` (6, real
  containers).
- **Same scores:** re-testing the 420 saved A1 v2 translations in the hardened sandbox gives the same
  265 / 420 (211 first try), identical in all 21 directions.

**Fixture E2E**, served models (`--backend auto`), broken seed draft → model → sandbox:

- All 8 languages succeed on iteration 1: `rebirth-<lang>:7b` for JS/TS, Python, Go, Java, C#, Rust, C++
  and Swift (12–39 s each).
- `--to`: Python → Go against the Go fixture's tests succeeds on iteration 2 with `rebirth-xlang:14b` (v2).
- Phase 7 JS benches (`training/bench_js_repos.py`, plain 14B): 6 / 6 ok (success or human review, the bench gate).
  - `bench_express` now proves the refusal: a draft-only `pretty-ms` is refused, the model drops it, and
    the run ends in human review. The original passes the tests; the 14B kept reimplementing pretty-ms.
  - `bench_express_optin` (`REBIRTH_ALLOW_UNDECLARED_PACKAGES=1`) proves the install: queued, set up
    online, PASS.

**Real repos** (shallow clones in git-ignored `data/bench/`, MIT, no writes):

| Repo | File | Served model | Result |
|---|---|---|---|
| jonschlinkert/is-number | `index.js` (18 lines), tests: mocha via `npm test` | `rebirth-js-ts:7b` | tests pass, file **unchanged** |
| mitchellh/go-homedir | `homedir.go` (167 lines), `homedir_test.go` | `rebirth-go:7b` | tests pass, file **unchanged** |
| jpvanhal/inflection | `inflection/__init__.py` (429 lines), `test_inflection.py` | `rebirth-python:7b` | tests pass, one blank line added (**unchanged**) |
| is-number | `index.js` → Go (`--to go`) | `rebirth-xlang:14b` | compiles, iteration 1 |
| go-homedir | `homedir.go` → Rust (`--to rust`) | `rebirth-xlang:14b` | compiles, iteration 1 |

The same three files with the plain 14B (`--backend ollama`) are real modernizations, all passing the
projects' tests:

- **is-number:** an arrow function, and the `isFinite` fallback dropped.
- **go-homedir:** a `switch` and merged conditions, but the explanatory comments were deleted.
- **inflection:** f-strings, `typing` imports and an annotated `__version__`.

**Fixed from these runs:**

1. **Test files are found by what they import.** migrate-repo skipped is-number and inflection ("no
   matching test file"). It now falls back to a test that imports the file; JS runs the project's own
   `npm test`.
2. **Long files are no longer cut off.** The served 7B's chat path was capped at 1,024 output tokens, so
   go-homedir was cut off ("unexpected EOF" in all 5 drafts). The cap is now 8,192, the same as the plain
   client.
3. **A refused package is explained to the model**, so it stops importing it.
4. **A passing draft that changes nothing is reported as `unchanged`** instead of `success`, and is never
   written or put in a PR. Before, these real-repo runs reported "success".

**Measured on more files (owner's go, 2026-11-11):** `training/bench_real_repos.py`.

- **Set-up:** 13 MIT / BSD repos at pinned commits, 18 files, objective "modernize to current idioms,
  keep the public API, behavior and comments". The report is `data/reports/real_repos.json`.
- **Left out (4 files):** the project's own tests fail on the original in the sandbox:
  - fill-range: needs an undeclared `eslint`.
  - minimist: the tests download npm.
  - pkg/errors (2 files): the tests expect the GOPATH folder layout.
- **Fixed on the way:**
  - Python tests that import through the package now map to the test file, or the test folder.
  - Projects with no marker file are detected from their source files.
  - A Go project with no `go.mod` gets `go mod init` + `go mod tidy` in the sandbox copy.

| On 14 files | Changed, tests pass | Unchanged | Failed (5 drafts) |
|---|---:|---:|---:|
| Served 7B (`--backend auto`) | 4 | 10 | 0 |
| Plain 14B (`--backend ollama`) | 8 | 1 | 5 |
| 7B, then the 14B when unchanged | 9 | 1 | 4 |

Read by hand, the 9 fallback results are:

- **6 clean:** shortuuid and slugify `special.py` (7B); is-number, repeat-string, string-utils `generation.py`
  and go-humanize `comma.go` (14B).
- **3 pass the tests but are wrong:**
  - go-homedir (14B): on macOS, a failed `dscl` call no longer falls back to the shell. That branch is
    untested on Linux.
  - inflection (7B): `set[str]` breaks the Python ≥ 3.5 the project supports.
  - to-regex-range (7B): the exported `cache` becomes a `Map`.

Comments were kept: at most 2 comment lines lost per file, with the objective asking for it. The 7B alone
gives 2 clean results out of 14, the fallback 6. Cost: one 14B run per unchanged file, 24–290 s here, and
the longest when it fails.

**Fallback shipped (owner's go, 2026-11-11):** `src/rebirth_ai/graph/fallback.py`.

- **When:** a served 7B adapter passes the tests but changes nothing.
- **What:** one more run with a 14B. It tries `REBIRTH_FALLBACK_MODEL`, then the plain Qwen2.5-Coder 14B,
  then `rebirth-xlang:14b`, whichever Ollama has. If none is there, it is skipped with a note.
- **On by default:** `--no-fallback` or `REBIRTH_FALLBACK=0` turns it off. `migrate-repo` runs all
  fallbacks after the 7B pass, so the models swap once instead of once per file.
- **Measured:**
  - The shipped default, end to end on the 14 files: 9 changed and passing, 1 unchanged, 4 failed. That
    matches the estimate composed from separate runs.
  - `rebirth-xlang:14b` as the fallback, on the 10 files the 7B left unchanged: 2 clean modernizations
    and a docstring typo fix, with no regression. The plain 14B gets 4 clean and 1 regression, so it is
    tried first.
- **Also:**
  - The default objectives ask the model to keep comments and docstrings.
  - Reports, PR bodies and successful runs say "passing tests are evidence, not proof".

**Bug hunt (2026-11-11, before anything is published):** 10 rounds, each fix with tests (264 → 307 tests).

1. **Bad input:**
   - `migrate --file` accepted paths outside the project, secrets files and huge files. Binary and
     non-UTF-8 files crashed, and empty files went to the model.
   - Ollama or Docker down, a missing sandbox image, `--to klingon`, a malformed report, rollback of an
     unknown language, and `sandbox` on a folder with no code gave tracebacks or wrong messages.
   - All of these now end in a one-line error with exit code 2.
2. **`--open-pr`:**
   - It opened the PR on the repository you ran the command from, not the project's.
   - Paths were wrong for a project in a subfolder.
   - It committed the user's unrelated staged changes, and `-B` silently reset an existing branch.
   - It staged a report file outside the repo, so `git add` failed.
   - The PR body showed absolute local paths and unredacted logs.
   - Without `--write` it committed nothing. Now `--open-pr` implies `--write`, and git and `gh` are
     checked before any model runs. `tests/test_github_pr.py` covers this with a real repo and a fake `gh`.
3. **Downloads:** a truncated or corrupt model file, or a `.part` file longer than the release, failed on
   every later run. A dropped connection gave a traceback instead of "run again to resume".
4. **Writing files:**
   - `--write` turned CRLF files into LF (every line changed) and lost the BOM and the final newline.
   - The fixes for long files (chat output ceiling) and for passing drafts that change nothing are under
     "Fixed from these runs" above.
5. **Running tests:**
   - Paths with spaces broke the sandbox command.
   - Python ran bare `pytest`, so a package next to `tests/` could not be imported.
   - A `.py` file in a project with `package.json` got the JS adapter.
   - `rebirth migrate` defaulted `--file` and `--test` to a fixture's paths, so without `--test` it ran
     the wrong test.
   - Test discovery and language detection for real layouts are under the real-repo results above.
6. **Code search:** the index of whatever project was last indexed (`data/embeddings.db`) fed its chunks
   to every migration. Now an index is used only for its own project.
7. **Error classes:**
   - Go, Rust, Java, C#, C++ and Swift compile errors were classified with JS / Python rules. A Go syntax
     error came out as a "missing dependency" and took the run to the package-install step.
   - A low-confidence guess no longer goes to install.
8. **Comments:** a passing draft that deletes the original's comments (at least 3 lines and over 20 %) goes
   back to the model with a note. The 14B fallback had deleted 9 of go-homedir's 22 comment lines.
9. **Other:**
   - `OLLAMA_HOST` without a scheme broke the clients, and `rebirth doctor` ignored `OLLAMA_HOST`.
   - `install --from <missing folder>` pulled models first.
   - The pyproject `license` table is deprecated.
   - bootstrap asked the Hugging Face Hub for an embedding model it already had.
   - INSTALL now says the 7B base still comes from the Ollama library.

Final regression on the finished code (2026-11-11):

- 8 / 8 fixture languages on iteration 1; Python → Go on iteration 2.
- Phase 7 benches 6 / 6.
- The 14 real files with the shipped defaults: 8 changed and passing, 1 unchanged, 5 failed.
  - go-homedir is now human review: the comment guard turned down 5 drafts that stripped its comments,
    including the one that also removed its macOS fallback.
  - Every other file kept its comments.
  - Of the 8, 6 are clean and 2 are the 7B issues listed above (inflection, to-regex-range).

Checked clean:
- `--help` of all 24 commands and their subcommands.
- `bootstrap.sh` (shellcheck).
- A clean-copy bootstrap: 38 s, `doctor --strict` green.

**`--open-pr` on a throwaway repository** (owner's go, 2026-11-11): a new **private** GitHub repo,
`rebirth-openpr-test`, with two small dummy legacy projects in subfolders (`js-legacy/` with
`node --test`, `py-legacy/` with pytest). `migrate-repo <clone>/<project> --open-pr` was run from the
Rebirth checkout, the case that used to target the wrong repository:

| PR | Branch | Files committed | Result |
|---|---|---|---|
| #1 | `rebirth/migrate-js` | `js-legacy/src/array.js` (7B), `js-legacy/src/strings.js` (14B fallback) | correct files, repo-relative paths, body with no local paths, review note |
| #2 | `rebirth/migrate-py` | `py-legacy/legacy/textutil.py` (7B); empty `__init__.py` skipped | correct |

The PR #1 diff shows why the review note is there. All tests pass, but:
- `!seen.hasOwnProperty(x)` → `!seen[x]` changes `unique(["toString"])`.
- `Array(n).fill(str)` throws on a negative `n`.

Neither case is tested by the dummy project. The Rebirth checkout was left untouched. Deleting the
repository is the owner's call (`gh auth refresh -s delete_repo`, then `gh repo delete`).

### Step 5 — docs

`docs/INSTALL.md` (requirements, packs, offline install, troubleshooting, uninstall), README "Install",
Hugging Face model card drafts in `docs/release/`.

### Pre-launch review and fixes (2026-11-11, evening)

PR #6 (steps 1–5) was merged as `fa37aef`. CI first failed on two tests that depended on the developer
machine (no sandbox images on GitHub; Rich colours output on Actions); `3ec18c9` fixed them. Then came a
review of what a new user would meet, and these fixes on `rebirth/p19-prelaunch`:

| Found | Fix |
|---|---|
| The sandboxes ran Go 1.22, Rust 1.82, Swift 5.10 and .NET 8 (support ended 2026-11-10). A `go.mod` declaring `go 1.24` failed before any test ran. | Go 1.27, Rust 1.99, Swift 6.4, and .NET 10 with the .NET 8 SDK and packs, so `net8.0` projects build offline. Checked: a `go 1.24` module, a Rust 2024-edition crate, the C# fixture as `net8.0` and as `net10.0`. Java stays on JDK 21 (supported to 2029). |
| When the 14B fallback failed after the 7B left a file unchanged, the untouched, passing file was reported as `human_review` (exit 1; "needs review" in PRs). | It is reported as unchanged, with the 14B's attempts in the trace. The 14B gets 2 tries (`REBIRTH_FALLBACK_TRIES`). |
| Run logs, the code-search index, the migrate-repo report and the dashboard were written relative to the current folder, so running from your own project wrote `data/` into it. The error classifier was looked up the same way, and is not in git, so a fresh clone ran on keyword rules only. | Defaults live in `<Rebirth>/data` (`paths.py`). The classifier trains itself on first use (< 1 s, same 0.926 accuracy), and `bootstrap.sh` trains it. |
| `migrate-repo` looked only in `src/`, so it found nothing in most Go and many Python repos. | It looks in `src/` if the project has one, else in the whole project. |
| `migrate` printed no change on success. | It prints the diff, and where the last draft is when no draft passed. |
| A container Docker could not start gave a 40-line traceback. | One-line error with the rebuild command (`SandboxError`). |
| Such a container was left behind in Docker as *Created* (3 found after the C# rebuild). | Each sandbox container has a unique name; one that fails to start is removed. |
| On a Linux host whose user is not uid 1000, the sandbox user (uid 1000) could not write the project copy: every Rust and Java file failed (simulated: Rust `Permission denied`, Java exit 1). | The copy is opened to the sandbox user (it sits in a 0700 temp folder), and what the sandbox created is handed back before the container is removed. All 8 languages pass in that simulation. |
| `bootstrap.sh` would have stopped on a Mac: no `sha256sum`, and bash 3.2 rejects an empty `"${from[@]}"` under `set -u` (reproduced in a `bash:3.2` container). | `shasum -a 256` when `sha256sum` is missing (same label); the bash-3.2-safe form; on macOS a warning that it is untested, with the report link; on Windows outside WSL2 a clear stop. |
| `--to go` never verified a `package main` translation in `src/` (most Python projects): `go build ./...` wrote the program to a file named `src`, where the folder is (`build output "src" already exists`). Found when the demo's `examples/billing` failed 5 tries. The evaluation suites use `go build -o prog .` and are not affected. | `go build -o /dev/null ./...` (compiles everything, keeps nothing). The example now passes on the first try (17 s); sandbox tests for main packages in `src/`, the root and a subfolder. |
| `bootstrap.sh` kept existing images after a Dockerfile changed. | Images carry a hash of their folder and are rebuilt when it changes. |
| Dependencies had no upper bounds; a fresh install already got newer versions than the tested ones. | Upper bounds at the next breaking release. The suite passes on Python 3.11 with the newest allowed versions (typer 0.29, langchain-core 1.8, sentence-transformers 6.3). |
| README was a development log; `--help` started with maintainer commands. | README is a user guide. `rebirth` alone opens a start screen (`docs/assets/start-screen.png`, `scripts/render_start_screen.py`). `--help` is grouped (Migrate, Setup, Tools, Maintainers). `--version` added. |
| `.env.example` set a relative index path and listed unused variables (`GITHUB_TOKEN`, `REBIRTH_CLASSIFIER_DIR`). | Rewritten: every variable the code reads, with its default, all commented out. |
| CI used Node 20 actions (deprecated). | `actions/checkout@v7`, `actions/setup-python@v7`. |
| GitHub: no private vulnerability reporting (SECURITY.md points there), no Dependabot alerts, no topics. | All three enabled. |

**Regression on the finished code (`22d730c`, new images):**
- Fixtures: 8 / 8 with the served models, plus python→go (iteration 2).
- Phase 7 benches: 6 / 6.
- Real repos with the shipped default (`data/reports/real_repos_fallback.json`):
  - 14 files: **8 changed with the tests passing** (4 by the 7B alone, 4 by the 14B fallback), **6 unchanged,
    0 failed**;
  - before this fix: 8 / 1 / 5. The 5 files whose fallback found nothing (kind-of, go-homedir, 3 go-humanize
    files) are now reported unchanged and take 45 % less time;
  - the 14 files take 20.7 min in all (was 29.3);
  - the Go files ran on the new Go 1.27 sandbox.
- Tests: 327 passed (Python 3.12 and 3.11); ruff clean.

## What shipped / how to run / samples

**Version 1.0.0** (2026-11-11).

- **Models on Hugging Face** (user `belebindahugo`, published 2026-11-11 by `scripts/publish_hf.py`):
  - [`rebirth-modernize-7b-lora-gguf`](https://huggingface.co/belebindahugo/rebirth-modernize-7b-lora-gguf): the
    8 LoRA adapters (10 MB each) and the card;
  - [`rebirth-xlang-14b-gguf`](https://huggingface.co/belebindahugo/rebirth-xlang-14b-gguf): the 14B Q6_K
    (12.1 GB), a ready Ollama `Modelfile` and the card.
  - Checked without a login: every file's size and sha256 match `configs/models.json`. The installer's own
    download code fetched the 8 adapters (sha256 OK, 24 s), and the 14B answers a range request (`GGUF`, right
    size, so downloads resume).
- **How to run:** `git clone … && bash scripts/bootstrap.sh`, then `rebirth` (README § Install, Quick start).
- **Sample:** [`examples/billing`](../../examples/billing), an old-style invoice module with 4 tests.
  - The README demo (`docs/assets/demo.gif`, recorded by `scripts/record_demo.py`) is a real run on it.
  - The 7B modernizes it on the first try in 12 s, with the tests passing.
  - `--to go` compiles on the first try in 17 s (after the Go `src/` fix below).

### Found while preparing the release (all fixed on `rebirth/v1.0.0`)

| Finding | Fix |
|---|---|
| `--to go` never verified a `package main` translation in `src/` (most Python projects). `go build ./...` wrote the program to a file named `src`, where the folder is. Found when the demo's example failed 5 tries. | `go build -o /dev/null ./...`. Sandbox tests cover main packages in `src/`, the root and a subfolder. The evaluation suites use `go build -o prog .` and are not affected. |
| On a Linux host whose user is not uid 1000, Rust and Java always failed: the sandbox user could not write the project copy. | The copy is opened to the sandbox user, and its files are handed back before the container is removed. Simulated on all 8 languages. |
| A container Docker created but could not start was left behind as *Created*. | Unique container names; a container that fails to start is removed. |
| `REBIRTH_VECTOR_DB_PATH` was documented, but the CLI and the graph ignored it. | One default (`paths.DEFAULT_DB`) that honours it. |
| A reasoning model's `<think>` block reached the code reader on the Ollama path. A mistyped `REBIRTH_XLANG_MODEL` silently used the plain 14B. | `<think>` is stripped; the typo is reported. Checked end to end with a model Rebirth doesn't ship. |
| `bootstrap.sh` would have stopped on a Mac (`sha256sum`, bash 3.2 with `set -u`). | Fixed. macOS gets an "untested" warning; Windows outside WSL2 a clear stop. |
| Printed paths showed the full home folder (the user name in screenshots). | `~` in printed paths and in the start screen's command. |
| *Outside review (read-only):* `python -m rebirth_ai.cli` crashed on a passing migrate (`NameError: _print_diff`) and lost the grouped `--help`: the `__main__` block sat above the last definitions. | The block is the last statement (an AST test keeps it there). The installed `rebirth` was never affected. |
| *Review:* `rebirth parse` without `--lang` only knew `.py`; `.go`, `.rs`, `.swift`, `.cpp` were rejected. | The language comes from the registry's suffix map, like everywhere else. |
| *Review:* `rebirth generate` sent every non-TypeScript file to the JS/TS model with a JavaScript prompt (a `.py` file went to `rebirth-js-ts:7b`). | The file's own language and model; its code search parses with that language too. |
| *Review:* `--out` and the `--to` output were written whatever the result: an empty file when the model returned no code (reproduced), and for same-language runs a draft that failed the tests. | Never an empty file. Same-language `--out` only keeps a draft that passed; an unverified translation is still written, marked "not verified". |
| *Review:* `mine-dataset` and `eval` need `training/`, which a plain `pip install` doesn't copy. | By design Rebirth runs from a clone (configs, sandboxes, data), so any command outside one now says how to install. |
| *Same kind, found while checking:* `eval` defaults were relative to the current folder (`data/reports/…`, `fixtures/eval_suite/…`). | Defaults in the Rebirth folder, your own paths resolved from your folder, scripts run from the clone. |
| *Same kind:* the code-search retriever, without an adapter, parsed every non-`.py` file as JS/TS. | The registry's language, else the file's text as the query. |
| *Same kind:* `migrate-repo`'s dependency order parsed Go, Java, Rust… files with the JS parser; the README promised dependency order for all languages. | Import edges for JS/TS and Python only (their imports name files); the others keep path order. The README says so. |
