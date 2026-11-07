# Rust / C++ training-data sources (Phase 12 note → Phase 15/17)

> Written 2026-11-06. Status: research only — nothing downloaded or installed yet.

## Where we are

| Lang | Rows (expanded) | Source | Real "modernization" rows | Frozen eval rows |
|------|----------------:|--------|--------------------------:|-----------------:|
| Rust | 2,001 | all of CommitPackFT `rust` (2,996 scanned) | **24** (keyword match) | 56 |
| C++  | 2,906 | all of CommitPackFT `c++` (4,992 scanned) | **20** | 93 |

CommitPackFT is exhausted for both. Almost all rows are *generic* instructed edits, not
legacy→modern rewrites. Go and Python showed that more generic edits stop helping at 7B
(Go 1.4K→2.95K: +1 soft-exact; Python 5K→12K: −1). So the useful data is **(a) much more of it
for the 14B run** and **(b) real modernization pairs**, not just more generic commits.

## Candidates, best first

### 1. Tool-generated, compiler-checked pairs (highest value, unlimited)

Run a deterministic modernizer over permissively licensed old code; the legacy file and the
tool's output form a pair, and the objective is the check's description. Every pair is
correct by construction and is *exactly* the modernization task.

| Lang | Tool | Fixes that apply automatically (examples) |
|------|------|-------------------------------------------|
| C++  | `clang-tidy -fix` with `modernize-*` | `use-nullptr`, `use-auto`, `loop-convert`, `use-override`, `make-unique`/`make-shared`, `use-using`, `use-emplace`, `use-equals-default`/`-delete`, `use-std-print`, `use-ranges`, `concat-nested-namespaces`, `return-braced-init-list` |
| Rust | `cargo fix --edition` (2015→2018→2021→2024) + `cargo clippy --fix` | `try!`→`?`, bare trait objects → `dyn`, `extern crate` removal, closure captures, `let`-`else`, clippy machine-applicable lints (`manual_*`, `needless_*`, `redundant_*`, `map_unwrap_or`, …) |

- Input code: pre-2018 crates from crates.io (MIT/Apache in `Cargo.toml`), and old C++ file
  versions (`old_contents`) from CommitPack with permissive licenses.
- Cost: local CPU only. Needs `clang-tidy` (distro `clang-tools` package) and `rustup`
  (stable + old toolchains), and neither is installed yet. C++ needs a compile database per
  project, so start with single-file snippets that compile on their own.
- Risk: the model only learns what the tools know. Mix it with real commits, cap each check's
  share, and keep the frozen eval sets unchanged so we can see the effect.
- The same idea works for the other six languages (`pyupgrade`/`ruff --fix`, `go fix` + gopls
  `modernize`, OpenRewrite (Java), Roslyn analyzers (C#), `lebab` (JS), Swift migrator). That
  is the cheapest way to get *real* modernization pairs everywhere.

### 2. Full CommitPack (≈100× the FT subset)

[`bigcode/commitpack`](https://huggingface.co/datasets/bigcode/commitpack): MIT dataset card,
per-row repo license (filter to permissive, as the miner already does).

| Lang | CommitPack (full) | CommitPackFT (used) |
|------|------------------:|--------------------:|
| Rust | 296,214 commits, ~15 GB | 2,996 |
| C++  | 2,402,294 commits, ~187 GB | 4,992 |
| Swift (for later) | 319,289 commits, ~5.9 GB | — |

- The miner already reads the same schema (`old_contents`/`new_contents`/`subject`/`license`),
  so it mostly needs a new `--source commitpack` flag and **streaming** (C++ is larger than the
  67 GB of free disk). Process one shard at a time and keep only the rows that pass.
- The full set is *not* instruction-filtered, so apply a commit-message quality filter
  (imperative subject, single file, small diff) plus our parse, license and size filters.
  Expected yield is unknown; a dry run on one shard will tell. Even 5 % would be ~15K Rust
  and ~120K C++ rows, and the modernization-keyword matches alone should be roughly 2K Rust and
  10K C++.
- Download: Rust ≈ 15 GB (fits); C++ shards streamed. CPU-only; can run while the GPU trains.

### 3. Other public sets (lower priority)

- [JetBrains `commit-chronicle`](https://huggingface.co/datasets/JetBrains-Research/commit-chronicle)
  has ~10M commits in many languages. The license is "other" (check terms before use), and it
  stores diffs rather than full files, so pairs would need reconstruction.
- The Stack v1 dedup (gated, permissive only) gives *modern* Rust/C++ files for
  back-translation (rule-based "de-modernize": `?`→`try!`, `nullptr`→`NULL`, `using`→`typedef`,
  range-for → index loops). Unlimited, but synthetic in the other direction.

## Recommendation

1. **Now (CPU, no cost):** add `--source commitpack --stream` to the miner and dry-run one Rust
   shard to measure the yield. If it's good, mine Rust fully and C++ by streaming.
2. **Before Phase 15 (14B):** build the tool-pair generator for C++ (clang-tidy) and Rust
   (cargo fix/clippy). It needs your OK to install `clang-tools` and `rustup`.
3. Grow the Rust and C++ **eval sets** from the new data: 56 and 93 rows are thin. Freeze new
   v2 sets and keep the old ones for continuity.
