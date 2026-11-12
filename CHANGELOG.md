# Changelog

All notable changes to Rebirth AI. Versions follow [Semantic Versioning](https://semver.org).

## [1.0.0] — unreleased

The first release.

### Modernize and translate
- `rebirth migrate <project> --file <path>`: modernize one file in its own language.
- `rebirth migrate-repo <project>`:
  - modernizes every source file, in dependency order;
  - `--dry-run` shows the plan, `--write` keeps the changes that pass, `--open-pr` opens a GitHub pull
    request.
- `--to <language>`: translates a file between any two of the 8 languages (56 directions). The translation is
  a new file under `<project>/rebirth_out/<language>/`.
- **Every draft is checked by the project's tests** in a Docker sandbox and retried with the errors, up to 5
  times. A draft that fails the tests is never written.
- **Supported languages:** JavaScript / TypeScript, Python, Go, Java, C#, Rust, C++, Swift.

### Models (local, in Ollama)
- **modernize pack:** Qwen2.5-Coder 7B with one LoRA adapter per language (`rebirth-<lang>:7b`). On 749
  held-out real commits they match the human change 107 times, against 42 for the base model.
- **migrate pack:** Qwen2.5-Coder 14B fine-tuned for translation (`rebirth-xlang:14b`, Q6_K). Verified
  whole-program translations go from 551 to 667 of 931, and HumanEval from 91.1 % to 92.6 %.
- **Fallback:** when the 7B passes the tests but changes nothing, a 14B tries once more (2 tries).
  `--no-fallback` turns it off.
- **Any Ollama model** can take any of the three steps:
  - `--backend ollama` with `OLLAMA_MODEL` for modernizing;
  - `REBIRTH_FALLBACK_MODEL` for the fallback;
  - `REBIRTH_XLANG_MODEL` for `--to`.

  Reasoning models work too: their thinking is dropped before the code is read.

### Using it
- `rebirth` with no command opens a guided start screen. It checks your setup and builds the command for
  you.
- `scripts/bootstrap.sh` installs everything: the Python environment, 8 sandbox images, both model packs
  (sha256-checked) and `rebirth doctor`. Offline install: `--from <folder>`.
- `rebirth doctor` checks Docker, Ollama, the models, the sandbox images, the GPU and disk space.
- A passing change is shown as a diff. Results are `success`, `unchanged` or `human_review`, with clear
  exit codes.

### Safety
- Model-written code never runs with network access. The sandbox is a copy of your project, run as a
  non-root user with no Linux capabilities and with memory, process and time limits.
- Only packages your project declares are installed. A package that only the model asks for is refused.
- Files outside the project, secrets files and binaries are refused. `--write` keeps line endings and
  encoding. `--open-pr` commits only the files that passed, on a new branch.
- Nothing is sent anywhere while Rebirth runs. Run logs stay local, with secrets redacted.

### Sandbox toolchains
Node 22, Python 3.12, Go 1.27, JDK 21 with Maven 3.9, .NET 10 with .NET 8, Rust 1.99, GCC 14 with CMake,
Swift 6.4.

[1.0.0]: https://github.com/hugobelebinda/rebirth-ai/releases/tag/v1.0.0
