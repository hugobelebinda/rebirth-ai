# Contributing to Rebirth AI

Thanks for helping. One person maintains Rebirth, so small pull requests that do one thing are the easiest to
review and merge.

## Most wanted

1. **Hardware reports.** Rebirth has only been tested on Linux with an AMD GPU. If you run it on an NVIDIA GPU,
   an 8 GB GPU, macOS or Windows (WSL2), open a
   [hardware report](https://github.com/hugobelebinda/rebirth-ai/issues/new?template=hardware_report.md). It
   asks for the output of `rebirth doctor` and of one run on a sample project. A report that says "it works"
   is just as useful as one that says it doesn't.
2. **Real projects for the benchmark.**
   - `training/bench_real_repos.py` runs Rebirth on pinned commits of small, permissively licensed libraries,
     judged by their own tests.
   - Each entry gives a name, the URL, a commit, the license and the files to migrate.
   - Java, C#, Rust, C++ and Swift have no entries yet.
3. **A new language.** See [Add a language](#add-a-language).
4. **Third-party packages in translations.** `--to` only uses the target language's standard library. Mapping
   common packages (for example `requests` to Go's `net/http`) is open.
5. **Bugs,** with the command you ran and its output. `rebirth report <file>` prints a run report with secrets
   redacted.

## Set up

```bash
git clone https://github.com/hugobelebinda/rebirth-ai && cd rebirth-ai
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q        # no GPU, Docker or models needed; the Docker tests are skipped without Docker
ruff check .
```

For full runs, with the models and the sandboxes, use `bash scripts/bootstrap.sh` instead (see the
[README](README.md#install)).

## Add a language

1. **The adapter:** `src/rebirth_ai/adapters/<language>.py`.
   - It implements `LanguageAdapter` (`adapters/base.py`): which files are source files, how to parse them, the
     test command, and the sandbox it runs in.
   - `adapters/rust.py` is a good example to copy.
2. **The registry:** `adapters/registry.py`. Add the adapter, the files that mark a project in that language
   (like `Cargo.toml`) and the file suffixes.
3. **The sandbox:** `docker/<language>_sandbox/Dockerfile`, with:
   - the toolchain;
   - a non-root user `rebirth` with uid 1000;
   - `/workspace` as the working directory.
4. **A sample project:** `fixtures/<language>_sample/`, a small project with an old-style file and tests that
   pass.
5. **Tests,** next to the existing ones (`tests/test_sandbox.py`, `tests/test_discovery.py`).
6. **For `--to`:** an entry in `src/rebirth_ai/xlang/languages.py`.

A new language has no fine-tuned adapter, so it uses the plain model in Ollama. That works; an adapter can come
later.

## Pull requests

- One change per pull request, with a test that fails without it.
- `pytest -q` and `ruff check .` pass. CI runs both on every pull request.
- Don't commit datasets, model files, run logs or anything secret. Data and models are versioned outside git.
- A change to a model or a prompt needs before-and-after numbers on the same evaluation. The reports in
  [docs/phases/](docs/phases/) show how each one was measured.

## Security

Report vulnerabilities privately, not in a public issue: [docs/SECURITY.md](docs/SECURITY.md).

## License

By contributing, you agree that your contribution is licensed under the MIT License, like the rest of the code
([LICENSE](LICENSE)).
