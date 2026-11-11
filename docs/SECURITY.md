# Security

Rebirth runs code that a language model wrote. This page says what it trusts, how it confines the rest, and
what it does not protect against.

## Trust model

| Trusted | Not trusted |
|---|---|
| You, your machine, and the project you point Rebirth at (its code, tests, build files and declared dependencies) | Everything the model writes: drafts, translations, and any package name that only a draft asks for |

Model output is only ever executed inside a sandbox container, and is only written into your project when you ask
(`--write`, `--open-pr`).

## The sandbox

Every test run (`rebirth migrate`, `rebirth eval`, the translation checks) gets a fresh container from one of
the 8 `rebirth-ai/<lang>-sandbox` images, which is deleted afterwards. The runner is
[`src/rebirth_ai/sandbox/runner.py`](../src/rebirth_ai/sandbox/runner.py).

| Measure | Setting | Override |
|---|---|---|
| Your project is copied, never mounted | the container works on a temporary copy; `.git`, `node_modules`, virtualenvs and build outputs are left out | — |
| Non-root user | each image's own user (uid 1000) | — |
| No Linux capabilities | `cap_drop: ALL` | — |
| No privilege escalation | `no-new-privileges` (setuid binaries cannot raise privileges) | — |
| Memory | 4 GB, no swap on top | `REBIRTH_SANDBOX_MEMORY` (e.g. `2g`) |
| Processes | 1,024 (fork bombs stop there) | `REBIRTH_SANDBOX_PIDS` |
| Time | `timeout` inside the container (120–300 s per test run depending on the language; the host kills the container 30 s later if it is still running) | `REBIRTH_SANDBOX_TIMEOUT_SEC` |
| System calls | Docker's default seccomp profile (and AppArmor `docker-default` where the host has it) | — |
| Network | none for model-written code (below) | — |

### Network

A run has up to two steps in the same container:

1. **Setup** (only when the project has dependencies): runs on your project *as it is, before the model's
   draft is written in*, with network, to download what the project declares.
2. **Test**: the container is disconnected from every network first. Rebirth checks this and refuses to run
   the tests if the container is still attached. Then the draft is written in and the tests run offline.

So **model-written code never runs with network access**, in any language. Translation compile checks and the
eval suites have no setup step at all; `--to` with `--target-project` runs that project's setup step first, as
above.

| Language | Setup step, when the project has dependencies | Test step (offline) |
|---|---|---|
| JS / TS | `npm ci` (with a lockfile) or `npm install`, for the packages in `package.json` | `npm test` / `node --test` |
| Python | `pip install --user` of `requirements*.txt` and the `pyproject.toml` dependencies (not the project itself) | `pytest` |
| Java | **your project's own tests, once**, online (`mvn test`): Maven only downloads its test-runner plugin when tests actually run | `mvn -o test` |
| C# | `dotnet restore` | `dotnet test --no-restore` |
| Go | `go mod download` | `go test` |
| Rust | `cargo fetch` | `cargo test` |
| Swift | `swift package resolve` | `swift test` |
| C++ | — | the project's CMake build and tests |

The setup step runs your own build tooling with network: npm lifecycle scripts of your declared dependencies,
Maven plugins, and for Java your original tests. That is the same trust as running them yourself, and none of
it includes the model's code.

### Packages a draft asks for

When the tests report a missing module (JS/TS, Python), Rebirth installs it in the next run's setup step only
if:

- your project declares it (`package.json`, `requirements*.txt`, `pyproject.toml`), or
- your original file already imports it.

A package that only the model's draft imports is refused, because model-invented names can be typo-squatted
on npm and PyPI. The model is asked for a new draft instead. Names must be valid npm / PyPI names and are
shell-quoted. Python import names are mapped to their real package (`yaml` → `PyYAML`, `cv2` →
`opencv-python`, …) so the wrong package is never fetched. `REBIRTH_ALLOW_UNDECLARED_PACKAGES=1` lifts the
restriction; only use it on a machine you can throw away.

## Your data

- **Nothing leaves your machine** while Rebirth runs. The models run in Ollama on `127.0.0.1:11434`
  (`OLLAMA_HOST` changes that). The code-search model runs in-process and loads from the local cache without
  contacting Hugging Face. Rebirth sends no telemetry, and the C# sandbox opts out of .NET's.
- **Network use**: the model download at install time (Hugging Face; every file is checked against the sha256
  in [`configs/models.json`](../configs/models.json) before Ollama sees it), the code-search model's
  one-time download (bootstrap does it; otherwise on first use), the sandbox setup step above, and
  `--open-pr`.
- **Run logs** (`data/runs/`, `rebirth dashboard`) stay local, and tokens and keys are redacted before they
  are written ([`redact.py`](../src/rebirth_ai/redact.py)).

## Writing to your project

- Without `--write`, nothing in your project changes: drafts are shown and kept in the run report, and
  translations (`--to`) are written to a new `rebirth_out/<language>/` folder.
- `--write` replaces only the files whose tests passed in the sandbox.
- `--open-pr` creates a branch, commits only the files that passed, and opens a pull request (`--draft` for a
  draft) with your own `gh` login (`gh auth login`; Rebirth never sees a token). Secret-looking paths (`.env*`, `*.pem`,
  `*.key`, `credentials*`, `secrets*`, SSH keys) are never committed
  ([`safety.py`](../src/rebirth_ai/safety.py)). Review the pull request before merging it, like any other.

## Limits

- **Docker access is root-equivalent.** Being in the `docker` group lets any process of your user control
  the Docker daemon. This is how Docker works, not something Rebirth adds.
- The container's root filesystem is writable (compilers write caches into it). It is deleted after every
  run.
- CPU is not capped: a busy loop runs until the time limit.
- Tests pass ≠ correct. Rebirth keeps a draft when your tests pass, so weak tests let wrong code through.
  Review what you merge.
- A passing translation (`--to`) only means it compiled, or passed the target tests you gave
  (`--target-test`).
- Projects whose tests need network (live APIs, services started over the network) fail in the test step.
  Use test doubles, or run those tests yourself.

## Reporting a vulnerability

Use **Security → Report a vulnerability** on the GitHub repository (private), not a public issue. Please
include the version (`rebirth version`), the command, and a minimal reproduction.
