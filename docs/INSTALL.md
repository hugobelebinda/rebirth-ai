# Installing Rebirth AI

Rebirth runs entirely on your machine: the models run in [Ollama](https://ollama.com), the code it writes is
tested in Docker sandboxes with no network, and nothing is sent anywhere.

## What you need

| | Minimum | Notes |
|---|---|---|
| OS | Linux x86-64 | Tested on Ubuntu 24.04. macOS / Windows (WSL2) are untested. |
| Python | 3.11+ with `venv` | `sudo apt install python3 python3-venv` |
| Docker | Engine 24+ | Your user must be in the `docker` group. The 8 sandbox images take ~8 GB. |
| Ollama | 0.30+ | Must be running (`ollama list` works). |
| GPU | see the packs below | NVIDIA (CUDA) or AMD (ROCm, see [OLLAMA_AMD_GPU.md](OLLAMA_AMD_GPU.md)). Without a GPU Ollama runs on the CPU, which is very slow. |
| Disk | ~30 GB free | Models ~17 GB, sandbox images ~8 GB, Python environment ~3 GB. |
| Internet | for the install only | Afterwards everything works offline. |

### The two model packs

Rebirth uses two models (see [ADR-001](research/ADR-001_UNIFIED_VS_SPECIALISTS.md) for why):

| Pack | What it does | Download | GPU memory |
|---|---|---:|---:|
| **modernize** | Rewrites old code into modern code in the same language (`rebirth migrate … --lang go`). One 7B base model + 8 small language adapters. | ~4.8 GB | 8 GB |
| **migrate** | Translates code between any two of the 8 languages (`rebirth migrate … --to rust`). One 14B model. | ~12.1 GB | 16 GB |

Both are installed by default. Switching between the two in one session costs a 10–20 s model reload.

## Install

```bash
git clone https://github.com/hugobelebinda/rebirth-ai
cd rebirth-ai
bash scripts/bootstrap.sh                    # both packs
# or: bash scripts/bootstrap.sh --pack modernize   (8 GB GPU)
```

`bootstrap.sh` never uses `sudo`. It:

1. checks for git, Python ≥ 3.11, Docker (daemon reachable) and Ollama (server reachable), and tells you
   what to install if one is missing;
2. creates `.venv` and installs Rebirth with a CPU-only PyTorch (only the small code-search model runs in
   Python; the LLMs run in Ollama);
3. downloads the code-search embedding model (~330 MB) so later runs work offline;
4. builds the 8 sandbox images (`rebirth-ai/<lang>-sandbox`);
5. runs `rebirth install`: downloads the model files from Hugging Face, checks each one against the
   sha256 in [`configs/models.json`](../configs/models.json), and registers them in Ollama;
6. runs `rebirth doctor --strict`.

Re-running it is safe; finished steps are skipped. An interrupted download resumes where it stopped.

### Offline or mirrored install

Copy the release files (the `file` names in `configs/models.json`) to a folder or USB stick, then:

```bash
bash scripts/bootstrap.sh --from /media/usb/rebirth-models
```

The files are still checked against their sha256 before Ollama sees them.

## Check the install

```bash
.venv/bin/rebirth doctor
```

A healthy machine ends with `all checks passed`. It reports Python, Docker, Ollama, the GPU and its memory,
free disk, each model pack (`8/8`, `1/1`) and each sandbox image.

## First run

```bash
source .venv/bin/activate
# modernize one file; the draft is kept only if the project's tests pass (--write puts it back in place)
rebirth migrate fixtures/python_sample --file src/math_ops.py --test tests/test_math_ops.py --lang python
# translate it to Go: a new file under fixtures/python_sample/rebirth_out/go/, compile-checked
# (add --target-project/--target-test to check it against Go tests instead)
rebirth migrate fixtures/python_sample --file src/math_ops.py --lang python --to go
# your own code: rebirth migrate <project dir> --file <path in it> --test <its test file> [--to <lang>] [--write]
# health of your runs (local only)
rebirth dashboard
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Docker daemon not reachable` | Start Docker; `sudo usermod -aG docker $USER`, then log out and in. |
| `Ollama server not reachable` | `ollama serve`, or `sudo systemctl start ollama`. |
| `pack migrate incomplete` / model missing | `.venv/bin/rebirth install --pack migrate` |
| `sha256 … ≠ expected` | The file is damaged or not the release file: delete it and run the install again. |
| Translations are very slow | The 14B does not fit your GPU and runs partly on the CPU: `ollama ps` shows the CPU / GPU split. Use the modernize pack only, or a 16 GB GPU. |
| GPU memory errors when using `--backend peft` | Unload Ollama first (`ollama stop <model>`); see `rebirth doctor`'s VRAM warning. |
| Swift sandbox build is slow / large | It is the biggest image (~2.6 GB). `--skip-images` skips all image builds; build one later with `docker build -t rebirth-ai/swift-sandbox:latest docker/swift_sandbox`. |
| AMD GPU not used by Ollama | [OLLAMA_AMD_GPU.md](OLLAMA_AMD_GPU.md) |

## Uninstall

```bash
for m in $(ollama list | awk '/^rebirth-/{print $1}'); do ollama rm "$m"; done
ollama rm qwen2.5-coder:7b-instruct          # the modernize pack's base, if nothing else uses it
docker rmi $(docker images 'rebirth-ai/*' -q)
rm -rf rebirth-ai                            # the clone, its .venv and your local run logs (data/runs)
```
