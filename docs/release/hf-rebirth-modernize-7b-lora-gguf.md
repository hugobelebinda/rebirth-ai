---
license: apache-2.0
base_model: Qwen/Qwen2.5-Coder-7B-Instruct
library_name: gguf
tags:
  - code
  - lora
  - gguf
  - ollama
  - code-modernization
datasets:
  - bigcode/commitpackft
  - bigcode/commitpack
language:
  - code
---

# Rebirth AI — modernize pack (8 LoRA adapters for Qwen2.5-Coder-7B)

> Draft model card for the Hugging Face repo `<namespace>/rebirth-modernize-7b-lora-gguf` (Phase 19). Review
> before publishing.

Eight small LoRA adapters (GGUF, f16, ~10 MB each) that turn **Qwen2.5-Coder-7B-Instruct** into a
same-language code modernizer: given a legacy file and an objective, it rewrites the code in modern idioms
of the same language. They are the "modernize" pack of [Rebirth AI](https://github.com/hugobelebinda/rebirth-ai),
an offline agent that only keeps a rewrite if the project's own tests pass in a Docker sandbox.

Languages: JavaScript/TypeScript, Python, Go, Java, C#, Rust, C++, Swift.

## Use

The normal way is the installer, which downloads these files, checks their sha256 and registers them in Ollama:

```bash
git clone https://github.com/hugobelebinda/rebirth-ai && cd rebirth-ai
bash scripts/bootstrap.sh --pack modernize
```

By hand, for one language (the adapters expect the raw prompt format below, not a chat template):

```text
FROM qwen2.5-coder:7b-instruct
ADAPTER ./rebirth-go-7b-v1.lora.f16.gguf
TEMPLATE """{{ .Prompt }}"""
PARAMETER temperature 0
PARAMETER num_ctx 4096
PARAMETER stop "
###"
```

Prompt: `### Objective\n<what to change>\n### Legacy\n<code>\n### Modern\n`.

## Files

| File | Language | sha256 |
|---|---|---|
| `rebirth-js-ts-7b-v1.lora.f16.gguf` | JS/TS | `76e05a5f…` |
| `rebirth-python-7b-v1.lora.f16.gguf` | Python | `20c567b1…` |
| `rebirth-go-7b-v1.lora.f16.gguf` | Go | `5f932146…` |
| `rebirth-java-7b-v1.lora.f16.gguf` | Java | `604723a6…` |
| `rebirth-csharp-7b-v1.lora.f16.gguf` | C# | `cf657b28…` |
| `rebirth-rust-7b-v1.lora.f16.gguf` | Rust | `61deb5a2…` |
| `rebirth-cpp-7b-v1.lora.f16.gguf` | C++ | `ca80b439…` |
| `rebirth-swift-7b-v1.lora.f16.gguf` | Swift | `33c9a6a3…` |

Full hashes: `configs/models.json` in the GitHub repo.

## Training

- LoRA r 8 / α 16 on the attention projections, QLoRA 4-bit, sequence length 512, completion-only loss, 1 epoch,
  on an AMD RX 7800 XT; 1,187 (Rust) to 8,308 (JS/TS) training rows per language, 22,372 in total.
- Data: the training split of 24,640 single-file before/after commit pairs mined from
  [CommitPackFT](https://huggingface.co/datasets/bigcode/commitpackft) and, for Rust and C++,
  [CommitPack](https://huggingface.co/datasets/bigcode/commitpack); **permissively licensed repositories only**,
  filtered for size, parseability (Tree-sitter) and commit quality. Details: `docs/DATASET.md`.

## Evaluation

Frozen held-out sets of real commits (not used in training), served exactly as shipped (Q4_K_M base + this
adapter in Ollama). **Soft-exact** = the output matches the human commit after normalizing whitespace; it
undercounts, since other valid rewrites score as misses.

| Language | Base 7B | **This adapter** | Rows |
|---|---:|---:|---:|
| JS/TS | 0 | **9** | 100 |
| Python | 6 | **18** | 100 |
| Go | 5 | **13** | 100 |
| Java | 10 | **20** | 100 |
| C# | 2 | **11** | 100 |
| Rust | 5 | **7** | 56 |
| C++ | 8 | **16** | 93 |
| Swift | 6 | **13** | 100 |
| **All** | **42** | **107** | 749 |

A Qwen2.5-Coder-14B with the same kind of adapters only tied these 7B adapters (Go 15 vs 14, Python 19 vs 20
per 100), so the 7B is the shipped model for this job.

## Limitations

- Trained on single-file commits of ≤ 512 tokens; long files are handled by Rebirth's chat prompt path.
- Same-language modernization only; for translation between languages use the migrate pack
  (`rebirth-xlang-14b-gguf`).
- Output is a suggestion: Rebirth only keeps it when the project's tests pass. Used on its own, review it.

## License

Adapters: Apache-2.0 (same as the base model). Training data: permissively licensed repositories from
CommitPack / CommitPackFT (BigCode). Base model: [Qwen2.5-Coder-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct),
Apache-2.0.
