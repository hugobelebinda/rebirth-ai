---
license: apache-2.0
base_model: Qwen/Qwen2.5-Coder-14B-Instruct
base_model_relation: finetune
library_name: gguf
tags:
  - code
  - gguf
  - ollama
  - code-translation
  - lora-merged
datasets:
  - deepmind/code_contests
  - bigcode/commitpackft
  - bigcode/commitpack
language:
  - code
---

# Rebirth AI — migrate pack (Qwen2.5-Coder-14B for code translation, Q6_K)

**Qwen2.5-Coder-14B-Instruct** fine-tuned to translate code between 8 languages (Python, JavaScript /
TypeScript, Go, Java, C#, Rust, C++, Swift: all 56 directions), with the LoRA merged into the weights and
quantized once to Q6_K (one 12.1 GB GGUF). It is the "migrate" pack of
[Rebirth AI](https://github.com/hugobelebinda/rebirth-ai), an offline agent that only keeps a translation if
it compiles, or passes the target project's tests, in a Docker sandbox, and retries with the errors otherwise.

## Use

The normal way is the installer, which downloads the file, checks its sha256 and registers it in Ollama as
`rebirth-xlang:14b`:

```bash
git clone https://github.com/hugobelebinda/rebirth-ai && cd rebirth-ai
bash scripts/bootstrap.sh --pack migrate
source .venv/bin/activate
rebirth migrate path/to/project --file src/app.py --to go
```

By hand: download the GGUF and the `Modelfile` from this repo into one folder (the Modelfile carries
Qwen2.5-Coder's chat template, which the model was evaluated with), then
`ollama create rebirth-xlang:14b -f Modelfile`. Rebirth sends this system prompt:

```text
You are Rebirth AI, a precise code translation engine.
Translate source code from one programming language to another.
Rules:
- Keep the behavior identical: same functions and types, same results for the same inputs.
- Write idiomatic, modern code in the target language using only its standard library.
- Output ONLY the target-language code in a single fenced code block — no explanations.
```

and a user message starting `Translate this <Source> file to <Target>.` followed by the code (the full prompt
is `build_translation_prompt` in `src/rebirth_ai/xlang/prompts.py`).

## Files

| File | Size | sha256 |
|---|---:|---|
| `rebirth-xlang-14b-v2.Q6_K.gguf` | 12,124,684,384 bytes | `bfa04408e2156a51e56b9b1c8d3a41418c9eca524498c2d97adcc21295a45883` |
| `Modelfile` | | for `ollama create` |

## Training

- **Data:** a mix of 14,000 rows, of which the 13,616 of at most 2,048 tokens were trained on:
  - 12,000 translation pairs over 63 directions, at most 190 per direction;
  - 2,000 same-language rows from the Rebirth corpus (commits from permissively licensed repositories in
    [CommitPackFT](https://huggingface.co/datasets/bigcode/commitpackft) and
    [CommitPack](https://huggingface.co/datasets/bigcode/commitpack)), so file-level code is not forgotten.
- **Where the pairs come from:**
  1. Programs from the [CodeContests](https://huggingface.co/datasets/deepmind/code_contests) **train** split,
     each with a Python 3, C++ or Java reference solution that passes 10 of the problem's tests.
  2. Qwen2.5-Coder-14B-Instruct translated each one into the other languages, with up to 2 retries.
  3. Only translations that pass the same tests were kept: 9,387 translated, 50.1 % verified.
  4. Every two verified versions of one program form a pair, so the human-written reference is also used
     as a target. This gave 30,211 pairs, from which the 12,000 above were sampled.
- **Leak guard:** problems whose title or statement matches one in the evaluation suite were dropped.
- **LoRA:** r 16 / α 32 on all linear layers, bf16, 2 epochs (3,404 steps, 4 h 38 min on one L40S).
- **Merge:** merged into the bf16 weights shard by shard in fp32 (`training/merge_lora.py`, checked against
  PEFT's `merge_and_unload()`), converted with llama.cpp (`6c01da6`) and quantized once to Q6_K.

## Evaluation

All runs are local, served exactly as shipped (Ollama, Q6_K), on the same translations for every model.

- **Programs:** whole Codeforces programs (20–150 lines, stdin → stdout) from the CodeContests **valid +
  test** splits, which training never saw. They are judged by 10 tests each, like an online judge, with at
  most 2 retries on errors.
- **HumanEval:** functions, judged by the MultiPL-E / HumanEval tests, first try only.

| Suite | Translations | Plain Qwen2.5-Coder-14B Q6_K | **This model** |
|---|---:|---:|---:|
| Programs (56 directions) | 931 | 551 (59.2 %) | **667 (71.6 %)** |
| HumanEval (56 directions) | 2,229 | 2,030 (91.1 %) | **2,065 (92.6 %)** |

The same adapter applied at run time, instead of merged, scores 672 and 2,064: the merge costs nothing
beyond run-to-run noise. Merged, it generates 24 % faster on an RX 7800 XT (40.5 vs 32.7 tokens/s).

## Limitations

- **What it learned from:** mostly competitive-programming programs. They teach input parsing, strict
  compilers and integer widths, but not project idioms or frameworks. Rebirth checks every translation
  (target tests, else a compile check), and you should still review what you keep.
- **Standard library only:** third-party dependencies are not mapped to equivalents in the target language.
- **Go:** about 1 Go draft in 9 mixes `bufio.Scanner` with `fmt.Fscan` on the same input. Rebirth's retry
  loop usually fixes it.
- **Hardware:** needs about 12 GB of VRAM plus context; fits a 16 GB GPU.

## License and attribution

- **This model:** Apache-2.0, the license of the base model,
  [Qwen2.5-Coder-14B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-14B-Instruct).
- **Training and evaluation data:**
  - Derived from DeepMind **CodeContests** (Li et al. 2022, "Competition-Level Code Generation with
    AlphaCode"), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), with problems from
    codeforces.com and other sites.
  - Changes made: problems filtered, tests subsampled, solutions translated by a model and kept only if they
    pass.
  - Same-language rows: CommitPackFT / CommitPack (BigCode), permissively licensed repositories only.
  - HumanEval tests: MultiPL-E and OpenAI HumanEval (MIT).
