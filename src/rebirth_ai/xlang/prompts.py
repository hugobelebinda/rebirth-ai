"""Prompts for cross-language translation.

- :func:`build_translation_prompt` — chat prompt for zero-shot models (base 7B chat, Ollama 14B),
  used by ``rebirth migrate --to`` and its retry loop (previous attempt + target errors).
- :func:`build_function_prompt` — the eval-harness variant: translate one function so it matches a
  given target signature (the target-language tests call that exact signature).
- :func:`build_xlang_sft_prompt` — the shared cross-language adapter format (Phase 17 plan), for
  training data and a future adapter.
"""

from __future__ import annotations

import re

from rebirth_ai.xlang.languages import XLang

_FENCE_RE = re.compile(r"^```[ \t]*([A-Za-z0-9_+#.-]*)[ \t]*\n(.*?)^```[ \t]*$", re.DOTALL | re.MULTILINE)
_TAGS = {
    "python": {"python", "py", "python3"},
    "javascript": {"javascript", "js", "jsx", "node", "mjs"},
    "typescript": {"typescript", "ts", "tsx"},
    "go": {"go", "golang"},
    "java": {"java"},
    "csharp": {"csharp", "cs", "c#"},
    "rust": {"rust", "rs"},
    "cpp": {"cpp", "c++", "cxx", "cc", "hpp"},
    "swift": {"swift"},
}


def extract_code(raw: str, target: XLang) -> str:
    """The target-language code in a model reply.

    First fenced block tagged with the target language, else the longest fenced block, else an
    unterminated opening fence, else the whole reply.
    """
    blocks = [(tag.lower(), body) for tag, body in _FENCE_RE.findall(raw)]
    tagged = [b for t, b in blocks if t in _TAGS.get(target.id, {target.fence})]
    if tagged:
        code = tagged[0]
    elif blocks:
        code = max((b for _, b in blocks), key=len)
    else:
        text = raw.strip()
        m = re.match(r"^```[^\n]*\n", text)
        code = text[m.end():] if m else text  # truncated reply: opening fence, no closing one
    return code.strip("\n") + "\n"

TRANSLATE_SYSTEM_PROMPT = """You are Rebirth AI, a precise code translation engine.
Translate source code from one programming language to another.
Rules:
- Keep the behavior identical: same functions and types, same results for the same inputs.
- Write idiomatic, modern code in the target language using only its standard library.
- Output ONLY the target-language code in a single fenced code block — no explanations.
"""

MAX_LOG_CHARS = 3000


def _fence(lang: XLang, code: str) -> str:
    return f"```{lang.fence}\n{code.rstrip()}\n```"


def build_translation_prompt(
    *,
    source: XLang,
    target: XLang,
    source_code: str,
    source_path: str,
    target_path: str,
    target_tests: str | None = None,
    previous: str | None = None,
    error_class: str | None = None,
    logs: str | None = None,
    is_program: bool = False,
) -> str:
    parts = [
        f"Translate this {source.label} file to {target.label}.",
        f"Source file: {source_path}\nTarget file: {target_path}",
        "Keep every public function, class and constant; use the naming conventions of "
        f"{target.label} unless the target tests below require specific names.",
        "The result must be self-contained: other files of the source project are not translated, "
        f"so inline what you need from them or use the {target.label} standard library.",
    ]
    if is_program:
        parts.append(
            f"This is a complete program: it reads standard input and writes standard output. The "
            f"{target.label} version must be a complete program too (with its entry point) that prints "
            "exactly the same output for every input, and is about as fast (keep the algorithm)."
        )
    if target_tests:
        parts.append(
            f"The {target.label} translation must make these {target.label} tests pass. They are fixed "
            "and run unchanged against your file: match the package / namespace / module, names and "
            "signatures they use, change your code — never the tests — and do not include them in "
            "your answer. Put each item exactly where the tests look for it: declare the package / "
            "namespace / module the tests import or live in, and put functions the tests call "
            "directly at the top level of it — do not add an extra module, namespace or class that "
            "only copies the source language's structure:\n" + _fence(target, target_tests)
        )
    parts.append(f"{source.label} source:\n" + _fence(source, source_code))
    if previous:
        clipped = (logs or "")[-MAX_LOG_CHARS:]
        parts.append(
            f"Your previous {target.label} translation failed"
            + (f" ({error_class})" if error_class else "")
            + ":\n"
            + _fence(target, previous)
            + ("\nCompiler / test output:\n```\n" + clipped.strip() + "\n```" if clipped.strip() else "")
            + f"\nFix it and output the complete corrected {target.label} file."
        )
    else:
        parts.append(f"Output the complete {target.label} file.")
    return "\n\n".join(parts)


def build_function_prompt(
    *,
    source: XLang,
    target: XLang,
    source_code: str,
    target_signature: str,
    class_wrapped: bool = False,
    target_header: str = "",
) -> str:
    where = (
        " It will be placed inside an existing `class Problem`, so output only the method(s) "
        "(make helper methods `static`), not the class."
        if class_wrapped
        else ""
    )
    return (
        f"Translate this {source.label} code to {target.label}.\n\n"
        f"{source.label} source:\n{_fence(source, source_code)}\n\n"
        f"The {target.label} version must have exactly this signature:\n"
        f"{_fence(target, target_signature)}\n\n"
        + (
            f"The {target.label} file already starts with this — do not repeat it, but add any other "
            "imports your code needs:\n"
            f"{_fence(target, target_header)}\n\n"
            if target_header.strip()
            else ""
        )
        + f"Output the complete {target.label} function, with any helper functions and imports it "
        f"needs. Do not add a main function or tests.{where}"
    )


def build_xlang_sft_prompt(*, source: XLang, target: XLang, source_code: str) -> str:
    """Training / adapter format: one shared adapter, both languages named in the prompt."""
    return (
        "### Source language\n"
        f"{source.id}\n"
        "### Target language\n"
        f"{target.id}\n"
        "### Source\n"
        f"{source_code.rstrip()}\n"
        "### Target\n"
    )
