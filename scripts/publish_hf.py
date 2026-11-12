"""Publish a model pack to Hugging Face (maintainers; log in first with `.venv/bin/hf auth login`).

    .venv/bin/python scripts/publish_hf.py modernize|migrate [--private]

For the pack in configs/models.json, this:
1. creates `<hf_namespace>/<hf_repo>`, private at first;
2. uploads the model card (docs/release/hf-<hf_repo>.md, as README.md) and every model file, renamed as in
   the manifest. The migrate pack also gets a ready-to-use Ollama `Modelfile`;
3. checks each file's size and sha256 on the Hub against the manifest;
4. makes the repo public, unless --private or a check failed.

Upload again to replace the files; a repo that is already public stays public.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs" / "models.json"


def hf_modelfile(pack: dict) -> bytes:
    """The Ollama Modelfile template with FROM pointing at the GGUF next to it."""
    model = pack["models"][0]
    text = (ROOT / model["modelfile"]).read_text(encoding="utf-8")
    text = text.replace("`rebirth install` fills in FROM.", "FROM is the GGUF next to this file: "
                        f"`ollama create {model['alias']} -f Modelfile`.")
    return text.replace("{gguf}", f"./{model['file']}").encode("utf-8")


def operations(pack: dict) -> list[tuple[str, str | bytes]]:
    """(path in the repo, local path or bytes) for every file of the pack."""
    ops: list[tuple[str, str | bytes]] = [("README.md", str(ROOT / "docs" / "release" / f"hf-{pack['hf_repo']}.md"))]
    if any(m.get("modelfile") for m in pack["models"]):
        ops.append(("Modelfile", hf_modelfile(pack)))
    ops += [(m["file"], str(ROOT / m["local"])) for m in pack["models"]]
    return ops


def main(argv: list[str]) -> int:
    if not argv or argv[0] not in ("modernize", "migrate"):
        print(__doc__)
        return 2
    from huggingface_hub import CommitOperationAdd, HfApi

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    pack = manifest["packs"][argv[0]]
    repo = f"{manifest['hf_namespace']}/{pack['hf_repo']}"
    ops = operations(pack)
    missing = [p for _, p in ops if isinstance(p, str) and not Path(p).is_file()]
    if missing:
        print(f"missing local files: {missing}")
        return 1

    api = HfApi()
    print(f"signed in as {api.whoami()['name']}; publishing {repo}", flush=True)
    api.create_repo(repo, repo_type="model", private=True, exist_ok=True)
    start = time.time()
    print(f"uploading {len(ops)} files ...", flush=True)
    commit = api.create_commit(repo, operations=[CommitOperationAdd(n, p) for n, p in ops],
                               commit_message=f"Rebirth AI: {argv[0]} pack")
    print(f"uploaded in {time.time() - start:.0f}s (commit {commit.oid[:12]})", flush=True)

    files = {f.rfilename: f for f in api.model_info(repo, files_metadata=True).siblings}
    bad = [n for n, _ in ops if n not in files]
    for m in pack["models"]:
        f = files.get(m["file"])
        ok = f is not None and f.lfs is not None and f.lfs.sha256 == m["sha256"] and f.size == m["size"]
        print(f"  {m['file']}: {'sha256 OK' if ok else 'MISMATCH'}", flush=True)
        if not ok:
            bad.append(m["file"])
    if bad:
        print(f"NOT made public, check: {sorted(set(bad))}")
        return 1
    if "--private" in argv:
        print(f"done, still private: https://huggingface.co/{repo}")
        return 0
    api.update_repo_settings(repo, private=False)
    print(f"public: https://huggingface.co/{repo}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
