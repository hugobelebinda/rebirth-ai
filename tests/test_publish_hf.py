"""scripts/publish_hf.py: what each pack uploads (no network)."""

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("publish_hf", ROOT / "scripts" / "publish_hf.py")
publish_hf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(publish_hf)
PACKS = json.loads((ROOT / "configs" / "models.json").read_text())["packs"]


def test_modernize_uploads_the_card_and_the_8_adapters_under_their_manifest_names() -> None:
    ops = publish_hf.operations(PACKS["modernize"])
    names = [n for n, _ in ops]
    assert names[0] == "README.md" and ops[0][1].endswith("docs/release/hf-rebirth-modernize-7b-lora-gguf.md")
    assert names[1:] == [m["file"] for m in PACKS["modernize"]["models"]] and len(names) == 9
    assert Path(ops[0][1]).is_file()


def test_migrate_gets_a_modelfile_that_points_at_the_gguf_next_to_it() -> None:
    ops = dict(publish_hf.operations(PACKS["migrate"]))
    assert set(ops) == {"README.md", "Modelfile", "rebirth-xlang-14b-v2.Q6_K.gguf"}
    text = ops["Modelfile"].decode()
    assert "FROM ./rebirth-xlang-14b-v2.Q6_K.gguf\n" in text and "{gguf}" not in text
    assert "ollama create rebirth-xlang:14b -f Modelfile" in text and "<|im_start|>" in text


def test_the_cards_are_not_drafts() -> None:
    for pack in PACKS.values():
        card = (ROOT / "docs" / "release" / f"hf-{pack['hf_repo']}.md").read_text()
        assert card.startswith("---\nlicense: apache-2.0\n") and "Draft" not in card and "base_model_relation:" in card


def test_no_pack_name_prints_usage(capsys) -> None:
    assert publish_hf.main([]) == 2 and "publish_hf.py modernize|migrate" in capsys.readouterr().out
