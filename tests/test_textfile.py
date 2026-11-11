"""`write_like` keeps the line endings, BOM and final newline of the file it overwrites."""

from __future__ import annotations

import codecs

from rebirth_ai.textfile import write_like


def test_crlf_bom_and_final_newline_are_kept(tmp_path) -> None:
    f = tmp_path / "a.js"
    f.write_bytes(codecs.BOM_UTF8 + b"var a = 1;\r\nvar b = 2;\r\n")
    write_like(f, "let a = 1;\nlet b = 2;\n\n\n")  # what the model returns (LF, extra blank lines)
    assert f.read_bytes() == codecs.BOM_UTF8 + b"let a = 1;\r\nlet b = 2;\r\n"


def test_lf_file_without_final_newline(tmp_path) -> None:
    f = tmp_path / "a.py"
    f.write_bytes(b"x = 1\ny = 2")
    write_like(f, "x = 1\ny = 3\n")
    assert f.read_bytes() == b"x = 1\ny = 3"


def test_new_file_is_plain_utf8_lf(tmp_path) -> None:
    f = tmp_path / "out" / "b.go"
    write_like(f, "package b\r\n")
    assert f.read_bytes() == b"package b\n"
