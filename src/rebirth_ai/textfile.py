"""Write a draft over a source file the way that file was written (Phase 19 bug hunt).

Files are read with universal newlines, so the model only ever sees ``\\n``. Writing its draft back
as-is turned a CRLF file into LF, which shows up as every line changed. ``write_like`` keeps the file's
line endings (CRLF when most of its lines use them), its UTF-8 byte-order mark, and whether it ends
with a newline.
"""

from __future__ import annotations

import codecs
from pathlib import Path


def write_like(target: Path, text: str) -> None:
    target = Path(target)
    try:
        raw = target.read_bytes()
    except OSError:  # a new file: plain UTF-8, LF
        raw = b""
    body = text.replace("\r\n", "\n").lstrip("﻿")
    bom = raw.startswith(codecs.BOM_UTF8)
    if raw:
        crlf = raw.count(b"\r\n")
        lf = raw.count(b"\n") - crlf
        if body.strip():
            body = body.rstrip("\n") + ("\n" if raw.endswith(b"\n") else "")
        if crlf > lf:
            body = body.replace("\n", "\r\n")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes((codecs.BOM_UTF8 if bom else b"") + body.encode("utf-8"))
