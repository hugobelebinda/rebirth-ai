"""Record the README demo: a real Rebirth session in a terminal, drawn as an animated GIF.

    pip install pexpect pyte pillow        # any environment; Rebirth itself must be installed (.venv)
    python scripts/record_demo.py record   # runs the session  → docs/assets/demo.cast (asciicast v2)
    python scripts/record_demo.py render   # demo.cast          → docs/assets/demo.gif

`record` runs on this machine's models and sandboxes, in a throwaway home folder holding a copy of
examples/billing (nothing in the repo changes). The session: the start screen modernizes the example,
then `--to go` translates it. `render` draws the terminal on an exact character grid, like
render_start_screen.py; waits longer than MAX_WAIT seconds (the model writing) are shortened in the
GIF, and the times Rebirth prints are the real ones.
"""

from __future__ import annotations

import codecs
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CAST = ROOT / "docs" / "assets" / "demo.cast"
GIF = ROOT / "docs" / "assets" / "demo.gif"
COLS, ROWS = 100, 50
PROMPT = "\x1b[1;38;2;249;113;44m$\x1b[0m "
MAX_WAIT = 1.6  # seconds: longer gaps in the recording are cut to this in the GIF
TITLE = "rebirth  ·  a real run (model waits shortened)"

# --------------------------------------------------------------------------- record


class Recorder:
    """pexpect logfile: keeps every chunk the terminal shows, with its time."""

    def __init__(self) -> None:
        self.t0 = time.monotonic()
        self.events: list[list] = []
        self._dec = codecs.getincrementaldecoder("utf-8")(errors="replace")

    def write(self, data: bytes) -> None:
        text = self._dec.decode(data)
        if text:
            self.events.append([round(time.monotonic() - self.t0, 4), "o", text])

    def mark(self, label: str) -> None:
        self.events.append([round(time.monotonic() - self.t0, 4), "m", label])

    def flush(self) -> None:
        pass


def record(cast: Path = CAST) -> None:
    import pexpect

    home = Path(tempfile.mkdtemp(prefix="rebirth-demo-"))
    shutil.copytree(ROOT / "examples" / "billing", home / "code" / "billing")
    env = {
        "HOME": str(home), "PATH": f"{ROOT / '.venv' / 'bin'}:{os.environ['PATH']}", "TERM": "xterm-256color",
        "COLORTERM": "truecolor", "LANG": "C.UTF-8",
        "PS1": "\\[\x1b[1;38;2;249;113;44m\\]$\\[\x1b[0m\\] ",  # bash shows exactly PROMPT
        "REBIRTH_RUNS_DIR": str(home / ".rebirth" / "runs"),  # the demo's run logs stay in its home
        "REBIRTH_VECTOR_DB_PATH": str(home / ".rebirth" / "embeddings.db"),  # no index: none of this repo's
        "HF_HOME": os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface")), "HF_HUB_OFFLINE": "1",
    }
    rec = Recorder()
    child = pexpect.spawn("bash", ["--norc", "--noprofile"], cwd=str(home), env=env, dimensions=(ROWS, COLS))
    child.logfile_read = rec

    seen = bytearray()  # what the terminal printed since the last wait_for

    def pump(seconds: float) -> None:
        """Read the terminal for ``seconds``: every chunk is recorded with the time it appeared."""
        stop = time.monotonic() + seconds
        while (left := stop - time.monotonic()) > 0:
            try:
                seen.extend(child.read_nonblocking(65536, timeout=min(0.03, left)))
            except pexpect.TIMEOUT:
                pass

    def wait_for(text: str, timeout: float = 900) -> None:
        want, stop = text.encode(), time.monotonic() + timeout
        while want not in seen:
            if time.monotonic() > stop:
                raise TimeoutError(f"no {text!r} after {timeout} s; the screen ended with {bytes(seen[-300:])!r}")
            pump(0.05)
        del seen[: seen.index(want) + len(want)]

    def hold(seconds: float) -> None:  # a pause the GIF keeps (the viewer reads the screen)
        rec.mark(f"hold {seconds}")
        pump(seconds)

    def type_(text: str, enter: bool = True) -> None:
        for ch in text:
            child.send(ch)
            pump(0.07)
        if enter:
            pump(0.3)
            child.send("\r")

    try:
        wait_for(PROMPT)
        hold(0.8)
        type_("rebirth")
        wait_for("What do you want to do", 60)
        hold(3.5)
        type_("modernize")
        wait_for("Project folder", 30)
        hold(0.5)
        type_("code/billing")
        wait_for("File", 30)
        hold(0.4)
        type_("src/invoice.py")
        wait_for("Write the changes", 30)
        hold(0.8)
        type_("")
        wait_for(PROMPT)
        hold(6.0)
        type_("clear")
        wait_for(PROMPT)
        hold(0.5)
        type_("rebirth migrate code/billing --file src/invoice.py --to go")
        wait_for(PROMPT)
        hold(2.5)
        type_("head -32 code/billing/rebirth_out/go/src/invoice.go")
        wait_for(PROMPT)
        hold(5.0)
    finally:
        child.close(force=True)
        shutil.rmtree(home, ignore_errors=True)
    header = {"version": 2, "width": COLS, "height": ROWS, "timestamp": int(time.time()), "title": "Rebirth AI",
              "env": {"TERM": "xterm-256color", "SHELL": "/bin/bash"}}
    cast.write_text(json.dumps(header) + "\n" + "".join(json.dumps(e) + "\n" for e in rec.events), encoding="utf-8")
    print(cast, f"{len(rec.events)} events, {rec.events[-1][0]:.0f} s")


# --------------------------------------------------------------------------- render

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
SIZE = 16
BG, FG = (30, 31, 34), (226, 226, 222)
ANSI = {  # the start screenshot's look (Monokai-like)
    "black": (39, 40, 34), "red": (249, 38, 114), "green": (166, 226, 46), "brown": (244, 191, 117),
    "yellow": (244, 191, 117), "blue": (102, 217, 239), "magenta": (174, 129, 255), "cyan": (161, 239, 228),
    "white": (248, 248, 242), "brightblack": (117, 113, 94), "brightred": (249, 38, 114),
    "brightgreen": (166, 226, 46), "brightbrown": (244, 191, 117), "brightyellow": (244, 191, 117),
    "brightblue": (102, 217, 239), "brightmagenta": (174, 129, 255), "brightcyan": (161, 239, 228),
    "brightwhite": (249, 248, 245),
}
SINGLE = {"─": "LR", "│": "UD", "┌": "RD", "┐": "LD", "└": "RU", "┘": "LU", "┬": "LRD", "┴": "LRU",
          "├": "UDR", "┤": "UDL", "┼": "LRUD", "╭": "RD", "╮": "LD", "╰": "RU", "╯": "LU"}
DOUBLE = {"═", "║", "╔", "╗", "╚", "╝"}


def _screen():
    import pyte

    class Screen(pyte.Screen):
        """pyte ignores SGR 2 (dim), which Rich uses: carry it in the unused blink flag."""

        def select_graphic_rendition(self, *attrs, private=False):
            out, i = [], 0
            while i < len(attrs):
                a = attrs[i]
                if a in (38, 48) and i + 1 < len(attrs):  # 38;2;r;g;b / 38;5;n: copy the colour as is
                    n = 5 if attrs[i + 1] == 2 else 3
                    out += attrs[i:i + n]
                    i += n
                    continue
                out += [5] if a == 2 else [22, 25] if a == 22 else [a]
                i += 1
            super().select_graphic_rendition(*out)

    return pyte, Screen(COLS, ROWS)


def _colour(c, default):
    if c == "default":
        return default
    if c in ANSI:
        return ANSI[c]
    try:
        return tuple(int(c[i:i + 2], 16) for i in (0, 2, 4))
    except ValueError:
        return default


class Painter:
    def __init__(self) -> None:
        from PIL import ImageFont

        self.reg, self.bold = ImageFont.truetype(FONT, SIZE), ImageFont.truetype(BOLD, SIZE)
        self.asc, desc = self.reg.getmetrics()
        self.cw, self.ch = self.reg.getlength("0"), self.asc + desc + 3
        self.pad, self.bar = 22, 38
        self.W = int(COLS * self.cw + 2 * self.pad)
        self.H = int(ROWS * self.ch + self.bar + 2 * self.pad - 8)

    def frame(self, screen):
        from PIL import Image, ImageDraw

        img = Image.new("RGB", (self.W, self.H), (16, 17, 19))
        d = ImageDraw.Draw(img)
        d.rounded_rectangle([0, 0, self.W - 1, self.H - 1], radius=12, fill=BG)
        for i, c in enumerate([(255, 95, 87), (254, 188, 46), (40, 200, 64)]):
            d.ellipse([16 + i * 22, 13, 28 + i * 22, 25], fill=c)
        d.text((self.W / 2, 19), TITLE, font=self.reg, fill=(150, 150, 150), anchor="mm")
        for y in range(ROWS):
            line = screen.buffer[y]
            top = round(self.bar + self.pad - 6 + y * self.ch)
            for x in range(COLS):
                cell = line[x]
                fg, bg = _colour(cell.fg, FG), _colour(cell.bg, BG)
                if cell.reverse:
                    fg, bg = bg, fg
                if cell.blink:  # dim
                    fg = tuple(int(v * 0.55 + b * 0.45) for v, b in zip(fg, BG, strict=True))
                x0, x1 = round(self.pad + x * self.cw), round(self.pad + (x + 1) * self.cw)
                if bg != BG:
                    d.rectangle([x0, top, x1 - 1, top + self.ch - 1], fill=bg)
                ch = cell.data
                if ch == "█":
                    d.rectangle([x0, top, x1 - 1, top + self.ch - 1], fill=fg)
                elif ch in SINGLE or ch in DOUBLE:
                    self._box(d, x0, top, x1, top + self.ch, ch, fg)
                elif ch.strip():
                    d.text((x0 + self.cw / 2, top + self.asc + 1), ch, font=self.bold if cell.bold else self.reg,
                           fill=fg, anchor="ms")
        if not screen.cursor.hidden:
            cx, cy = screen.cursor.x, screen.cursor.y
            if 0 <= cx < COLS and 0 <= cy < ROWS:
                x0 = round(self.pad + cx * self.cw)
                top = round(self.bar + self.pad - 6 + cy * self.ch)
                d.rectangle([x0, top + 2, x0 + round(self.cw) - 1, top + self.ch - 2], fill=(200, 200, 200))
        return img

    @staticmethod
    def _line(d, a, b, fill, w=2):
        (xa, ya), (xb, yb) = a, b
        if ya == yb:
            d.rectangle([min(xa, xb), ya - w // 2, max(xa, xb), ya - w // 2 + w - 1], fill=fill)
        else:
            d.rectangle([xa - w // 2, min(ya, yb), xa - w // 2 + w - 1, max(ya, yb)], fill=fill)

    def _box(self, d, x0, y0, x1, y1, char, fill, w=2):
        x0, x1, y0, y1 = int(x0), int(x1), int(y0), int(y1)
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        x1, y1 = x1 - 1, y1 - 1
        if char in SINGLE:
            for dirn in SINGLE[char]:
                self._line(d, (cx, cy), {"L": (x0, cy), "R": (x1, cy), "U": (cx, y0), "D": (cx, y1)}[dirn], fill, w)
            return
        g = max(2, round((x1 - x0) / 5))
        paths = {
            "═": [[(x0, cy - g), (x1, cy - g)], [(x0, cy + g), (x1, cy + g)]],
            "║": [[(cx - g, y0), (cx - g, y1)], [(cx + g, y0), (cx + g, y1)]],
            "╔": [[(x1, cy - g), (cx - g, cy - g), (cx - g, y1)], [(x1, cy + g), (cx + g, cy + g), (cx + g, y1)]],
            "╗": [[(x0, cy - g), (cx + g, cy - g), (cx + g, y1)], [(x0, cy + g), (cx - g, cy + g), (cx - g, y1)]],
            "╚": [[(cx - g, y0), (cx - g, cy + g), (x1, cy + g)], [(cx + g, y0), (cx + g, cy - g), (x1, cy - g)]],
            "╝": [[(cx + g, y0), (cx + g, cy + g), (x0, cy + g)], [(cx - g, y0), (cx - g, cy - g), (x0, cy - g)]],
        }[char]
        for path in paths:
            for a, b in zip(path, path[1:], strict=False):
                self._line(d, a, b, fill, w)


def timeline(events: list[list]) -> list[tuple[float, str]]:
    """(time in the GIF, text) for each output chunk: gaps cut to MAX_WAIT unless a hold mark asks for more."""
    out, t_gif, last_t, allow = [], 0.0, None, MAX_WAIT
    for t, kind, data in events:
        if kind == "m":
            allow = max(allow, float(data.split()[1]) + 0.1)
            continue
        if last_t is not None:
            t_gif += min(t - last_t, allow)
        out.append((t_gif, data))
        last_t, allow = t, MAX_WAIT
    return out


def render(cast: Path = CAST, gif: Path = GIF, frames_dir: Path | None = None) -> None:
    from PIL import Image

    lines = cast.read_text(encoding="utf-8").splitlines()
    events = [json.loads(ln) for ln in lines[1:] if ln.strip()]
    pyte, screen = _screen()
    stream = pyte.Stream(screen)
    painter = Painter()
    shots: list[tuple[float, Image.Image]] = []
    chunks = timeline(events)
    i = 0
    while i < len(chunks):
        t0 = chunks[i][0]
        while i < len(chunks) and chunks[i][0] - t0 < 0.05:  # one frame per 50 ms at most
            stream.feed(chunks[i][1])
            i += 1
        shots.append((t0, painter.frame(screen)))
    end = shots[-1][0] + 3.0
    durations = [max(40, round((b - a) * 1000)) for (a, _), (b, _) in zip(shots, shots[1:] + [(end, None)], strict=True)]
    # one palette for every frame (from a few spread-out frames) keeps the colours steady and the file small
    sample = [shots[k][1] for k in range(0, len(shots), max(1, len(shots) // 6))]
    strip = Image.new("RGB", (painter.W, painter.H * len(sample)))
    for k, im in enumerate(sample):
        strip.paste(im, (0, k * painter.H))
    palette = strip.quantize(colors=255, method=Image.Quantize.MEDIANCUT)
    frames = [im.quantize(palette=palette, dither=Image.Dither.NONE) for _, im in shots]
    frames[0].save(gif, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=False,
                   disposal=1)
    if frames_dir:
        frames_dir.mkdir(parents=True, exist_ok=True)
        for k in range(0, len(shots), max(1, len(shots) // 12)):
            shots[k][1].save(frames_dir / f"frame_{k:04d}.png")
        shots[-1][1].save(frames_dir / "frame_last.png")
    print(gif, f"{len(frames)} frames, {sum(durations) / 1000:.0f} s, {gif.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else ""
    if step == "record":
        record()
    elif step == "render":
        render(frames_dir=Path(sys.argv[2]) if len(sys.argv) > 2 else None)
    else:
        print(__doc__)
        sys.exit(2)
