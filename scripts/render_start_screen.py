"""Render the start screen (`rebirth` with no command) to docs/assets/start-screen.png for the README.

Draws a recorded Rich console on an exact character grid, as a terminal does: block and box-drawing characters
are drawn as shapes, so the logo and the boxes line up. Run it on a machine where `rebirth doctor` passes (the
screen shows the live status) after changing the start screen or the version:

    pip install pillow
    python scripts/render_start_screen.py              # → docs/assets/start-screen.png
"""
import io
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from rich.console import Console
from rich.markup import escape
from rich.segment import Segment
from rich.terminal_theme import MONOKAI

from rebirth_ai import start

COLS = 84
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
SIZE = 24
THEME = MONOKAI
BG = (30, 31, 34)
FG = (226, 226, 222)

console = Console(file=io.StringIO(), record=True, width=COLS, force_terminal=True, color_system="truecolor", highlight=False)
start.render(console, rows=start.machine_status(), menu=True)
for question, default, answer in [
    ("What do you want to do", "modernize", "modernize"),
    ("Project folder", ".", "~/code/billing"),
    ("File (Enter = the whole project)", "", "src/invoice.py"),
    ("Write the changes into your files? (y/N)", "N", ""),
]:
    console.print(f"  [bold]{escape(question)}[/bold] {escape(f'[{default}]')}: {escape(answer)}")
console.print()
console.print(f"  [bold {start.ACCENT}]→[/] [bold]rebirth migrate /home/you/code/billing --file src/invoice.py[/bold]")
lines = list(Segment.split_lines(console._record_buffer))
while lines and not "".join(s.text for s in lines[-1]).strip():
    lines.pop()

reg, bold = ImageFont.truetype(FONT, SIZE), ImageFont.truetype(BOLD, SIZE)
asc, desc = reg.getmetrics()
cw, ch = reg.getlength("0"), asc + desc + 2
pad, bar = 34, 46
W = int(COLS * cw + 2 * pad)
H = int(len(lines) * ch + bar + 2 * pad - 10)
img = Image.new("RGB", (W, H), (0, 0, 0, 0))
d = ImageDraw.Draw(img)
d.rounded_rectangle([0, 0, W - 1, H - 1], radius=14, fill=BG)
for i, c in enumerate([(255, 95, 87), (254, 188, 46), (40, 200, 64)]):
    d.ellipse([20 + i * 26, 16, 34 + i * 26, 30], fill=c)
d.text((W / 2, 23), "rebirth", font=reg, fill=(150, 150, 150), anchor="mm")

SINGLE = {"─": "LR", "│": "UD", "┌": "RD", "┐": "LD", "└": "RU", "┘": "LU", "┬": "LRD", "┴": "LRU",
          "├": "UDR", "┤": "UDL", "┼": "LRUD"}
DOUBLE = {"═", "║", "╔", "╗", "╚", "╝"}


def rgb(color, dim=False):
    if color is None:
        c = FG
    else:
        c = tuple(color.get_truecolor(THEME))
    return tuple(int(v * 0.55 + b * 0.45) for v, b in zip(c, BG, strict=True)) if dim else tuple(c)


def rect_line(a, b, fill, w=2):
    """Axis-aligned line from a to b as a w-pixel rectangle (direction-independent, gap-free)."""
    (xa, ya), (xb, yb) = a, b
    if ya == yb:
        d.rectangle([min(xa, xb), ya - w // 2, max(xa, xb), ya - w // 2 + w - 1], fill=fill)
    else:
        d.rectangle([xa - w // 2, min(ya, yb), xa - w // 2 + w - 1, max(ya, yb)], fill=fill)


def box(x0, y0, x1, y1, char, fill, w=2):
    x0, x1, y0, y1 = int(x0), int(x1), int(y0), int(y1)
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    x1, y1 = x1 - 1, y1 - 1  # last pixel of this cell; the next cell starts at x1 + 1
    if char in SINGLE:
        for dirn in SINGLE[char]:
            rect_line((cx, cy), {"L": (x0, cy), "R": (x1, cy), "U": (cx, y0), "D": (cx, y1)}[dirn], fill, w)
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
            rect_line(a, b, fill, w)


for row, segs in enumerate(lines):
    col = 0
    top = round(bar + pad - 6 + row * ch)
    for seg in segs:
        style = seg.style
        fill = rgb(style.color if style else None, bool(style and style.dim))
        font = bold if style and style.bold else reg
        for char in seg.text:
            x0, x1 = round(pad + col * cw), round(pad + (col + 1) * cw)
            if char == "█":
                d.rectangle([x0, top, x1 - 1, top + round(ch) - 1], fill=fill)
            elif char in SINGLE or char in DOUBLE:
                box(x0, top, x1, top + round(ch), char, fill)
            elif char != " ":
                d.text((x0 + cw / 2, top + asc + 1), char, font=font, fill=fill, anchor="ms")
            col += 1

out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "docs" / "assets" / "start-screen.png"
img.save(out, optimize=True)
print(out, img.size)
