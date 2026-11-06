#!/usr/bin/env python3
"""Sample AMD dGPU telemetry to JSONL, fsync'd every line (survives a hard GPU hang).

Reads sysfs only (no ROCm/torch import). Picks the card with the largest VRAM
(the RX 7800 XT, not the iGPU). Stops on SIGTERM/SIGINT.

  python training/gpu_telemetry.py --out logs/train/<run>/telemetry.jsonl --interval 2
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import time
from pathlib import Path


def _read(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except OSError:
        return None


def _int(path: Path) -> int | None:
    v = _read(path)
    try:
        return int(v) if v is not None else None
    except ValueError:
        return None


def find_dgpu() -> Path | None:
    best: tuple[int, Path] | None = None
    for dev in Path("/sys/class/drm").glob("card[0-9]*/device"):
        total = _int(dev / "mem_info_vram_total")
        if total and (best is None or total > best[0]):
            best = (total, dev)
    return best[1] if best else None


def mem_available_mb() -> int | None:
    for line in (_read(Path("/proc/meminfo")) or "").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) // 1024
    return None


def sample(dev: Path) -> dict:
    hw = next(iter((dev / "hwmon").glob("hwmon*")), None)
    rec: dict = {
        "t": round(time.time(), 2),
        "vram_used_mb": (_int(dev / "mem_info_vram_used") or 0) // (1 << 20),
        "gtt_used_mb": (_int(dev / "mem_info_gtt_used") or 0) // (1 << 20),
        "gpu_busy_pct": _int(dev / "gpu_busy_percent"),
        "ram_avail_mb": mem_available_mb(),
    }
    if hw is not None:
        for i in (1, 2, 3):
            label = _read(hw / f"temp{i}_label") or f"temp{i}"
            milli = _int(hw / f"temp{i}_input")
            if milli is not None:
                rec[f"temp_{label}_c"] = milli / 1000
        power = _int(hw / "power1_average") or _int(hw / "power1_input")
        if power is not None:
            rec["power_w"] = round(power / 1e6, 1)
        sclk = _int(hw / "freq1_input")
        if sclk is not None:
            rec["sclk_mhz"] = sclk // 1_000_000
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--interval", type=float, default=2.0)
    args = ap.parse_args()

    dev = find_dgpu()
    if dev is None:
        print("gpu_telemetry: no amdgpu card with mem_info_vram_total found")
        return 1
    stop = False

    def _stop(*_):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"event": "start", "device": str(dev.resolve()), "t": time.time()}) + "\n")
        while not stop:
            fh.write(json.dumps(sample(dev)) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
            time.sleep(args.interval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
