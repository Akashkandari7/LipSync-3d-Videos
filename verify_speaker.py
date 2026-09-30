#!/usr/bin/env python3
"""Did the composite keep the lipsync on the person who is talking?

The compositor used to mask the biggest face in frame. On s01 and s09
the speaker is not the biggest face, so the patch landed on a bystander
and the speaker kept their original, barely-moving mouth - the paid-for
re-run was thrown away at the last step and nothing in the frame-count
or seam checks noticed.

This follows the speaker's own mouth, frame by frame, and asks one
question: does the composited clip differ from the original there by as
much as the raw sync output does? If it does not, the lipsync was lost.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
W, H = 540, 960


def read(path, a=None, b=None):
    vf = f"scale={W}:{H}"
    if a is not None:
        vf = f"select='between(n,{a},{b})',{vf}"
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path),
                        "-fps_mode", "passthrough", "-vf", vf,
                        "-pix_fmt", "bgr24", "-f", "rawvideo", "-"],
                       capture_output=True).stdout
    return np.frombuffer(r, np.uint8).reshape(-1, H, W, 3).astype(np.float32)


def main():
    spk = {p["n"]: p for p in json.loads((ROOT / "work2" / "speakers.json").read_text())}
    plan = json.loads((ROOT / "work2" / "plan_refined.json").read_text())
    want = {int(x) for x in sys.argv[1:]}
    print(f"{'seg':6}{'spk':8}{'sync vs orig':>14}{'composited vs orig':>20}{'kept':>7}")
    for p in plan:
        if want and p["n"] not in want:
            continue
        sid = f"s{p['n']:02d}"
        trk = spk[p["n"]].get("track") or {}
        a, b = p["final"]
        o = read(ROOT / "segments2" / "input" / sid / "seg.mp4")
        s = read(ROOT / "segments2" / "output" / sid / "synced.mp4")
        c = read(ROOT / "segments2" / "output" / sid / "composited.mp4")
        n = min(len(o), len(s), len(c))
        ds, dc = [], []
        for i in range(n):
            k = str(i)
            if k not in trk:
                continue
            cx, cy = trk[k][0] * W, trk[k][1] * H
            hw, hh = 34, 26
            x0, y0 = max(0, int(cx - hw)), max(0, int(cy - hh))
            x1, y1 = min(W, int(cx + hw)), min(H, int(cy + hh))
            if x1 <= x0 or y1 <= y0:
                continue
            ds.append(np.abs(s[i, y0:y1, x0:x1] - o[i, y0:y1, x0:x1]).mean())
            dc.append(np.abs(c[i, y0:y1, x0:x1] - o[i, y0:y1, x0:x1]).mean())
        if not ds:
            print(f"{sid:6}{p['spk']:8}{'no track':>14}")
            continue
        ms, mc = float(np.mean(ds)), float(np.mean(dc))
        ratio = mc / max(ms, 1e-6)
        print(f"{sid:6}{p['spk']:8}{ms:14.2f}{mc:20.2f}{ratio:6.0%}"
              f"{'' if ratio > 0.6 else '   LOST'}")


if __name__ == "__main__":
    main()
