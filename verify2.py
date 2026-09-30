#!/usr/bin/env python3
"""Layer 3: prove the rebuilt video2 differs from the original only inside
the segments we meant to replace.

Frame count alone is not proof. On the first video the count matched
while a duplicated frame had shifted everything after it by one, so this
compares the final against the original frame by frame: outside the
segments the difference must be re-encode noise, and every seam must be
no more abrupt than the same join in the original.
"""

import json
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
SRC = ROOT / "source" / "video2.mp4"
FIN = ROOT / "final" / "FINAL_video2.mp4"
TOTAL = 3727
W, H = 108, 192


def gray(p):
    # Two traps here, both of which produced false readings earlier.
    # passthrough: a rawvideo pipe defaults to CFR and invents a duplicate
    # frame to cover the master's ~0.02s container overhang, which made this
    # check read the ORIGINAL as 3728 frames.
    # The Y plane, not gray/bgr: the master is tagged with an sRGB transfer
    # and the final is too, but any RGB conversion applies it, so comparing
    # through RGB showed a ~4 Y difference that is not in the pixels.
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", str(p),
                        "-fps_mode", "passthrough", "-vf", f"scale={W}:{H}",
                        "-pix_fmt", "yuv420p", "-f", "rawvideo", "-"],
                       capture_output=True).stdout
    fs = W * H * 3 // 2
    n = len(r) // fs
    return (np.frombuffer(r[:n * fs], np.uint8).reshape(n, fs)[:, :W * H]
            .reshape(n, H, W).astype(np.int16))


def probe(p):
    o = subprocess.run(["ffprobe", "-v", "error", "-count_frames",
                        "-select_streams", "v:0", "-show_entries",
                        "stream=nb_read_frames,r_frame_rate,width,height",
                        "-of", "csv=p=0", str(p)], capture_output=True, text=True).stdout
    return o.strip()


def scan(p, filt, token):
    e = subprocess.run(["ffmpeg", "-v", "info", "-i", str(p), "-vf", filt,
                        "-an", "-f", "null", "-"], capture_output=True, text=True).stderr
    return e.count(token)


def main():
    usable = set(json.loads((ROOT / "work2" / "usable.json").read_text()))
    plan = json.loads((ROOT / "work2" / "plan_refined.json").read_text())
    segs = sorted((p["final"][0], p["final"][1], f"s{p['n']:02d}", p["spk"])
                  for p in plan if f"s{p['n']:02d}" in usable)

    print("=== container ===")
    print("  original:", probe(SRC))
    print("  final   :", probe(FIN))

    new, old = gray(FIN), gray(SRC)
    n = min(len(new), len(old))
    print(f"  decoded frames: final={len(new)} original={len(old)}"
          f"  {'MATCH' if len(new) == len(old) == TOTAL else 'MISMATCH'}")

    d = np.abs(new[:n] - old[:n]).mean(axis=(1, 2))
    inside = np.zeros(n, bool)
    for a, b, _, _ in segs:
        inside[a:b + 1] = True
    out = d[~inside]
    stray = int(((~inside) & (d > 3)).sum())
    print("\n=== frames outside the replaced segments ===")
    print(f"  mean diff {out.mean():.3f}   max {out.max():.2f}   "
          f"frames altered: {stray}  {'PASS' if stray == 0 else 'FAIL'}")

    dn = np.abs(np.diff(new, axis=0)).mean(axis=(1, 2))
    do = np.abs(np.diff(old, axis=0)).mean(axis=(1, 2))
    print("\n=== seams (each join vs the same join in the original) ===")
    worst = 0.0
    for a, b, sid, spk in segs:
        for lbl, i in ((f"in @{a}", a - 1), (f"out @{b + 1}", b)):
            if 0 <= i < len(dn):
                delta = dn[i] - do[i]
                worst = max(worst, abs(delta))
                mark = "" if abs(delta) < do.std() else "  CHECK"
                print(f"  {sid} {spk:7}{lbl:>12}  final={dn[i]:6.2f}"
                      f"  orig={do[i]:6.2f}  delta={delta:+6.2f}{mark}")
    print(f"\n  largest seam delta {worst:.2f}   (original std dev {do.std():.2f})"
          f"  {'PASS' if worst < do.std() else 'CHECK'}")

    print("\n=== artefact scan ===")
    print(f"  black frames  : {scan(FIN, 'blackdetect=d=0.03:pic_th=0.98', 'black_start')}")
    print(f"  freeze events : {scan(FIN, 'freezedetect=n=0.001:d=0.5', 'freeze_start')}")


if __name__ == "__main__":
    main()
