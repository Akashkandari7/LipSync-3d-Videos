#!/usr/bin/env python3
"""Cross-check the finished video3 against the master.

Every check here exists because its absence let a real defect through
on an earlier build:

  frame alignment  - a duplicated frame once shifted everything after it
                     while the frame count still looked right.
  opacity/level    - sync.so returns darker frames; untreated, every
                     segment boundary flickers.
  speaker mouth    - the compositor once masked the wrong face, quietly
                     discarding the lipsync it was meant to keep.
  seams, black, freeze, tags.

Measurements are taken on the raw Y plane. Reading through RGB applies
the transfer tag and invents differences that are not in the pixels.
"""

import json
import subprocess

import cv2
import numpy as np

import cfg3 as C

W, H = 270, 480
DW, DH = 540, 960


def yplane(path, w=W, h=H):
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path),
                        "-map", "0:v:0", "-fps_mode", "passthrough",
                        "-vf", f"scale={w}:{h}", "-pix_fmt", "yuv420p",
                        "-f", "rawvideo", "-"], capture_output=True).stdout
    fs = w * h * 3 // 2
    n = len(r) // fs
    return (np.frombuffer(r[:n * fs], np.uint8).reshape(n, fs)[:, :w * h]
            .reshape(n, h, w).astype(np.float32))


def bgr(path, a, b):
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path),
                        "-map", "0:v:0", "-fps_mode", "passthrough",
                        "-vf", f"select='between(n,{a},{b})',scale={DW}:{DH}",
                        "-pix_fmt", "bgr24", "-f", "rawvideo", "-"],
                       capture_output=True).stdout
    return np.frombuffer(r, np.uint8).reshape(-1, DH, DW, 3).astype(np.float32)


def scan(path, filt, token):
    e = subprocess.run(["ffmpeg", "-v", "info", "-i", str(path), "-vf", filt,
                        "-an", "-f", "null", "-"], capture_output=True, text=True).stderr
    return e.count(token)


def main():
    plan = json.loads((C.WORK / "speakers.json").read_text())
    usable = set(json.loads((C.WORK / "usable.json").read_text()))
    segs = [p for p in plan if f"s{p['n']:02d}" in usable]
    fails = []

    print("=== container ===")
    for lbl, f in (("original", C.SRC), ("final", C.OUT)):
        o = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                            "-show_entries",
                            "stream=width,height,r_frame_rate,avg_frame_rate,"
                            "nb_frames,color_transfer",
                            "-of", "csv=p=0", str(f)],
                           capture_output=True, text=True).stdout.strip().splitlines()[0]
        print(f"  {lbl:9}{o}")

    new, old = yplane(C.OUT), yplane(C.SRC)
    n = min(len(new), len(old))
    same = len(new) == len(old) == C.TOTAL
    print(f"  decoded frames: final={len(new)} original={len(old)} "
          f"{'MATCH' if same else 'MISMATCH'}")
    if not same:
        fails.append("frame count")

    d = np.abs(new[:n] - old[:n]).mean(axis=(1, 2))
    inside = np.zeros(n, bool)
    for p in segs:
        a, b = p["final"]
        inside[a:b + 1] = True
    stray = int(((~inside) & (d > 3)).sum())
    print("\n=== frames outside the replaced segments ===")
    print(f"  mean {d[~inside].mean():.3f}  max {d[~inside].max():.2f}  "
          f"altered {stray}  {'PASS' if stray == 0 else 'FAIL'}")
    if stray:
        fails.append("untouched frames changed")

    print("\n=== opacity / level shift per segment ===")
    print(f"  {'seg':6}{'spk':8}{'level vs original':>20}")
    worst = 0.0
    for p in segs:
        a, b = p["final"]
        sh = float((new[a:b + 1] - old[a:b + 1]).mean())
        worst = max(worst, abs(sh))
        print(f"  s{p['n']:02d}    {p['spk']:8}{sh:>20.2f}")
    print(f"  worst {worst:.2f} Y  {'PASS' if worst < 1.0 else 'FAIL'}")
    if worst >= 1.0:
        fails.append("level shift")

    print("\n=== lipsync present on the speaker ===")
    print(f"  {'seg':6}{'spk':8}{'mouth diff':>12}{'frame avg':>11}{'ratio':>8}")
    for p in segs:
        a, b = p["final"]
        trk = p.get("track") or {}
        o, f = bgr(C.SRC, a, b), bgr(C.OUT, a, b)
        m = min(len(o), len(f))
        vals = []
        for i in range(m):
            k = str(i)
            if k not in trk:
                continue
            cx, cy = trk[k][0] * DW, trk[k][1] * DH
            x0, y0 = max(0, int(cx - 34)), max(0, int(cy - 26))
            x1, y1 = min(DW, int(cx + 34)), min(DH, int(cy + 26))
            if x1 > x0 and y1 > y0:
                vals.append(np.abs(f[i, y0:y1, x0:x1] - o[i, y0:y1, x0:x1]).mean())
        mouth = float(np.mean(vals)) if vals else 0.0
        rest = float(np.abs(f[:m] - o[:m]).mean())
        ratio = mouth / max(rest, 1e-6)
        bad = mouth < 3.0
        print(f"  s{p['n']:02d}    {p['spk']:8}{mouth:12.2f}{rest:11.2f}{ratio:7.1f}x"
              f"{'   NO LIPSYNC' if bad else ''}")
        if bad:
            fails.append(f"s{p['n']:02d} no lipsync")

    dn = np.abs(np.diff(new, axis=0)).mean(axis=(1, 2))
    do = np.abs(np.diff(old, axis=0)).mean(axis=(1, 2))
    print("\n=== seams ===")
    worst_seam = 0.0
    for p in segs:
        a, b = p["final"]
        for lbl, i in ((f"in @{a}", a - 1), (f"out @{b + 1}", b)):
            if 0 <= i < len(dn):
                delta = dn[i] - do[i]
                worst_seam = max(worst_seam, abs(delta))
                print(f"  s{p['n']:02d} {lbl:>12}  final={dn[i]:6.2f} "
                      f"orig={do[i]:6.2f}  delta={delta:+6.2f}")
    ok = worst_seam < do.std()
    print(f"  worst {worst_seam:.2f}  (original std {do.std():.2f})  "
          f"{'PASS' if ok else 'CHECK'}")
    if not ok:
        fails.append("seam")

    print("\n=== artefacts ===")
    for lbl, f in (("original", C.SRC), ("final", C.OUT)):
        bl = scan(f, "blackdetect=d=0.03:pic_th=0.98", "black_start")
        fz = scan(f, "freezedetect=n=0.001:d=0.5", "freeze_start")
        print(f"  {lbl:9} black={bl}  freeze={fz}")

    print("\n" + ("ALL CHECKS PASSED" if not fails else "FAILED: " + ", ".join(fails)))


if __name__ == "__main__":
    main()
