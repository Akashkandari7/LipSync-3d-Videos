#!/usr/bin/env python3
"""Find the real mouth-movement span for each segment.

Layer 2 showed mouth motion running past the dialogue windows, which is
the defect from the first video: lipsync stops, the rendered animation
keeps mouthing, and the join shows.

Rather than trusting a fixed threshold, this calibrates per shot. It
walks the whole shot, scores each frame as mouth-change over
whole-frame-change, and calls a frame active when it sits above that
shot's own quiet level. The active run overlapping the dialogue is the
span that actually needs replacing.

The layer 1 rule still wins: a span is never allowed to grow into
another speaker's audio, because that is what made the character mouth
the narrator's words last time.
"""

import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).parent
SRC = ROOT / "source" / "video2.mp4"
FPS, TOTAL = 30, 3727
W, H = 480, 853
DET = cv2.FaceDetectorYN.create(str(ROOT / "models" / "yunet.onnx"), "",
                                (W, H), 0.6, 0.3, 5000)
CUTS = sorted(float(x) for x in (ROOT / "work2" / "cuts.txt").read_text().split())
TX = json.loads((ROOT / "work2" / "transcript.json").read_text())


def shot(t):
    return (max([c for c in CUTS if c <= t], default=0.0),
            min([c for c in CUTS if c > t], default=TOTAL / FPS))


def ratios(t0, t1):
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", f"{t0}", "-to", f"{t1}", "-i", str(SRC),
         "-vf", f"scale={W}:{H}", "-pix_fmt", "bgr24", "-f", "rawvideo", "-"],
        capture_output=True).stdout
    fr = np.frombuffer(raw, np.uint8).reshape(-1, H, W, 3)
    boxes = []
    for f in fr:
        _, faces = DET.detect(np.ascontiguousarray(f))
        if faces is None or not len(faces):
            boxes.append(None)
            continue
        fa = max(faces, key=lambda r: r[2] * r[3])
        rx, ry, lx, ly = fa[10], fa[11], fa[12], fa[13]
        cx, cy = (rx + lx) / 2, (ry + ly) / 2
        half = max(abs(lx - rx), 8.0)
        boxes.append((int(cx - half), int(cy - half * .7),
                      int(cx + half), int(cy + half * .7)))
    out = np.full(len(fr) - 1, np.nan)
    for i in range(len(fr) - 1):
        a, c = fr[i].astype(np.int16), fr[i + 1].astype(np.int16)
        glob = float(np.abs(a - c).mean())
        b = boxes[i] or boxes[i + 1]
        if b is None or glob > 40:
            continue
        x0, y0, x1, y1 = max(0, b[0]), max(0, b[1]), min(W, b[2]), min(H, b[3])
        if x1 > x0 and y1 > y0:
            out[i] = float(np.abs(a[y0:y1, x0:x1] - c[y0:y1, x0:x1]).mean()) / (glob + 1e-3)
    return out


def smooth(x, k=5):
    y = x.copy()
    for i in range(len(x)):
        w = x[max(0, i - k // 2):i + k // 2 + 1]
        w = w[~np.isnan(w)]
        y[i] = w.mean() if len(w) else np.nan
    return y


def main():
    plan = json.loads((ROOT / "work2" / "plan_final.json").read_text())
    print(f"{'#':3}{'spk':7}{'dialogue':>15}{'mouth-active span':>21}{'fr':>6}{'cr':>6}  bound")
    out, tot = [], 0
    for p in plan:
        a, b = p["dialogue"]
        lo, hi = shot((a + b) / 2)
        r = smooth(ratios(lo, hi))
        base = np.nanmedian(r)
        thr = base * 1.25
        act = np.nan_to_num(r, nan=0.0) > thr
        i_a, i_b = int((a - lo) * FPS), min(int((b - lo) * FPS), len(act) - 1)
        # grow outward from the dialogue while the mouth is still going
        s = i_a
        while s > 0 and act[max(0, s - 1)]:
            s -= 1
        e = i_b
        while e < len(act) - 1 and act[min(len(act) - 1, e + 1)]:
            e += 1
        na, nb = lo + s / FPS, lo + (e + 1) / FPS
        # layer 1 guard: never reach into another voice
        prev = max([t['end'] for t in TX if t['end'] <= a + .01], default=0.0)
        nxt = min([t['start'] for t in TX if t['start'] >= b - .01], default=TOTAL / FPS)
        bound = []
        if na < prev:
            na, _ = prev, bound.append("prev-voice")
        if nb > nxt:
            nb, _ = nxt, bound.append("next-voice")
        if na <= lo:
            bound.append("shot-in")
        if nb >= hi:
            bound.append("shot-out")
        sf, ef = int(na * FPS), int(nb * FPS) - 1
        n = ef - sf + 1
        tot += n
        grew = (b - a) < (nb - na) - 0.05
        out.append({**p, "final": [sf, ef], "frames": n,
                    "window": [round(na, 2), round(nb, 2)],
                    "grew": grew, "bound": bound})
        print(f"{p['n']:<3}{p['spk']:7}{a:6.2f}-{b:<8.2f}{na:9.2f}-{nb:<8.2f}"
              f"{'+' if grew else ' '}{n:6d}{n * .5333:6.0f}  {','.join(bound)}")
    (ROOT / "work2" / "plan_refined.json").write_text(json.dumps(out, indent=1))
    print(f"\nrefined: {tot} frames = {tot / FPS:.1f}s = {tot * .5333:.0f} credits "
          f"(${tot * .5333 / 100:.2f})")


if __name__ == "__main__":
    main()
