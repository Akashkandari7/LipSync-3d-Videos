#!/usr/bin/env python3
"""Layer 2: does the mouth actually move, and does it stop inside our window?

Layer 1 proved each window carries only its own speaker's audio. That is
not enough: the rendered animation moves the mouth on its own schedule,
so a window that fits the words can still cut away while the mouth is
still going - which is what went wrong on the first video.

For each planned segment this looks 1s past both edges, locates faces
with YuNet (which returns mouth-corner landmarks, so the mouth box is
anchored rather than guessed), and measures frame-to-frame change in
that box. Activity continuing past an edge means the segment ends
mid-motion and needs extending.
"""

import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).parent
SRC = ROOT / "source" / "video2.mp4"
FPS = 30
PAD = 1.0                      # how far past each edge to look
W, H = 480, 853                # detection resolution, keeps YuNet fast on 4K
DET = cv2.FaceDetectorYN.create(str(ROOT / "models" / "yunet.onnx"), "",
                                (W, H), 0.6, 0.3, 5000)


def read_window(t0, t1):
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", f"{t0}", "-to", f"{t1}", "-i", str(SRC),
         "-vf", f"scale={W}:{H}", "-pix_fmt", "bgr24", "-f", "rawvideo", "-"],
        capture_output=True).stdout
    return np.frombuffer(out, np.uint8).reshape(-1, H, W, 3)


def mouth_boxes(frames):
    """Mouth box per frame, anchored on YuNet's two mouth-corner landmarks."""
    boxes = []
    for f in frames:
        _, faces = DET.detect(np.ascontiguousarray(f))
        if faces is None or not len(faces):
            boxes.append(None)
            continue
        face = max(faces, key=lambda r: r[2] * r[3])
        rx, ry, lx, ly = face[10], face[11], face[12], face[13]
        cx, cy = (rx + lx) / 2, (ry + ly) / 2
        half = max(abs(lx - rx), 8.0)
        boxes.append((int(cx - half), int(cy - half * 0.7),
                      int(cx + half), int(cy + half * 0.7)))
    return boxes


def activity(frames, boxes):
    """Mouth change divided by whole-frame change.

    Raw mouth difference is useless here: a shot cut or a camera move
    lights up every pixel, mouth included, so an unchanged mouth next to
    a cut scores higher than a talking one mid-shot. The ratio cancels
    that out - only the mouth moving faster than its own frame counts.
    """
    act = []
    for i in range(len(frames) - 1):
        a_f = frames[i].astype(np.int16)
        c_f = frames[i + 1].astype(np.int16)
        glob = float(np.abs(a_f - c_f).mean())
        b = boxes[i] or boxes[i + 1]
        if b is None or glob > 40:            # >40 means a cut, not motion
            act.append(np.nan)
            continue
        x0, y0, x1, y1 = (max(0, b[0]), max(0, b[1]), min(W, b[2]), min(H, b[3]))
        if x1 <= x0 or y1 <= y0:
            act.append(np.nan)
            continue
        loc = float(np.abs(a_f[y0:y1, x0:x1] - c_f[y0:y1, x0:x1]).mean())
        act.append(loc / (glob + 1e-3))
    return np.array(act)


def main():
    plan = json.loads((ROOT / "work2" / "plan_final.json").read_text())
    cuts = sorted(float(x) for x in (ROOT / "work2" / "cuts.txt").read_text().split())

    def clamp(a, b):
        """Keep the look-around inside the segment's own shot."""
        lo = max([c for c in cuts if c <= a], default=0.0)
        hi = min([c for c in cuts if c >= b], default=3727 / FPS)
        return max(lo, a - PAD), min(hi, b + PAD)

    print(f"{'#':3}{'spk':7}{'window':>16}{'face%':>7}{'inside':>8}"
          f"{'before':>8}{'after':>8}  verdict")
    report, need = [], []
    for p in plan:
        a, b = p["window"]
        t0, t1 = clamp(a, b)
        frames = read_window(t0, t1)
        boxes = mouth_boxes(frames)
        act = activity(frames, boxes)
        n = len(act)
        i0 = int((a - t0) * FPS)
        i1 = n - int((t1 - b) * FPS)
        m = lambda s: float(np.nanmean(s)) if len(s) and not np.all(np.isnan(s)) else np.nan
        inside, before, after = m(act[i0:i1]), m(act[:i0]), m(act[i1:])
        frac = sum(x is not None for x in boxes) / max(len(boxes), 1)

        def spill(x):
            return (not np.isnan(x)) and (not np.isnan(inside)) and x > inside * 0.70

        v = []
        if frac < 0.5:
            v.append("FACE-WEAK")
        if spill(before):
            v.append("SPILL-BEFORE")
        if spill(after):
            v.append("SPILL-AFTER")
        if v and v != ["FACE-WEAK"]:
            need.append(p["n"])
        report.append({**p, "face_frac": round(frac, 2),
                       "act_inside": None if np.isnan(inside) else round(inside, 2),
                       "act_before": None if np.isnan(before) else round(before, 2),
                       "act_after": None if np.isnan(after) else round(after, 2),
                       "layer2": v or ["OK"]})
        f = lambda x: "   n/a" if np.isnan(x) else f"{x:6.2f}"
        print(f"{p['n']:<3}{p['spk']:7}{a:7.2f}-{b:<8.2f}{frac*100:6.0f}%"
              f"{f(inside):>8}{f(before):>8}{f(after):>8}  {','.join(v) or 'OK'}")
    (ROOT / "work2" / "layer2.json").write_text(json.dumps(report, indent=1))
    ok = sum(1 for r in report if r["layer2"] == ["OK"])
    print(f"\nLAYER 2: {ok}/{len(report)} clean; needs widening: {need or 'none'}")


if __name__ == "__main__":
    main()
