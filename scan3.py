#!/usr/bin/env python3
"""Find shots the plan missed.

The plan only took lines that read as in-scene dialogue and treated the
rest as narration. That assumption is wrong whenever the animation has
a character mouthing the narration on camera - and those shots come
back unsynced, which is exactly what a viewer notices.

So this ignores the transcript's grammar and asks the footage instead:
walk every frame, find the face, score mouth movement against
whole-frame movement, and report every run where a visible face is
moving its mouth while there is speech in the audio and no segment
covers it.
"""

import json
import subprocess

import cv2
import numpy as np

import cfg3 as C

DW, DH = 540, 960
DET = cv2.FaceDetectorYN.create("models/yunet.onnx", "", (DW, DH), 0.6, 0.3, 5000)
CHUNK = 300


def stream():
    p = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", str(C.SRC), "-map", "0:v:0",
         "-fps_mode", "passthrough", "-vf", f"scale={DW}:{DH}",
         "-pix_fmt", "bgr24", "-f", "rawvideo", "-"], stdout=subprocess.PIPE)
    size = DW * DH * 3
    while True:
        b = p.stdout.read(size)
        if len(b) < size:
            break
        yield np.frombuffer(b, np.uint8).reshape(DH, DW, 3)
    p.stdout.close()
    p.wait()


def main():
    prev = None
    act = np.full(C.TOTAL, np.nan)
    face = np.zeros(C.TOTAL, bool)
    area = np.zeros(C.TOTAL)
    for i, f in enumerate(stream()):
        if i >= C.TOTAL:
            break
        _, faces = DET.detect(np.ascontiguousarray(f))
        box = None
        if faces is not None and len(faces):
            fa = max(faces, key=lambda r: r[2] * r[3])
            face[i] = True
            area[i] = float(fa[2] * fa[3])
            rx, ry, lx, ly = fa[10], fa[11], fa[12], fa[13]
            cx, cy = (rx + lx) / 2, (ry + ly) / 2
            half = max(abs(lx - rx), 8.0)
            box = (max(0, int(cx - half)), max(0, int(cy - half * .7)),
                   min(DW, int(cx + half)), min(DH, int(cy + half * .7)))
        if prev is not None and box is not None:
            a, c = prev.astype(np.int16), f.astype(np.int16)
            glob = float(np.abs(a - c).mean())
            if glob <= 40:
                x0, y0, x1, y1 = box
                if x1 > x0 and y1 > y0:
                    act[i] = (float(np.abs(a[y0:y1, x0:x1] - c[y0:y1, x0:x1]).mean())
                              / (glob + 1e-3))
        prev = f
        if i % 500 == 0:
            print(f"  .. {i}/{C.TOTAL}", flush=True)

    # speech mask from the word timings
    words = json.loads((C.WORK / "words.json").read_text())
    speech = np.zeros(C.TOTAL, bool)
    for s in words:
        for w in s["words"]:
            speech[int(w["s"] * C.FPS):int(w["e"] * C.FPS) + 1] = True

    plan = json.loads((C.WORK / "speakers.json").read_text())
    covered = np.zeros(C.TOTAL, bool)
    for p in plan:
        a, b = p["final"]
        covered[a:b + 1] = True

    base = np.nanmedian(act)
    thr = base * 1.25
    moving = np.nan_to_num(act, nan=0.0) > thr
    big = area > (DW * DH * 0.01)          # ignore specks in the distance
    cand = moving & face & big & speech & ~covered

    print(f"\nbaseline mouth ratio {base:.2f}, threshold {thr:.2f}")
    print(f"frames with a face: {face.sum()}  speech: {speech.sum()}  "
          f"already covered: {covered.sum()}")
    print(f"candidate frames missed: {cand.sum()}")

    runs, i = [], 0
    while i < C.TOTAL:
        if cand[i]:
            j = i
            while j + 1 < C.TOTAL and cand[j + 1]:
                j += 1
            if j - i + 1 >= 8:             # ignore flickers shorter than ~0.27s
                runs.append((i, j))
            i = j + 1
        else:
            i += 1

    def said(a, b):
        t0, t1 = a / C.FPS, (b + 1) / C.FPS
        ws = [w["w"] for s in words for w in s["words"]
              if w["e"] > t0 and w["s"] < t1]
        return " ".join(ws)[:70]

    print(f"\n{'frames':>14}{'time':>16}{'sec':>6}{'cr':>6}  words in that window")
    tot = 0
    for a, b in runs:
        n = b - a + 1
        tot += n
        print(f"{a:6d}-{b:<7d}{a/C.FPS:7.2f}-{b/C.FPS:<8.2f}{n/C.FPS:6.1f}"
              f"{n*0.5333:6.0f}  {said(a, b)}")
    print(f"\n{len(runs)} missed runs, {tot} frames = {tot/C.FPS:.1f}s "
          f"= {tot*0.5333:.0f} credits")
    json.dump([{"start_frame": a, "end_frame": b, "frames": b - a + 1,
                "text": said(a, b)} for a, b in runs],
              open(C.WORK / "missed.json", "w"), indent=1)


if __name__ == "__main__":
    main()
