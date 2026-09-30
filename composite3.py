#!/usr/bin/env python3
"""Put back only the speaker's mouth, matched to the original's levels.

Three things this gets right that earlier versions did not:

  - it follows the SPEAKER's face, from the per-frame track, not the
    biggest face in frame. Masking the biggest face once threw away a
    correct re-run and left the talker's mouth unsynced.
  - it works in yuv444p. A bgr24 roundtrip moved the picture by +2.3 Y
    on its own, worse than the flicker being fixed.
  - it writes no colour tags. Passing -color_trc to the encoder converts
    pixels rather than labelling them, costing another -4.5 Y. Tags go
    on once, at the end, via a bitstream filter.

sync.so returns frames measurably darker than it was given, and more so
outside the face than on it, which is what makes every untreated
segment boundary flicker.
"""

import json
import subprocess
import sys

import cv2
import numpy as np

import cfg3 as C

DW, DH = 540, 960
SCX, SCY = C.NW / DW, C.NH / DH
PLANE = C.NW * C.NH
FRAME = PLANE * 3
DET = cv2.FaceDetectorYN.create("models/yunet.onnx", "", (DW, DH), 0.6, 0.3, 5000)


def reader(path):
    return subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", str(path), "-fps_mode", "passthrough",
         "-pix_fmt", "yuv444p", "-f", "rawvideo", "-"], stdout=subprocess.PIPE)


def writer(path, audio_from):
    return subprocess.Popen(
        ["ffmpeg", "-v", "error",
         "-f", "rawvideo", "-pix_fmt", "yuv444p", "-s", f"{C.NW}x{C.NH}",
         "-r", "30", "-i", "-", "-i", str(audio_from),
         "-map", "0:v", "-map", "1:a?",
         "-c:v", "libx264", "-crf", "12", "-preset", "medium",
         "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
         "-fps_mode", "passthrough", "-y", str(path)], stdin=subprocess.PIPE)


def small_bgr(planes):
    yuv = np.stack([cv2.resize(p, (DW, DH)) for p in planes], axis=-1)
    return cv2.cvtColor(yuv.astype(np.uint8), cv2.COLOR_YUV2BGR)


def mask_for(small, last, want):
    _, faces = DET.detect(np.ascontiguousarray(small))
    if faces is None or not len(faces):
        return last
    if want is not None:
        f = min(faces, key=lambda r: (r[0] + r[2] / 2 - want[0]) ** 2
                                     + (r[1] + r[3] / 2 - want[1]) ** 2)
    else:
        f = max(faces, key=lambda r: r[2] * r[3])
    x, y, w, h = [float(v) for v in f[:4]]
    cx, cy = (x + w / 2) * SCX, (y + h * 0.72) * SCY
    ax, ay = w * 0.62 * SCX, h * 0.42 * SCY
    m = np.zeros((C.NH, C.NW), np.uint8)
    cv2.ellipse(m, (int(cx), int(cy)), (int(ax), int(ay)), 0, 0, 360, 255, -1)
    k = int(max(ax, ay) * 0.25) | 1
    return cv2.GaussianBlur(m, (k, k), 0).astype(np.float32) / 255.0


def run(p):
    sid = f"s{p['n']:02d}"
    src = C.SEGIN / sid / "seg.mp4"
    syn = C.SEGOUT / sid / "synced.mp4"
    dst = C.SEGOUT / sid / "composited.mp4"
    trk = {int(k): (v[0] * DW, v[1] * DH) for k, v in (p.get("track") or {}).items()}
    a, b, out = reader(src), reader(syn), writer(dst, src)
    last = np.zeros((C.NH, C.NW), np.float32)
    want, n = None, 0
    while True:
        ra, rb = a.stdout.read(FRAME), b.stdout.read(FRAME)
        if len(ra) < FRAME or len(rb) < FRAME:
            break
        po = [np.frombuffer(ra, np.uint8, PLANE, i * PLANE)
              .reshape(C.NH, C.NW).astype(np.float32) for i in range(3)]
        ps = [np.frombuffer(rb, np.uint8, PLANE, i * PLANE)
              .reshape(C.NH, C.NW).astype(np.float32) for i in range(3)]
        want = trk.get(n, want)
        last = mask_for(small_bgr(po), last, want)
        outside = last < 0.02                 # identical content in both frames
        keep = outside.sum() > 1000
        buf = bytearray()
        for c in range(3):
            s = ps[c]
            if keep:
                oc, sc = po[c][outside], s[outside]
                sd = sc.std()
                g = float(np.clip((oc.std() / sd) if sd > 1e-3 else 1.0, 0.9, 1.1))
                s = s * g + (oc.mean() - g * sc.mean())
            buf += np.clip(po[c] * (1 - last) + s * last, 0, 255).astype(np.uint8).tobytes()
        out.stdin.write(bytes(buf))
        n += 1
    for q in (a, b):
        q.stdout.close(); q.wait()
    out.stdin.close(); out.wait()
    return n


if __name__ == "__main__":
    plan = json.loads((C.WORK / "speakers.json").read_text())
    usable = set(json.loads((C.WORK / "usable.json").read_text()))
    want = {int(x) for x in sys.argv[1:]}
    for p in plan:
        sid = f"s{p['n']:02d}"
        if sid not in usable or (want and p["n"] not in want):
            continue
        n = run(p)
        print(f"  {sid} {n} frames  "
              f"{'OK' if n == p['frames'] else f'MISMATCH {n} != ' + str(p['frames'])}",
              flush=True)
