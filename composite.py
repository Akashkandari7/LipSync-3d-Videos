#!/usr/bin/env python3
"""Put only the mouth back, not the whole frame.

sync.so re-encodes everything it is given and hands back a darker frame
- measured in the raw Y plane at -1.1 to -4.2 across all twelve
segments, and the shift is *larger* outside the face than on it. Drop
those frames into the timeline whole and every segment boundary
flickers.

So each output frame is the original everywhere, with a feathered patch
over the lower face taken from the synced frame. The patch is
level-matched first, per plane, using the region outside it where both
frames carry identical content.

Everything stays in yuv444p. An earlier version routed frames through
bgr24 and that roundtrip alone moved the picture by +2.28 Y - worse
than the problem being fixed - because the master is tagged with an
sRGB transfer that a raw RGB pipe does not carry. In YUV the same
roundtrip measures +0.006, and the source's colour tags are written
back out explicitly.
"""

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).parent
NW, NH = 2160, 3840
DW, DH = 540, 960                       # detection scale
SC = NW / DW
PLANE = NW * NH
FRAME = PLANE * 3                       # yuv444p: three full-size planes
DET = cv2.FaceDetectorYN.create(str(ROOT / "models" / "yunet.onnx"), "",
                                (DW, DH), 0.6, 0.3, 5000)
# No colour tags here on purpose. Passing -color_trc to the encoder does not
# merely label the stream, it converts the pixels: tagging this output cost
# -4.47 Y against an untagged +0.006. The master's tags are restored once, on
# the final splice, with a bitstream filter that rewrites the header only.


def reader(path):
    return subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", str(path), "-fps_mode", "passthrough",
         "-pix_fmt", "yuv444p", "-f", "rawvideo", "-"], stdout=subprocess.PIPE)


def writer(path, audio_from):
    return subprocess.Popen(
        ["ffmpeg", "-v", "error",
         "-f", "rawvideo", "-pix_fmt", "yuv444p", "-s", f"{NW}x{NH}",
         "-r", "30", "-i", "-", "-i", str(audio_from),
         "-map", "0:v", "-map", "1:a?",
         "-c:v", "libx264", "-crf", "12", "-preset", "medium",
         "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-b:a", "192k",
         "-fps_mode", "passthrough", "-y", str(path)], stdin=subprocess.PIPE)


def small_bgr(planes):
    yuv = np.stack([cv2.resize(p, (DW, DH)) for p in planes], axis=-1)
    return cv2.cvtColor(yuv.astype(np.uint8), cv2.COLOR_YUV2BGR)


def mask_for(small, last, want=None):
    """Feathered patch over the lower face, in full-resolution coordinates.

    `want` is the speaker's expected centre in detection coordinates. Without
    it this took the biggest face, which on s01 and s09 is a bystander - so
    the patch landed on the wrong person and left the speaker's mouth showing
    the original animation, undoing the whole point of the re-run.
    """
    _, faces = DET.detect(np.ascontiguousarray(small))
    if faces is None or not len(faces):
        return last
    if want is not None:
        f = min(faces, key=lambda r: (r[0] + r[2] / 2 - want[0]) ** 2
                                     + (r[1] + r[3] / 2 - want[1]) ** 2)
    else:
        f = max(faces, key=lambda r: r[2] * r[3])
    x, y, w, h = [float(v) for v in f[:4]]
    cx = (x + w / 2) * SC
    cy = (y + h * 0.72) * SC            # mouth and jaw, not the eyes
    ax, ay = w * 0.62 * SC, h * 0.42 * SC
    m = np.zeros((NH, NW), np.uint8)
    cv2.ellipse(m, (int(cx), int(cy)), (int(ax), int(ay)), 0, 0, 360, 255, -1)
    k = int(max(ax, ay) * 0.25) | 1
    return cv2.GaussianBlur(m, (k, k), 0).astype(np.float32) / 255.0


def speaker_track(sid):
    """Expected speaker centre per frame, in detection coordinates."""
    data = json.loads((ROOT / "work2" / "speakers.json").read_text())
    for p in data:
        if f"s{p['n']:02d}" == sid and p.get("track"):
            return {int(k): (v[0] * DW, v[1] * DH) for k, v in p["track"].items()}
    return {}


def run(sid):
    src = ROOT / "segments2" / "input" / sid / "seg.mp4"
    syn = ROOT / "segments2" / "output" / sid / "synced.mp4"
    dst = ROOT / "segments2" / "output" / sid / "composited.mp4"
    a, b, out = reader(src), reader(syn), writer(dst, src)
    trk = speaker_track(sid)
    last = np.zeros((NH, NW), np.float32)
    want = None
    n = 0
    while True:
        ra, rb = a.stdout.read(FRAME), b.stdout.read(FRAME)
        if len(ra) < FRAME or len(rb) < FRAME:
            break
        po = [np.frombuffer(ra, np.uint8, PLANE, i * PLANE).reshape(NH, NW).astype(np.float32)
              for i in range(3)]
        ps = [np.frombuffer(rb, np.uint8, PLANE, i * PLANE).reshape(NH, NW).astype(np.float32)
              for i in range(3)]
        want = trk.get(n, want)
        last = mask_for(small_bgr(po), last, want)
        outside = last < 0.02                 # identical content in both frames
        keep = outside.sum() > 1000
        buf = bytearray()
        for c in range(3):
            s = ps[c]
            if keep:
                oc, sc_ = po[c][outside], s[outside]
                sd = sc_.std()
                g = float(np.clip((oc.std() / sd) if sd > 1e-3 else 1.0, 0.9, 1.1))
                s = s * g + (oc.mean() - g * sc_.mean())
            buf += np.clip(po[c] * (1 - last) + s * last, 0, 255).astype(np.uint8).tobytes()
        out.stdin.write(bytes(buf))
        n += 1
    for p in (a, b):
        p.stdout.close(); p.wait()
    out.stdin.close(); out.wait()
    return n


if __name__ == "__main__":
    plan = json.loads((ROOT / "work2" / "plan_refined.json").read_text())
    want = {f"s{int(x):02d}" for x in sys.argv[1:]}
    for p in plan:
        sid = f"s{p['n']:02d}"
        if want and sid not in want:
            continue
        n = run(sid)
        print(f"  {sid} {n} frames  "
              f"{'OK' if n == p['frames'] else f'MISMATCH {n} != ' + str(p['frames'])}",
              flush=True)
