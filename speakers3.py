#!/usr/bin/env python3
"""Find which face is talking in each planned segment, and track it.

Left alone, sync.so syncs "the most prominent face". On the previous
video that meant a bystander twice over, and the compositor then made
it worse by masking the biggest face too. So each segment carries both
a point to hand sync.so and a per-frame track for the compositor to
follow.
"""

import json
import subprocess

import cv2
import numpy as np

import cfg3 as C

DW, DH = 540, 960
SX, SY = C.NW / DW, C.NH / DH
DET = cv2.FaceDetectorYN.create("models/yunet.onnx", "", (DW, DH), 0.6, 0.3, 5000)


def frames_of(a, b):
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(C.SRC), "-map", "0:v:0",
         "-fps_mode", "passthrough",
         "-vf", f"select='between(n,{a},{b})',scale={DW}:{DH}",
         "-pix_fmt", "bgr24", "-f", "rawvideo", "-"], capture_output=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, DH, DW, 3)


def detect(frames):
    out = []
    for f in frames:
        _, faces = DET.detect(np.ascontiguousarray(f))
        row = []
        if faces is not None:
            for fa in faces:
                x, y, w, h = fa[:4]
                rx, ry, lx, ly = fa[10], fa[11], fa[12], fa[13]
                mx, my = (rx + lx) / 2, (ry + ly) / 2
                half = max(abs(lx - rx), 8.0)
                row.append((float(x + w / 2), float(y + h / 2), float(w * h),
                            (int(mx - half), int(my - half * .7),
                             int(mx + half), int(my + half * .7))))
        out.append(row)
    return out


def track(dets, tol=60):
    tracks = []
    for i, row in enumerate(dets):
        for cx, cy, area, mb in row:
            best, bd = None, tol
            for t in tracks:
                if t["last"] == i:
                    continue
                d = ((t["cx"] - cx) ** 2 + (t["cy"] - cy) ** 2) ** .5
                if d < bd:
                    best, bd = t, d
            if best is None:
                tracks.append({"cx": cx, "cy": cy, "area": area, "last": i,
                               "frames": {i: mb}, "n": 1})
            else:
                best.update(cx=cx, cy=cy, last=i, n=best["n"] + 1)
                best["area"] = max(best["area"], area)
                best["frames"][i] = mb
    return tracks


def score(frames, t):
    vals = []
    for i in sorted(t["frames"]):
        if i + 1 >= len(frames) or (i + 1) not in t["frames"]:
            continue
        a, c = frames[i].astype(np.int16), frames[i + 1].astype(np.int16)
        glob = float(np.abs(a - c).mean())
        if glob > 40:
            continue
        x0, y0, x1, y1 = t["frames"][i]
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(DW, x1), min(DH, y1)
        if x1 > x0 and y1 > y0:
            vals.append(float(np.abs(a[y0:y1, x0:x1] - c[y0:y1, x0:x1]).mean())
                        / (glob + 1e-3))
    return float(np.mean(vals)) if vals else 0.0


def main():
    plan = json.loads((C.WORK / "plan.json").read_text())
    print(f"{'seg':6}{'spk':8}{'faces':>6}{'coords':>14}{'score':>7}{'2nd':>7}  note")
    out = []
    for p in plan:
        a, b = p["final"]
        fr = frames_of(a, b)
        tracks = [t for t in track(detect(fr)) if t["n"] >= max(3, len(fr) * .2)]
        if not tracks:
            print(f"s{p['n']:02d}    {p['spk']:8}{0:>6}   NO FACE")
            out.append({**p, "coords": None, "faces": 0, "track": {}})
            continue
        scored = sorted(((score(fr, t), t) for t in tracks), key=lambda x: -x[0])
        best_s, best = scored[0]
        second = scored[1][0] if len(scored) > 1 else 0.0
        biggest = max(tracks, key=lambda t: t["area"])
        # biggest frame of this face, not the middle one - a small face at the
        # wrong moment got a whole generation rejected last time
        fnum = max(best["frames"],
                   key=lambda i: (best["frames"][i][2] - best["frames"][i][0])
                                 * (best["frames"][i][3] - best["frames"][i][1]))
        mb = best["frames"][fnum]
        cx, cy = ((mb[0] + mb[2]) / 2) * SX, ((mb[1] + mb[3]) / 2) * SY
        note = ""
        if len(tracks) > 1:
            note = "MULTI-FACE"
            if best is not biggest:
                note += " - speaker is NOT biggest"
        trk = {str(i): [round(((v[0] + v[2]) / 2) / DW, 5),
                        round(((v[1] + v[3]) / 2) / DH, 5)]
               for i, v in best["frames"].items()}
        out.append({**p, "faces": len(tracks), "coords": [round(cx), round(cy)],
                    "frame_number": int(fnum), "face_box_small": [int(v) for v in mb],
                    "score": round(best_s, 2), "runner_up": round(second, 2),
                    "note": note, "track": trk})
        print(f"s{p['n']:02d}    {p['spk']:8}{len(tracks):>6}"
              f"{str([round(cx), round(cy)]):>14}{best_s:7.2f}{second:7.2f}  {note}")
    (C.WORK / "speakers.json").write_text(json.dumps(out, indent=1))
    print(f"\nmulti-face: {sum(1 for o in out if o.get('faces', 0) > 1)}/{len(out)}")


if __name__ == "__main__":
    main()
