#!/usr/bin/env python3
"""Cut video3's segments and run them through sync.so.

Extraction is a single decode pass with one output pair per segment. The
previous version ran one ffmpeg per segment, each scanning the whole
master - twelve full scans of a 4K file, and the slowest step in the
pipeline by far.

Every request names the speaker explicitly (point plus a face crop).
Left to itself sync.so picks the most prominent face, which on the last
video was the wrong person twice.
"""

import base64
import json
import subprocess
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor

import cfg3 as C
import lipsync as L

PLAN = json.loads((C.WORK / "speakers.json").read_text())


def extract_all(plan):
    fc, maps = [], []
    for i, p in enumerate(plan):
        a, b = p["final"]
        t0, t1 = a / C.FPS, (b + 1) / C.FPS
        d = C.SEGIN / f"s{p['n']:02d}"
        d.mkdir(parents=True, exist_ok=True)
        fc.append(f"[0:v:0]select='between(n,{a},{b})',setpts=PTS-STARTPTS[v{i}]")
        fc.append(f"[0:a:0]aselect='between(t,{t0},{t1})',asetpts=PTS-STARTPTS,"
                  f"asplit=2[a{i}][w{i}]")
        maps += ["-map", f"[v{i}]", "-map", f"[a{i}]",
                 "-c:v", "libx264", "-crf", "12", "-preset", "medium",
                 "-pix_fmt", "yuv420p", "-r", "30",
                 "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
                 "-movflags", "+faststart", str(d / "seg.mp4")]
        maps += ["-map", f"[w{i}]", "-vn", "-c:a", "pcm_s16le", "-ar", "48000",
                 str(d / "seg.wav")]
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(C.SRC),
                    "-filter_complex", ";".join(fc)] + maps + ["-y"], check=True)
    for p in plan:
        d = C.SEGIN / f"s{p['n']:02d}"
        got = L.ffprobe(d / "seg.mp4")
        assert got["frames"] == p["frames"], \
            f"s{p['n']:02d}: {got['frames']} != {p['frames']}"


def face_image(seg_mp4, frame_number, box_small):
    """Crop of the speaker's face as a data URI - coordinates alone once got
    a small background face rejected."""
    sx = C.NW / 540
    x0, y0, x1, y1 = [v * sx for v in box_small]
    cx, cy = int((x0 + x1) / 2), int((y0 + y1) / 2)
    half = int(max(x1 - x0, y1 - y0))
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(seg_mp4),
         "-vf", f"select='eq(n,{frame_number})',"
                f"crop={half*2}:{half*2}:{max(0,cx-half)}:{max(0,cy-half)},"
                f"scale=128:128",
         "-frames:v", "1", "-f", "webp", "-"], capture_output=True).stdout
    return "data:image/webp;base64," + base64.b64encode(out).decode() if out else None


def process(p):
    sid = f"s{p['n']:02d}"
    d = C.SEGIN / sid
    try:
        v, a = L.upload(d / "seg.mp4"), L.upload(d / "seg.wav")
        asd = None
        if p.get("coords"):
            asd = {"auto_detect": False, "frame_number": p["frame_number"],
                   "coordinates": p["coords"]}
            fi = face_image(d / "seg.mp4", p["frame_number"], p["face_box_small"])
            if fi:
                asd["face_image"] = fi
        idem = str(uuid.uuid5(uuid.NAMESPACE_URL,
                              f"v3:{sid}:{p['final'][0]}:{p.get('coords')}"))
        g = L.submit(v, a, idem, asd)
        gid = g["id"]
        print(f"  {sid} submitted {gid[:8]}", flush=True)
        g = L.poll(gid)
        rec = {"segment_id": f"v3_{sid}", "generation_id": gid,
               "status": g["status"], "speaker": p["spk"], "final": p["final"],
               "frames": p["frames"], "text": p["text"], "idempotency_key": idem}
        if g["status"] != "COMPLETED":
            rec["error"] = g.get("error") or g.get("potentialError")
            L.save_job(rec)
            print(f"  {sid} {g['status']} -> KEEP ORIGINAL: {rec['error']}", flush=True)
            return sid, False
        out = C.SEGOUT / sid
        out.mkdir(parents=True, exist_ok=True)
        dst = out / "synced.mp4"
        subprocess.run(["curl", "-sfL", "-o", str(dst), g["outputMediaUrl"]], check=True)
        checks, _, _ = L.validate(d / "seg.mp4", dst)
        ok = all(c[2] for c in checks.values())
        rec["validation"] = {k: c[2] for k, c in checks.items()}
        L.save_job(rec)
        bad = [k for k, c in checks.items() if not c[2]]
        print(f"  {sid} {'OK' if ok else 'VALIDATION FAILED ' + str(bad)}", flush=True)
        return sid, ok
    except Exception as e:
        print(f"  {sid} ERROR {e} -> KEEP ORIGINAL", flush=True)
        return sid, False


if __name__ == "__main__":
    want = {int(x) for x in sys.argv[1:]}
    todo = [p for p in PLAN if not want or p["n"] in want]
    print(f"extracting {len(todo)} segments in one pass...")
    extract_all(todo)
    print("submitting...")
    with ThreadPoolExecutor(max_workers=7) as ex:
        res = list(ex.map(process, todo))
    good = [s for s, ok in res if ok]
    f = C.WORK / "usable.json"
    prev = json.loads(f.read_text()) if f.exists() else []
    f.write_text(json.dumps(sorted(set(prev) | set(good))))
    print(f"\nusable: {len(good)}/{len(todo)} -> {good}")
