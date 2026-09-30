#!/usr/bin/env python3
"""Process every planned segment: extract, lipsync, validate.

One ffmpeg pass cuts all segments (decoding the 448MB master once).
Generations then run concurrently, since each spends ~100s server-side.
A segment that fails any check is left out of the splice entirely.
"""

import json
import subprocess
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import lipsync as L

ROOT = Path(__file__).parent
SRC = next((ROOT / "source").glob("*.mp4"))
PLAN = json.loads((ROOT / "metadata" / "segments.json").read_text())
FPS = 30


def extract_all():
    """Cut every segment in a single decode pass, frame-exact."""
    fc, maps = [], []
    for i, s in enumerate(PLAN):
        a, b = s["start_frame"], s["end_frame"]
        t0, t1 = a / FPS, (b + 1) / FPS
        d = ROOT / "segments" / "input" / s["id"]
        d.mkdir(parents=True, exist_ok=True)
        fc.append(f"[0:v]select='between(n,{a},{b})',setpts=PTS-STARTPTS[v{i}]")
        # asplit: the same audio feeds both the mp4 and the wav, and ffmpeg
        # refuses to consume one filter label twice.
        fc.append(f"[0:a]aselect='between(t,{t0},{t1})',asetpts=PTS-STARTPTS,"
                  f"asplit=2[a{i}][w{i}]")
        maps += ["-map", f"[v{i}]", "-map", f"[a{i}]",
                 "-c:v", "libx264", "-crf", "12", "-preset", "medium",
                 "-pix_fmt", "yuv420p", "-r", "30",
                 "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
                 "-movflags", "+faststart", str(d / "seg.mp4")]
        maps += ["-map", f"[w{i}]", "-vn", "-c:a", "pcm_s16le", "-ar", "48000",
                 str(d / "seg.wav")]
    cmd = ["ffmpeg", "-v", "error", "-i", str(SRC),
           "-filter_complex", ";".join(fc)] + maps + ["-y"]
    subprocess.run(cmd, check=True)


def process(s):
    sid = s["id"]
    d = ROOT / "segments" / "input" / sid
    try:
        v = L.upload(d / "seg.mp4")
        a = L.upload(d / "seg.wav")
        idem = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{sid}:{s['start_frame']}"))
        g = L.submit(v, a, idem)
        gid = g["id"]
        print(f"  {sid} submitted {gid[:8]}", flush=True)
        g = L.poll(gid)
        rec = {"segment_id": sid, "generation_id": gid, "status": g["status"],
               "idempotency_key": idem, **{k: s[k] for k in
               ("start_frame", "end_frame", "frames", "text")}}
        if g["status"] != "COMPLETED":
            rec["error"] = g.get("error") or g.get("potentialError")
            L.save_job(rec)
            print(f"  {sid} {g['status']} -> KEEP ORIGINAL", flush=True)
            return sid, False
        out = ROOT / "segments" / "output" / sid
        out.mkdir(parents=True, exist_ok=True)
        dst = out / "synced.mp4"
        subprocess.run(["curl", "-sfL", "-o", str(dst), g["outputMediaUrl"]], check=True)
        checks, _, _ = L.validate(d / "seg.mp4", dst)
        ok = all(c[2] for c in checks.values())
        rec["validation"] = {k: c[2] for k, c in checks.items()}
        rec["safe_to_splice"] = ok
        L.save_job(rec)
        bad = [k for k, c in checks.items() if not c[2]]
        print(f"  {sid} {'OK' if ok else 'VALIDATION FAILED ' + str(bad)}", flush=True)
        return sid, ok
    except Exception as e:
        print(f"  {sid} ERROR {e} -> KEEP ORIGINAL", flush=True)
        return sid, False


if __name__ == "__main__":
    print(f"extracting {len(PLAN)} segments in one pass...")
    extract_all()
    print("submitting...")
    with ThreadPoolExecutor(max_workers=9) as ex:
        results = list(ex.map(process, PLAN))
    good = [sid for sid, ok in results if ok]
    print(f"\nusable: {len(good)}/{len(PLAN)} -> {good}")
    (ROOT / "metadata" / "usable.json").write_text(json.dumps(good))
