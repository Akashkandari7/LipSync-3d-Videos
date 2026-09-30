#!/usr/bin/env python3
"""Run the video2 plan through sync.so, optionally a chosen subset.

Usage: python batch2.py [n ...]   (segment numbers; all of them if none)

Segment windows come from refine.py, which already guaranteed two
things: the window carries only its own speaker's audio, and it spans
the whole stretch where that character's mouth is moving. Nothing here
pads or widens them again - that padding is what made the character
mouth the narrator's lines on the first video.
"""

import json
import subprocess
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import lipsync as L


def face_image(seg_mp4, frame_number, box_small):
    """A crop of the speaker's face as a data URI.

    Coordinates alone got s09 rejected - the protagonist stands in the
    background there and sync.so could not find a face at the point we
    named. Sync's own web app sends this crop alongside the point, so we
    do the same.
    """
    import base64
    import subprocess as sp
    x0, y0, x1, y1 = [v * 4 for v in box_small]     # 540-wide detect -> 2160
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    half = max(x1 - x0, y1 - y0)                    # square, generous
    png = sp.run(["ffmpeg", "-v", "error", "-i", str(seg_mp4),
                  "-vf", f"select='eq(n,{frame_number})',"
                         f"crop={half*2}:{half*2}:{max(0,cx-half)}:{max(0,cy-half)},"
                         f"scale=128:128",
                  "-frames:v", "1", "-f", "webp", "-"], capture_output=True).stdout
    if not png:
        return None
    return "data:image/webp;base64," + base64.b64encode(png).decode()

ROOT = Path(__file__).parent
SRC = ROOT / "source" / "video2.mp4"
PLAN = json.loads((ROOT / "work2" / "plan_refined.json").read_text())
SPK = {p["n"]: p for p in json.loads((ROOT / "work2" / "speakers.json").read_text())}
FPS = 30


def extract(p):
    a, b = p["final"]
    t0, t1 = a / FPS, (b + 1) / FPS
    d = ROOT / "segments2" / "input" / f"s{p['n']:02d}"
    d.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(SRC),
         "-vf", f"select='between(n,{a},{b})',setpts=PTS-STARTPTS",
         "-af", f"aselect='between(t,{t0},{t1})',asetpts=PTS-STARTPTS",
         "-r", "30", "-c:v", "libx264", "-crf", "12", "-preset", "medium",
         "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
         "-movflags", "+faststart", "-y", str(d / "seg.mp4")], check=True)
    subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(SRC),
         "-af", f"aselect='between(t,{t0},{t1})',asetpts=PTS-STARTPTS",
         "-vn", "-c:a", "pcm_s16le", "-ar", "48000", "-y", str(d / "seg.wav")],
        check=True)
    got = L.ffprobe(d / "seg.mp4")
    assert got["frames"] == p["frames"], f"s{p['n']}: {got['frames']} != {p['frames']}"
    return d


def process(p):
    sid = f"s{p['n']:02d}"
    d = ROOT / "segments2" / "input" / sid
    try:
        v, a = L.upload(d / "seg.mp4"), L.upload(d / "seg.wav")
        # Point sync.so at the face that is actually talking. Left to itself
        # it takes the most prominent face, which on s01 and s09 was a
        # bystander, so those lines came back unsynced.
        sp = SPK.get(p["n"]) or {}
        asd = None
        if sp.get("coords"):
            asd = {"auto_detect": False, "frame_number": sp["frame_number"],
                   "coordinates": sp["coords"]}
            if sp.get("face_box_small"):
                fi = face_image(d / "seg.mp4", sp["frame_number"],
                                sp["face_box_small"])
                if fi:
                    asd["face_image"] = fi
        idem = str(uuid.uuid5(uuid.NAMESPACE_URL, f"v2:{sid}:{p['final'][0]}:"
                              f"{asd and asd['coordinates']}:{bool(asd and asd.get('face_image'))}"))
        g = L.submit(v, a, idem, asd)
        gid = g["id"]
        print(f"  {sid} submitted {gid[:8]}", flush=True)
        g = L.poll(gid)
        rec = {"segment_id": sid, "generation_id": gid, "status": g["status"],
               "speaker": p["spk"], "final": p["final"], "frames": p["frames"],
               "text": p["text"], "idempotency_key": idem}
        if g["status"] != "COMPLETED":
            rec["error"] = g.get("error") or g.get("potentialError")
            L.save_job(rec)
            print(f"  {sid} {g['status']} -> KEEP ORIGINAL", flush=True)
            return sid, False
        out = ROOT / "segments2" / "output" / sid
        out.mkdir(parents=True, exist_ok=True)
        dst = out / "synced.mp4"
        subprocess.run(["curl", "-sfL", "-o", str(dst), g["outputMediaUrl"]], check=True)
        checks, _, _ = L.validate(d / "seg.mp4", dst)
        ok = all(c[2] for c in checks.values())
        rec["validation"] = {k: c[2] for k, c in checks.items()}
        rec["safe_to_splice"] = ok
        L.save_job(rec)
        print(f"  {sid} {'OK' if ok else 'VALIDATION FAILED ' + str([k for k,c in checks.items() if not c[2]])}",
              flush=True)
        return sid, ok
    except Exception as e:
        print(f"  {sid} ERROR {e} -> KEEP ORIGINAL", flush=True)
        return sid, False


if __name__ == "__main__":
    want = {int(x) for x in sys.argv[1:]}
    todo = [p for p in PLAN if not want or p["n"] in want]
    print(f"extracting {len(todo)} segment(s)...")
    for p in todo:
        extract(p)
    print("submitting...")
    with ThreadPoolExecutor(max_workers=6) as ex:
        res = list(ex.map(process, todo))
    good = [s for s, ok in res if ok]
    f = ROOT / "work2" / "usable.json"
    prev = json.loads(f.read_text()) if f.exists() else []
    f.write_text(json.dumps(sorted(set(prev) | set(good))))
    print(f"\nusable: {len(good)}/{len(todo)} -> {good}")
