#!/usr/bin/env python3
"""Rebuild the master with every validated segment swapped in.

Builds the whole timeline in one ffmpeg pass using the concat *filter*.
The concat demuxer was stitching parts by timestamp and duplicated a
frame wherever a part was very short (the single original frame between
two adjacent segments), which pushed everything after it out by one.
Trimming straight from the master avoids intermediate parts entirely and
re-encodes exactly once.

Audio is taken whole from the original: music, SFX and ambience never go
through the lipsync stage.
"""

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).parent
SRC = next((ROOT / "source").glob("*.mp4"))
TOTAL = 3540


def main():
    usable = json.loads((ROOT / "metadata" / "usable.json").read_text())
    plan = {s["id"]: s for s in json.loads((ROOT / "metadata" / "segments.json").read_text())}
    segs = sorted([(plan[s]["start_frame"], plan[s]["end_frame"], s) for s in usable]
                  + [(3100, 3209, "seg001")])

    inputs = ["-i", str(SRC)]
    for _, _, sid in segs:
        inputs += ["-i", str(ROOT / "segments" / "output" / sid / "synced.mp4")]

    fc, labels, cursor = [], [], 0
    for idx, (a, b, sid) in enumerate(segs, start=1):
        if a > cursor:                       # untouched original run
            lbl = f"o{idx}"
            fc.append(f"[0:v]trim=start_frame={cursor}:end_frame={a},"
                      f"setpts=PTS-STARTPTS[{lbl}]")
            labels.append(lbl)
        lbl = f"s{idx}"
        fc.append(f"[{idx}:v]setpts=PTS-STARTPTS[{lbl}]")
        labels.append(lbl)
        cursor = b + 1
    if cursor < TOTAL:
        fc.append(f"[0:v]trim=start_frame={cursor}:end_frame={TOTAL},"
                  f"setpts=PTS-STARTPTS[oz]")
        labels.append("oz")

    fc.append("".join(f"[{l}]" for l in labels) + f"concat=n={len(labels)}:v=1:a=0[out]")

    out = ROOT / "final" / "FINAL_lipsynced_all.mp4"
    out.parent.mkdir(exist_ok=True)
    cmd = ["ffmpeg", "-v", "error", *inputs,
           "-filter_complex", ";".join(fc),
           "-map", "[out]", "-map", "0:a",
           "-c:v", "libx264", "-crf", "14", "-preset", "fast",
           "-pix_fmt", "yuv420p", "-r", "30",
           "-c:a", "aac", "-b:a", "192k",
           "-movflags", "+faststart", "-y", str(out)]
    subprocess.run(cmd, check=True)
    print(f"{len(segs)} segments spliced into {out}  ({len(labels)} runs)")


if __name__ == "__main__":
    main()
