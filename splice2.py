#!/usr/bin/env python3
"""Rebuild video2 with every validated segment swapped in.

Single ffmpeg pass over the concat filter, trimming straight from the
master. The concat demuxer is deliberately avoided: on the first video
it duplicated a frame wherever a run was very short, which silently
pushed everything after it out by one and was only caught by comparing
frames against the original.

Audio is taken whole from the original, so music and ambience never pass
through the lipsync stage.

Every piece has its timestamps rebuilt from its frame index
(setpts=N/30/TB) rather than merely rebased. sync.so returned one clip
with the right frame count but a 29fps average, and rebasing kept that
spacing, so concat padded the gap with a duplicate frame and shifted
everything after it.

The output carries no -r/-fps_mode cfr either. The master's container
duration runs ~0.02s past 3727/30, and CFR resampling filled that gap
with one more duplicate frame. setpts already makes every piece exact
CFR, so the muxer is told to pass the timestamps through untouched.
"""

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).parent
SRC = ROOT / "source" / "video2.mp4"
TOTAL = 3727
OUT = ROOT / "final" / "FINAL_video2.mp4"


def main():
    usable = set(json.loads((ROOT / "work2" / "usable.json").read_text()))
    plan = json.loads((ROOT / "work2" / "plan_refined.json").read_text())
    segs = sorted((p["final"][0], p["final"][1], f"s{p['n']:02d}")
                  for p in plan if f"s{p['n']:02d}" in usable)
    if not segs:
        raise SystemExit("nothing validated to splice")

    inputs = ["-i", str(SRC)]
    for _, _, sid in segs:
        # composited, not synced: the raw sync.so frame is ~4 Y darker than the
        # master even where it changed nothing, which flickers at every join.
        c = ROOT / "segments2" / "output" / sid / "composited.mp4"
        inputs += ["-i", str(c if c.exists()
                             else ROOT / "segments2" / "output" / sid / "synced.mp4")]

    fc, labels, cursor = [], [], 0
    for idx, (a, b, sid) in enumerate(segs, start=1):
        if a > cursor:
            fc.append(f"[0:v]trim=start_frame={cursor}:end_frame={a},"
                      f"setparams=color_primaries=bt709:color_trc=iec61966-2-1:colorspace=bt709:range=tv,setpts=N/30/TB[o{idx}]")
            labels.append(f"o{idx}")
        # setparams DECLARES the composited clip's colour properties without
        # touching a pixel. Without it concat sees untagged inputs beside the
        # tagged master and quietly converts them: measured -5.57 Y.
        fc.append(f"[{idx}:v]setparams=color_primaries=bt709:color_trc=iec61966-2-1:colorspace=bt709:range=tv,setpts=N/30/TB[s{idx}]")
        labels.append(f"s{idx}")
        cursor = b + 1
    if cursor < TOTAL:
        fc.append(f"[0:v]trim=start_frame={cursor}:end_frame={TOTAL},"
                  f"setparams=color_primaries=bt709:color_trc=iec61966-2-1:colorspace=bt709:range=tv,setpts=N/30/TB[oz]")
        labels.append("oz")
    fc.append("".join(f"[{l}]" for l in labels) + f"concat=n={len(labels)}:v=1:a=0[out]")

    OUT.parent.mkdir(exist_ok=True)
    subprocess.run(["ffmpeg", "-v", "error", *inputs,
                    "-filter_complex", ";".join(fc),
                    "-map", "[out]", "-map", "0:a",
                    "-c:v", "libx264", "-crf", "14", "-preset", "fast",
                    "-pix_fmt", "yuv420p", "-fps_mode", "passthrough",
                    "-c:a", "aac", "-b:a", "192k",
                    # h264_metadata rewrites the header only. Passing
                    # -color_trc to the encoder instead converts the pixels,
                    # which measured -4.47 Y on a no-op roundtrip.
                    "-bsf:v", "h264_metadata=colour_primaries=1:"
                              "transfer_characteristics=13:"
                              "matrix_coefficients=1:video_full_range_flag=0",
                    "-movflags", "+faststart", "-y", str(OUT)], check=True)
    print(f"{len(segs)} segments spliced into {OUT} ({len(labels)} runs)")


if __name__ == "__main__":
    main()
