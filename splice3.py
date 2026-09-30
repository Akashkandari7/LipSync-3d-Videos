#!/usr/bin/env python3
"""Rebuild video3 with the composited segments in place.

One ffmpeg pass over the concat filter, trimmed straight from the
master. Four things here exist because each one silently broke an
earlier build:

  - the concat *filter*, not the demuxer: the demuxer stitches by
    timestamp and duplicated a frame wherever a run was very short.
  - setpts=N/30/TB on every piece: rebasing alone keeps whatever odd
    spacing sync.so returned (29fps on one clip, 32.3 on another) and
    concat pads the gap with a duplicate.
  - setparams on every piece: concat sees untagged composites beside the
    tagged master and quietly colour-converts them, worth -5.6 Y.
  - no -r / -fps_mode cfr on the output: CFR resampling invents a frame
    to cover any container overhang.

Colour tags are written last with h264_metadata, which rewrites the
header only. video3 is bt709 transfer (video2 was sRGB) - the wrong
value here shifts the whole picture.
"""

import json
import subprocess

import cfg3 as C


def main():
    usable = set(json.loads((C.WORK / "usable.json").read_text()))
    plan = json.loads((C.WORK / "speakers.json").read_text())
    segs = sorted((p["final"][0], p["final"][1], f"s{p['n']:02d}")
                  for p in plan if f"s{p['n']:02d}" in usable)

    inputs = ["-i", str(C.SRC)]
    for _, _, sid in segs:
        c = C.SEGOUT / sid / "composited.mp4"
        inputs += ["-i", str(c if c.exists() else C.SEGOUT / sid / "synced.mp4")]

    fc, labels, cursor = [], [], 0
    for idx, (a, b, sid) in enumerate(segs, start=1):
        if a > cursor:
            fc.append(f"[0:v:0]trim=start_frame={cursor}:end_frame={a},"
                      f"{C.SETPARAMS}setpts=N/30/TB[o{idx}]")
            labels.append(f"o{idx}")
        fc.append(f"[{idx}:v]{C.SETPARAMS}setpts=N/30/TB[s{idx}]")
        labels.append(f"s{idx}")
        cursor = b + 1
    if cursor < C.TOTAL:
        fc.append(f"[0:v:0]trim=start_frame={cursor}:end_frame={C.TOTAL},"
                  f"{C.SETPARAMS}setpts=N/30/TB[oz]")
        labels.append("oz")
    fc.append("".join(f"[{l}]" for l in labels) + f"concat=n={len(labels)}:v=1:a=0[out]")

    C.OUT.parent.mkdir(exist_ok=True)
    subprocess.run(["ffmpeg", "-v", "error", *inputs,
                    "-filter_complex", ";".join(fc),
                    "-map", "[out]", "-map", "0:a:0",
                    "-c:v", "libx264", "-crf", "14", "-preset", "fast",
                    "-pix_fmt", "yuv420p", "-fps_mode", "passthrough",
                    "-c:a", "aac", "-b:a", "192k",
                    "-bsf:v", f"h264_metadata=colour_primaries=1:"
                              f"transfer_characteristics={C.TRC_H264}:"
                              f"matrix_coefficients=1:video_full_range_flag=0",
                    "-movflags", "+faststart", "-y", str(C.OUT)], check=True)
    print(f"{len(segs)} segments spliced into {C.OUT} ({len(labels)} runs)")


if __name__ == "__main__":
    main()
