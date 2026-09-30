"""Per-video settings. Everything the pipeline needs to know that differs
between masters - kept in one place because the last run's worst bugs came
from a constant that was right for one video and wrong for the next."""
from pathlib import Path
ROOT = Path(__file__).parent
SRC = ROOT / "source" / "video3.mp4"
WORK = ROOT / "work3"
SEGIN = ROOT / "segments3" / "input"
SEGOUT = ROOT / "segments3" / "output"
OUT = ROOT / "final" / "FINAL_video3.mp4"
FPS = 30
TOTAL = 3330
NW, NH = 1080, 1920
# video3 is tagged bt709 transfer; video2 was iec61966-2-1. Writing the wrong
# one back shifts the whole picture.
TRC_FF = "bt709"
TRC_H264 = 1          # transfer_characteristics for h264_metadata
SETPARAMS = "setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709:range=tv,"
