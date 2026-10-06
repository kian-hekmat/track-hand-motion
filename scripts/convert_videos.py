"""Convert iPhone HEVC/HDR .mov takes to H.264 SDR .mp4.

- zscale+tonemap maps the HLG/BT.2020 HDR signal to BT.709 SDR so colors aren't washed out.
- -fps_mode passthrough keeps the original (variable) frame timestamps; no frames are
  duplicated or dropped to force a constant 30 fps. -enc_time_base demux plus a 600
  track timescale stop the encoder quantizing those timestamps onto a 1/30 s grid.
"""
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import CONVERTED_DIR, VIDS_DIR

VF = (
    "zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,"
    "tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv,format=yuv420p"
)


def source_timescale(src: Path) -> int:
    """The video stream's time base (ticks per second) as ffmpeg reports it ('600 tbn', '90k tbn')."""
    err = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-i", str(src)], capture_output=True, text=True).stderr
    line = next(l for l in err.splitlines() if "Video:" in l)
    tok = line.split(" tbn")[0].split()[-1]
    return int(float(tok[:-1]) * 1000) if tok.endswith("k") else int(tok)


def convert(src: Path, dst: Path) -> None:
    # Keep the source's own time base so the original (variable) frame times survive exactly. vid1-5 and vid7 use 600;
    # vid6 uses 90000, and forcing 600 on it rounded its timestamps by up to 1.2 ms (caught by tests/test_extraction.py).
    timescale = source_timescale(src)
    cmd = [
        imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(src), "-vf", VF, "-fps_mode", "passthrough", "-enc_time_base", "demux",
        "-c:v", "libx264", "-crf", "18", "-preset", "medium",
        "-video_track_timescale", str(timescale),  # source time base; keeps exact VFR pts
        "-c:a", "aac", "-b:a", "160k", str(dst),
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    CONVERTED_DIR.mkdir(parents=True, exist_ok=True)
    for src in sorted(VIDS_DIR.glob("*.mov")):
        dst = CONVERTED_DIR / f"{src.stem}.mp4"
        print(f"{src.name} -> {dst.relative_to(CONVERTED_DIR.parents[1])}")
        convert(src, dst)
