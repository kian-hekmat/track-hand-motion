import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import RAW_DIR, VIDS_DIR

TAKES = sorted(p.stem for p in VIDS_DIR.glob("*.mov"))


def _source_pts_seconds(take: str) -> np.ndarray:
    """Per-frame presentation timestamps of the ORIGINAL iPhone .mov, from ffmpeg."""
    err = subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-i", str(VIDS_DIR / f"{take}.mov"),
         "-map", "0:v:0", "-vf", "showinfo", "-f", "null", "-"],
        capture_output=True, text=True, check=True,
    ).stderr
    return np.array([float(tok.split(":")[1]) for tok in err.split()
                     if tok.startswith("pts_time:")])


@pytest.fixture(scope="session", params=TAKES)
def take(request):
    return request.param


@pytest.fixture(scope="session")
def keypoints():
    cache = {}

    def load(take):
        if take not in cache:
            path = RAW_DIR / f"{take}_keypoints.csv"
            if not path.exists():
                pytest.fail(f"{path} missing: run scripts/extract_take.py {take}")
            cache[take] = pd.read_csv(path)
        return cache[take]
    return load


@pytest.fixture(scope="session")
def source_pts():
    cache = {}

    def load(take):
        if take not in cache:
            cache[take] = _source_pts_seconds(take)
        return cache[take]
    return load
