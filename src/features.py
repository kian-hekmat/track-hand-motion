"""Phase 2 v2: per-frame multi-scale features from the derived signals, and time-scale augmentation.

Everything is computed from the signals of ONE take (no ground truth, no other takes), so it can be
applied to a held-out take. Windows are in grid samples (30 Hz): 3, 8, 20, 45 = 0.1, 0.27, 0.67, 1.5 s.
"""
import numpy as np
import pandas as pd

from config import GRID_HZ
from src import ground_truth as G
from src.signals import SignalParams, derive_signals, load_frames

BASE = ["speed", "aperture", "aperture_slope", "p", "dp", "q", "dq", "ydev", "dy"]
MEAN_WINDOWS = (3, 8, 20, 45)
TREND_CHANNELS = ("aperture", "ydev", "p", "speed")
TREND_WINDOWS = (5, 15, 45)


def _fill(v: np.ndarray) -> np.ndarray:
    fin = np.isfinite(v)
    return np.interp(np.arange(len(v)), np.flatnonzero(fin), v[fin]) if fin.any() else np.zeros(len(v))


def build_features(sig: pd.DataFrame) -> pd.DataFrame:
    """Feature matrix, one row per grid sample, no NaN. Long gaps are interpolated and flagged."""
    ch = {c: _fill(sig[c].to_numpy(float)) for c in BASE}
    ch["speed"] = np.sqrt(np.clip(ch["speed"], 0, None))  # compress spikes
    f = {}
    for c, v in ch.items():
        f[c] = v
        s = pd.Series(v)
        for w in MEAN_WINDOWS:
            f[f"{c}_m{w}"] = s.rolling(2 * w + 1, center=True, min_periods=1).mean().to_numpy()
    for c in TREND_CHANNELS:
        s = pd.Series(ch[c])
        for w in TREND_WINDOWS:
            past = s.rolling(w, min_periods=1).mean()
            fut = s[::-1].rolling(w, min_periods=1).mean()[::-1]
            f[f"{c}_trend{w}"] = (fut - past).to_numpy()
    sp = pd.Series(ch["speed"])
    f["speed_max8"] = sp.rolling(17, center=True, min_periods=1).max().to_numpy()
    f["speed_std8"] = sp.rolling(17, center=True, min_periods=1).std().fillna(0).to_numpy()
    f["ap_min20"] = pd.Series(ch["aperture"]).rolling(41, center=True, min_periods=1).min().to_numpy()
    f["missing"] = sig["missing"].to_numpy(float)
    return pd.DataFrame(f)


def time_scaled_frames(frames: pd.DataFrame, factor: float) -> pd.DataFrame:
    """Same motion performed `factor` times faster (factor 2 = half the duration). Training-set
    augmentation only; ground-truth times scale by 1/factor."""
    out = frames.copy()
    out["t"] = frames["t"] / factor
    return out


def scaled_ground_truth(gt: pd.DataFrame, factor: float) -> pd.DataFrame:
    g = gt.copy()
    g["start_s"], g["end_s"] = gt["start_s"] / factor, gt["end_s"] / factor
    return g


def take_data(take: str, factor: float = 1.0, smooth_win: int = 5):
    """(signals, features, per-sample ground-truth labels, t_end) for a take, optionally time-scaled."""
    frames = time_scaled_frames(load_frames(take), factor)
    t_end = float(frames["t"].iloc[-1])
    sig = derive_signals(frames, SignalParams(smooth_win=smooth_win), t_end=t_end)
    labels = G.labels_at(scaled_ground_truth(G.load_ground_truth(take), factor), sig["t"].to_numpy())
    return sig, build_features(sig), labels, t_end
