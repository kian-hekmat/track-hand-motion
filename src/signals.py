"""Phase 2: derive the segmentation signals from raw keypoints.

Channels (all on a uniform GRID_HZ grid, zero-phase smoothed):
  speed     wrist speed from IMAGE x/y (world_* landmarks are hand-centred and do not encode
            translation), converted to pixels with the frame aspect, divided by hand size
            (take-level median wrist->middle-MCP pixel length)  -> hand-lengths / second
  aperture  |thumb tip - index tip| / |wrist - middle MCP|, both in world (metric) coordinates,
            so the ratio is scale- and rotation-invariant
  p, dp     progress of the wrist along the home->far axis (0 at home, ~1 at the far mark) and its
            rate; sign of dp separates REACH from RETRACT. Home/far are found from the take itself
            (no ground truth)
Gaps: runs of <= MAX_INTERP_FRAMES undetected frames are linearly interpolated (flag `interpolated`);
longer runs stay NaN (`missing`). The raw table is never modified.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from config import FRAME_H, FRAME_W, GRID_HZ, MAX_INTERP_FRAMES, RAW_DIR

KEYPOINTS = (0, 4, 8, 9)  # wrist, thumb tip, index tip, middle MCP
HOME_S = 0.5              # take starts at rest at A: home = median wrist position over this window
FAR_QUANTILE = 0.90


@dataclass(frozen=True)
class SignalParams:
    smooth_win: int = 7   # samples (odd); 7 samples at 30 Hz = 0.23 s
    polyorder: int = 2


def load_frames(take: str) -> pd.DataFrame:
    """One row per frame from data/raw/<take>_keypoints.csv (only the 4 landmarks needed)."""
    k = pd.read_csv(RAW_DIR / f"{take}_keypoints.csv")
    k = k[k["keypoint_id"].isin(KEYPOINTS)]
    base = k.drop_duplicates("frame_idx").set_index("frame_idx")[["timestamp_ms", "detected"]]
    out = pd.DataFrame({"t": base["timestamp_ms"] / 1000.0, "detected": base["detected"].astype(bool)})
    for kid in KEYPOINTS:
        sub = k[k["keypoint_id"] == kid].set_index("frame_idx")
        for c in ("x", "y"):
            out[f"i{kid}_{c}"] = sub[c]
        for c in ("x", "y", "z"):
            out[f"w{kid}_{c}"] = sub[f"world_{c}"]
    return out.reset_index(drop=True)


def _fill_short_gaps(t: np.ndarray, v: np.ndarray, max_run: int):
    """Linearly interpolate NaN runs of length <= max_run that have a finite sample on both sides.
    Returns (filled copy, bool mask of filled samples)."""
    v = v.copy()
    filled = np.zeros(len(v), bool)
    bad = ~np.isfinite(v)
    i = 0
    while i < len(v):
        if bad[i]:
            j = i
            while j < len(v) and bad[j]:
                j += 1
            if (j - i) <= max_run and i > 0 and j < len(v):
                v[i:j] = np.interp(t[i:j], [t[i - 1], t[j]], [v[i - 1], v[j]])
                filled[i:j] = True
            i = j
        else:
            i += 1
    return v, filled


def _to_grid(t: np.ndarray, v: np.ndarray, tg: np.ndarray, gap_s: float) -> np.ndarray:
    """Resample finite samples of v(t) onto tg; NaN where the bracketing finite samples are more
    than gap_s apart (a real gap) or the grid point lies outside the finite range."""
    fin = np.isfinite(v)
    tf, vf = t[fin], v[fin]
    out = np.interp(tg, tf, vf)
    j = np.searchsorted(tf, tg, side="left")
    lo = tf[np.clip(j - 1, 0, len(tf) - 1)]
    hi = tf[np.clip(j, 0, len(tf) - 1)]
    exact = np.isin(tg, tf)
    bad = (~exact) & (((hi - lo) > gap_s) | (tg < tf[0]) | (tg > tf[-1]))
    out[bad] = np.nan
    return out


def _fill_for_filter(x: np.ndarray) -> np.ndarray:
    fin = np.isfinite(x)
    return np.interp(np.arange(len(x)), np.flatnonzero(fin), x[fin])


def derive_signals(frames: pd.DataFrame, params: SignalParams = SignalParams(),
                   t_end: float | None = None) -> pd.DataFrame:
    t = frames["t"].to_numpy(float)
    dt = 1.0 / GRID_HZ
    t_end = float(t[-1]) if t_end is None else t_end
    tg = np.arange(t[0], t_end + 1e-9, dt)
    gap_s = (MAX_INTERP_FRAMES + 1.5) * dt
    det = frames["detected"].to_numpy(bool)

    def col(name):
        v = frames[name].to_numpy(float).copy()
        v[~det] = np.nan
        return v

    # --- native-frame quantities (NaN where undetected), then short-gap fill ---
    X, Y = col("i0_x") * FRAME_W, col("i0_y") * FRAME_H
    hand_px = np.hypot((col("i0_x") - col("i9_x")) * FRAME_W, (col("i0_y") - col("i9_y")) * FRAME_H)
    hand_size_px = float(np.nanmedian(hand_px))
    w = {k: np.stack([col(f"w{k}_x"), col(f"w{k}_y"), col(f"w{k}_z")], 1) for k in (0, 4, 8, 9)}
    ap = np.linalg.norm(w[4] - w[8], axis=1) / np.linalg.norm(w[0] - w[9], axis=1)

    filled_any = np.zeros(len(t), bool)
    chans = {}
    for name, v in (("X", X), ("Y", Y), ("ap", ap)):
        v2, f = _fill_short_gaps(t, v, MAX_INTERP_FRAMES)
        chans[name] = v2
        filled_any |= f
    grid = {k: _to_grid(t, v, tg, gap_s) for k, v in chans.items()}
    nearest = np.clip(np.searchsorted(t, tg), 0, len(t) - 1)
    interpolated = filled_any[nearest]

    missing = ~np.isfinite(grid["X"]) | ~np.isfinite(grid["Y"]) | ~np.isfinite(grid["ap"])

    # --- zero-phase smoothing + derivatives on gap-filled series, masked back to NaN afterwards ---
    win, po = params.smooth_win, params.polyorder
    fx, fy, fa = (_fill_for_filter(grid[k]) for k in ("X", "Y", "ap"))
    sg = lambda a, d=0: savgol_filter(a, win, po, deriv=d, delta=dt, mode="interp")
    Xs, Ys, vx, vy = sg(fx), sg(fy), sg(fx, 1), sg(fy, 1)
    speed = np.hypot(vx, vy) / hand_size_px
    aperture, ap_slope = sg(fa), sg(fa, 1)

    # --- progress along the home->far axis, found from the take itself ---
    valid = ~missing
    n_home = max(3, int(HOME_S * GRID_HZ))
    hv = np.flatnonzero(valid)[:n_home]
    home = np.array([np.median(Xs[hv]), np.median(Ys[hv])])
    pos = np.stack([Xs, Ys], 1)
    dist = np.linalg.norm(pos - home, axis=1)
    far_sel = valid & (dist >= np.quantile(dist[valid], FAR_QUANTILE))
    far = pos[far_sel].mean(0)
    L = float(np.linalg.norm(far - home))
    if L < 1e-6:
        p, dp, q, dq = (np.zeros_like(Xs) for _ in range(4))
    else:
        u = (far - home) / L
        n = np.array([-u[1], u[0]])                      # perpendicular to the home->far axis
        p = (pos - home) @ u / L
        dp = (vx * u[0] + vy * u[1]) / L
        q = (pos - home) @ n / hand_size_px              # off-axis offset, hand-lengths
        dq = (vx * n[0] + vy * n[1]) / hand_size_px
    ydev = -(Ys - home[1]) / hand_size_px                # height above home in the image (up = +)
    dy = -vy / hand_size_px                              # vertical velocity (up = +), hand-lengths/s

    out = pd.DataFrame({"t": tg, "speed": speed, "aperture": aperture, "aperture_slope": ap_slope,
                        "p": p, "dp": dp, "q": q, "dq": dq, "ydev": ydev, "dy": dy, "missing": missing, "interpolated": interpolated & ~missing})
    for c in ("speed", "aperture", "aperture_slope", "p", "dp", "q", "dq", "ydev", "dy"):
        out.loc[missing, c] = np.nan
    out.attrs.update(hand_size_px=hand_size_px, home=home.tolist(), far=far.tolist(), axis_len_px=L)
    return out
