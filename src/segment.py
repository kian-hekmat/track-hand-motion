"""Phase 2: segmentation (ruptures PELT, L2) + rule-based labelling.

Approach (committed, see docs/phase2_roadmap.md D1-D5):
  1. Boundaries: PELT with an L2 (piecewise-constant mean) cost on two channels, sqrt(speed) and
     aperture, each divided by a fixed physical scale. Penalty is one fixed number (`pen`), frozen after
     tuning on TUNING_TAKES only. NaN gaps are filled by interpolation for this step only.
  2. Labels: each segment gets a label from its own features (speed level, position p along the
     home->far axis, direction dp, aperture level and trend). The expected cycle order is NOT used.
     The only context used is the previous segment's label, for two ambiguous cases (see _label).
  3. Adjacent segments with the same label are merged (a hesitation inside REACH becomes one REACH).
"""
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import ruptures as rpt

import json

from config import GRID_HZ, SEGMENTS_DIR
from src.signals import SignalParams, derive_signals, load_frames


@dataclass(frozen=True)
class SegParams:
    # signal
    smooth_win: int = 7
    # changepoint
    pen: float = 1.0           # PELT penalty (L2 cost summed over both scaled channels)
    speed_scale: float = 1.0   # sqrt(speed) is divided by this
    ap_scale: float = 0.5      # aperture is divided by this
    min_size_s: float = 0.1    # minimum segment length (3 frames)
    # labelling (speed in hand-lengths/s, aperture in thumb-index / hand-size)
    still_thr: float = 0.35    # below: still (REST / HOLD)
    moving_thr: float = 1.0    # above: travelling (REACH / RETRACT)
    p_home: float = 0.25       # p below this: near home
    p_far: float = 0.6         # p above this: at the object
    ap_closed: float = 0.9     # mean aperture below this: fingers closed
    ap_delta: float = 0.12     # aperture change over a segment that counts as closing/opening
    dp_min: float = 0.05       # net progress change that counts as direction

    def to_dict(self):
        return asdict(self)


PARAMS_FILE = SEGMENTS_DIR / "params.json"


def load_params() -> SegParams:
    """Frozen parameters written by scripts/tune_segmentation.py (defaults if not tuned yet)."""
    if PARAMS_FILE.exists():
        return SegParams(**json.loads(PARAMS_FILE.read_text())["params"])
    return SegParams()


def _scaled_channels(sig: pd.DataFrame, params: "SegParams") -> np.ndarray:
    """sqrt(speed)/speed_scale and aperture/ap_scale. Fixed physical scales (speed is already in
    hand-lengths/s and aperture is a hand-size ratio), not per-take range scaling: range scaling
    would amplify noise on nearly flat signals."""
    speed = _interp_nan(np.sqrt(np.clip(sig["speed"].to_numpy(float), 0, None)))
    ap = _interp_nan(sig["aperture"].to_numpy(float))
    return np.stack([speed / params.speed_scale, ap / params.ap_scale], 1)


def _interp_nan(v: np.ndarray) -> np.ndarray:
    fin = np.isfinite(v)
    if fin.all():
        return v
    return np.interp(np.arange(len(v)), np.flatnonzero(fin), v[fin])


def find_boundaries(sig: pd.DataFrame, params: SegParams) -> list[int]:
    """Sample indices where a new segment starts (excluding 0)."""
    x = _scaled_channels(sig, params)
    algo = rpt.Pelt(model="l2", min_size=max(2, int(round(params.min_size_s * GRID_HZ))), jump=1).fit(x)
    return [int(i) for i in algo.predict(pen=params.pen) if i < len(x)]


def _seg_features(sig: pd.DataFrame, a: int, b: int) -> dict:
    s = sig.iloc[a:b]
    k = min(3, len(s))
    f = lambda c: _interp_nan(sig[c].to_numpy(float))[a:b]
    p, ap, sp = f("p"), f("aperture"), f("speed")
    return {
        "mean_speed": float(np.mean(sp)), "mean_aperture": float(np.mean(ap)),
        "p_mean": float(np.mean(p)), "dp": float(np.median(p[-k:]) - np.median(p[:k])),
        "d_ap": float(np.median(ap[-k:]) - np.median(ap[:k])),
        "frac_missing": float(s["missing"].mean()),
    }


def _label(ft: dict, prev: str | None, P: SegParams) -> str:
    sp, p, dp, ap, dap = ft["mean_speed"], ft["p_mean"], ft["dp"], ft["mean_aperture"], ft["d_ap"]
    if sp > P.moving_thr:                                   # travelling
        return "REACH" if dp >= 0 else "RETRACT"
    if p < P.p_home:                                        # near home, not travelling
        if sp < P.still_thr:
            return "REST"
        return "RETRACT" if dp < -P.dp_min else ("REACH" if dp > P.dp_min else "REST")
    if p > P.p_far:                                         # at the object
        closed = ap < P.ap_closed
        if dap < -P.ap_delta:
            return "GRASP"
        if dap > P.ap_delta:
            return "RELEASE"
        if closed:
            return "HOLD" if sp < P.still_thr else "GRASP"  # closed + slowly moving = lifting
        # open and stable at the object: tail of REACH, or the rest of RELEASE if it follows one
        return "RELEASE" if prev in ("HOLD", "RELEASE") else "REACH"
    # mid-way (neither home nor object): a pause in transit keeps the previous travel label
    if prev in ("REACH", "RETRACT"):
        return prev
    return "RETRACT" if dp < -P.dp_min else "REACH"


def label_segments(sig: pd.DataFrame, bounds: list[int], params: SegParams, t_end: float) -> pd.DataFrame:
    edges = [0] + list(bounds) + [len(sig)]
    labels, prev = [], None
    for a, b in zip(edges[:-1], edges[1:]):
        lab = _label(_seg_features(sig, a, b), prev, params)
        labels.append(lab)
        prev = lab
    # merge adjacent equal labels
    merged = [[labels[0], edges[0], edges[1]]]
    for lab, a, b in zip(labels[1:], edges[1:-1], edges[2:]):
        if lab == merged[-1][0]:
            merged[-1][2] = b
        else:
            merged.append([lab, a, b])
    t = sig["t"].to_numpy(float)
    rows = []
    for i, (lab, a, b) in enumerate(merged):
        start = float(t[a])
        end = float(t[b]) if b < len(t) else float(t_end)
        ft = _seg_features(sig, a, b)
        rows.append({"event_idx": i, "label": lab, "start_s": start, "end_s": end,
                     "duration_s": end - start, "n_samples": b - a,
                     "mean_speed": ft["mean_speed"], "mean_aperture": ft["mean_aperture"],
                     "frac_missing": ft["frac_missing"]})
    return pd.DataFrame(rows)


def segment_signals(sig: pd.DataFrame, params: SegParams, t_end: float) -> pd.DataFrame:
    return label_segments(sig, find_boundaries(sig, params), params, t_end)


def segment_take(take: str, params: SegParams):
    """Full Phase 2 run for one take. Returns (events, signals)."""
    frames = load_frames(take)
    t_end = float(frames["t"].iloc[-1])
    sig = derive_signals(frames, SignalParams(smooth_win=params.smooth_win), t_end=t_end)
    ev = segment_signals(sig, params, t_end)
    ev.insert(0, "take", take)
    return ev, sig
