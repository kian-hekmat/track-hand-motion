"""Decode the exported Tableau dashboard image (evidence/tableau_dashboard.png) and compare the coloured timeline bands with the
actual detected events. Layout assumptions (valid for the saved export; re-check if the dashboard is re-exported with a different layout):
legend swatches at the right of the timeline, five Gantt rows, one per take in order Take 1..5.

Time scale is calibrated from Take 1 only (first and last coloured pixel = 0 and its duration); the other four takes are then a
genuine out-of-sample check of the scale and the band positions.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
IMG = ROOT / "evidence" / "tableau_dashboard.png"
NAMES = ["At rest", "Grasping", "Holding", "Reaching", "Releasing", "Returning"]
LEGEND_Y = [95, 113, 131, 149, 167, 185]          # legend swatch rows, in the 2000-px-wide display of the 3348-px image
LEGEND_X = 1862
TO_LABEL = {"At rest": "REST", "Reaching": "REACH", "Grasping": "GRASP", "Holding": "HOLD", "Releasing": "RELEASE", "Returning": "RETRACT"}
S = 3348 / 2000


def decode(path: Path = IMG):
    a = np.array(Image.open(path).convert("RGB")).astype(int)
    leg = {n: a[int(y * S), int(LEGEND_X * S)] for n, y in zip(NAMES, LEGEND_Y)}
    x_lo, x_hi = int(1000 * S), int(1850 * S)
    # five rows: find vertical runs of non-white at the first bar column
    col = a[:, int(1030 * S)]
    ys = [y for y in range(int(90 * S), int(230 * S)) if np.abs(col[y] - 255).sum() > 60]
    rows, cur = [], []
    for y in ys:
        if cur and y - cur[-1] > 2:
            rows.append(cur); cur = []
        cur.append(y)
    rows.append(cur)
    out = []
    for r in rows:
        yc = (r[0] + r[-1]) // 2
        line = a[yc, x_lo:x_hi]
        labs = []
        for px in line:
            d = {n: np.abs(px - c).sum() for n, c in leg.items()}
            n = min(d, key=d.get)
            labs.append(n if d[n] < 45 else None)
        segs, start = [], None
        for i, l in enumerate(labs + [None]):
            if start is None and l is not None:
                start = (i, l)
            elif start is not None and l != start[1]:
                segs.append((x_lo + start[0], x_lo + i, start[1]))
                start = (i, l) if l is not None else None
        out.append(segs)
    return out


def compare(path: Path = IMG) -> dict:
    ev = pd.read_csv(ROOT / "data" / "export" / "events.csv")
    rows = decode(path)
    assert len(rows) == 5, f"expected 5 timeline rows, found {len(rows)}"
    take1 = rows[0]
    x0, x1 = take1[0][0], take1[-1][1]
    d1 = float(ev[ev["take"] == "vid1"]["end_s"].max())
    px_per_s = (x1 - x0) / d1
    res = {"px_per_second": px_per_s}
    for i, take in enumerate(["vid1", "vid2", "vid3", "vid4", "vid5"]):
        segs = rows[i]
        e = ev[ev["take"] == take].sort_values("event_idx")
        labels_img = [TO_LABEL[s[2]] for s in segs]
        match = labels_img == list(e["label"])
        starts_img = [(s[0] - x0) / px_per_s for s in segs]
        err = np.abs(np.array(starts_img) - e["start_s"].to_numpy()) if match else None
        end_img = (segs[-1][1] - x0) / px_per_s
        res[take] = {"n_segments_image": len(segs), "n_events": len(e), "labels_match": match,
                     "max_start_error_s": float(err.max()) if err is not None else None,
                     "end_error_s": abs(end_img - float(e["end_s"].iloc[-1]))}
    return res


def _mask(path: Path):
    a = np.array(Image.open(path).convert("RGB")).astype(int)
    return a, (np.abs(a - np.array([78, 121, 167])).sum(2) < 40)     # the single blue used for the speed lines and accuracy bars


def accuracy_bars(path: Path = IMG) -> dict:
    """Decode the five accuracy bars; scale calibrated on Take 1 only."""
    _, m = _mask(path)
    sub = m[int(500 * S):int(620 * S), int(1000 * S):int(1850 * S)]
    runs = []
    for y in np.where(sub.any(1))[0]:
        if runs and y - runs[-1][-1] <= 2:
            runs[-1].append(y)
        else:
            runs.append([y])
    assert len(runs) == 5, f"expected 5 accuracy bars, found {len(runs)}"
    spans = []
    for r in runs:
        xs = np.where(sub[(r[0] + r[-1]) // 2])[0]
        spans.append(xs.max() - xs.min())
    acc = pd.read_csv(ROOT / "data" / "tableau" / "tableau_accuracy.csv").sort_values("take")["percent_frames_matching_human_labels"].to_numpy()
    scale = spans[0] / acc[0]
    dec = np.array(spans) / scale
    return {"decoded_percent": [round(float(x), 1) for x in dec], "actual_percent": acc.tolist(), "max_abs_error_points": float(np.abs(dec - acc).max())}


def speed_lines(path: Path = IMG) -> dict:
    """Trace each speed panel and correlate line height with the real speed signal. x scale is calibrated on Take 3 only
    (its line starts at t = 0 and ends at its duration); Takes 1, 2, 4, 5 end where that scale predicts."""
    _, m = _mask(path)
    sig = pd.read_csv(ROOT / "data" / "tableau" / "tableau_signals.csv")
    ends, starts, res = {}, {}, {}
    panels = [(105, 245), (247, 388), (390, 531), (533, 673), (677, 817)]
    for k, (y0, y1) in enumerate(panels):
        sub = m[int(y0 * S):int(y1 * S), int(130 * S):int(920 * S)]
        xs = np.where(sub.any(0))[0]
        starts[k], ends[k] = xs.min() + int(130 * S), xs.max() + int(130 * S)
    x0 = starts[2]
    dur = {k: float(sig[sig["take"] == f"vid{k + 1}"]["t_s"].max()) for k in range(5)}
    ppx = (ends[2] - x0) / dur[2]
    for k, (y0, y1) in enumerate(panels):
        g = sig[sig["take"] == f"vid{k + 1}"]
        t, sp = g["t_s"].to_numpy(), g["speed"].to_numpy()
        sub = m[int(y0 * S):int(y1 * S), :]
        base = np.where(sub[:, 200:].any(1))[0].max()
        hs, sv = [], []
        for x in range(x0, x0 + int(ppx * t.max())):
            col = np.where(sub[:, x])[0]
            ts = (x - x0) / ppx
            i = min(max(np.searchsorted(t, ts), 1), len(t) - 1)
            if len(col) == 0 or np.isnan(sp[i]) or np.isnan(sp[i - 1]):
                continue
            hs.append(base - col.min()); sv.append(np.interp(ts, t, sp))
        r = float(np.corrcoef(hs, sv)[0, 1])
        res[f"vid{k + 1}"] = {"correlation": r, "pixels_per_speed_unit": float(np.polyfit(sv, hs, 1)[0]),
                              "start_x_error_px": int(abs(starts[k] - x0)), "end_x_error_s": abs((ends[k] - x0) / ppx - dur[k])}
    return res


if __name__ == "__main__":
    import json
    print(json.dumps({"timeline": compare(), "accuracy_bars": accuracy_bars(), "speed_lines": speed_lines()}, indent=2))
