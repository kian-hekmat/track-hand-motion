"""Decode the exported Tableau dashboard image (evidence/tableau_dashboard.png) and compare what it draws with the data.

Phase colours are read from the saved workbook (tableau/hand-motion-phases.twbx), not guessed from the picture. Each chart is
found inside a layout box (LAYOUT, in pixels of the 3348 x 1475 export); rows, bars and panels inside a box are detected
automatically. If the dashboard is re-exported with a different arrangement, update LAYOUT.

Every check calibrates its scale on ONE take (or one bar) and treats the others as out-of-sample:
  timeline       band colours and positions vs the detected events (scale from Take 1)
  accuracy_bars  bar lengths vs the scored accuracy (scale from Take 1)
  speed_lines    traced line height vs the real speed signal (time scale from Take 3)
  scatter        where each phase's points sit vs the data (per-phase centre of speed and hand opening). Overlapping hollow
                 circles hide much of the dense low-speed region, so this is a weak check along the speed axis; the scatter's
                 binding to the data is verified from the workbook instead (workbook()).
  workbook       packaged CSVs identical to data/tableau, the "Detected only" data-source filter, fields on each sheet
"""
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
IMG = ROOT / "evidence" / "tableau_dashboard.png"
TWBX = ROOT / "tableau" / "hand-motion-phases.twbx"
TAKES = ["vid1", "vid2", "vid3", "vid4", "vid5"]
TO_LABEL = {"At rest": "REST", "Reaching": "REACH", "Grasping": "GRASP", "Holding": "HOLD", "Releasing": "RELEASE", "Returning": "RETRACT"}

# (x0, y0, x1, y1) boxes in pixels of the 3348 x 1475 export of 2026-10-06
LAYOUT = {
    "size": (3348, 1475),
    "timeline": (1700, 150, 3080, 340),
    "accuracy": (160, 830, 1680, 990),
    "speed": (1812, 505, 3090, 1372),
    "scatter": (100, 75, 1655, 655),
}
LINE_BLUE = np.array([78, 121, 167])          # Tableau's default first colour, used for the speed lines


def phase_colours(twbx: Path = TWBX) -> dict:
    with zipfile.ZipFile(twbx) as z:
        twb = next(n for n in z.namelist() if n.endswith(".twb"))
        text = z.read(twb).decode("utf-8")
    pairs = re.findall(r"<map to='(#[0-9a-fA-F]{6})'>\s*<bucket>&quot;([A-Za-z ]+)&quot;</bucket>", text)
    cols = {}
    for hexc, name in pairs:
        if name in TO_LABEL:
            cols.setdefault(name, np.array([int(hexc[i:i + 2], 16) for i in (1, 3, 5)]))
    assert set(cols) == set(TO_LABEL), f"workbook colour map incomplete: {sorted(cols)}"
    return cols


def _image(path: Path = IMG) -> np.ndarray:
    im = Image.open(path).convert("RGB")
    assert im.size == LAYOUT["size"], f"image size {im.size} differs from the layout {LAYOUT['size']}: update LAYOUT"
    return np.array(im).astype(int)


def _runs(idx, gap=2):
    runs = []
    for v in idx:
        if runs and v - runs[-1][-1] <= gap:
            runs[-1].append(v)
        else:
            runs.append([v])
    return runs


def _nearest(px, cols, tol=30):
    d = {n: np.abs(px - c).sum() for n, c in cols.items()}
    n = min(d, key=d.get)
    return n if d[n] < tol else None


# ------------------------------------------------------------------ timeline
def timeline(path: Path = IMG) -> dict:
    a, cols = _image(path), phase_colours()
    x0, y0, x1, y1 = LAYOUT["timeline"]
    box = a[y0:y1, x0:x1]
    hit = np.zeros(box.shape[:2], bool)
    for c in cols.values():
        hit |= np.abs(box - c).sum(2) < 25
    rows = [r for r in _runs(np.where(hit.sum(1) > 100)[0]) if len(r) > 8]
    assert len(rows) == 5, f"expected 5 timeline rows, found {len(rows)}"
    decoded = []
    for r in rows:
        line = box[(r[0] + r[-1]) // 2]
        labs = [_nearest(px, cols) for px in line]
        segs, start = [], None
        for i, l in enumerate(labs + [None]):
            if start is None and l is not None:
                start = (i, l)
            elif start is not None and l != start[1]:
                segs.append((x0 + start[0], x0 + i, start[1]))
                start = (i, l) if l is not None else None
        decoded.append(segs)
    ev = pd.read_csv(ROOT / "data" / "export" / "events.csv")
    gx0, gx1 = decoded[0][0][0], decoded[0][-1][1]
    pps = (gx1 - gx0) / float(ev[ev["take"] == "vid1"]["end_s"].max())
    res = {"px_per_second": pps}
    for segs, take in zip(decoded, TAKES):
        e = ev[ev["take"] == take].sort_values("event_idx")
        labels = [TO_LABEL[s[2]] for s in segs]
        match = labels == list(e["label"])
        starts = np.array([(s[0] - gx0) / pps for s in segs])
        res[take] = {"n_segments_image": len(segs), "n_events": len(e), "labels_match": match,
                     "max_start_error_s": float(np.abs(starts - e["start_s"].to_numpy()).max()) if match else None,
                     "end_error_s": abs((segs[-1][1] - gx0) / pps - float(e["end_s"].iloc[-1]))}
    return res


# ------------------------------------------------------------------ accuracy bars
def accuracy_bars(path: Path = IMG) -> dict:
    a = _image(path)
    x0, y0, x1, y1 = LAYOUT["accuracy"]
    box = a[y0:y1, x0:x1]
    r, g, b = box[..., 0], box[..., 1], box[..., 2]
    bluish = (b - r > 25) & (box.sum(2) < 690)                 # bar fills (light to dark blue); excludes white, grey and black text
    rows = [rr for rr in _runs(np.where(bluish.sum(1) > 200)[0]) if len(rr) > 6]
    assert len(rows) == 5, f"expected 5 accuracy bars, found {len(rows)}"
    spans = []
    for rr in rows:
        cols = np.where(bluish[rr[0]:rr[-1] + 1].any(0))[0]
        spans.append(cols.max() - cols.min())
    acc = pd.read_csv(ROOT / "data" / "tableau" / "tableau_accuracy.csv").sort_values("take")["percent_frames_matching_human_labels"].to_numpy()
    scale = spans[0] / acc[0]
    dec = np.array(spans) / scale
    return {"decoded_percent": [round(float(v), 1) for v in dec], "actual_percent": acc.tolist(),
            "max_abs_error_points": float(np.abs(dec - acc).max())}


# ------------------------------------------------------------------ speed panels
def speed_lines(path: Path = IMG) -> dict:
    a = _image(path)
    x0, y0, x1, y1 = LAYOUT["speed"]
    box = a[y0:y1, x0:x1]
    m = np.abs(box - LINE_BLUE).sum(2) < 40
    panels = [p for p in _runs(np.where(m.any(1))[0], gap=12) if len(p) > 40]
    assert len(panels) == 5, f"expected 5 speed panels, found {len(panels)}"
    sig = pd.read_csv(ROOT / "data" / "tableau" / "tableau_signals.csv")
    dur = {t: float(sig[sig["take"] == t]["t_s"].max()) for t in TAKES}
    ext = []
    for p in panels:
        xs = np.where(m[p[0]:p[-1] + 1].any(0))[0]
        ext.append((xs.min(), xs.max()))
    sx0 = ext[2][0]
    pps = (ext[2][1] - sx0) / dur["vid3"]
    res = {}
    for k, (p, take) in enumerate(zip(panels, TAKES)):
        g = sig[sig["take"] == take]
        t, sp = g["t_s"].to_numpy(), g["speed"].to_numpy()
        sub = m[p[0]:p[-1] + 1]
        base = np.where(sub.any(1))[0].max()
        hs, sv = [], []
        for x in range(sx0, sx0 + int(pps * t.max())):
            col = np.where(sub[:, x])[0]
            ts = (x - sx0) / pps
            i = min(max(np.searchsorted(t, ts), 1), len(t) - 1)
            if len(col) == 0 or np.isnan(sp[i]) or np.isnan(sp[i - 1]):
                continue
            hs.append(base - col.min())
            sv.append(np.interp(ts, t, sp))
        res[take] = {"correlation": float(np.corrcoef(hs, sv)[0, 1]),
                     "pixels_per_speed_unit": float(np.polyfit(sv, hs, 1)[0]),
                     "start_x_error_px": int(abs(ext[k][0] - sx0)),
                     "end_x_error_s": abs((ext[k][1] - sx0) / pps - dur[take])}
    return res


# ------------------------------------------------------------------ scatter
def scatter(path: Path = IMG) -> dict:
    """Per-phase centre of the coloured outline pixels vs the per-phase mean of the data. Overplotting shifts pixel centres,
    so this checks ordering (rank correlation) across the six phases, not exact positions."""
    a, cols = _image(path), phase_colours()
    x0, y0, x1, y1 = LAYOUT["scatter"]
    box = a[y0:y1, x0:x1]
    sig = pd.read_csv(ROOT / "data" / "tableau" / "tableau_signals.csv").dropna(subset=["speed", "aperture"])
    data = sig.groupby("phase_name")[["speed", "aperture"]].median()
    pix = {}
    for name, c in cols.items():
        yy, xx = np.where(np.abs(box - c).sum(2) < 25)
        pix[name] = (float(np.median(xx)), float(-np.median(yy)), int(len(xx)))
    names = list(TO_LABEL)
    px = pd.DataFrame(pix, index=["x", "y", "n"]).T.loc[names]
    rank = lambda s: s.rank().to_numpy()
    rx = float(np.corrcoef(rank(px["x"]), rank(data.loc[names, "speed"]))[0, 1])
    ry = float(np.corrcoef(rank(px["y"]), rank(data.loc[names, "aperture"]))[0, 1])
    return {"rank_corr_speed": rx, "rank_corr_aperture": ry, "pixels_per_phase": px["n"].astype(int).to_dict(),
            "data_medians": data.round(3).to_dict()}


def workbook(twbx: Path = TWBX) -> dict:
    """What the saved workbook is built from: packaged CSVs (hashed and compared with data/tableau), data-source filters, and the
    fields on each sheet's shelves. Sheets are identified by mark type and fields, not by name."""
    import hashlib
    import xml.etree.ElementTree as ET
    with zipfile.ZipFile(twbx) as z:
        names = z.namelist()
        root = ET.fromstring(z.read(next(n for n in names if n.endswith(".twb"))))
        csv_hash = {n: hashlib.sha256(z.read(n)).hexdigest() for n in names if n.endswith(".csv")}
    local = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in (ROOT / "data" / "tableau").glob("*.csv")}
    packaged = [{"file": Path(n).name, "identical_to_data_tableau": local.get(Path(n).name) == h} for n, h in csv_hash.items()]
    ds_caption = {d.get("name"): d.get("caption") for d in root.find("datasources")}
    ds_filters = {}
    for d in root.find("datasources"):
        for f in d.findall("filter"):
            ds_filters.setdefault(d.get("caption"), []).append(
                (f.get("column"), [g.get("member") for g in f.iter("groupfilter") if g.get("member")]))
    sheets = []
    for w in root.iter("worksheet"):
        t = w.find(".//table")
        text = " ".join(filter(None, [t.findtext("rows"), t.findtext("cols")]))
        fields = sorted(set(re.findall(r"\[(?:none|sum|avg|attr):([a-z_]+):", text)))
        enc = sorted({re.search(r":([a-z_]+):", e.get("column")).group(1) for e in w.iter()
                      if e.tag in ("color", "size", "text") and e.get("column")})
        src = sorted({ds_caption.get(m) for m in re.findall(r"\[(federated\.[a-z0-9]+)\]", text)})
        sheets.append({"name": w.get("name"), "marks": [m.get("class") for m in w.iter("mark")], "shelf_fields": fields,
                       "encodings": enc, "datasources": src})
    dashboards = [d.get("name") for d in root.iter("dashboard") if d.get("name")]
    return {"packaged_csvs": packaged, "datasource_filters": ds_filters, "sheets": sheets, "dashboards": dashboards}


def compare(path: Path = IMG) -> dict:          # kept for backwards compatibility with older callers
    return timeline(path)


if __name__ == "__main__":
    import json
    print(json.dumps({"workbook": workbook(), "timeline": timeline(), "accuracy_bars": accuracy_bars(), "speed_lines": speed_lines(), "scatter": scatter()},
                     indent=2, default=str))
