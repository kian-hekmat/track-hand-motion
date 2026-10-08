"""Cloud path M2: run the bronze -> silver -> gold chain and compare every table with the verified local reference.

Used by the Databricks notebook (databricks/cloud/m2_build_tables.py) and by tests/test_cloud_m2.py, so both apply the same
checks. Tolerances are fixed here before any cloud run. M1 measured floating-point differences of at most 1.1e-13 (signals)
and 3.5e-18 (probabilities) between environments; anything discrete (labels, counts, event boundaries) must match exactly.
"""
import json

import numpy as np
import pandas as pd

from config import HOLDOUT_TAKES, LABELS, RAW_DIR, ROOT, TAKE_GROUPS
from src import ground_truth as G
from src.cloud import tables as C
from src.final import load_model_for_take
from src.signals import load_frames

TAKES = sorted(TAKE_GROUPS) + sorted(HOLDOUT_TAKES)
REF_DIR = ROOT / "data" / "cloud_reference"
FRAME_TOL = 1e-12        # frames are copied values; any difference comes only from CSV number parsing
SIGNAL_TOL = 1e-9        # signals and probabilities
EVENT_MEAN_TOL = 1e-12   # mean_confidence: an average, summed in a different order by Spark
SCORE_TOL = 1e-12        # accuracies are ratios of counts


def build(spark, landing: str, takes=TAKES) -> dict:
    """All M2 tables as Spark DataFrames. Models are loaded and hash-checked here, on the driver."""
    bronze = C.read_landing(spark, landing)
    frames = C.build_frames(bronze["raw_keypoints"]).filter(C.F.col("take").isin(list(takes)))
    models, meta = {}, None
    for take in takes:
        model, meta, name = load_model_for_take(take)
        models[take] = (model, name)
    signals = C.predict_signals(frames, models, meta)
    events = C.build_events(signals, frames)
    frame_labels = C.label_frames(frames, events, bronze["ground_truth"], bronze["out_of_frame_intervals"])
    return {**{f"bronze.{k}": v for k, v in bronze.items()}, "silver.frames": frames, "silver.signals": signals,
            "gold.events": events, "gold.frame_labels": frame_labels, "gold.frame_scores": C.frame_scores(frame_labels)}


def _max_abs_diff(a: pd.Series, b: pd.Series):
    x, y = a.to_numpy(dtype=float), b.to_numpy(dtype=float)
    if not np.array_equal(np.isnan(x), np.isnan(y)):
        return float("inf")
    return float(np.nanmax(np.abs(x - y))) if (~np.isnan(x)).any() else 0.0


def compare(t: dict, takes=TAKES) -> list:
    """t: the tables as pandas DataFrames (same keys as build()). Returns [(check, status, detail)]."""
    res = []

    def rec(check, ok, detail):
        res.append((check, "PASS" if ok else "FAIL", detail))

    manifest = json.loads((REF_DIR / "manifest.json").read_text())
    meta = t["bronze.take_meta"].set_index("take")
    raw_counts = t["bronze.raw_keypoints"].groupby("take").size()

    # bronze: complete uploads
    for take in takes:
        n_frames = int(meta.loc[take, "frames_extracted"])
        rec(f"bronze {take}: raw keypoint rows", raw_counts.get(take, 0) == 21 * n_frames,
            f"{raw_counts.get(take, 0)} rows = 21 x {n_frames} frames" if raw_counts.get(take, 0) == 21 * n_frames
            else f"{raw_counts.get(take, 0)} rows, expected 21 x {n_frames}")
    gt_local = pd.concat([G.load_ground_truth(k) for k in takes])
    gt_cloud = t["bronze.ground_truth"]
    rec("bronze: ground truth rows", len(gt_cloud[gt_cloud["take"].isin(takes)]) == len(gt_local),
        f"{len(gt_cloud[gt_cloud['take'].isin(takes)])} uploaded vs {len(gt_local)} in the repo")
    oof_local = pd.read_csv(RAW_DIR / "out_of_frame_intervals.csv")
    rec("bronze: out-of-frame intervals", len(t["bronze.out_of_frame_intervals"]) == len(oof_local),
        f"{len(t['bronze.out_of_frame_intervals'])} uploaded vs {len(oof_local)} in the repo")

    frames, sig, ev, sc = t["silver.frames"], t["silver.signals"], t["gold.events"], t["gold.frame_scores"]
    for take in takes:
        # silver.frames vs src.signals.load_frames on the repo's copy of the same file
        f = frames[frames["take"] == take].sort_values("frame_idx").reset_index(drop=True)
        ref_f = load_frames(take)
        same_rows = len(f) == len(ref_f) == int(meta.loc[take, "frames_extracted"])
        d = max(_max_abs_diff(f[c], ref_f[c]) for c in C.FRAME_COLUMNS if c != "detected") if same_rows else float("inf")
        det = same_rows and f["detected"].astype(bool).tolist() == ref_f["detected"].tolist()
        rec(f"silver {take}: frames", same_rows and det and d <= FRAME_TOL,
            f"{len(f)} frames; max abs difference {d:.3g} (tolerance {FRAME_TOL:g}); detected flags {'identical' if det else 'DIFFER'}")

        # silver.signals vs reference signals, probabilities and per-sample labels
        s = sig[sig["take"] == take].sort_values("sample_idx").reset_index(drop=True)
        ref_s = pd.read_csv(REF_DIR / f"{take}_signals.csv", float_precision="round_trip")
        ref_p = pd.read_csv(REF_DIR / f"{take}_posteriors.csv", float_precision="round_trip")
        ref_e = pd.read_csv(REF_DIR / f"{take}_events.csv", float_precision="round_trip")
        if len(s) != len(ref_s):
            rec(f"silver {take}: signals", False, f"{len(s)} samples vs {len(ref_s)} in the reference")
        else:
            ds = max(_max_abs_diff(s[c], ref_s[c]) for c in C.SIGNAL_COLUMNS)
            dp = max(_max_abs_diff(s[c], ref_p[c]) for c in C.PROB_COLUMNS)
            flags = all(s[c].astype(bool).tolist() == ref_s[c].astype(bool).tolist() for c in C.FLAG_COLUMNS)
            labels = s["predicted_label"].tolist() == list(G.labels_at(ref_e, ref_s["t"].to_numpy()))
            model_ok = set(s["model"]) == {manifest["takes"][take]["model"]}
            rec(f"silver {take}: signals", ds <= SIGNAL_TOL and dp <= SIGNAL_TOL and flags and labels and model_ok,
                f"{len(s)} samples; signals max diff {ds:.3g}, probabilities {dp:.3g} (tolerance {SIGNAL_TOL:g}); "
                f"flags {'identical' if flags else 'DIFFER'}; labels {'identical' if labels else 'DIFFER'}; "
                f"model {s['model'].iloc[0]}{'' if model_ok else ' (WRONG MODEL)'}")

        # gold.events: built by Spark (gaps-and-islands) vs the events decoded in Python
        e = ev[ev["take"] == take].sort_values("event_idx").reset_index(drop=True)
        exact = (len(e) == len(ref_e) and e["label"].tolist() == ref_e["label"].tolist()
                 and e["event_idx"].tolist() == ref_e["event_idx"].tolist()
                 and all(np.array_equal(e[c].to_numpy(float), ref_e[c].to_numpy(float)) for c in ("start_s", "end_s", "duration_s"))
                 and e["n_samples"].tolist() == ref_e["n_samples"].tolist() and (e["labels_in_event"] == 1).all())
        dm = _max_abs_diff(e["mean_confidence"], ref_e["mean_confidence"]) if len(e) == len(ref_e) else float("inf")
        rec(f"gold {take}: events", exact and dm <= EVENT_MEAN_TOL,
            f"{len(e)} events vs {len(ref_e)}; label/start/end/duration/samples {'identical' if exact else 'DIFFER'}; "
            f"mean_confidence max diff {dm:.3g} (tolerance {EVENT_MEAN_TOL:g})")

    # gold.frame_scores vs src.evaluate.score_take
    ref_sc = pd.read_csv(REF_DIR / "scores_frame.csv", float_precision="round_trip")
    ref_sc = ref_sc[ref_sc["take"].isin(takes)].sort_values(["take", "scope"]).reset_index(drop=True)
    c_sc = sc.sort_values(["take", "scope"]).reset_index(drop=True)
    keys = c_sc[["take", "scope"]].values.tolist() == ref_sc[["take", "scope"]].values.tolist()
    counts = keys and all(c_sc[c].astype(int).tolist() == ref_sc[c].astype(int).tolist()
                          for c in ["n_frames", "n_unlabelled"] + [f"n_{l}" for l in LABELS])
    rates = ["frame_accuracy", "balanced_accuracy", "majority_baseline"] + [f"acc_{l}" for l in LABELS]
    dr = max(_max_abs_diff(c_sc[c], ref_sc[c]) for c in rates) if keys else float("inf")
    rec("gold: frame scores", counts and dr <= SCORE_TOL,
        f"{len(c_sc)} rows (take x scope) vs {len(ref_sc)}; counts {'identical' if counts else 'DIFFER'}; "
        f"accuracies max diff {dr:.3g} (tolerance {SCORE_TOL:g})")
    return res
