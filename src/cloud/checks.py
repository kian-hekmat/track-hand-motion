"""Cloud path M2: run the bronze -> silver -> gold chain and compare every table with the verified local reference.

Used by the Databricks notebook (databricks/cloud/m2_build_tables.py) and by tests/test_cloud_m2.py, so both apply the same
checks. Tolerances are fixed here before any cloud run. M1 measured floating-point differences of at most 1.1e-13 (signals)
and 3.5e-18 (probabilities) between environments; anything discrete (labels, counts, event boundaries) must match exactly.
"""
import json

import numpy as np
import pandas as pd

from config import ALL_GROUPS, HOLDOUT_DIR, HOLDOUT_TAKES, LABELS, RAW_DIR, ROOT, TAKE_GROUPS
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
            "gold.events": events, "gold.frame_labels": frame_labels, "gold.frame_scores": C.frame_scores(frame_labels),
            "gold.scores": C.boundary_scores(events, bronze["ground_truth"], frames, bronze["out_of_frame_intervals"])}


def build_serving(t: dict) -> dict:
    """The PIPELINE-layout tables for Snowflake, from the M2 tables (Spark DataFrames, keys as in build())."""
    return {
        "takes": C.serve_takes(t["bronze.take_meta"], t["bronze.ground_truth"], t["gold.events"], ALL_GROUPS),
        "events": C.serve_events(t["gold.events"], t["silver.frames"], t["silver.signals"], ALL_GROUPS),
        "ground_truth": C.serve_ground_truth(t["bronze.ground_truth"]),
        "signals": C.serve_signals(t["silver.signals"], t["bronze.ground_truth"]),
        "frames": C.serve_frames(t["silver.frames"]),
        "scores": t["gold.scores"],
        "raw_keypoints": C.serve_raw_keypoints(t["bronze.raw_keypoints"]),
    }


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


# ---------------- M2b: full scores ----------------
def compare_scores(cloud: pd.DataFrame, takes=TAKES) -> list:
    ref = pd.read_csv(REF_DIR / "scores_full.csv", float_precision="round_trip")
    ref = ref[ref["take"].isin(takes)].sort_values(["take", "scope"]).reset_index(drop=True)
    c = cloud.sort_values(["take", "scope"]).reset_index(drop=True)
    keys = c[["take", "scope"]].values.tolist() == ref[["take", "scope"]].values.tolist()
    text = keys and c["take_group"].tolist() == ref["take_group"].tolist()
    d = max(_max_abs_diff(c[col], ref[col]) for col in C.SCORES_COLUMNS if col not in ("take", "take_group", "scope")) if keys else float("inf")
    return [("gold: scores (boundaries, timing, chance, per cycle)", "PASS" if text and d <= SCORE_TOL else "FAIL",
             f"{len(c)} rows (take x scope incl. cycles) vs {len(ref)}; keys {'identical' if keys else 'DIFFER'}; "
             f"max diff {d:.3g} (tolerance {SCORE_TOL:g}; NaN must match NaN)")]


# ---------------- M3: serving tables vs the verified exports (the data behind Snowflake PIPELINE) ----------------
SERVING_KEYS = {"takes": ["take"], "events": ["take", "event_idx"], "ground_truth": ["take", "_pos"],
                "signals": ["take", "_pos"], "frames": ["take", "frame_idx"], "scores": ["take", "scope"],
                "raw_keypoints": ["take", "frame_idx", "keypoint_id"]}
# tables keyed by a time are matched by position in time order within the take (a time can differ in its last bit)
POSITION_BY = {"ground_truth": "start_s", "signals": "t_s"}
# tolerance per column; columns not listed must match exactly
SERVING_TOL = {
    "takes": {"duration_s": FRAME_TOL, "mean_fps": FRAME_TOL, "fraction_not_detected": FRAME_TOL},
    # start/end/duration: the verified export was written after a default-precision CSV read, so it carries last-bit
    # rounding (up to 3.6e-15); gold.events itself matches the reference exactly (checked in compare()).
    "events": {"start_s": FRAME_TOL, "end_s": FRAME_TOL, "duration_s": FRAME_TOL, "mean_speed": SIGNAL_TOL, "mean_aperture": SIGNAL_TOL, "frac_missing": SCORE_TOL, "mean_confidence": EVENT_MEAN_TOL},
    "ground_truth": {"start_s": FRAME_TOL, "end_s": FRAME_TOL},
    "signals": {c: SIGNAL_TOL for c in ("t_s", "speed", "aperture", "aperture_slope", "progress", "progress_rate", "offaxis",
                                        "offaxis_rate", "height", "vertical_velocity")},
    "frames": {c: FRAME_TOL for c in ("wrist_x", "wrist_y", "thumb_tip_x", "thumb_tip_y", "index_tip_x", "index_tip_y",
                                      "handedness_score")},
    "scores": {c: SCORE_TOL for c in C.SCORES_COLUMNS if c not in ("take", "take_group", "scope")},
    "raw_keypoints": {c: FRAME_TOL for c in ("x", "y", "z", "world_x", "world_y", "world_z", "handedness_score")},
}
SKIP = {"events": {"model_version"}}  # verified export says v2_frozen_oof / v2_frozen_holdout; cloud records the model file


def verified_exports() -> dict:
    """The verified tables behind Snowflake PIPELINE (vid1-5, data/export) plus the hold-out export (vid6-7)."""
    out = {}
    for name in SERVING_KEYS:
        parts = []
        for d in (ROOT / "data" / "export", HOLDOUT_DIR / "export"):
            f = d / (f"{name}.parquet" if name == "raw_keypoints" else f"{name}.csv")
            parts.append(pd.read_parquet(f) if f.suffix == ".parquet" else pd.read_csv(f, float_precision="round_trip"))
        out[name] = pd.concat(parts, ignore_index=True).rename(columns={"group": "take_group", "false": "false_boundaries"})
    return out


def compare_serving(cloud: dict, exports: dict | None = None) -> list:
    """Every verified row must exist in the cloud table with equal values (scores: the cloud also has per-cycle rows for
    takes whose export had none; those extra rows are counted and checked against the reference in compare_scores)."""
    exports = exports or verified_exports()
    res = []
    for name, keys in SERVING_KEYS.items():
        ref, c = exports[name], cloud[name]
        if name in POSITION_BY:
            ref, c = (df.sort_values(["take", POSITION_BY[name]]).assign(_pos=lambda d: d.groupby("take").cumcount())
                      for df in (ref, c))
        cols = [col for col in ref.columns if col not in SKIP.get(name, set())]
        missing_cols = [col for col in cols if col not in c.columns]
        if missing_cols:
            res.append((f"serving {name}", "FAIL", f"cloud table lacks columns {missing_cols}"))
            continue
        m = ref[cols].merge(c[cols], on=keys, how="left", suffixes=("_ref", "_cloud"), indicator=True)
        not_found = int((m["_merge"] != "both").sum())
        extra = len(c) - (len(m) - not_found)
        worst, bad = 0.0, []
        for col in cols:
            if col in keys:
                continue
            a, b = m[f"{col}_ref"], m[f"{col}_cloud"]
            tol = SERVING_TOL.get(name, {}).get(col)
            if tol is None:
                same = (a.isna() & b.isna()) | (a.astype(str) == b.astype(str))
                if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
                    same = (a.isna() & b.isna()) | (a.astype(float) == b.astype(float))
                if not same.all():
                    bad.append(f"{col} ({int((~same).sum())} rows)")
            else:
                d = _max_abs_diff(a.astype(float), b.astype(float))
                worst = max(worst, d)
                if d > tol:
                    bad.append(f"{col} (max diff {d:.3g} > {tol:g})")
        ok = not_found == 0 and not bad and (extra == 0 or name == "scores")
        res.append((f"serving {name}", "PASS" if ok else "FAIL",
                    f"{len(ref)} verified rows, {len(ref) - not_found} found in the cloud table ({len(c)} rows, {extra} extra); "
                    f"{'all values equal' if not bad else 'DIFFER: ' + ', '.join(bad)} (largest within-tolerance diff {worst:.3g})"))
    return res
