"""Build the canonical tables for Databricks / Snowflake / Tableau into data/export/.

Sources: the frozen-model out-of-fold events (data/segments/v2_frozen_oof), the derived signals, the raw
keypoints in Postgres, the ground truth and the per-take metadata. Writes a manifest with row counts and
SHA-256 hashes so later phases can verify nothing was dropped.
"""
import datetime
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import RAW_DIR, ROOT, SEGMENTS_DIR, TAKE_GROUPS
from src import ground_truth as G
from src.db import connect
from src.final import load_meta
from src.signals import load_frames

OUT = ROOT / "data" / "export"
SRC = SEGMENTS_DIR / "v2_frozen_oof"
MODEL_VERSION = "v2_frozen_oof"
TAKES = sorted(TAKE_GROUPS)


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    events, signals, frames, gts, takes = [], [], [], [], []
    for take in TAKES:
        ev = pd.read_csv(SRC / f"{take}_events.csv")
        sig = pd.read_csv(SRC / f"{take}_signals.csv")
        fr = load_frames(take)
        gt = G.load_ground_truth(take)
        meta = json.loads((RAW_DIR / f"{take}_meta.json").read_text())
        ft, st = fr["t"].to_numpy(), sig["t"].to_numpy()

        # --- per-sample labels ---
        sig["predicted_label"] = G.labels_at(ev, st)
        sig["ground_truth_label"] = G.labels_at(gt, st)
        # --- events enriched with signal summaries and frame counts ---
        rows = []
        for r in ev.itertuples():
            last = r.Index == len(ev) - 1
            m = (st >= r.start_s) & ((st < r.end_s) if not last else (st <= r.end_s + 1e-9))
            fm = (ft >= r.start_s) & ((ft < r.end_s) if not last else (ft <= r.end_s + 1e-9))
            s = sig[m]
            rows.append({
                "take": take, "group": TAKE_GROUPS[take], "event_idx": r.event_idx, "label": r.label,
                "start_s": r.start_s, "end_s": r.end_s, "duration_s": r.duration_s, "n_frames": int(fm.sum()),
                "mean_speed": s["speed"].mean(), "mean_aperture": s["aperture"].mean(),
                "mean_confidence": r.mean_confidence, "frac_missing": float(s["missing"].mean()),
                "model_version": MODEL_VERSION})
        events.append(pd.DataFrame(rows))
        sig.insert(0, "take", take)
        signals.append(sig.rename(columns={"t": "t_s", "p": "progress", "dp": "progress_rate", "q": "offaxis",
                                           "dq": "offaxis_rate", "ydev": "height", "dy": "vertical_velocity"}))
        # --- frame-level aggregate of raw keypoints (wrist + thumb/index tips), one row per frame ---
        keep = fr[["t", "detected", "i0_x", "i0_y", "i4_x", "i4_y", "i8_x", "i8_y"]].copy()
        hd = pd.read_csv(RAW_DIR / f"{take}_keypoints.csv").drop_duplicates("frame_idx").sort_values("frame_idx")
        keep.insert(0, "frame_idx", hd["frame_idx"].to_numpy())
        keep.insert(0, "take", take)
        keep["handedness"], keep["handedness_score"] = hd["handedness"].to_numpy(), hd["handedness_score"].to_numpy()
        frames.append(keep.rename(columns={"t": "t_s", "i0_x": "wrist_x", "i0_y": "wrist_y", "i4_x": "thumb_tip_x",
                                           "i4_y": "thumb_tip_y", "i8_x": "index_tip_x", "i8_y": "index_tip_y"}))
        gts.append(gt)
        takes.append({"take": take, "group": TAKE_GROUPS[take], "n_frames": meta["frames_extracted"],
                      "duration_s": meta["last_timestamp_s"], "mean_fps": meta["mean_fps_from_timestamps"],
                      "frames_not_detected": meta["frames_not_detected"],
                      "fraction_not_detected": meta["fraction_not_detected"],
                      "left_handedness_frames": int(meta["handedness_counts"].get("Left", 0)),
                      "n_ground_truth_segments": len(gt), "n_events": len(ev)})
    tables = {
        "events": pd.concat(events, ignore_index=True), "signals": pd.concat(signals, ignore_index=True),
        "frames": pd.concat(frames, ignore_index=True), "ground_truth": pd.concat(gts, ignore_index=True),
        "takes": pd.DataFrame(takes),
        "scores": pd.read_csv(SRC / "scores.csv"),
    }
    for name, df in tables.items():
        df.to_csv(OUT / f"{name}.csv", index=False)

    # --- raw keypoints straight from the Postgres landing table (for Spark/Snowflake) ---
    with connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT take, frame_idx, timestamp_ms, keypoint_id, x, y, z, world_x, world_y, world_z, "
                    "detected, handedness, handedness_score FROM raw_keypoints ORDER BY take, frame_idx, keypoint_id")
        cols = [d[0] for d in cur.description]
        raw = pd.DataFrame(cur.fetchall(), columns=cols)
    raw.to_parquet(OUT / "raw_keypoints.parquet", index=False)

    files = {f"{n}.csv": df for n, df in tables.items()}
    manifest = {"created": datetime.date.today().isoformat(), "model_version": MODEL_VERSION,
                "model_sha256": load_meta()["model_sha256"], "files": {}}
    for fn, df in files.items():
        manifest["files"][fn] = {"rows": len(df), "columns": list(df.columns), "sha256": hashlib.sha256((OUT / fn).read_bytes()).hexdigest()}
    manifest["files"]["raw_keypoints.parquet"] = {"rows": len(raw), "columns": list(raw.columns),
                                                  "sha256": hashlib.sha256((OUT / "raw_keypoints.parquet").read_bytes()).hexdigest()}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    for fn, m in manifest["files"].items():
        print(f"{fn:24s} {m['rows']:>7} rows  {len(m['columns'])} cols")


if __name__ == "__main__":
    build()
