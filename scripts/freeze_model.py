"""Freeze the v2 segmenter.

1. Leave-one-take-out (LOTO) posteriors over ALL five takes (classifier = gradient boosting, the family
   selected in every nested-CV fold).
2. Choose ONE PELT penalty = the value with the best mean objective across the five LOTO predictions.
   (Objective = 0.5*boundary F1 + 0.5*frame accuracy, tolerance from config.) This uses all five takes, so
   the frozen penalty is mildly optimistic for those takes; takes recorded later are a genuine hold-out.
3. Train the final model on all five takes (+ time-compressed copies), save it with metadata + hash.
4. Write the canonical events: each take's events come from its LOTO model (never trained on that take)
   decoded with the frozen penalty -> data/segments/v2_frozen_oof/.
"""
import datetime
import json
import platform
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import ruptures
import sklearn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv_segmentation as cv
from config import BOUNDARY_TOLERANCE_S, SEGMENTS_DIR
from src.final import META_PATH, MODEL_PATH, sha256
from src.learned import AUG_FACTORS, decode_pelt, fit, predict_proba

PENS = (0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 8.0)
SMOOTH, MIN_SIZE_S = 5, 0.1
TAKES = cv.TAKES


def main():
    oof = {}
    for held in TAKES:
        clf = fit([t for t in TAKES if t != held], "hgb")
        assert held not in clf.train_takes_
        oof[held] = predict_proba(clf, cv.td(held)[1])
        print("LOTO posteriors:", held, flush=True)
    grid = {}
    for pen in PENS:
        per = {}
        for t in TAKES:
            sig, _, t_end, _ = cv.td(t)
            per[t] = cv.objective(t, decode_pelt(oof[t], sig["t"].to_numpy(), t_end, t, pen=pen, smooth=SMOOTH))
        grid[pen] = {"mean": float(np.mean(list(per.values()))), "per_take": per}
        print(f"pen={pen}: mean objective {grid[pen]['mean']:.4f}", flush=True)
    pen = max(PENS, key=lambda p: (round(grid[p]["mean"], 6), p))  # ties -> larger penalty (fewer segments)

    final = fit(TAKES, "hgb")
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(final, MODEL_PATH)
    feat_cols = list(cv.td(TAKES[0])[1].columns)
    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    META_PATH.write_text(json.dumps({
        "name": "segmenter_v2", "created": datetime.date.today().isoformat(), "git_commit_at_freeze": git,
        "classifier": "HistGradientBoostingClassifier(max_depth=3, max_iter=150, learning_rate=0.08, l2_regularization=1.0, class-balanced)",
        "train_takes": list(final.train_takes_), "augmentation_time_factors": list(AUG_FACTORS),
        "pelt_pen": pen, "posterior_smooth": SMOOTH, "min_size_s": MIN_SIZE_S, "smooth_win": 5,
        "boundary_tolerance_s": BOUNDARY_TOLERANCE_S, "feature_columns": feat_cols,
        "model_sha256": sha256(MODEL_PATH),
        "pen_selection": {"method": "best mean objective over LOTO predictions of all five takes",
                          "grid": {str(k): v for k, v in grid.items()}},
        "libraries": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                      "ruptures": ruptures.__version__, "numpy": np.__version__, "pandas": pd.__version__},
        "note": "Trained on all five takes. A take not in train_takes is a genuine hold-out. The frozen penalty was "
                "chosen using all five takes' leave-one-out predictions, so scores on those five are mildly optimistic.",
    }, indent=2))
    out = SEGMENTS_DIR / "v2_frozen_oof"
    out.mkdir(parents=True, exist_ok=True)
    for t in TAKES:
        sig, _, t_end, _ = cv.td(t)
        decode_pelt(oof[t], sig["t"].to_numpy(), t_end, t, pen=pen, smooth=SMOOTH).to_csv(out / f"{t}_events.csv", index=False)
        sig.to_csv(out / f"{t}_signals.csv", index=False)
    print(f"frozen: pen={pen}, model -> {MODEL_PATH} ({MODEL_PATH.stat().st_size/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
