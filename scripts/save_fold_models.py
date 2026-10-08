"""Cloud-path milestone M0: save the five leave-one-take-out (LOTO) models that produce the canonical events.

scripts/freeze_model.py trains these five fold models (each on four takes, never on the take it labels) but saves only
the full model. The cloud pipeline must label vid1-5 with the fold models (labelling a take with the full model would score
the model on its own training data), so they are trained again here with the same code and seed and saved with hashes.

Nothing is saved unless every fold model's events, decoded with the frozen settings, are IDENTICAL to the verified
canonical events in data/segments/v2_frozen_oof/. Output: models/folds/segmenter_v2_fold_<take>.joblib + folds.json.
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
from config import SEGMENTS_DIR
from src.final import FOLDS_DIR, FOLDS_META_PATH, fold_model_path, load_meta, sha256
from src.learned import decode_pelt, fit, predict_proba

CANONICAL = SEGMENTS_DIR / "v2_frozen_oof"


def fold_events(clf, take, meta):
    sig, feat, t_end, _ = cv.td(take)
    assert list(feat.columns) == meta["feature_columns"], "feature columns changed since the model was frozen"
    P = predict_proba(clf, feat)
    # same call as scripts/freeze_model.py wrote the canonical events with (min_size_s left at its default)
    return decode_pelt(P, sig["t"].to_numpy(), t_end, take, pen=meta["pelt_pen"], smooth=meta["posterior_smooth"])


def main():
    meta = load_meta()
    takes = list(meta["train_takes"])
    assert takes == cv.TAKES, (takes, cv.TAKES)
    models, report = {}, {}
    for held in takes:
        clf = fit([t for t in takes if t != held], "hgb", seed=0)
        assert held not in clf.train_takes_
        ev = fold_events(clf, held, meta)
        ref_text = (CANONICAL / f"{held}_events.csv").read_text()
        identical = ev.to_csv(index=False) == ref_text  # byte-for-byte, as freeze_model.py wrote it
        n_ref = len(pd.read_csv(CANONICAL / f"{held}_events.csv"))
        report[held] = {"events": len(ev), "canonical_events": n_ref, "identical_csv": identical}
        print(f"{held}: trained on {clf.train_takes_}; {len(ev)} events vs {n_ref} canonical; "
              f"{'IDENTICAL' if identical else 'DIFFERENT'}", flush=True)
        models[held] = clf
    if not all(r["identical_csv"] for r in report.values()):
        sys.exit("STOP: a fold model does not reproduce the canonical events; nothing was saved. Find out why first.")

    FOLDS_DIR.mkdir(parents=True, exist_ok=True)
    folds = {}
    for held, clf in models.items():
        path = fold_model_path(held)
        joblib.dump(clf, path)
        folds[held] = {"file": path.name, "sha256": sha256(path), "train_takes": list(clf.train_takes_),
                       "labels_take": held, **report[held]}
    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    FOLDS_META_PATH.write_text(json.dumps({
        "name": "segmenter_v2 leave-one-take-out fold models",
        "created": datetime.date.today().isoformat(), "git_commit": git,
        "purpose": "Label each development take with the model that never saw it (cloud path). New takes use the full "
                   "model models/segmenter_v2.joblib.",
        "training": "src.learned.fit(other four takes, 'hgb', seed=0), same as scripts/freeze_model.py",
        "decoding": {"pelt_pen": meta["pelt_pen"], "posterior_smooth": meta["posterior_smooth"], "min_size_s": 0.1},
        "check": "events decoded from each fold model are byte-identical (CSV) to data/segments/v2_frozen_oof/<take>_events.csv",
        "libraries": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                      "ruptures": ruptures.__version__, "numpy": np.__version__, "pandas": pd.__version__},
        "folds": folds,
    }, indent=2))
    print(f"saved {len(folds)} fold models -> {FOLDS_DIR}")


if __name__ == "__main__":
    main()
