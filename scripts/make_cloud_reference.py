"""Cloud-path milestone M1: save the local outputs the Databricks run must reproduce.

For every take (vid1-5 with their leave-one-take-out fold model, vid6-7 with the full model), runs the exact call the cloud
pipeline makes (src.final.load_model_for_take + segment_frames) and writes, in data/cloud_reference/:
  <take>_signals.csv      the derived signals (model input before feature building)
  <take>_posteriors.csv   per-sample phase probabilities from the model
  <take>_events.csv       the decoded events
  scores_frame.csv        frame-level scores per take (src.evaluate.score_take on the native frame times; scopes
                          'all' and 'out_of_frame'), the numbers the cloud gold.frame_scores must reproduce
  manifest.json           model per take, row counts, library versions used to make the reference
Signals and probabilities are saved so that, if the cloud events differ, the run shows where the difference starts.
"""
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import ruptures
import scipy
import sklearn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import HOLDOUT_TAKES, LABELS, ROOT, TAKE_GROUPS
from src.evaluate import score_take
from src.final import load_model_for_take, segment_frames
from src.signals import load_frames

OUT = ROOT / "data" / "cloud_reference"
TAKES = sorted(TAKE_GROUPS) + sorted(HOLDOUT_TAKES)
SCORE_COLUMNS = (["take", "scope", "n_frames", "n_unlabelled", "frame_accuracy"]
                 + [c for l in LABELS for c in (f"acc_{l}", f"n_{l}")] + ["balanced_accuracy", "majority_baseline"])


def frame_scores(take, ev):
    rows = score_take(take, ev, load_frames(take)["t"].to_numpy())
    return pd.DataFrame(rows)[SCORE_COLUMNS]


def reference(take):
    model, meta, name = load_model_for_take(take)
    ev, sig, P = segment_frames(load_frames(take), take, model=model, meta=meta)
    post = pd.DataFrame(P, columns=[f"p_{l}" for l in LABELS])
    post.insert(0, "t", sig["t"].to_numpy())
    return ev, sig, post, name


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    takes, scores = {}, []
    for take in TAKES:
        ev, sig, post, name = reference(take)
        sig.to_csv(OUT / f"{take}_signals.csv", index=False)
        post.to_csv(OUT / f"{take}_posteriors.csv", index=False)
        ev.to_csv(OUT / f"{take}_events.csv", index=False)
        scores.append(frame_scores(take, ev))
        takes[take] = {"model": name, "samples": len(sig), "events": len(ev)}
        print(f"{take}: model {name}, {len(sig)} samples, {len(ev)} events")
    pd.concat(scores, ignore_index=True).to_csv(OUT / "scores_frame.csv", index=False)
    (OUT / "manifest.json").write_text(json.dumps({
        "made_by": "scripts/make_cloud_reference.py",
        "libraries": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                      "scipy": scipy.__version__, "scikit-learn": sklearn.__version__, "ruptures": ruptures.__version__},
        "takes": takes,
    }, indent=2))


if __name__ == "__main__":
    main()
