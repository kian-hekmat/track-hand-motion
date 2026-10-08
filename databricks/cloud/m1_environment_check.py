# Databricks notebook source
# MAGIC %md
# MAGIC # Cloud path M1: does the frozen model give the same answers on Databricks?
# MAGIC Runs the repo's own code (`src/`, imported from this Git folder) on serverless compute, for all seven takes, with the
# MAGIC model the cloud pipeline will use for each (`src.final.load_model_for_take`: vid1-5 their leave-one-take-out model,
# MAGIC vid6-7 the full model). The outputs are compared with the local reference in `data/cloud_reference/`
# MAGIC (`scripts/make_cloud_reference.py`):
# MAGIC
# MAGIC - **events:** must be byte-identical to the reference CSV (this is the pass/fail check);
# MAGIC - **signals** and **phase probabilities:** the largest difference is measured and reported, so that if events differ the run
# MAGIC   shows where the difference starts (feature input or model output).
# MAGIC
# MAGIC Library versions: scikit-learn and ruptures are pinned to the versions the model was frozen with
# MAGIC (`databricks/cloud/requirements-cloud.txt`). numpy, pandas and scipy are whatever serverless provides; they are recorded
# MAGIC and compared with the reference's versions. Nothing is written anywhere; this notebook only reads and compares.

# COMMAND ----------

# MAGIC %pip install -q scikit-learn==1.9.1 ruptures==1.1.10

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Find the repo in this Git folder and import its code

# COMMAND ----------

import json
import os
import platform
import sys
from pathlib import Path


def find_repo_root():
    starts = [Path(os.getcwd())]
    try:
        nb_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
        starts.append(Path("/Workspace" + nb_path).parent)
    except Exception:
        pass
    for start in starts:
        for p in (start, *start.parents):
            if (p / "config.py").exists() and (p / "src" / "final.py").exists():
                return p
    raise RuntimeError(f"repo root not found from {starts}; open this notebook from the Git folder of the repo")


REPO = find_repo_root()
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np
import pandas as pd
import ruptures
import scipy
import sklearn

from config import HOLDOUT_TAKES, LABELS, TAKE_GROUPS
from src.final import load_model_for_take, segment_frames
from src.signals import load_frames

REF_DIR = REPO / "data" / "cloud_reference"
REF = json.loads((REF_DIR / "manifest.json").read_text())
TAKES = sorted(TAKE_GROUPS) + sorted(HOLDOUT_TAKES)
assert TAKES == sorted(REF["takes"], key=TAKES.index), (TAKES, list(REF["takes"]))

RESULTS = []  # (check, status, detail)


def record(check, status, detail=""):
    RESULTS.append((check, status, detail))
    print(f"{status:4}  {check}" + (f": {detail}" if detail else ""))


cloud_versions = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                  "scipy": scipy.__version__, "scikit-learn": sklearn.__version__, "ruptures": ruptures.__version__}
print(f"repo: {REPO}")
for lib, ref_version in REF["libraries"].items():
    same = cloud_versions[lib] == ref_version
    pinned = lib in ("scikit-learn", "ruptures")
    record(f"library {lib}", ("PASS" if same else "FAIL") if pinned else "INFO",
           f"Databricks {cloud_versions[lib]} / reference {ref_version}" + ("" if same else " (differs)"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Run every take and compare with the reference

# COMMAND ----------


def read_ref(take, kind):
    return pd.read_csv(REF_DIR / f"{take}_{kind}.csv", float_precision="round_trip")


def max_abs_diff(a: pd.DataFrame, b: pd.DataFrame):
    """Largest absolute difference over numeric columns; NaN must sit in the same places (else inf)."""
    worst, where = 0.0, None
    for col in a.columns:
        if not pd.api.types.is_numeric_dtype(a[col]) or pd.api.types.is_bool_dtype(a[col]):
            if not a[col].astype(str).equals(b[col].astype(str)):
                return float("inf"), col
            continue
        x, y = a[col].to_numpy(dtype=float), b[col].to_numpy(dtype=float)
        if not np.array_equal(np.isnan(x), np.isnan(y)):
            return float("inf"), f"{col} (NaN positions differ)"
        d = np.nanmax(np.abs(x - y)) if np.isfinite(x).any() else 0.0
        if d > worst:
            worst, where = float(d), col
    return worst, where


for take in TAKES:
    model, meta, name = load_model_for_take(take)  # checks the model file hash
    expected_model = REF["takes"][take]["model"]
    if name != expected_model:
        record(f"{take}: model", "FAIL", f"{name}, reference used {expected_model}")
        continue
    ev, sig, P = segment_frames(load_frames(take), take, model=model, meta=meta)
    post = pd.DataFrame(P, columns=[f"p_{l}" for l in LABELS])
    post.insert(0, "t", sig["t"].to_numpy())

    ref_sig, ref_post = read_ref(take, "signals"), read_ref(take, "posteriors")
    shape_ok = list(sig.columns) == list(ref_sig.columns) and len(sig) == len(ref_sig) and len(post) == len(ref_post)
    if not shape_ok:
        record(f"{take}: signals shape", "FAIL", f"{sig.shape} vs reference {ref_sig.shape}")
        continue
    sig_d, sig_col = max_abs_diff(sig.reset_index(drop=True), ref_sig)
    post_d, _ = max_abs_diff(post, ref_post)
    agree = float((P.argmax(1) == ref_post[[f"p_{l}" for l in LABELS]].to_numpy().argmax(1)).mean())
    record(f"{take}: signals vs reference", "INFO", f"max abs difference {sig_d:.3g}" + (f" (in {sig_col})" if sig_d else ""))
    record(f"{take}: phase probabilities vs reference", "INFO",
           f"max abs difference {post_d:.3g}; most-likely phase agrees on {agree:.4f} of {len(P)} samples")

    ref_text = (REF_DIR / f"{take}_events.csv").read_text()
    if ev.to_csv(index=False) == ref_text:
        record(f"{take}: events", "PASS", f"{len(ev)} events, byte-identical to the reference ({name})")
    else:
        ref_ev = pd.read_csv(REF_DIR / f"{take}_events.csv", float_precision="round_trip")
        same_labels = ev["label"].tolist() == ref_ev["label"].tolist()
        detail = f"{len(ev)} events vs {len(ref_ev)}; label sequence {'same' if same_labels else 'DIFFERENT'}"
        if same_labels:
            detail += f"; largest boundary shift {np.abs(ev['start_s'].to_numpy() - ref_ev['start_s'].to_numpy()).max():.3g} s"
        record(f"{take}: events", "FAIL", detail)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Summary

# COMMAND ----------

print(f"{'status':6} check / detail")
for check, status, detail in RESULTS:
    print(f"{status:6} {check}" + (f"\n         {detail}" if detail else ""))
failures = [c for c, s, _ in RESULTS if s == "FAIL"]
n_events_pass = sum(1 for c, s, _ in RESULTS if c.endswith(": events") and s == "PASS")
print()
print(f"events identical for {n_events_pass} of {len(TAKES)} takes")
print(f"M1 CHECKS: {'ALL PASSED' if not failures else f'{len(failures)} FAILED: ' + '; '.join(failures)}")
