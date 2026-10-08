"""Check the three Tableau tables downloaded by hand from Snowflake (MOTION_INTENT.CLOUD views) before they go into Tableau.

  python scripts/check_cloud_tableau_exports.py

Reads data/tableau_cloud/tableau_phases.csv, tableau_signals.csv, tableau_accuracy.csv (the Snowsight downloads) and compares
them with the verified Tableau tables made from the same view SQL: data/tableau/ (vid1-5) plus data/holdout/tableau/ (vid6-7).
Rows are matched by position after sorting on the key columns; text and whole numbers must be identical, decimals within 1e-9
(the cloud signals differ from the local ones by at most ~1.4e-13). Exit code 1 if anything differs.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DOWNLOADS = ROOT / "data" / "tableau_cloud"
TOL = 1e-9
TABLES = {"tableau_phases": ["take", "source", "start_s"], "tableau_signals": ["take", "t_s"], "tableau_accuracy": ["take"]}


def _norm(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=str.lower)
    for c in df.columns:  # Snowsight writes booleans as true/false, pandas wrote True/False
        if df[c].dtype == object and set(df[c].dropna().astype(str).str.lower()) <= {"true", "false"}:
            df[c] = df[c].astype(str).str.lower() == "true"
    return df


def check(name: str, keys: list[str]) -> tuple[bool, str]:
    path = DOWNLOADS / f"{name}.csv"
    if not path.exists():
        return False, f"{path.relative_to(ROOT)} not found"
    got = _norm(pd.read_csv(path, float_precision="round_trip"))
    ref = _norm(pd.concat([pd.read_csv(ROOT / d / f"{name}.csv", float_precision="round_trip")
                           for d in ("data/tableau", "data/holdout/tableau")], ignore_index=True))
    if list(got.columns) != list(ref.columns):
        return False, f"columns {list(got.columns)}, expected {list(ref.columns)}"
    if len(got) != len(ref):
        return False, f"{len(got)} rows, expected {len(ref)}"
    got, ref = (d.sort_values(keys, kind="stable").reset_index(drop=True) for d in (got, ref))
    bad, worst = [], 0.0
    for c in ref.columns:
        a, b = got[c], ref[c]
        if pd.api.types.is_float_dtype(b) or pd.api.types.is_float_dtype(a):
            x, y = a.to_numpy(float), b.to_numpy(float)
            if not np.array_equal(np.isnan(x), np.isnan(y)):
                bad.append(f"{c} (empty cells in different rows)")
                continue
            d = float(np.nanmax(np.abs(x - y))) if (~np.isnan(x)).any() else 0.0
            worst = max(worst, d)
            if d > TOL:
                bad.append(f"{c} (max diff {d:.3g})")
        elif not (a.astype(str) == b.astype(str)).all():
            bad.append(f"{c} ({int((a.astype(str) != b.astype(str)).sum())} rows)")
    return not bad, f"{len(got)} rows; " + ("all values equal" if not bad else "DIFFER: " + ", ".join(bad)) + \
        f" (largest decimal difference {worst:.3g})"


def main() -> int:
    ok_all = True
    for name, keys in TABLES.items():
        ok, detail = check(name, keys)
        ok_all &= ok
        print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")
    print("TABLEAU EXPORT CHECK:", "ALL PASSED" if ok_all else "FAILED")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
