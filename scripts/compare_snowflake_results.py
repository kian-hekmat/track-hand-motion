"""Compare results downloaded from Snowflake (snowflake/actual_results/*.csv) with the expected results produced
locally (snowflake/expected_results/*.csv). Column names are compared case-insensitively (Snowflake returns upper
case); rows are compared after sorting; numbers to 1e-6.

Usage: python scripts/compare_snowflake_results.py            # exit code 1 on any mismatch or missing file
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SF = Path(__file__).resolve().parents[1] / "snowflake"
TOLERANCE = {"fingerprint.csv": 2e-4}   # rounded sums of up to 83k floats; everything else must match to 1e-6
OPTIONAL = {"fingerprint.csv"}          # 04_fingerprint.sql is an extra check; its absence is reported, not a failure


def normalise(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().strip('"').lower() for c in df.columns]
    return df


def compare_frames(actual: pd.DataFrame, expected: pd.DataFrame, tol: float = 1e-6) -> list[str]:
    a, e = normalise(actual), normalise(expected)
    problems = []
    if list(a.columns) != list(e.columns):
        return [f"columns differ: actual {list(a.columns)} vs expected {list(e.columns)}"]
    if len(a) != len(e):
        return [f"row count differs: actual {len(a)} vs expected {len(e)}"]
    keys = list(e.columns)
    a = a.sort_values(keys, kind="mergesort").reset_index(drop=True)
    e = e.sort_values(keys, kind="mergesort").reset_index(drop=True)
    for c in keys:
        ea, ee = e[c], a[c]
        if pd.api.types.is_numeric_dtype(ea) and pd.api.types.is_numeric_dtype(ee):
            bad = ~np.isclose(ee.astype(float), ea.astype(float), atol=tol, rtol=0, equal_nan=True)
        else:
            bad = ee.astype(str).str.strip() != ea.astype(str).str.strip()
        if bad.any():
            i = int(np.flatnonzero(bad)[0])
            problems.append(f"column {c}: first difference at sorted row {i}: actual {ee.iloc[i]!r} vs expected {ea.iloc[i]!r} ({int(bad.sum())} rows differ)")
    return problems


def main() -> int:
    exp_dir, act_dir = SF / "expected_results", SF / "actual_results"
    bad = 0
    for e in sorted(exp_dir.glob("*.csv")):
        a = act_dir / e.name
        if not a.exists():
            if e.name in OPTIONAL:
                print(f"SKIPPED  {e.name}  (optional check not run)")
                continue
            print(f"MISSING  {e.name}  (download the result from Snowflake into snowflake/actual_results/)")
            bad += 1
            continue
        problems = compare_frames(pd.read_csv(a), pd.read_csv(e), tol=TOLERANCE.get(e.name, 1e-6))
        if e.name == "verify.csv":
            act = normalise(pd.read_csv(a))
            if "status" in act and not (act["status"].astype(str).str.strip() == "PASS").all():
                problems.append("verify.csv contains a status other than PASS")
        print(("OK       " if not problems else "MISMATCH ") + e.name)
        for p in problems:
            print("   ", p)
        bad += bool(problems)
    print("\nALL MATCH" if not bad else f"\n{bad} file(s) with problems")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
