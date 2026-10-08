"""Cloud path M3: SQL that checks the published cloud tables inside Snowflake.

Generated from the same keys and tolerances as the Python checks (src/cloud/checks.py), so there is one definition of
"equal". Portable SQL (ROW_NUMBER, COUNT_IF, IS NOT DISTINCT FROM): the tests run it in DuckDB, the notebook in Snowflake.

parity_sql()      one query per table: every verified PIPELINE row must have a CLOUD row with the same key and equal values
                  (within the stated tolerance for floating-point columns). Counts the rows that fail, per column.
compare_queries() runs each analytical query in queries.sql against both schemas and compares the rows that belong to the
                  verified takes (the cloud schema also holds the hold-out takes vid6 and vid7).
"""
import re

import numpy as np
import pandas as pd

from config import ROOT, TAKE_GROUPS
from src.cloud import checks

QUERIES = ROOT / "queries.sql"
VIEWS = ROOT / "snowflake" / "05_tableau_views.sql"
DEV_TAKES = sorted(TAKE_GROUPS)
DEV_GROUPS = sorted(set(TAKE_GROUPS.values()))


def _keyed(prefix: str, name: str, takes_from: str | None = None) -> str:
    where = f" WHERE take IN (SELECT DISTINCT take FROM {takes_from})" if takes_from else ""
    if name in checks.POSITION_BY:
        order = checks.POSITION_BY[name]
        return (f"SELECT *, ROW_NUMBER() OVER (PARTITION BY take ORDER BY {order}) - 1 AS _pos "
                f"FROM {prefix}{name}{where}")
    return f"SELECT * FROM {prefix}{name}{where}"


def parity_sql(name: str, columns: list[str], ref_prefix: str, cloud_prefix: str) -> str:
    keys = checks.SERVING_KEYS[name]
    tol = checks.SERVING_TOL.get(name, {})
    tests = []
    for col in columns:
        if col in keys or col in checks.SKIP.get(name, set()):
            continue
        if col in tol:
            ok = (f"COALESCE((r.{col} IS NULL AND c.{col} IS NULL) OR ABS(r.{col} - c.{col}) <= {tol[col]!r}, FALSE)")
        else:
            ok = f"(r.{col} IS NOT DISTINCT FROM c.{col})"
        tests.append(f"COUNT_IF(c.take IS NOT NULL AND NOT {ok}) AS bad_{col}")
    join = " AND ".join(f"r.{k} = c.{k}" for k in keys)
    return (f"WITH r AS ({_keyed(ref_prefix, name)}),\n"
            f"     c AS ({_keyed(cloud_prefix, name, takes_from=f'{ref_prefix}{name}')})\n"
            f"SELECT COUNT(*) AS n_ref, COUNT_IF(c.take IS NULL) AS n_missing,\n       " + ",\n       ".join(tests)
            + f"\nFROM r LEFT JOIN c ON {join}")


def run_parity(fetch_df, columns_by_table: dict, ref_prefix: str, cloud_prefix: str) -> list:
    """fetch_df(sql) -> pandas DataFrame. Returns [(check, status, detail)]."""
    res = []
    for name, columns in columns_by_table.items():
        row = fetch_df(parity_sql(name, columns, ref_prefix, cloud_prefix)).iloc[0]
        row.index = [i.lower() for i in row.index]
        bad = {k[4:]: int(v) for k, v in row.items() if k.startswith("bad_") and int(v)}
        ok = int(row["n_missing"]) == 0 and not bad and int(row["n_ref"]) > 0
        res.append((f"snowflake parity {name}", "PASS" if ok else "FAIL",
                    f"{int(row['n_ref'])} PIPELINE rows; {int(row['n_missing'])} without a CLOUD row; "
                    + ("all values equal" if not bad else "DIFFER: " + ", ".join(f"{k} ({v} rows)" for k, v in bad.items()))))
    return res


def named_queries() -> dict:
    out, name, buf = {}, None, []
    for line in QUERIES.read_text().splitlines():
        m = re.match(r"--\s*name:\s*(\w+)", line)
        if m:
            if name:
                out[name] = "\n".join(buf).strip().rstrip(";")
            name, buf = m.group(1), []
        elif name:
            buf.append(line)
    if name:
        out[name] = "\n".join(buf).strip().rstrip(";")
    return out


def view_statements() -> list[str]:
    return re.findall(r"CREATE OR REPLACE VIEW .*?;", VIEWS.read_text(), flags=re.S)


def _dev_rows(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=str.lower)
    if "take" in df.columns:
        df = df[df["take"].isin(DEV_TAKES)]
    elif "take_group" in df.columns:
        df = df[df["take_group"].isin(DEV_GROUPS)]
    df = df.drop(columns=[c for c in ("ambiguity_rank",) if c in df.columns])  # a rank over all takes in the schema
    return df.sort_values(list(df.columns)).reset_index(drop=True)


def compare_queries(fetch_df, use_ref: str, use_cloud: str) -> tuple[list, dict]:
    """Each query from queries.sql, unchanged, in both schemas. Returns (checks, cloud results)."""
    res, cloud_results = [], {}
    for qname, sql in named_queries().items():
        fetch_df(use_ref)
        ref = fetch_df(sql)
        fetch_df(use_cloud)
        cloud = fetch_df(sql)
        cloud_results[qname] = cloud
        a, b = _dev_rows(ref), _dev_rows(cloud)
        same = list(a.columns) == list(b.columns) and len(a) == len(b)
        if same:
            for col in a.columns:
                x, y = a[col], b[col]
                if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
                    same &= bool(np.allclose(x.astype(float), y.astype(float), rtol=0, atol=1e-9, equal_nan=True))
                else:
                    same &= x.astype(str).tolist() == y.astype(str).tolist()
        res.append((f"snowflake query {qname}", "PASS" if same else "FAIL",
                    f"verified-take rows: PIPELINE {len(a)}, CLOUD {len(b)}, {'identical' if same else 'DIFFER'}; "
                    f"CLOUD total {len(cloud)} rows (includes vid6-7)"))
    return res, cloud_results
