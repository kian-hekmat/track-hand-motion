"""Run the Snowflake scripts locally in DuckDB (same table definitions, positional CSV load, portable SQL).
Used by tests and to produce the expected query results."""
import re
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
SF, EXP = ROOT / "snowflake", ROOT / "data" / "export"
CSV_TABLES = ["takes", "events", "ground_truth", "signals", "frames", "scores"]


def connect(exp: Path = EXP) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(":memory:")
    setup = (SF / "01_setup.sql").read_text()
    for stmt in re.findall(r"CREATE OR REPLACE TABLE .*?\n\);", setup, flags=re.S):
        con.execute(stmt)
    for t in CSV_TABLES:   # positional mapping, header skipped: same semantics as COPY INTO with SKIP_HEADER = 1
        con.execute(f"COPY {t} FROM '{exp / (t + '.csv')}' (FORMAT csv, HEADER true)")
    con.execute(f"COPY raw_keypoints FROM '{exp / 'raw_keypoints.parquet'}' (FORMAT parquet)")
    return con


def create_tableau_views(con) -> None:
    sql = (SF / "05_tableau_views.sql").read_text()
    for stmt in re.findall(r"CREATE OR REPLACE VIEW .*?;", sql, flags=re.S):
        con.execute(stmt)


def run_file(con, path: Path):
    """Execute the single (last) SELECT in a script, ignoring USE statements."""
    sql = "\n".join(l for l in path.read_text().splitlines() if not l.strip().upper().startswith("USE "))
    return con.execute(sql).fetchdf()


def named_queries(path: Path) -> dict[str, str]:
    """Split queries.sql on '-- name: <id>' markers."""
    out, name, buf = {}, None, []
    for line in path.read_text().splitlines():
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
