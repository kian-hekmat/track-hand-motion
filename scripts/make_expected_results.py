"""Run queries.sql and 03_verify.sql locally (DuckDB) and save the results to snowflake/expected_results/.
These are what Snowflake should return for the same exported data."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import local_sql as L

OUT = L.SF / "expected_results"


def main():
    OUT.mkdir(exist_ok=True)
    con = L.connect()
    ver = L.run_file(con, L.SF / "03_verify.sql")
    assert (ver["status"] == "PASS").all(), ver[ver["status"] != "PASS"]
    ver.to_csv(OUT / "verify.csv", index=False)
    fp = L.run_file(con, L.SF / "04_fingerprint.sql")
    fp.to_csv(OUT / "fingerprint.csv", index=False)
    print(f"fingerprint: {len(fp)} values")
    for name, q in L.named_queries(L.ROOT / "queries.sql").items():
        q = "\n".join(l for l in q.splitlines() if not l.strip().upper().startswith("USE "))
        df = con.execute(q).fetchdf()
        df.to_csv(OUT / f"{name}.csv", index=False)
        print(f"{name}: {len(df)} rows")
    print(f"verify: {len(ver)} checks, all PASS")


if __name__ == "__main__":
    main()
