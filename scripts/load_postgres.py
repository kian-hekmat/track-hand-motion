"""Load data/raw/*_keypoints.csv into Postgres and verify row counts against the CSVs.

Usage: python scripts/load_postgres.py [take ...]   (default: every take in data/raw)
Exits non-zero if any loaded row count differs from the CSV's data-row count.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import RAW_DIR
from src.db import connect, init_schema, load_take


def csv_rows(path: Path) -> int:
    with open(path) as f:
        return sum(1 for _ in f) - 1  # minus header


def main(takes: list[str]) -> int:
    takes = takes or sorted(p.name.removesuffix("_keypoints.csv") for p in RAW_DIR.glob("*_keypoints.csv"))
    conn = connect()
    init_schema(conn)
    bad = 0
    for take in takes:
        expected = csv_rows(RAW_DIR / f"{take}_keypoints.csv")
        got = load_take(conn, take)
        ok = got == expected
        bad += not ok
        print(f"{take}: csv={expected} postgres={got} {'OK' if ok else 'MISMATCH'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
