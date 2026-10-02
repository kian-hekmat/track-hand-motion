"""Phase 1: load raw keypoint CSVs into the local Postgres landing table `raw_keypoints`."""
import os
from pathlib import Path

import psycopg2

from config import RAW_DIR, ROOT

DSN = os.environ.get("MOTION_DATABASE_URL", "postgresql://motion:motion_pass@localhost:5433/motion")
SCHEMA_FILE = ROOT / "sql" / "001_raw_keypoints.sql"
COLUMNS = ("take, frame_idx, timestamp_ms, keypoint_id, x, y, z, world_x, world_y, world_z, "
           "detected, handedness, handedness_score")


def connect():
    return psycopg2.connect(DSN)


def init_schema(conn) -> None:
    with conn, conn.cursor() as cur:
        cur.execute(SCHEMA_FILE.read_text())


def load_take(conn, take: str, csv_path: Path | None = None) -> int:
    """Replace all rows for `take` with the CSV contents, in one transaction (idempotent).

    Returns the number of rows in the table for this take after the load. A failed COPY
    (e.g. constraint violation) rolls back and leaves the previous data intact.
    """
    csv_path = csv_path or RAW_DIR / f"{take}_keypoints.csv"
    with conn, conn.cursor() as cur:
        cur.execute("DELETE FROM raw_keypoints WHERE take = %s", (take,))
        with open(csv_path) as f:
            cur.copy_expert(f"COPY raw_keypoints ({COLUMNS}) FROM STDIN WITH (FORMAT csv, HEADER true)", f)
        cur.execute("SELECT count(*) FROM raw_keypoints WHERE take = %s", (take,))
        return cur.fetchone()[0]
