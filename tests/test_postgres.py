"""Phase 1 landing-table checks. Require the local Postgres (docker compose up -d db) and a
prior `python scripts/load_postgres.py`. Fail (not skip) if the DB is unreachable so a missing
landing table can never look like a pass."""
import pandas as pd
import pytest

from config import RAW_DIR
from src.db import connect
from src.landmarks import NUM_LANDMARKS


@pytest.fixture(scope="module")
def conn():
    try:
        c = connect()
    except Exception as e:  # noqa: BLE001
        pytest.fail(f"Postgres unreachable ({e}); run `docker compose up -d db`")
    yield c
    c.close()


def _scalar(conn, sql, *args):
    with conn.cursor() as cur:
        cur.execute(sql, args)
        return cur.fetchone()[0]


def test_row_count_matches_csv(conn, take):
    expected = len(pd.read_csv(RAW_DIR / f"{take}_keypoints.csv"))
    assert _scalar(conn, "SELECT count(*) FROM raw_keypoints WHERE take = %s", take) == expected


def test_frame_count_matches_meta(conn, take):
    import json
    meta = json.loads((RAW_DIR / f"{take}_meta.json").read_text())
    n = _scalar(conn, "SELECT count(DISTINCT frame_idx) FROM raw_keypoints WHERE take = %s", take)
    assert n == meta["frames_extracted"]
    assert _scalar(conn, "SELECT count(*) FROM raw_keypoints WHERE take = %s", take) == n * NUM_LANDMARKS


def test_undetected_fraction_matches_meta(conn, take):
    import json
    meta = json.loads((RAW_DIR / f"{take}_meta.json").read_text())
    n_undet = _scalar(conn, "SELECT count(DISTINCT frame_idx) FROM raw_keypoints "
                            "WHERE take = %s AND NOT detected", take)
    assert n_undet == meta["frames_not_detected"]


def test_no_null_coords_when_detected(conn, take):
    assert _scalar(conn, "SELECT count(*) FROM raw_keypoints WHERE take = %s AND detected AND "
                         "(x IS NULL OR y IS NULL OR z IS NULL OR world_x IS NULL)", take) == 0


def test_load_is_idempotent(conn, take):
    from src.db import load_take
    before = _scalar(conn, "SELECT count(*) FROM raw_keypoints WHERE take = %s", take)
    assert load_take(conn, take) == before
