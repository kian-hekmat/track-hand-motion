"""Phase 3: the PySpark transforms and the generated Databricks notebook, run on a LOCAL Spark session
(Java 11/17 + pyspark 3.5). The notebook itself is executed end to end against data/export."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

from config import ROOT

sys.path.insert(0, str(ROOT / "databricks"))
sys.path.insert(0, str(ROOT / "scripts"))

NOTEBOOK = ROOT / "databricks" / "motion_pipeline_spark.py"


@pytest.fixture(scope="session")
def spark(tmp_path_factory):
    if "JAVA_HOME" not in os.environ:
        out = subprocess.run(["/usr/libexec/java_home", "-v", "11"], capture_output=True, text=True)
        if out.returncode != 0:
            pytest.fail("Java 11/17 needed for local Spark tests")
        os.environ["JAVA_HOME"] = out.stdout.strip()
    os.environ["PYSPARK_PYTHON"] = os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable  # workers = this venv
    from pyspark.sql import SparkSession
    s = (SparkSession.builder.master("local[2]").appName("tests").config("spark.ui.enabled", "false")
         .config("spark.sql.shuffle.partitions", "4")
         .config("spark.sql.warehouse.dir", str(tmp_path_factory.mktemp("warehouse"))).getOrCreate())
    yield s
    s.stop()


def _raw_rows(take="t", n=60, vx_px=600.0, hand_px=150.0, gap=()):
    """Synthetic raw_keypoints: wrist moves at vx_px px/s (30 fps), middle MCP hand_px above it,
    thumb/index tips 0.045 m apart in world coordinates, hand length 0.09 m (aperture 0.5)."""
    rows = []
    for i in range(n):
        t_ms = round(i * 1000 / 30)
        X, Y = 400 + vx_px * (i / 30), 500.0
        detected = i not in gap
        for k, (ix, iy, wx, wy) in {0: (X, Y, 0.0, 0.0), 9: (X, Y - hand_px, 0.0, 0.09),
                                    4: (X, Y, -0.0225, 0.0), 8: (X, Y, 0.0225, 0.0)}.items():
            if detected:
                rows.append(("t", i, t_ms, k, ix / 1920, iy / 1080, 0.0, wx, wy, 0.0, True, "Right", 0.9))
            else:
                rows.append(("t", i, t_ms, k, None, None, None, None, None, None, False, None, None))
    return rows


from pyspark.sql.types import BooleanType, DoubleType, IntegerType, LongType, StringType, StructField, StructType

COLS = StructType([StructField("take", StringType()), StructField("frame_idx", IntegerType()),
                   StructField("timestamp_ms", LongType()), StructField("keypoint_id", IntegerType()),
                   *[StructField(c, DoubleType()) for c in ("x", "y", "z", "world_x", "world_y", "world_z")],
                   StructField("detected", BooleanType()), StructField("handedness", StringType()),
                   StructField("handedness_score", DoubleType())])


def _signals(spark, **kw):
    import spark_transforms as T
    raw = spark.createDataFrame(_raw_rows(**kw), COLS)
    return T.derive_signals(T.build_frames(raw)), raw


def test_speed_and_aperture_known_values(spark):
    sig, raw = _signals(spark)
    rows = sig.collect()
    assert len(rows) == 60 and raw.count() == 60 * 4
    mid = [r for r in rows if 5 <= r["frame_idx"] <= 54]
    assert all(r["speed"] == pytest.approx(600 / 150, rel=0.02) for r in mid)      # 4 hand-lengths/s
    assert all(r["aperture"] == pytest.approx(0.5, rel=1e-6) for r in mid)
    assert rows[0]["speed"] is None and rows[-1]["speed"] is None                    # no neighbour at the ends
    assert rows[0]["hand_size_px"] == pytest.approx(150.0)


def test_undetected_frames_stay_null_and_poison_only_their_neighbours(spark):
    sig, _ = _signals(spark, gap=(30,))
    r = {x["frame_idx"]: x for x in sig.collect()}
    assert r[30]["missing"] and r[30]["speed"] is None and r[30]["aperture"] is None
    assert r[30]["speed_smooth"] is None and r[30]["aperture_smooth"] is None   # smoothing does not fill the gap
    assert r[29]["speed"] is None and r[31]["speed"] is None   # central difference needs both neighbours
    assert r[28]["speed"] is not None and r[32]["speed"] is not None
    assert len(r) == 60                                         # no row dropped


def test_nan_in_parquet_style_input_is_treated_as_null(spark):
    import spark_transforms as T
    rows = _raw_rows(gap=(10,))
    rows = [tuple(float("nan") if (v is None and i in (4, 5, 6, 7, 8, 9)) else v for i, v in enumerate(r)) for r in rows]
    assert any(v != v for r in rows for v in r if isinstance(v, float))  # really contains NaN
    sig = T.derive_signals(T.build_frames(spark.createDataFrame(rows, COLS)))
    r = {x["frame_idx"]: x for x in sig.collect()}
    assert r[10]["speed"] is None and r[10]["aperture"] is None


def test_gaps_and_islands_events_and_enrichment(spark):
    import spark_transforms as T
    labs = ["REST"] * 5 + ["REACH"] * 3 + ["REST"] * 2          # a label repeats later: must be separate events
    lab = spark.createDataFrame([("t", i / 30, l) for i, l in enumerate(labs)], ["take", "t_s", "predicted_label"])
    takes = spark.createDataFrame([("t", 9 / 30)], ["take", "duration_s"])
    ev = T.build_events(lab, takes).orderBy("event_idx").collect()
    assert [(e["label"], e["event_idx"]) for e in ev] == [("REST", 0), ("REACH", 1), ("REST", 2)]
    assert ev[0]["start_s"] == 0 and ev[0]["end_s"] == pytest.approx(5 / 30) and ev[1]["end_s"] == pytest.approx(8 / 30)
    assert ev[2]["end_s"] == pytest.approx(9 / 30)               # last event ends at the take's last frame
    assert sum(e["n_samples"] for e in ev) == 10

    sig, _ = _signals(spark, n=10)
    en = T.enrich_events(T.build_events(lab, takes), sig).orderBy("event_idx").collect()
    assert [e["n_frames"] for e in en] == [5, 3, 2]              # every frame in exactly one event, last inclusive


def test_generated_notebook_is_up_to_date_and_valid_python():
    import ast
    from build_databricks_notebook import build
    assert NOTEBOOK.read_text() == build(), "run scripts/build_databricks_notebook.py"
    ast.parse(NOTEBOOK.read_text())
    assert NOTEBOOK.read_text().startswith("# Databricks notebook source")
    src = (ROOT / "databricks" / "spark_transforms.py").read_text()
    assert src.strip() in NOTEBOOK.read_text()                    # tested code == notebook code


def test_notebook_runs_end_to_end_locally_and_all_checks_pass(spark, tmp_path):
    """Executes the exact notebook text against data/export (BASE/OUT redirected, display stubbed)."""
    text = NOTEBOOK.read_text()
    base_line = 'BASE = "dbfs:/FileStore/tables/motion"'
    assert base_line in text
    text = text.replace(base_line, f'BASE = "{ROOT / "data" / "export"}"').replace(
        'OUT = BASE + "/spark_output"', f'OUT = "{tmp_path}/out"')
    ns = {"spark": spark, "display": lambda df: None, "__name__": "notebook"}
    exec(compile(text, "motion_pipeline_spark.py", "exec"), ns)
    checks = ns["checks"]
    failed = [c for c in checks if not c[1]]
    assert not failed, failed
    assert len(checks) >= 12
    assert ns["events"].count() == 101
    assert (tmp_path / "out" / "events").exists()
