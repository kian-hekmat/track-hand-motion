"""Generate databricks/motion_pipeline_spark.py (Databricks 'source' notebook format) from
databricks/spark_transforms.py so the notebook always contains exactly the tested code.
Import into Databricks: Workspace -> Import -> upload the .py file (it becomes a notebook)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEP = "\n\n# COMMAND ----------\n\n"


def md(text: str) -> str:
    return "\n".join("# MAGIC " + l if l else "# MAGIC" for l in ("%md\n" + text.strip()).splitlines())


def build() -> str:
    transforms = (ROOT / "databricks" / "spark_transforms.py").read_text()
    cells = [
        "# Databricks notebook source\n" + md("""
# Motion Intent Pipeline: Phase 3 (Spark)
Reads the raw hand keypoints, derives the motion signals with **Spark window functions**, builds the event table with a
**gaps-and-islands** query, and checks the result against the pandas pipeline (Phase 2).

What runs in Spark: signal derivation (wrist speed, hand aperture) and event construction/aggregation.
What does not: the gradient-boosting phase classifier and `ruptures` segmentation stay in Python; their per-frame labels
(`signals.csv`, column `predicted_label`) are an input here.

**Before running:** upload the 5 input files (see the next cell) and set `BASE`."""),
        md("""
## 0. Inputs
Upload these files from `data/export/` of the repo into one folder (Catalog / Data -> Upload, or a Unity Catalog Volume):
`raw_keypoints.parquet`, `signals.csv`, `events.csv`, `frames.csv`, `takes.csv`.
Set `BASE` to that folder: `dbfs:/FileStore/tables/motion` (classic DBFS) or `/Volumes/<catalog>/<schema>/<volume>/motion`."""),
        'BASE = "dbfs:/FileStore/tables/motion"  # EDIT THIS to the folder you uploaded the files to\n'
        'OUT = BASE + "/spark_output"\n'
        'try:\n    display(dbutils.fs.ls(BASE))\nexcept NameError:\n    pass  # not running on Databricks',
        md("## 1. Spark transforms (generated from `databricks/spark_transforms.py`, tested locally)"),
        transforms,
        md("## 2. Load"),
        'from pyspark.sql import functions as F\n\n'
        'rd = lambda name: spark.read.csv(f"{BASE}/{name}.csv", header=True, inferSchema=True)\n'
        'raw = spark.read.parquet(f"{BASE}/raw_keypoints.parquet")\n'
        'pd_signals, pd_events, pd_frames, takes = rd("signals"), rd("events"), rd("frames"), rd("takes")\n'
        'print("raw keypoint rows:", raw.count())',
        md("## 3. Stage A: frames and signals (window functions partitioned by take)"),
        'frames = build_frames(raw)\nsignals = derive_signals(frames)\n'
        'display(signals.filter("take = \'vid1\' AND frame_idx BETWEEN 150 AND 160"))',
        md("## 4. Checks A: row counts and missing data"),
        'checks = []\n\n'
        'def check(name, ok, detail=""):\n'
        '    checks.append((name, bool(ok), str(detail)))\n'
        '    print(("PASS " if ok else "FAIL ") + name + ("  " + str(detail) if detail else ""))\n\n'
        'raw_n = {r["take"]: r["n"] for r in raw.groupBy("take").agg(F.count("*").alias("n")).collect()}\n'
        'fr_n = {r["take"]: r["n"] for r in frames.groupBy("take").agg(F.count("*").alias("n")).collect()}\n'
        'sg_n = {r["take"]: r["n"] for r in signals.groupBy("take").agg(F.count("*").alias("n")).collect()}\n'
        'pd_fr_n = {r["take"]: r["n"] for r in pd_frames.groupBy("take").agg(F.count("*").alias("n")).collect()}\n'
        'tk = {r["take"]: r for r in takes.collect()}\n\n'
        'check("raw rows == 21 x frames (every take)", all(raw_n[t] == 21 * fr_n[t] for t in raw_n), raw_n)\n'
        'check("total raw keypoint rows == 83160 (Postgres / CSV count)", sum(raw_n.values()) == 83160, sum(raw_n.values()))\n'
        'check("Spark frames per take == pandas frames table", fr_n == pd_fr_n, fr_n)\n'
        'check("Spark frames per take == takes.n_frames", all(fr_n[t] == tk[t]["n_frames"] for t in fr_n))\n'
        'check("Spark signal rows == frame rows (no row lost or added)", sg_n == fr_n, sg_n)\n'
        'und = {r["take"]: r["n"] for r in signals.filter("missing").groupBy("take").agg(F.count("*").alias("n")).collect()}\n'
        'check("undetected frames == takes.frames_not_detected", all(und.get(t, 0) == tk[t]["frames_not_detected"] for t in tk), und)\n'
        'check("no zero-filling: missing frames have NULL speed/aperture",\n'
        '      signals.filter("missing AND (speed IS NOT NULL OR aperture IS NOT NULL)").count() == 0)',
        md("## 5. Agreement of Spark signals with the pandas signals (informational)\n"
           "The methods differ on purpose (Spark: central difference + 5-frame moving average on native frames; pandas: 30 Hz "
           "grid + Savitzky-Golay), so equality is not expected; a high correlation shows both measure the same motion."),
        'pd_sig = pd_signals.withColumn("k", F.round(F.col("t_s") * 30).cast("int")).select("take", "k", F.col("speed").alias("p_speed"), F.col("aperture").alias("p_ap"))\n'
        'agree = (signals.withColumn("k", F.round(F.col("t_s") * 30).cast("int")).join(pd_sig, ["take", "k"])\n'
        '         .groupBy("take").agg(F.count("*").alias("n_joined"), F.corr("speed_smooth", "p_speed").alias("r_speed"),\n'
        '                              F.corr("aperture_smooth", "p_ap").alias("r_aperture")).orderBy("take"))\n'
        'display(agree)\n'
        'rows = agree.collect()\n'
        'check("Pearson r (Spark vs pandas) > 0.8 for speed and aperture on every take", all(r["r_speed"] > 0.8 and r["r_aperture"] > 0.8 for r in rows),\n'
        '      {r["take"]: (round(r["r_speed"], 3), round(r["r_aperture"], 3)) for r in rows})',
        md("## 6. Stage B: events by gaps-and-islands, then range-join enrichment"),
        'labelled = pd_signals.select("take", "t_s", "predicted_label")\n'
        'spark_events = build_events(labelled, takes)\n'
        'events = enrich_events(spark_events, signals)\n'
        'display(events.filter("take = \'vid1\'"))',
        md("## 7. Checks B: events vs the pandas pipeline"),
        'pe = pd_events.alias("b")\n'
        'j = events.alias("a").join(pe, ["take", "event_idx"], "full_outer")\n'
        'check("event count equal (Spark vs pandas)", events.count() == pd_events.count(), (events.count(), pd_events.count()))\n'
        'mism = j.filter((F.col("a.label") != F.col("b.label")) | (F.abs(F.col("a.start_s") - F.col("b.start_s")) > 1e-6) |\n'
        '                (F.abs(F.col("a.end_s") - F.col("b.end_s")) > 1e-6) | F.col("a.label").isNull() | F.col("b.label").isNull()).count()\n'
        'check("every event matches on label, start_s, end_s", mism == 0, f"{mism} mismatches")\n'
        'nf = j.filter(F.col("a.n_frames") != F.col("b.n_frames")).count()\n'
        'check("n_frames per event equals the pandas value", nf == 0, f"{nf} mismatches")\n'
        'tot = events.groupBy("take").agg(F.sum("n_frames").alias("n")).collect()\n'
        'check("events partition every frame exactly once", all(r["n"] == fr_n[r["take"]] for r in tot))\n\n'
        'print()\nprint("PHASE 3 CHECKS:", "ALL PASSED" if all(ok for _, ok, _ in checks) else "FAILURES - see FAIL lines above")\n'
        'print(len(checks), "checks run")',
        md("## 8. Write outputs (Delta tables if the workspace allows, plus CSV for export)"),
        'try:\n'
        '    signals.write.mode("overwrite").saveAsTable("motion_signals_spark")\n'
        '    events.write.mode("overwrite").saveAsTable("motion_events_spark")\n'
        '    print("saved tables motion_signals_spark, motion_events_spark")\n'
        'except Exception as e:\n'
        '    print("saveAsTable not available here:", type(e).__name__)\n'
        'try:\n'
        '    events.coalesce(1).write.mode("overwrite").option("header", True).csv(OUT + "/events")\n'
        '    signals.coalesce(1).write.mode("overwrite").option("header", True).csv(OUT + "/signals")\n'
        '    print("wrote CSV to", OUT)\n'
        'except Exception as e:\n'
        '    print("CSV write failed:", type(e).__name__, "- use display(events) and the Download CSV button")\n'
        'display(events)',
        md("""
## 9. Spark concepts used
- **Window functions partitioned by take, ordered by frame_idx:** `lag`/`lead` for central-difference wrist velocity with real (variable) frame timestamps; `avg` over `rowsBetween(-2, 2)` for smoothing.
- **groupBy/agg with an exact `percentile`:** per-take median hand size, joined back with a broadcast join.
- **Conditional aggregation:** pivots 21 landmark rows per frame into one wide row without `pivot()`.
- **Gaps-and-islands:** `lag` to flag label changes, a cumulative-sum window to number the runs, `groupBy/agg` to collapse each run into an event, `lead` for the end time.
- **Range join:** events joined to per-frame signals on `start_s <= t_s < end_s` for per-event statistics."""),
    ]
    return SEP.join(c.strip("\n") if i else c for i, c in enumerate(cells)).rstrip() + "\n"


if __name__ == "__main__":
    out = ROOT / "databricks" / "motion_pipeline_spark.py"
    out.write_text(build())
    print("wrote", out, f"({out.stat().st_size} bytes)")
