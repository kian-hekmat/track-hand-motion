# Databricks notebook source
# MAGIC %md
# MAGIC # Motion Intent Pipeline: Phase 3 (Spark)
# MAGIC Reads the raw hand keypoints, derives the motion signals with **Spark window functions**, builds the event table with a
# MAGIC **gaps-and-islands** query, and checks the result against the pandas pipeline (Phase 2).
# MAGIC
# MAGIC What runs in Spark: signal derivation (wrist speed, hand aperture) and event construction/aggregation.
# MAGIC What does not: the gradient-boosting phase classifier and `ruptures` segmentation stay in Python; their per-frame labels
# MAGIC (`signals.csv`, column `predicted_label`) are an input here.
# MAGIC
# MAGIC **Before running:** upload the 5 input files (see the next cell) and set `BASE`.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Inputs
# MAGIC Upload these files from `data/export/` of the repo into one folder (Catalog / Data -> Upload, or a Unity Catalog Volume):
# MAGIC `raw_keypoints.parquet`, `signals.csv`, `events.csv`, `frames.csv`, `takes.csv`.
# MAGIC Set `BASE` to that folder: `dbfs:/FileStore/tables/motion` (classic DBFS) or `/Volumes/<catalog>/<schema>/<volume>/motion`.

# COMMAND ----------

BASE = "dbfs:/FileStore/tables/motion"  # EDIT THIS to the folder you uploaded the files to
OUT = BASE + "/spark_output"
try:
    display(dbutils.fs.ls(BASE))
except NameError:
    pass  # not running on Databricks

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Spark transforms (generated from `databricks/spark_transforms.py`, tested locally)

# COMMAND ----------

"""PySpark transforms for the Motion Intent Pipeline (Phase 3).

Pure DataFrame API (no RDDs, no .cache(), no project imports) so the same source runs on a local Spark,
a Databricks cluster and Databricks serverless.

Stage A  build_frames + derive_signals:  raw keypoints (one row per frame x landmark)
         -> one row per frame -> wrist speed and hand aperture (window functions over take)
Stage B  build_events + enrich_events:   per-sample labels -> events (gaps-and-islands) -> per-event stats
"""
from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

FRAME_W, FRAME_H = 1920, 1080          # all takes are 1920x1080 landscape
KEYPOINTS = (0, 4, 8, 9)               # wrist, thumb tip, index tip, middle-finger MCP
SMOOTH_HALF_WINDOW = 2                 # +/-2 frames moving average


def _nan_to_null(df: DataFrame) -> DataFrame:
    """pandas-written parquet stores missing floats as NaN; Spark aggregates/windows want NULL."""
    for name, dtype in df.dtypes:
        if dtype in ("double", "float"):
            df = df.withColumn(name, F.when(F.isnan(F.col(name)), None).otherwise(F.col(name)))
    return df


# ------------------------------------------------------------------ Stage A
def build_frames(raw: DataFrame) -> DataFrame:
    """raw_keypoints (take, frame_idx, timestamp_ms, keypoint_id, x, y, z, world_x/y/z, detected, ...)
    -> one row per (take, frame_idx) with the four landmarks we need as columns.
    Conditional aggregation = a pivot without the shuffle of pivot()."""
    raw = _nan_to_null(raw).filter(F.col("keypoint_id").isin(*KEYPOINTS))
    aggs = [F.first("timestamp_ms").alias("timestamp_ms"), F.first("detected").alias("detected"),
            F.first("handedness").alias("handedness"), F.first("handedness_score").alias("handedness_score")]
    for k in KEYPOINTS:
        is_k = F.col("keypoint_id") == k
        for c in ("x", "y"):
            aggs.append(F.max(F.when(is_k, F.col(c))).alias(f"i{k}_{c}"))
        for c in ("world_x", "world_y", "world_z"):
            aggs.append(F.max(F.when(is_k, F.col(c))).alias(f"w{k}_{c[-1]}"))
    return (raw.groupBy("take", "frame_idx").agg(*aggs)
               .withColumn("t_s", F.col("timestamp_ms") / 1000.0))


def derive_signals(frames: DataFrame) -> DataFrame:
    """Per-frame wrist speed (hand-lengths/s) and hand aperture (thumb-index / hand length).

    Window functions partitioned by take, ordered by frame_idx:
      * lag/lead  -> central-difference wrist velocity using the REAL frame timestamps (iPhone video is VFR)
      * avg over rowsBetween(-2, 2) -> moving-average smoothing (NULLs are ignored by avg)
    Undetected frames stay NULL in every signal column (never zero-filled, never interpolated); a NULL neighbour makes
    a neighbouring frame's central difference NULL, and the moving average ignores NULLs. hand size = per-take median wrist->middle-MCP pixel length (exact percentile)."""
    w = Window.partitionBy("take").orderBy("frame_idx")
    x = F.col("i0_x") * FRAME_W
    y = F.col("i0_y") * FRAME_H
    hand_px = F.sqrt(F.pow((F.col("i0_x") - F.col("i9_x")) * FRAME_W, 2) + F.pow((F.col("i0_y") - F.col("i9_y")) * FRAME_H, 2))

    df = frames.withColumn("wrist_x_px", x).withColumn("wrist_y_px", y).withColumn("hand_px", hand_px)
    hand_size = df.groupBy("take").agg(F.expr("percentile(hand_px, 0.5)").alias("hand_size_px"))   # groupBy/agg
    df = df.join(F.broadcast(hand_size), "take")

    dx = F.lead("wrist_x_px", 1).over(w) - F.lag("wrist_x_px", 1).over(w)
    dy = F.lead("wrist_y_px", 1).over(w) - F.lag("wrist_y_px", 1).over(w)
    dt = F.lead("t_s", 1).over(w) - F.lag("t_s", 1).over(w)
    # central difference skips the centre frame, so mask undetected frames explicitly (no silent interpolation)
    df = df.withColumn("speed", F.when(F.col("detected"), F.sqrt(dx * dx + dy * dy) / dt / F.col("hand_size_px")))

    def dist3(a, b):
        return F.sqrt(sum(F.pow(F.col(f"w{a}_{c}") - F.col(f"w{b}_{c}"), 2) for c in "xyz"))
    df = df.withColumn("aperture", dist3(4, 8) / dist3(0, 9))

    sm = w.rowsBetween(-SMOOTH_HALF_WINDOW, SMOOTH_HALF_WINDOW)
    return (df.withColumn("speed_smooth", F.when(F.col("detected"), F.avg("speed").over(sm)))
              .withColumn("aperture_smooth", F.when(F.col("detected"), F.avg("aperture").over(sm)))
              .withColumn("missing", ~F.col("detected"))
              .select("take", "frame_idx", "t_s", "detected", "missing", "hand_size_px",
                      "speed", "speed_smooth", "aperture", "aperture_smooth")
              .orderBy("take", "frame_idx"))


# ------------------------------------------------------------------ Stage B
def build_events(labelled: DataFrame, takes: DataFrame) -> DataFrame:
    """Per-sample labels (take, t_s, predicted_label) -> events, by 'gaps and islands':
       1. lag(label) over (take order by t_s)           -> flag the first sample of every run
       2. cumulative sum of the flag (unbounded window) -> island id
       3. groupBy(take, island).agg(min t_s, first label)
       4. end_s = start of the next event (lead); the last event ends at the take's last frame time."""
    w = Window.partitionBy("take").orderBy("t_s")
    flagged = (labelled.withColumn("prev", F.lag("predicted_label").over(w))
                       .withColumn("is_start", F.when(F.col("prev").isNull() | (F.col("prev") != F.col("predicted_label")), 1).otherwise(0))
                       .withColumn("island", F.sum("is_start").over(w.rowsBetween(Window.unboundedPreceding, Window.currentRow))))
    ev = flagged.groupBy("take", "island").agg(F.min("t_s").alias("start_s"), F.first("predicted_label").alias("label"),
                                               F.count("*").alias("n_samples"))
    w2 = Window.partitionBy("take").orderBy("start_s")
    ev = (ev.join(F.broadcast(takes.select("take", F.col("duration_s").alias("take_end_s"))), "take")
            .withColumn("end_s", F.coalesce(F.lead("start_s").over(w2), F.col("take_end_s")))
            .withColumn("event_idx", F.row_number().over(w2) - 1)
            .withColumn("duration_s", F.col("end_s") - F.col("start_s")))
    return ev.select("take", "event_idx", "label", "start_s", "end_s", "duration_s", "n_samples")


def enrich_events(events: DataFrame, signals: DataFrame) -> DataFrame:
    """Range join of events to the per-frame Spark signals: frames with start_s <= t_s < end_s (the last event of a
    take includes its final frame) -> n_frames and mean speed/aperture computed from the Spark-derived signals."""
    last = F.col("event_idx") == F.max("event_idx").over(Window.partitionBy("take"))
    ev = events.withColumn("is_last", last)
    cond = ((F.col("s.take") == F.col("e.take")) & (F.col("s.t_s") >= F.col("e.start_s")) &
            ((F.col("s.t_s") < F.col("e.end_s")) | (F.col("e.is_last") & (F.col("s.t_s") <= F.col("e.end_s") + F.lit(1e-9)))))
    j = ev.alias("e").join(signals.alias("s"), cond, "left")
    return (j.groupBy("e.take", "e.event_idx", "e.label", "e.start_s", "e.end_s", "e.duration_s")
             .agg(F.count("s.frame_idx").alias("n_frames"),
                  F.avg("s.speed_smooth").alias("spark_mean_speed"),
                  F.avg("s.aperture_smooth").alias("spark_mean_aperture"),
                  F.avg(F.col("s.missing").cast("double")).alias("spark_frac_missing"))
             .orderBy("take", "event_idx"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Load

# COMMAND ----------

from pyspark.sql import functions as F

rd = lambda name: spark.read.csv(f"{BASE}/{name}.csv", header=True, inferSchema=True)
raw = spark.read.parquet(f"{BASE}/raw_keypoints.parquet")
pd_signals, pd_events, pd_frames, takes = rd("signals"), rd("events"), rd("frames"), rd("takes")
print("raw keypoint rows:", raw.count())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Stage A: frames and signals (window functions partitioned by take)

# COMMAND ----------

frames = build_frames(raw)
signals = derive_signals(frames)
display(signals.filter("take = 'vid1' AND frame_idx BETWEEN 150 AND 160"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Checks A: row counts and missing data

# COMMAND ----------

checks = []

def check(name, ok, detail=""):
    checks.append((name, bool(ok), str(detail)))
    print(("PASS " if ok else "FAIL ") + name + ("  " + str(detail) if detail else ""))

raw_n = {r["take"]: r["n"] for r in raw.groupBy("take").agg(F.count("*").alias("n")).collect()}
fr_n = {r["take"]: r["n"] for r in frames.groupBy("take").agg(F.count("*").alias("n")).collect()}
sg_n = {r["take"]: r["n"] for r in signals.groupBy("take").agg(F.count("*").alias("n")).collect()}
pd_fr_n = {r["take"]: r["n"] for r in pd_frames.groupBy("take").agg(F.count("*").alias("n")).collect()}
tk = {r["take"]: r for r in takes.collect()}

check("raw rows == 21 x frames (every take)", all(raw_n[t] == 21 * fr_n[t] for t in raw_n), raw_n)
check("total raw keypoint rows == 83160 (Postgres / CSV count)", sum(raw_n.values()) == 83160, sum(raw_n.values()))
check("Spark frames per take == pandas frames table", fr_n == pd_fr_n, fr_n)
check("Spark frames per take == takes.n_frames", all(fr_n[t] == tk[t]["n_frames"] for t in fr_n))
check("Spark signal rows == frame rows (no row lost or added)", sg_n == fr_n, sg_n)
und = {r["take"]: r["n"] for r in signals.filter("missing").groupBy("take").agg(F.count("*").alias("n")).collect()}
check("undetected frames == takes.frames_not_detected", all(und.get(t, 0) == tk[t]["frames_not_detected"] for t in tk), und)
check("no zero-filling: missing frames have NULL speed/aperture",
      signals.filter("missing AND (speed IS NOT NULL OR aperture IS NOT NULL)").count() == 0)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Agreement of Spark signals with the pandas signals (informational)
# MAGIC The methods differ on purpose (Spark: central difference + 5-frame moving average on native frames; pandas: 30 Hz grid + Savitzky-Golay), so equality is not expected; a high correlation shows both measure the same motion.

# COMMAND ----------

pd_sig = pd_signals.withColumn("k", F.round(F.col("t_s") * 30).cast("int")).select("take", "k", F.col("speed").alias("p_speed"), F.col("aperture").alias("p_ap"))
agree = (signals.withColumn("k", F.round(F.col("t_s") * 30).cast("int")).join(pd_sig, ["take", "k"])
         .groupBy("take").agg(F.count("*").alias("n_joined"), F.corr("speed_smooth", "p_speed").alias("r_speed"),
                              F.corr("aperture_smooth", "p_ap").alias("r_aperture")).orderBy("take"))
display(agree)
rows = agree.collect()
check("Pearson r (Spark vs pandas) > 0.8 for speed and aperture on every take", all(r["r_speed"] > 0.8 and r["r_aperture"] > 0.8 for r in rows),
      {r["take"]: (round(r["r_speed"], 3), round(r["r_aperture"], 3)) for r in rows})

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Stage B: events by gaps-and-islands, then range-join enrichment

# COMMAND ----------

labelled = pd_signals.select("take", "t_s", "predicted_label")
spark_events = build_events(labelled, takes)
events = enrich_events(spark_events, signals)
display(events.filter("take = 'vid1'"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Checks B: events vs the pandas pipeline

# COMMAND ----------

pe = pd_events.alias("b")
j = events.alias("a").join(pe, ["take", "event_idx"], "full_outer")
check("event count equal (Spark vs pandas)", events.count() == pd_events.count(), (events.count(), pd_events.count()))
mism = j.filter((F.col("a.label") != F.col("b.label")) | (F.abs(F.col("a.start_s") - F.col("b.start_s")) > 1e-6) |
                (F.abs(F.col("a.end_s") - F.col("b.end_s")) > 1e-6) | F.col("a.label").isNull() | F.col("b.label").isNull()).count()
check("every event matches on label, start_s, end_s", mism == 0, f"{mism} mismatches")
nf = j.filter(F.col("a.n_frames") != F.col("b.n_frames")).count()
check("n_frames per event equals the pandas value", nf == 0, f"{nf} mismatches")
tot = events.groupBy("take").agg(F.sum("n_frames").alias("n")).collect()
check("events partition every frame exactly once", all(r["n"] == fr_n[r["take"]] for r in tot))

print()
print("PHASE 3 CHECKS:", "ALL PASSED" if all(ok for _, ok, _ in checks) else "FAILURES - see FAIL lines above")
print(len(checks), "checks run")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Write outputs (Delta tables if the workspace allows, plus CSV for export)

# COMMAND ----------

try:
    signals.write.mode("overwrite").saveAsTable("motion_signals_spark")
    events.write.mode("overwrite").saveAsTable("motion_events_spark")
    print("saved tables motion_signals_spark, motion_events_spark")
except Exception as e:
    print("saveAsTable not available here:", type(e).__name__)
try:
    events.coalesce(1).write.mode("overwrite").option("header", True).csv(OUT + "/events")
    signals.coalesce(1).write.mode("overwrite").option("header", True).csv(OUT + "/signals")
    print("wrote CSV to", OUT)
except Exception as e:
    print("CSV write failed:", type(e).__name__, "- use display(events) and the Download CSV button")
display(events)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Spark concepts used
# MAGIC - **Window functions partitioned by take, ordered by frame_idx:** `lag`/`lead` for central-difference wrist velocity with real (variable) frame timestamps; `avg` over `rowsBetween(-2, 2)` for smoothing.
# MAGIC - **groupBy/agg with an exact `percentile`:** per-take median hand size, joined back with a broadcast join.
# MAGIC - **Conditional aggregation:** pivots 21 landmark rows per frame into one wide row without `pivot()`.
# MAGIC - **Gaps-and-islands:** `lag` to flag label changes, a cumulative-sum window to number the runs, `groupBy/agg` to collapse each run into an event, `lead` for the end time.
# MAGIC - **Range join:** events joined to per-frame signals on `start_s <= t_s < end_s` for per-event statistics.
