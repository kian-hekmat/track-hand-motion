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
