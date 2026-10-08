"""Cloud path, milestone M2: bronze -> silver -> gold as Spark DataFrame transforms.

Pure functions over DataFrames (no I/O except reading the landing files), so the same code runs on a local Spark session in
the tests and on Databricks serverless. The notebook databricks/cloud/m2_build_tables.py writes the results as Delta tables.

bronze  read_landing()          the uploaded files, typed with fixed schemas, plus the source file name
silver  build_frames()          21 landmark rows -> one row per frame (conditional aggregation), same columns as
                                src.signals.load_frames
        predict_signals()       per take, in parallel (groupBy().applyInPandas): the repo's own signal, feature, model and
                                PELT code, with the model that may label that take honestly (src.final.load_model_for_take)
gold    build_events()          per-sample labels -> events (gaps-and-islands with window functions)
        label_frames()          native video frames x events / ground truth / out-of-frame intervals (range joins)
        frame_scores()          frame accuracy, per-label accuracy, balanced accuracy, majority baseline (groupBy/agg)

Interval rules copy src.ground_truth.labels_at and in_intervals exactly, including the inclusive end of the last segment
(numpy.isclose: |t - end| <= 1e-9 + 1e-5 * |end|), so the scores are comparable with the verified local ones.
"""
import numpy as np
import pandas as pd
from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql import types as T

from config import LABELS
from src.signals import KEYPOINTS

# ---------------- bronze: landing files with fixed schemas ----------------
RAW_KEYPOINTS_SCHEMA = T.StructType([
    T.StructField("take", T.StringType()), T.StructField("frame_idx", T.IntegerType()),
    T.StructField("timestamp_ms", T.LongType()), T.StructField("keypoint_id", T.IntegerType()),
    *[T.StructField(c, T.DoubleType()) for c in ("x", "y", "z", "world_x", "world_y", "world_z")],
    T.StructField("detected", T.BooleanType()), T.StructField("handedness", T.StringType()),
    T.StructField("handedness_score", T.DoubleType()),
])
GROUND_TRUTH_SCHEMA = T.StructType([
    T.StructField("take", T.StringType()), T.StructField("cycle", T.IntegerType()), T.StructField("label", T.StringType()),
    T.StructField("start_s", T.DoubleType()), T.StructField("end_s", T.DoubleType()), T.StructField("source", T.StringType()),
])
OUT_OF_FRAME_SCHEMA = T.StructType([
    T.StructField("take", T.StringType()), T.StructField("start_s", T.DoubleType()),
    T.StructField("end_s", T.DoubleType()), T.StructField("cause", T.StringType()),
])
TAKE_META_SCHEMA = T.StructType([
    T.StructField("take", T.StringType()), T.StructField("frames_extracted", T.LongType()),
    T.StructField("nominal_fps", T.DoubleType()), T.StructField("last_timestamp_s", T.DoubleType()),
    T.StructField("mean_fps_from_timestamps", T.DoubleType()), T.StructField("frames_not_detected", T.LongType()),
    T.StructField("fraction_not_detected", T.DoubleType()),
    T.StructField("handedness_counts", T.StructType([T.StructField("Right", T.LongType()), T.StructField("Left", T.LongType())])),
])


def _csv(spark: SparkSession, path: str, schema: T.StructType) -> DataFrame:
    # enforceSchema=False: the header must match the schema's column names, so a reordered file fails instead of loading wrong
    return (spark.read.csv(path, header=True, schema=schema, enforceSchema=False, mode="FAILFAST")
            .withColumn("_source_file", F.col("_metadata.file_name")))


def read_landing(spark: SparkSession, landing: str) -> dict:
    """landing/raw/: <take>_keypoints.csv, <take>_meta.json, out_of_frame_intervals.csv; landing/ground_truth/take_N.csv."""
    meta = (spark.read.option("multiLine", True).schema(TAKE_META_SCHEMA).json(f"{landing}/raw/*_meta.json")
            .withColumn("_source_file", F.col("_metadata.file_name")))
    return {
        "raw_keypoints": _csv(spark, f"{landing}/raw/*_keypoints.csv", RAW_KEYPOINTS_SCHEMA),
        "take_meta": meta,
        "ground_truth": _csv(spark, f"{landing}/ground_truth/take_*.csv", GROUND_TRUTH_SCHEMA),
        "out_of_frame_intervals": _csv(spark, f"{landing}/raw/out_of_frame_intervals.csv", OUT_OF_FRAME_SCHEMA),
    }


# ---------------- silver: frames ----------------
FRAME_COLUMNS = ["t", "detected"] + [c for k in KEYPOINTS for c in
                                     (f"i{k}_x", f"i{k}_y", f"w{k}_x", f"w{k}_y", f"w{k}_z")]


def build_frames(raw_keypoints: DataFrame) -> DataFrame:
    """One row per (take, frame_idx). Each landmark value is taken from the single row that holds it (max over one
    non-null value returns it unchanged). t = timestamp_ms / 1000, as in src.signals.load_frames."""
    k = raw_keypoints.filter(F.col("keypoint_id").isin(list(KEYPOINTS)))
    cols = []
    for kid in KEYPOINTS:
        for c in ("x", "y"):
            cols.append(F.max(F.when(F.col("keypoint_id") == kid, F.col(c))).alias(f"i{kid}_{c}"))
        for c in ("x", "y", "z"):
            cols.append(F.max(F.when(F.col("keypoint_id") == kid, F.col(f"world_{c}"))).alias(f"w{kid}_{c}"))
    frames = (k.groupBy("take", "frame_idx")
              .agg(F.max("timestamp_ms").alias("timestamp_ms"), F.max("detected").alias("detected"),
                   F.max("handedness").alias("handedness"), F.max("handedness_score").alias("handedness_score"), *cols)
              .withColumn("t", F.col("timestamp_ms") / F.lit(1000.0)))
    return frames.select("take", "frame_idx", *FRAME_COLUMNS, "handedness", "handedness_score")


# ---------------- silver: signals and predictions (the repo's own Python, one take per group) ----------------
SIGNAL_COLUMNS = ["t", "speed", "aperture", "aperture_slope", "p", "dp", "q", "dq", "ydev", "dy"]
FLAG_COLUMNS = ["missing", "interpolated"]
PROB_COLUMNS = [f"p_{l}" for l in LABELS]
SIGNALS_SCHEMA = T.StructType(
    [T.StructField("take", T.StringType()), T.StructField("sample_idx", T.IntegerType())]
    + [T.StructField(c, T.DoubleType()) for c in SIGNAL_COLUMNS]
    + [T.StructField(c, T.BooleanType()) for c in FLAG_COLUMNS]
    + [T.StructField(c, T.DoubleType()) for c in PROB_COLUMNS]
    + [T.StructField("predicted_label", T.StringType()), T.StructField("p_label_smoothed", T.DoubleType()),
       T.StructField("model", T.StringType())])


def make_predictor(models: dict, meta: dict):
    """models: take -> (fitted model, model file name), loaded and hash-checked on the driver. The function is shipped to
    the workers with the models inside it, so workers never read model files."""

    def predict(pdf: pd.DataFrame) -> pd.DataFrame:
        from src import ground_truth as G
        from src.final import segment_frames
        from src.learned import _smooth

        take = pdf["take"].iloc[0]
        model, name = models[take]
        frames = pdf.sort_values("frame_idx")[FRAME_COLUMNS].reset_index(drop=True)
        frames["detected"] = frames["detected"].astype(bool)
        ev, sig, P = segment_frames(frames, take, model=model, meta=meta)
        Ps = _smooth(P, meta["posterior_smooth"])
        labels = G.labels_at(ev, sig["t"].to_numpy())
        lab_idx = np.array([LABELS.index(l) for l in labels])
        out = sig[SIGNAL_COLUMNS + FLAG_COLUMNS].reset_index(drop=True)
        out.insert(0, "sample_idx", np.arange(len(out), dtype=np.int32))
        out.insert(0, "take", take)
        for i, c in enumerate(PROB_COLUMNS):
            out[c] = P[:, i]
        out["predicted_label"] = labels
        out["p_label_smoothed"] = Ps[np.arange(len(Ps)), lab_idx]
        out["model"] = name
        return out

    return predict


def predict_signals(frames: DataFrame, models: dict, meta: dict) -> DataFrame:
    cols = ["take", "frame_idx"] + FRAME_COLUMNS
    return frames.select(*cols).groupBy("take").applyInPandas(make_predictor(models, meta), schema=SIGNALS_SCHEMA)


# ---------------- gold: events (gaps-and-islands) ----------------
def build_events(signals: DataFrame, frames: DataFrame) -> DataFrame:
    """A new event starts where the label differs from the previous sample's; a running sum of those starts numbers the
    events. end = next event's start, or the take's last frame time for the last event (as src.learned._events)."""
    w = Window.partitionBy("take").orderBy("sample_idx")
    prev = F.lag("predicted_label").over(w)
    s = (signals.withColumn("is_start", F.when(prev.isNull() | (prev != F.col("predicted_label")), 1).otherwise(0))
         .withColumn("event_idx", (F.sum("is_start").over(w.rowsBetween(Window.unboundedPreceding, 0)) - 1).cast("int")))
    ev = s.groupBy("take", "event_idx").agg(
        F.first("predicted_label").alias("label"), F.min("t").alias("start_s"), F.count("*").cast("int").alias("n_samples"),
        F.avg("p_label_smoothed").alias("mean_confidence"), F.countDistinct("predicted_label").alias("labels_in_event"),
        F.first("model").alias("model"))
    t_end = frames.groupBy("take").agg(F.max("t").alias("_t_end"))
    we = Window.partitionBy("take").orderBy("event_idx")
    return (ev.join(t_end, "take")
            .withColumn("end_s", F.coalesce(F.lead("start_s").over(we), F.col("_t_end")))
            .withColumn("duration_s", F.col("end_s") - F.col("start_s"))
            .select("take", "event_idx", "label", "start_s", "end_s", "duration_s", "n_samples", "mean_confidence",
                    "model", "labels_in_event"))


# ---------------- gold: labels per native frame, and scores ----------------
def _label_at(points: DataFrame, segments: DataFrame, out_col: str) -> DataFrame:
    """Range join: the segment with start <= t < end; the last segment of a take also holds t ~= its end
    (numpy.isclose defaults, as src.ground_truth.labels_at). Frames outside every segment get NULL."""
    last = Window.partitionBy("take")
    seg = (segments.select("take", "label", "start_s", "end_s")
           .withColumn("_is_last", F.col("start_s") == F.max("start_s").over(last))
           .withColumnRenamed("take", "_seg_take"))
    inside = (F.col("t") >= F.col("start_s")) & (
        (F.col("t") < F.col("end_s"))
        | (F.col("_is_last") & (F.abs(F.col("t") - F.col("end_s")) <= F.lit(1e-9) + F.lit(1e-5) * F.abs(F.col("end_s")))))
    j = points.join(seg, (points["take"] == seg["_seg_take"]) & inside, "left")
    return j.select(*points.columns, F.col("label").alias(out_col))


def label_frames(frames: DataFrame, events: DataFrame, ground_truth: DataFrame, out_of_frame: DataFrame) -> DataFrame:
    f = frames.select("take", "frame_idx", "t")
    f = _label_at(f, events, "predicted_label")
    f = _label_at(f, ground_truth, "true_label")
    oof = out_of_frame.select(F.col("take").alias("_o_take"), F.col("start_s").alias("_o_a"), F.col("end_s").alias("_o_b"))
    hit = (f.join(oof, (f["take"] == oof["_o_take"]) & (F.col("t") >= F.col("_o_a") - 1e-9) & (F.col("t") < F.col("_o_b") - 1e-9))
           .select("take", "frame_idx").distinct().withColumn("out_of_frame", F.lit(True)))
    return (f.join(hit, ["take", "frame_idx"], "left").fillna({"out_of_frame": False})
            .select("take", "frame_idx", "t", "predicted_label", "true_label", "out_of_frame"))


SCORE_COLUMNS = (["take", "scope", "n_frames", "n_unlabelled", "frame_accuracy"]
                 + [c for l in LABELS for c in (f"acc_{l}", f"n_{l}")] + ["balanced_accuracy", "majority_baseline"])


def frame_scores(frame_labels: DataFrame) -> DataFrame:
    """Scope 'all' = frames with hand data (out-of-frame frames excluded), scope 'out_of_frame' = only those frames (a row
    only for takes that have them), as src.evaluate.score_take."""
    fl = frame_labels.withColumn("scope", F.when(F.col("out_of_frame"), "out_of_frame").otherwise("all"))
    labelled = F.col("predicted_label").isNotNull() & F.col("true_label").isNotNull()
    counts = fl.groupBy("take", "scope").agg(
        F.sum(labelled.cast("int")).alias("n_frames"), F.sum((~labelled).cast("int")).alias("n_unlabelled"),
        F.avg(F.when(labelled, (F.col("predicted_label") == F.col("true_label")).cast("int"))).alias("frame_accuracy"))
    per = (fl.filter(labelled).groupBy("take", "scope", "true_label")
           .agg(F.count("*").alias("n"), F.avg((F.col("predicted_label") == F.col("true_label")).cast("int")).alias("acc")))
    wide = per.groupBy("take", "scope").pivot("true_label", list(LABELS)).agg(F.first("n").alias("n"), F.first("acc").alias("acc"))
    summary = per.groupBy("take", "scope").agg(F.avg("acc").alias("balanced_accuracy"),
                                               (F.max("n") / F.sum("n")).alias("majority_baseline"))
    out = counts.join(wide, ["take", "scope"], "left").join(summary, ["take", "scope"], "left")
    for l in LABELS:
        out = out.withColumn(f"acc_{l}", F.col(f"{l}_acc")).withColumn(f"n_{l}", F.coalesce(F.col(f"{l}_n"), F.lit(0)))
    return out.select(*SCORE_COLUMNS)
