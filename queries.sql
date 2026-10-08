-- Motion Intent Pipeline: analytical queries for Snowflake.
-- Tables: takes, events, ground_truth, signals, frames, scores, raw_keypoints. Two schemas have this layout:
--   MOTION_INTENT.CLOUD     published by the Databricks job (all seven takes, vid1-7)
--   MOTION_INTENT.PIPELINE  the verified Phase 4 load of vid1-5, kept as the reference
-- Each query is marked with "-- name:". The M3 notebook runs every query, unchanged, in both schemas and requires identical
-- answers for vid1-5 (src/cloud/snowflake_sql.py). Columns renamed because they are reserved words: group -> take_group,
-- false -> false_boundaries. Takes are reported by kind (clean / fast / hard / holdout_*); averages are never pooled across kinds.

USE SCHEMA MOTION_INTENT.CLOUD;

-- name: q1_avg_duration_by_event_type
-- Question: how long does each kind of event last, for clean, fast and hard takes?
SELECT take_group,
       label,
       COUNT(*)                     AS n_events,
       ROUND(AVG(duration_s), 3)    AS avg_duration_s,
       ROUND(MIN(duration_s), 3)    AS min_duration_s,
       ROUND(MAX(duration_s), 3)    AS max_duration_s
FROM events
GROUP BY take_group, label
ORDER BY take_group,
         CASE label WHEN 'REST' THEN 1 WHEN 'REACH' THEN 2 WHEN 'GRASP' THEN 3
                    WHEN 'HOLD' THEN 4 WHEN 'RELEASE' THEN 5 ELSE 6 END;

-- name: q2_most_ambiguous_takes
-- Question: which take has the most ambiguous events? Ambiguous = shorter than 0.3 s, or detected with a mean phase
-- probability below 0.7 (thresholds fixed before looking at results). Rank 1 = most ambiguous.
WITH per_take AS (
    SELECT take,
           take_group,
           COUNT(*) AS n_events,
           SUM(CASE WHEN duration_s < 0.3 THEN 1 ELSE 0 END)        AS short_events,
           SUM(CASE WHEN mean_confidence < 0.7 THEN 1 ELSE 0 END)   AS low_confidence_events,
           SUM(CASE WHEN duration_s < 0.3 OR mean_confidence < 0.7 THEN 1 ELSE 0 END) AS ambiguous_events,
           ROUND(AVG(mean_confidence), 3) AS avg_confidence
    FROM events
    GROUP BY take, take_group
)
SELECT take, take_group, n_events, short_events, low_confidence_events, ambiguous_events,
       ROUND(ambiguous_events * 1.0 / n_events, 3) AS ambiguous_share,
       avg_confidence,
       RANK() OVER (ORDER BY ambiguous_events * 1.0 / n_events DESC) AS ambiguity_rank
FROM per_take
ORDER BY ambiguity_rank, take;

-- name: q3_time_between_consecutive_reaches
-- Question: how long is one full cycle? Window function: LAG over each take's REACH events, ordered by time.
WITH reaches AS (
    SELECT take,
           start_s,
           LAG(start_s) OVER (PARTITION BY take ORDER BY start_s) AS previous_reach_start_s
    FROM events
    WHERE label = 'REACH'
)
SELECT take,
       ROUND(previous_reach_start_s, 3)           AS previous_reach_start_s,
       ROUND(start_s, 3)                          AS reach_start_s,
       ROUND(start_s - previous_reach_start_s, 3) AS seconds_between_reaches
FROM reaches
WHERE previous_reach_start_s IS NOT NULL
ORDER BY take, start_s;

-- name: q4_phase_transitions
-- Question: which phase follows which, and does it match the protocol order
-- (REST > REACH > GRASP > HOLD > RELEASE > RETRACT > REST)? Window function: LEAD over each take's events.
WITH pairs AS (
    SELECT take_group,
           label AS from_label,
           LEAD(label) OVER (PARTITION BY take ORDER BY event_idx) AS to_label
    FROM events
)
SELECT take_group,
       from_label,
       to_label,
       COUNT(*) AS n_transitions,
       CASE WHEN (from_label = 'REST'    AND to_label = 'REACH')   OR (from_label = 'REACH'   AND to_label = 'GRASP')
              OR (from_label = 'GRASP'   AND to_label = 'HOLD')    OR (from_label = 'HOLD'    AND to_label = 'RELEASE')
              OR (from_label = 'RELEASE' AND to_label = 'RETRACT') OR (from_label = 'RETRACT' AND to_label = 'REST')
            THEN 'yes' ELSE 'no' END AS follows_protocol_order
FROM pairs
WHERE to_label IS NOT NULL
GROUP BY take_group, from_label, to_label
ORDER BY take_group, follows_protocol_order DESC, n_transitions DESC, from_label, to_label;

-- name: q5_prediction_agreement_by_label
-- Question: for each hand-labelled phase, how often does the detected phase agree, sample by sample (30 Hz grid)?
-- Samples with no hand data are excluded. Grid samples, not video frames, so values differ slightly from the
-- frame accuracy reported in NOTES.md.
SELECT take,
       ground_truth_label AS label,
       COUNT(*)           AS n_samples,
       ROUND(AVG(CASE WHEN predicted_label = ground_truth_label THEN 1.0 ELSE 0.0 END), 3) AS agreement
FROM signals
WHERE NOT missing AND ground_truth_label IS NOT NULL
GROUP BY take, ground_truth_label
ORDER BY take,
         CASE ground_truth_label WHEN 'REST' THEN 1 WHEN 'REACH' THEN 2 WHEN 'GRASP' THEN 3
                                 WHEN 'HOLD' THEN 4 WHEN 'RELEASE' THEN 5 ELSE 6 END;
