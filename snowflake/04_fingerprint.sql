-- Phase 4 / 04_fingerprint.sql  (generated, OPTIONAL). Value-level check: row counts can match while values are wrong
-- (a mis-parsed boolean, a truncated number). Each row is one aggregate over loaded data; the expected values come from the
-- source files (snowflake/expected_results/fingerprint.csv). Sums are rounded to 4 decimals.
USE SCHEMA MOTION_INTENT.PIPELINE;

SELECT table_name, take, metric, value FROM (
    SELECT 'raw_keypoints' AS table_name, take, 'sum_x' AS metric, ROUND(SUM(x), 4) AS value FROM raw_keypoints GROUP BY take
    UNION ALL SELECT 'raw_keypoints', take, 'sum_y', ROUND(SUM(y), 4) FROM raw_keypoints GROUP BY take
    UNION ALL SELECT 'raw_keypoints', take, 'sum_z', ROUND(SUM(z), 4) FROM raw_keypoints GROUP BY take
    UNION ALL SELECT 'raw_keypoints', take, 'sum_world_x', ROUND(SUM(world_x), 4) FROM raw_keypoints GROUP BY take
    UNION ALL SELECT 'raw_keypoints', take, 'sum_world_y', ROUND(SUM(world_y), 4) FROM raw_keypoints GROUP BY take
    UNION ALL SELECT 'raw_keypoints', take, 'sum_world_z', ROUND(SUM(world_z), 4) FROM raw_keypoints GROUP BY take
    UNION ALL SELECT 'raw_keypoints', take, 'sum_handedness_score', ROUND(SUM(handedness_score), 4) FROM raw_keypoints GROUP BY take
    UNION ALL SELECT 'raw_keypoints', take, 'detected_rows', SUM(CASE WHEN detected THEN 1 ELSE 0 END) FROM raw_keypoints GROUP BY take
    UNION ALL SELECT 'raw_keypoints', take, 'left_handed_rows', SUM(CASE WHEN handedness = 'Left' THEN 1 ELSE 0 END) FROM raw_keypoints GROUP BY take
    UNION ALL SELECT 'raw_keypoints', take, 'sum_timestamp_ms', SUM(timestamp_ms) FROM raw_keypoints GROUP BY take
    UNION ALL SELECT 'raw_keypoints', take, 'sum_keypoint_id', SUM(keypoint_id) FROM raw_keypoints GROUP BY take
    UNION ALL SELECT 'frames', take, 'sum_wrist_x', ROUND(SUM(wrist_x), 4) FROM frames GROUP BY take
    UNION ALL SELECT 'frames', take, 'sum_wrist_y', ROUND(SUM(wrist_y), 4) FROM frames GROUP BY take
    UNION ALL SELECT 'frames', take, 'sum_thumb_tip_x', ROUND(SUM(thumb_tip_x), 4) FROM frames GROUP BY take
    UNION ALL SELECT 'frames', take, 'sum_index_tip_y', ROUND(SUM(index_tip_y), 4) FROM frames GROUP BY take
    UNION ALL SELECT 'frames', take, 'sum_t_s', ROUND(SUM(t_s), 4) FROM frames GROUP BY take
    UNION ALL SELECT 'frames', take, 'detected_frames', SUM(CASE WHEN detected THEN 1 ELSE 0 END) FROM frames GROUP BY take
    UNION ALL SELECT 'signals', take, 'sum_speed', ROUND(SUM(speed), 4) FROM signals GROUP BY take
    UNION ALL SELECT 'signals', take, 'sum_aperture', ROUND(SUM(aperture), 4) FROM signals GROUP BY take
    UNION ALL SELECT 'signals', take, 'sum_progress', ROUND(SUM(progress), 4) FROM signals GROUP BY take
    UNION ALL SELECT 'signals', take, 'sum_height', ROUND(SUM(height), 4) FROM signals GROUP BY take
    UNION ALL SELECT 'signals', take, 'missing_samples', SUM(CASE WHEN missing THEN 1 ELSE 0 END) FROM signals GROUP BY take
    UNION ALL SELECT 'signals', take, 'interpolated_samples', SUM(CASE WHEN interpolated THEN 1 ELSE 0 END) FROM signals GROUP BY take
    UNION ALL SELECT 'signals', take, 'null_ground_truth_labels', SUM(CASE WHEN ground_truth_label IS NULL THEN 1 ELSE 0 END) FROM signals GROUP BY take
    UNION ALL SELECT 'events', take, 'sum_duration_s', ROUND(SUM(duration_s), 4) FROM events GROUP BY take
    UNION ALL SELECT 'events', take, 'sum_mean_speed', ROUND(SUM(mean_speed), 4) FROM events GROUP BY take
    UNION ALL SELECT 'events', take, 'sum_mean_aperture', ROUND(SUM(mean_aperture), 4) FROM events GROUP BY take
    UNION ALL SELECT 'events', take, 'sum_frac_missing', ROUND(SUM(frac_missing), 4) FROM events GROUP BY take
    UNION ALL SELECT 'ground_truth', take, 'sum_start_s', ROUND(SUM(start_s), 4) FROM ground_truth GROUP BY take
    UNION ALL SELECT 'ground_truth', take, 'sum_end_s', ROUND(SUM(end_s), 4) FROM ground_truth GROUP BY take
    UNION ALL SELECT 'ground_truth', take, 'max_cycle', MAX(cycle) FROM ground_truth GROUP BY take
    UNION ALL SELECT 'takes', take, 'n_frames', n_frames FROM takes
    UNION ALL SELECT 'takes', take, 'duration_s', ROUND(duration_s, 4) FROM takes
    UNION ALL SELECT 'takes', take, 'fraction_not_detected', ROUND(fraction_not_detected, 6) FROM takes
    UNION ALL SELECT 'scores', take, 'sum_frame_accuracy', ROUND(SUM(frame_accuracy), 6) FROM scores GROUP BY take
    UNION ALL SELECT 'scores', take, 'sum_balanced_accuracy', ROUND(SUM(balanced_accuracy), 6) FROM scores GROUP BY take
) f
ORDER BY table_name, metric, take;
