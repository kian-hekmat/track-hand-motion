-- Phase 4 / 03_verify.sql  (generated). Expected values come from data/export/manifest.json and the per-take
-- metadata, NOT from the tables being checked. Every row must say PASS.
USE SCHEMA MOTION_INTENT.PIPELINE;

WITH expected AS (
    SELECT * FROM (VALUES
        ('takes', 5),
        ('events', 101),
        ('ground_truth', 101),
        ('signals', 3960),
        ('frames', 3960),
        ('scores', 10),
        ('raw_keypoints', 83160)
    ) AS v(table_name, expected_rows)
), actual AS (
    SELECT 'takes' AS table_name, COUNT(*) AS actual_rows FROM takes UNION ALL
    SELECT 'events', COUNT(*) FROM events UNION ALL
    SELECT 'ground_truth', COUNT(*) FROM ground_truth UNION ALL
    SELECT 'signals', COUNT(*) FROM signals UNION ALL
    SELECT 'frames', COUNT(*) FROM frames UNION ALL
    SELECT 'scores', COUNT(*) FROM scores UNION ALL
    SELECT 'raw_keypoints', COUNT(*) FROM raw_keypoints
)
SELECT 'row count: ' || e.table_name AS check_name, e.expected_rows AS expected, a.actual_rows AS actual,
       CASE WHEN e.expected_rows = a.actual_rows THEN 'PASS' ELSE 'FAIL' END AS status
FROM expected e JOIN actual a ON a.table_name = e.table_name

UNION ALL
SELECT 'events per take: ' || x.take, x.exp, COALESCE(c.n, 0), CASE WHEN x.exp = COALESCE(c.n, 0) THEN 'PASS' ELSE 'FAIL' END
FROM (SELECT * FROM (VALUES
        ('vid1', 19),
        ('vid2', 19),
        ('vid3', 19),
        ('vid4', 16),
        ('vid5', 28)
    ) AS v(take, exp)) x LEFT JOIN (SELECT take, COUNT(*) AS n FROM events GROUP BY take) c ON c.take = x.take

UNION ALL
SELECT 'frames per take: ' || x.take, x.exp, COALESCE(c.n, 0), CASE WHEN x.exp = COALESCE(c.n, 0) THEN 'PASS' ELSE 'FAIL' END
FROM (SELECT * FROM (VALUES
        ('vid1', 976),
        ('vid2', 1015),
        ('vid3', 775),
        ('vid4', 281),
        ('vid5', 913)
    ) AS v(take, exp)) x LEFT JOIN (SELECT take, COUNT(*) AS n FROM frames GROUP BY take) c ON c.take = x.take

UNION ALL
SELECT 'raw keypoint rows per take (21 x frames): ' || x.take, x.exp, COALESCE(c.n, 0), CASE WHEN x.exp = COALESCE(c.n, 0) THEN 'PASS' ELSE 'FAIL' END
FROM (SELECT * FROM (VALUES
        ('vid1', 20496),
        ('vid2', 21315),
        ('vid3', 16275),
        ('vid4', 5901),
        ('vid5', 19173)
    ) AS v(take, exp)) x LEFT JOIN (SELECT take, COUNT(*) AS n FROM raw_keypoints GROUP BY take) c ON c.take = x.take

UNION ALL
SELECT 'signal rows per take: ' || x.take, x.exp, COALESCE(c.n, 0), CASE WHEN x.exp = COALESCE(c.n, 0) THEN 'PASS' ELSE 'FAIL' END
FROM (SELECT * FROM (VALUES
        ('vid1', 976),
        ('vid2', 1015),
        ('vid3', 775),
        ('vid4', 281),
        ('vid5', 913)
    ) AS v(take, exp)) x LEFT JOIN (SELECT take, COUNT(*) AS n FROM signals GROUP BY take) c ON c.take = x.take

UNION ALL
SELECT 'ground-truth segments per take: ' || x.take, x.exp, COALESCE(c.n, 0), CASE WHEN x.exp = COALESCE(c.n, 0) THEN 'PASS' ELSE 'FAIL' END
FROM (SELECT * FROM (VALUES
        ('vid1', 19),
        ('vid2', 19),
        ('vid3', 19),
        ('vid4', 19),
        ('vid5', 25)
    ) AS v(take, exp)) x LEFT JOIN (SELECT take, COUNT(*) AS n FROM ground_truth GROUP BY take) c ON c.take = x.take

UNION ALL
-- every video frame belongs to exactly one event
SELECT 'sum of event n_frames per take: ' || x.take, x.exp, COALESCE(c.n, 0), CASE WHEN x.exp = COALESCE(c.n, 0) THEN 'PASS' ELSE 'FAIL' END
FROM (SELECT * FROM (VALUES
        ('vid1', 976),
        ('vid2', 1015),
        ('vid3', 775),
        ('vid4', 281),
        ('vid5', 913)
    ) AS v(take, exp)) x LEFT JOIN (SELECT take, SUM(n_frames) AS n FROM events GROUP BY take) c ON c.take = x.take

UNION ALL
SELECT 'undetected frames per take: ' || x.take, x.exp, COALESCE(c.n, 0), CASE WHEN x.exp = COALESCE(c.n, 0) THEN 'PASS' ELSE 'FAIL' END
FROM (SELECT * FROM (VALUES
        ('vid1', 1),
        ('vid2', 30),
        ('vid3', 21),
        ('vid4', 16),
        ('vid5', 58)
    ) AS v(take, exp)) x LEFT JOIN (SELECT take, SUM(CASE WHEN detected THEN 0 ELSE 1 END) AS n FROM frames GROUP BY take) c ON c.take = x.take

UNION ALL
-- events are contiguous: each start equals the previous end, and every take starts at 0
SELECT 'event gaps or overlaps (rows)', 0, COUNT(*), CASE WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'FAIL' END
FROM (SELECT start_s, LAG(end_s) OVER (PARTITION BY take ORDER BY event_idx) AS prev_end, event_idx FROM events) g
WHERE event_idx > 0 AND ABS(start_s - prev_end) > 0.000001

UNION ALL
SELECT 'takes not starting at 0 s', 0, COUNT(*), CASE WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'FAIL' END
FROM (SELECT take, MIN(start_s) AS s FROM events GROUP BY take) m WHERE s <> 0

UNION ALL
SELECT 'NULLs in events key columns', 0, COUNT(*), CASE WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'FAIL' END
FROM events WHERE take IS NULL OR label IS NULL OR start_s IS NULL OR end_s IS NULL OR duration_s IS NULL

UNION ALL
SELECT 'detected raw keypoints with NULL coordinates', 0, COUNT(*), CASE WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'FAIL' END
FROM raw_keypoints WHERE detected AND (x IS NULL OR y IS NULL OR z IS NULL)

ORDER BY 1;
