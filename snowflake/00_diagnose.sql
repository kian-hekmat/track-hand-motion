-- Phase 4 / 00_diagnose.sql  (generated). Snowflake-only. Run each numbered statement on its own and read the result.
-- Use this when 03_verify.sql shows 0 rows loaded.

-- 1. Which role / warehouse / database / schema is this worksheet using? Expect MOTION_INTENT and PIPELINE, and a warehouse name.
SELECT CURRENT_ROLE() AS role, CURRENT_WAREHOUSE() AS warehouse, CURRENT_DATABASE() AS db, CURRENT_SCHEMA() AS schema_name;

-- 2. Which MOTION_STAGE stages exist? If you see more than one (another database or schema), your upload may have gone to the wrong one.
SHOW STAGES LIKE 'MOTION_STAGE' IN ACCOUNT;

-- 3. What is inside the stage? Expect 7 files: takes, events, ground_truth, signals, frames, scores (csv) and raw_keypoints.parquet.
--    No rows here = the files were never uploaded to THIS stage.
LIST @MOTION_INTENT.PIPELINE.motion_stage;

-- 4. Row counts right now (all 0 = nothing loaded, or 01_setup.sql was re-run after loading, which recreates empty tables).
SELECT 'takes' AS table_name, COUNT(*) AS n FROM MOTION_INTENT.PIPELINE.takes UNION ALL SELECT 'events', COUNT(*) FROM MOTION_INTENT.PIPELINE.events UNION ALL
SELECT 'signals', COUNT(*) FROM MOTION_INTENT.PIPELINE.signals UNION ALL SELECT 'raw_keypoints', COUNT(*) FROM MOTION_INTENT.PIPELINE.raw_keypoints;

-- 5. What did each COPY do in the last 24 hours? One row per file loaded. No rows = the COPY matched no file.
--    status LOAD_FAILED / PARTIALLY_LOADED shows the error text.
SELECT 'takes' AS table_name, file_name, status, row_count, row_parsed, first_error_message, first_error_line_number
FROM TABLE(MOTION_INTENT.INFORMATION_SCHEMA.COPY_HISTORY(TABLE_NAME => 'TAKES', START_TIME => DATEADD(hours, -24, CURRENT_TIMESTAMP())))
UNION ALL
SELECT 'events' AS table_name, file_name, status, row_count, row_parsed, first_error_message, first_error_line_number
FROM TABLE(MOTION_INTENT.INFORMATION_SCHEMA.COPY_HISTORY(TABLE_NAME => 'EVENTS', START_TIME => DATEADD(hours, -24, CURRENT_TIMESTAMP())))
UNION ALL
SELECT 'ground_truth' AS table_name, file_name, status, row_count, row_parsed, first_error_message, first_error_line_number
FROM TABLE(MOTION_INTENT.INFORMATION_SCHEMA.COPY_HISTORY(TABLE_NAME => 'GROUND_TRUTH', START_TIME => DATEADD(hours, -24, CURRENT_TIMESTAMP())))
UNION ALL
SELECT 'signals' AS table_name, file_name, status, row_count, row_parsed, first_error_message, first_error_line_number
FROM TABLE(MOTION_INTENT.INFORMATION_SCHEMA.COPY_HISTORY(TABLE_NAME => 'SIGNALS', START_TIME => DATEADD(hours, -24, CURRENT_TIMESTAMP())))
UNION ALL
SELECT 'frames' AS table_name, file_name, status, row_count, row_parsed, first_error_message, first_error_line_number
FROM TABLE(MOTION_INTENT.INFORMATION_SCHEMA.COPY_HISTORY(TABLE_NAME => 'FRAMES', START_TIME => DATEADD(hours, -24, CURRENT_TIMESTAMP())))
UNION ALL
SELECT 'scores' AS table_name, file_name, status, row_count, row_parsed, first_error_message, first_error_line_number
FROM TABLE(MOTION_INTENT.INFORMATION_SCHEMA.COPY_HISTORY(TABLE_NAME => 'SCORES', START_TIME => DATEADD(hours, -24, CURRENT_TIMESTAMP())))
UNION ALL
SELECT 'raw_keypoints' AS table_name, file_name, status, row_count, row_parsed, first_error_message, first_error_line_number
FROM TABLE(MOTION_INTENT.INFORMATION_SCHEMA.COPY_HISTORY(TABLE_NAME => 'RAW_KEYPOINTS', START_TIME => DATEADD(hours, -24, CURRENT_TIMESTAMP())))
ORDER BY 1, 2;

-- 6. Dry run (loads nothing): should return 5 parsed rows. An empty result = the PATTERN matched no file in the stage.
COPY INTO MOTION_INTENT.PIPELINE.events FROM @MOTION_INTENT.PIPELINE.motion_stage PATTERN = '.*events[.]csv([.]gz)?'
    FILE_FORMAT = (FORMAT_NAME = MOTION_INTENT.PIPELINE.motion_csv) VALIDATION_MODE = RETURN_5_ROWS;
