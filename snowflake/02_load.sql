-- Phase 4 / 02_load.sql  (generated). Upload the 7 files from data/export/ to @motion_stage first (see snowflake/README.md). Patterns match the file with or without a .gz suffix.
USE SCHEMA MOTION_INTENT.PIPELINE;
LIST @motion_stage;

TRUNCATE TABLE takes;
COPY INTO takes FROM @motion_stage PATTERN = '.*takes[.]csv([.]gz)?' FILE_FORMAT = (FORMAT_NAME = motion_csv) ON_ERROR = ABORT_STATEMENT FORCE = TRUE;

TRUNCATE TABLE events;
COPY INTO events FROM @motion_stage PATTERN = '.*events[.]csv([.]gz)?' FILE_FORMAT = (FORMAT_NAME = motion_csv) ON_ERROR = ABORT_STATEMENT FORCE = TRUE;

TRUNCATE TABLE ground_truth;
COPY INTO ground_truth FROM @motion_stage PATTERN = '.*ground_truth[.]csv([.]gz)?' FILE_FORMAT = (FORMAT_NAME = motion_csv) ON_ERROR = ABORT_STATEMENT FORCE = TRUE;

TRUNCATE TABLE signals;
COPY INTO signals FROM @motion_stage PATTERN = '.*signals[.]csv([.]gz)?' FILE_FORMAT = (FORMAT_NAME = motion_csv) ON_ERROR = ABORT_STATEMENT FORCE = TRUE;

TRUNCATE TABLE frames;
COPY INTO frames FROM @motion_stage PATTERN = '.*frames[.]csv([.]gz)?' FILE_FORMAT = (FORMAT_NAME = motion_csv) ON_ERROR = ABORT_STATEMENT FORCE = TRUE;

TRUNCATE TABLE scores;
COPY INTO scores FROM @motion_stage PATTERN = '.*scores[.]csv([.]gz)?' FILE_FORMAT = (FORMAT_NAME = motion_csv) ON_ERROR = ABORT_STATEMENT FORCE = TRUE;

TRUNCATE TABLE raw_keypoints;
COPY INTO raw_keypoints FROM @motion_stage PATTERN = '.*raw_keypoints[.]parquet' FILE_FORMAT = (FORMAT_NAME = motion_parquet)
    MATCH_BY_COLUMN_NAME = CASE_INSENSITIVE ON_ERROR = ABORT_STATEMENT FORCE = TRUE;

-- Summary: if any count below is 0, the COPY found no file. Run 00_diagnose.sql.
SELECT 'takes' AS table_name, COUNT(*) AS loaded_rows FROM takes UNION ALL SELECT 'events', COUNT(*) FROM events UNION ALL
SELECT 'ground_truth', COUNT(*) FROM ground_truth UNION ALL SELECT 'signals', COUNT(*) FROM signals UNION ALL
SELECT 'frames', COUNT(*) FROM frames UNION ALL SELECT 'scores', COUNT(*) FROM scores UNION ALL
SELECT 'raw_keypoints', COUNT(*) FROM raw_keypoints;
