-- Landing table for Phase 1 output: one row per (take, frame, keypoint).
-- Undetected frames are kept with NULL coordinates and detected = FALSE (never zero-filled).
CREATE TABLE IF NOT EXISTS raw_keypoints (
    take             TEXT             NOT NULL,
    frame_idx        INTEGER          NOT NULL,
    timestamp_ms     INTEGER          NOT NULL,
    keypoint_id      SMALLINT         NOT NULL CHECK (keypoint_id BETWEEN 0 AND 20),
    x                DOUBLE PRECISION,
    y                DOUBLE PRECISION,
    z                DOUBLE PRECISION,
    world_x          DOUBLE PRECISION,
    world_y          DOUBLE PRECISION,
    world_z          DOUBLE PRECISION,
    detected         BOOLEAN          NOT NULL,
    handedness       TEXT,
    handedness_score DOUBLE PRECISION,
    PRIMARY KEY (take, frame_idx, keypoint_id),
    -- detected frames must have coordinates; undetected frames must not
    CHECK (detected = (x IS NOT NULL AND y IS NOT NULL AND z IS NOT NULL))
);
CREATE INDEX IF NOT EXISTS raw_keypoints_take_ts ON raw_keypoints (take, timestamp_ms);
