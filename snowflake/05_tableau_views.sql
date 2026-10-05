-- Phase 5 / 05_tableau_views.sql  (generated). Tableau-ready views with plain-language names. Tableau Public cannot connect to
-- Snowflake (to the author's knowledge), so the views are exported as CSV; data/tableau/*.csv are produced from these same views.
USE SCHEMA MOTION_INTENT.PIPELINE;

-- One row per phase segment, for the timeline: detected phases and hand-labelled phases side by side.
CREATE OR REPLACE VIEW v_tableau_phases AS
WITH p AS (
    SELECT take, 'Detected' AS source, label, start_s, end_s FROM events
    UNION ALL
    SELECT take, 'Hand-labelled' AS source, label, start_s, end_s FROM ground_truth
)
SELECT p.take,
       'Take ' || SUBSTR(p.take, 4) || ' (' || t.take_group || ')' AS take_label,
       t.take_group,
       p.source,
       p.label,
       CASE p.label WHEN 'REST' THEN 'At rest' WHEN 'REACH' THEN 'Reaching' WHEN 'GRASP' THEN 'Grasping'
                 WHEN 'HOLD' THEN 'Holding' WHEN 'RELEASE' THEN 'Releasing' WHEN 'RETRACT' THEN 'Returning' END AS phase_name,
       CASE p.label WHEN 'REST' THEN 1 WHEN 'REACH' THEN 2 WHEN 'GRASP' THEN 3 WHEN 'HOLD' THEN 4
                 WHEN 'RELEASE' THEN 5 WHEN 'RETRACT' THEN 6 END AS phase_order,
       p.start_s,
       p.end_s,
       p.end_s - p.start_s AS duration_s
FROM p JOIN takes t ON t.take = p.take;

-- One row per 30 Hz sample, for the speed line. Speed is empty (NULL) where the hand was not visible: Tableau shows a gap.
CREATE OR REPLACE VIEW v_tableau_signals AS
SELECT s.take,
       'Take ' || SUBSTR(s.take, 4) || ' (' || t.take_group || ')' AS take_label,
       t.take_group,
       s.t_s,
       s.speed,
       s.aperture,
       s.predicted_label,
       CASE s.predicted_label WHEN 'REST' THEN 'At rest' WHEN 'REACH' THEN 'Reaching' WHEN 'GRASP' THEN 'Grasping'
                 WHEN 'HOLD' THEN 'Holding' WHEN 'RELEASE' THEN 'Releasing' WHEN 'RETRACT' THEN 'Returning' END AS phase_name,
       CASE s.predicted_label WHEN 'REST' THEN 1 WHEN 'REACH' THEN 2 WHEN 'GRASP' THEN 3 WHEN 'HOLD' THEN 4
                 WHEN 'RELEASE' THEN 5 WHEN 'RETRACT' THEN 6 END AS phase_order,
       s.missing AS hand_not_visible
FROM signals s JOIN takes t ON t.take = s.take;

-- One row per take: how often the automatic labels agree with the hand labels (frames with no hand data excluded).
CREATE OR REPLACE VIEW v_tableau_accuracy AS
SELECT sc.take,
       'Take ' || SUBSTR(sc.take, 4) || ' (' || sc.take_group || ')' AS take_label,
       sc.take_group,
       ROUND(sc.frame_accuracy * 100, 1) AS percent_frames_matching_human_labels
FROM scores sc
WHERE sc.scope = 'all';
