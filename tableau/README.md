# Phase 5: building the Tableau dashboard

**Status: data prepared and tested; the dashboard itself is built by you in Tableau Public.** Target picture:
`target_dashboard.png` (drawn with matplotlib from the same data; your Tableau version will look different in style but should
show the same story).

## Data (all in `data/tableau/`)

| File | Rows | Use |
|---|---|---|
| `tableau_phases.csv` | 202 | one row per phase segment: `source` (Detected / Hand-labelled), `phase_name`, `phase_order`, `start_s`, `end_s`, `duration_s`, `take_label` |
| `tableau_signals.csv` | 3,960 | one row per 30 Hz sample: `t_s`, `speed`, `aperture`, `phase_name`; `speed` is empty where the hand was out of view (Tableau shows a gap) |
| `tableau_accuracy.csv` | 5 | `percent_frames_matching_human_labels` per take (optional third chart) |

They are produced by the views in `snowflake/05_tableau_views.sql`. Tableau Public (the free edition) cannot connect to Snowflake, to
the author's knowledge: check the **Connect** pane; if Snowflake is listed, you may connect live instead. Otherwise use the CSVs.
**Optional, keeps Snowflake as the source:** run `05_tableau_views.sql` in Snowflake, run `SELECT * FROM v_tableau_phases ORDER BY take, source, start_s;` (same for `v_tableau_signals` ORDER BY take, t_s and `v_tableau_accuracy`),
download each result as `snowflake/actual_results/tableau_phases.csv` etc., run `python scripts/compare_snowflake_results.py`, and use those downloads
in Tableau.

Two things to know about the data:
- The first 0.1 s of takes 1 and 4 show a short speed spike (2.2 and 1.8 hand-lengths/s while the hand is labelled at rest). It could be the table tap that
  starts each take or an edge effect of the smoothing filter; this has not been investigated. The data is left unchanged. If it distracts, filter `t_s` to 0.1 or more.
- Take 5 has two gaps in the speed line (9.3 to 9.8 s and 17.1 to 17.3 s): the hand left the frame. Leave them as gaps.

## Phase colours (use these so the dashboard matches the project's plots)

| Phase | Hex |
|---|---|
| At rest | `#b9c2ca` |
| Reaching | `#3d78b8` |
| Grasping | `#e08a2c` |
| Holding | `#3f9a5e` |
| Releasing | `#d1544f` |
| Returning | `#8f66a8` |

## What you do

1. Install **Tableau Public** (free, public.tableau.com -> Create -> Download the app) and sign in or create the free account (needed only to publish).
2. **Sheet 1 "What the hand was doing" (the timeline).**
   1. Connect -> **Text file** -> `data/tableau/tableau_phases.csv`. Open a new sheet.
   2. Drag **source** to Filters and keep only `Detected`.
   3. Marks card: change the type to **Gantt Bar**.
   4. Drag **start_s** to Columns. Click the pill's dropdown and make sure it shows `SUM(start_s)` and is continuous (green).
   5. Drag **take_label** to Rows.
   6. Drag **duration_s** to **Size** (it should read `SUM(duration_s)`). Bars should now span the correct time ranges.
   7. Drag **phase_name** to **Color**. Colour -> Edit Colors: assign the six hex values above (click a phase, then "More colors..." and paste the hex).
      Sort the legend: right-click `phase_name` -> Sort -> by field `phase_order`, Minimum, ascending.
   8. Edit the x-axis: title "Time (seconds)", Range **fixed 0 to 35**, tick interval 5. Remove gridlines.
   9. Tooltip: `<phase_name>: <SUM(duration_s)> seconds`.
3. **Sheet 2 "How fast the wrist was moving".**
   1. Connect -> Text file -> `tableau_signals.csv` (a second data source). Open a new sheet.
   2. Drag **t_s** to Columns; right-click the pill -> **Dimension**, then right-click again -> **Continuous**.
   3. Drag **take_label** to Rows, then **speed** to Rows to the right of it (`AVG(speed)`), so each take gets its own small panel.
   4. Mark type **Line**; colour a dark neutral (`#33414d`). Make the speed axis the same for all panels (Fixed 0 to 10).
   5. Edit the x-axis: Fixed range **0 to 35**, same ticks as sheet 1. Hide the speed axis title ("higher = faster" goes in the sheet title).
   6. Optional: filter `t_s >= 0.1` (see the data note above).
4. **Optional sheet 3 "How often the computer matched a human labeller":** `tableau_accuracy.csv`, bars of `percent_frames_matching_human_labels` by `take_label`.
5. **Dashboard:** new dashboard, Size: Fixed 1200 x 900 (or Automatic). Drag Sheet 1 on top and Sheet 2 below in a vertical layout; add a Text object on top:
   - **Title:** `How a hand reaches, grabs and sets down an object`
   - **Caption:** `Each row is one recording of a hand picking up an object and putting it back. The coloured bands show what the hand was doing at each moment, found automatically from video. The lines below show how fast the wrist was moving: it speeds up when reaching and returning, and is nearly still while holding.`
   Show the colour legend once, at the top. If the two charts' time axes do not line up, give both sheets the same row-label width (Format -> Row headers) or hide the lower labels and keep them on the top chart only.
6. **Save the evidence:** File -> Save to Tableau Public (this **publishes publicly**: do it only when you are happy for the dashboard to be public) or File -> Save As a `.twbx` into `tableau/`. Also export an image: Dashboard -> Export Image -> `evidence/tableau_dashboard.png`. Put the public URL (if you publish) in this file under "Published dashboard".
7. **Test it on someone new** with `user_test.md`. The project's rule: if they cannot describe what it shows, the dashboard is not done.

## Published dashboard
https://public.tableau.com/app/profile/kian.hekmatnejad/viz/hand-motion-phases/Dashboard2#1 
(URL, date and screenshot path to be added after you publish.)
