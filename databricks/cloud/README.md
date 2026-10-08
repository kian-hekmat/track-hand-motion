# Cloud path notebooks (Databricks)

Plan and milestones: `docs/cloud_pipeline_plan.md`. These notebooks run from a **Git folder**: Databricks' own clone of
`github.com/kian-hekmat/track-hand-motion`. They import the repo's `src/` directly, so Databricks runs the committed code.
No notebook is generated or copy-pasted.

## One-time setup: the Git folder

1. In Databricks: **Workspace** → your user folder → **Create** → **Git folder**.
2. Git repository URL: `https://github.com/kian-hekmat/track-hand-motion`, provider GitHub, branch `main`.
   - If the repo is private, Databricks asks for a GitHub token first: Settings → Linked accounts → add a GitHub personal
     access token with read access to this repo. Do not send it to me.
3. Before each run, update the folder: open it, click the branch name → **Pull**, so it has the latest commit.

If Free Edition does not offer Git folders, stop and tell me; the fallback is a wheel of `src/` uploaded to a Volume.

## M1: does the frozen model give the same answers on Databricks? (`m1_environment_check.py`)

**Status: written and passing locally (`tests/test_cloud_m1.py`); not yet run on Databricks.**

1. In the Git folder open `databricks/cloud/m1_environment_check.py` (it opens as a notebook). Attach **serverless** compute.
2. **Run all.** The first cell installs the pinned scikit-learn and ruptures (`requirements-cloud.txt`), the second restarts
   Python, the rest run all seven takes and compare with `data/cloud_reference/`. Nothing needs editing: the data and models come
   from the Git folder itself.
3. The last cell prints `M1 CHECKS: ALL PASSED` or the failed checks. Export: File → Export → HTML, save as
   `evidence/cloud_m1_environment_check.html`, and send me the summary output.

How to read it:
- **PASS on every take's events:** the model and decoding give byte-identical events on Databricks. M2 can build on them.
- **Event FAIL:** look at the INFO lines above it. A difference in **signals** means the input already differs (numpy, pandas or
  scipy behave differently on serverless). Probabilities differing while signals match points at the model or scikit-learn. This is
  a real finding: send it to me and nothing gets adjusted until the cause is known.
- **Library FAIL:** the pinned scikit-learn or ruptures did not install at the pinned version.
