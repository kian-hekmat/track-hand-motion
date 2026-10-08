# Snowflake write test: can Databricks write straight to Snowflake?

**Status: written, not yet run.** This is the first and riskiest test for moving the pipeline into the cloud (Databricks → Snowflake → Tableau with no local hop). Nothing in the verified Phase 3/4 results depends on it.

Why it might fail:
- Databricks Free Edition limits outbound internet access to a set of trusted domains.
- Databricks labels its Snowflake connector experimental and supports only some options on serverless compute.

The notebook checks network access first, so a failure says *where* it stops.

| File | What it is |
|---|---|
| `databricks/snowflake_write_check.py` | the notebook (Databricks source format) |
| `snowflake/10_connector_test_setup.sql` | creates a test schema, a limited role and a key-pair service user in Snowflake |

The test writes only to `MOTION_INTENT.CONNECTOR_TEST`. The verified Phase 4 tables in `MOTION_INTENT.PIPELINE` are never touched.

**Credentials:** the notebook signs in with a key pair stored in a Databricks secret. No password, no MFA prompt, and nothing typed into the notebook. Do not paste the private key into a notebook cell or send it to me. The `.p8` file stays outside the repo, and `*.p8` is gitignored as a backstop.

## What you do

### 1. Make a key pair (on your laptop, outside the repo)

```bash
mkdir -p ~/.snowflake && chmod 700 ~/.snowflake
```

```bash
openssl genrsa 2048 | openssl pkcs8 -topk8 -inform PEM -out ~/.snowflake/databricks_svc_key.p8 -nocrypt
```

```bash
openssl rsa -in ~/.snowflake/databricks_svc_key.p8 -pubout -out ~/.snowflake/databricks_svc_key.pub && chmod 600 ~/.snowflake/databricks_svc_key.p8
```

Print the public key body (one line, no BEGIN/END lines). You will paste it in step 2:

```bash
grep -v -- '-----' ~/.snowflake/databricks_svc_key.pub | tr -d '\n'; echo
```

The key is unencrypted (`-nocrypt`) so the connectors can use it directly. It is protected by file permissions locally and by the secret store in Databricks.

### 2. Set up Snowflake

1. Check the trial's remaining days in Snowsight (trials last about 30 days).
2. Open a worksheet, paste `snowflake/10_connector_test_setup.sql`, replace `PASTE_PUBLIC_KEY_BODY_HERE` with the line from step 1, and run it.
3. Note two things from the output:
   - `DESC USER DATABRICKS_SVC`: the value of `RSA_PUBLIC_KEY_FP` (starts with `SHA256:`).
   - `SELECT SYSTEM$ALLOWLIST()`: the `host` values. Save the output; it lists the server and the cloud-storage (`STAGE`) hosts.
4. Find the host for the notebook: account menu (bottom left) → your account → **View account details** → **Account/Server URL**. It looks like `ORGNAME-ACCOUNTNAME.snowflakecomputing.com`.

### 3. Put the private key in a Databricks secret

Install and sign in to the Databricks CLI. `--host` takes only the scheme and domain from your browser's address bar, e.g. `https://dbc-xxxxxxxx-xxxx.cloud.databricks.com`, with no path (`/browse/...`) and no `?o=...`. When asked for a profile name, any short name works; press Enter to accept the default.

```bash
brew install databricks
```

```bash
databricks auth login --host https://dbc-xxxxxxxx-xxxx.cloud.databricks.com
```

```bash
databricks secrets create-scope motion
```

```bash
databricks secrets put-secret motion snowflake_private_key --string-value "$(cat ~/.snowflake/databricks_svc_key.p8)"
```

Check it is there (this lists key names, not values):

```bash
databricks secrets list-secrets motion
```

If Free Edition refuses to create a secret scope, stop and tell me the error. Do not fall back to pasting the key into the notebook.

### 4. Run the notebook

1. Optional: make sure `events.csv` from `data/export/` is in the Volume used for Phase 3 (`/Volumes/workspace/default/motion`). It enables check 5, a real 101-row table.
2. Workspace → Import → `databricks/snowflake_write_check.py`. Attach serverless compute.
3. Run the first code cell once so the widgets appear as input boxes in a bar at the top of the notebook. Type the host from step 2.4 into box **1. Snowflake host** there, not into the code (`dbutils.widgets.text` only sets a default when the widget is first created, so editing the code later has no effect). The defaults fit everything else.
4. **Run all.** The last cell prints a verdict line and `SNOWFLAKE WRITE TEST: ALL PASSED` or the list of failed checks. Every check runs even after a failure.
5. File → Export → HTML and save it as `evidence/databricks_snowflake_write_check.html`. Send me the summary cell's output and any error text.

## What the result means

| Verdict | Meaning | Next step |
|---|---|---|
| `PATH A WORKS` | The Spark connector writes and reads Snowflake tables from this workspace | Build the cloud pipeline with `df.write.format("snowflake")` from the gold tables |
| `PATH B ONLY` | The Spark connector fails here, but the Python connector can write | Publish the gold tables with `write_pandas` from the job (they are small: about 100 to 4,000 rows) |
| `SNOWFLAKE NOT REACHABLE` | Outbound access from this workspace is blocked | Verify the Databricks account to unlock outbound access and re-run, or keep one manual upload step |
| `REACHABLE BUT NO WRITE PATH WORKED` | Network is fine; sign-in or permissions fail | Usually the key (compare the fingerprint the notebook prints with `RSA_PUBLIC_KEY_FP`), the grants, or the account identifier |

What each check proves:
- **Spark read-back:** every value, including a NULL and full-precision doubles, survives the round trip.
- **Events table:** counts and sums are computed inside Snowflake and match Spark's numbers from the source file.
- **Cross-check:** the Spark-written rows are read back by a second, independent client, which proves they are really stored in Snowflake.
- **Stage hosts:** the cloud storage the connectors upload through is reachable. Data can fail to move even when the server is reachable.

## Cleanup

Set widget 8 to `yes` and re-run to drop the three test tables. To remove everything, run the commented `DROP` lines at the bottom of `snowflake/10_connector_test_setup.sql` and run `databricks secrets delete-scope motion`.
