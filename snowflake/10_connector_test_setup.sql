-- Connector test setup: can Databricks write straight to Snowflake? (first test for the cloud-only pipeline)
-- Run in a Snowsight worksheet as ACCOUNTADMIN, one statement at a time or Run all.
-- Creates a SEPARATE schema, role and key-pair service user for the test. It never touches MOTION_INTENT.PIPELINE
-- (the verified Phase 4 tables). Steps and meaning of the results: databricks/snowflake_write_check.md

USE ROLE ACCOUNTADMIN;
CREATE WAREHOUSE IF NOT EXISTS COMPUTE_WH WAREHOUSE_SIZE = XSMALL AUTO_SUSPEND = 60 AUTO_RESUME = TRUE;
CREATE DATABASE IF NOT EXISTS MOTION_INTENT;
CREATE SCHEMA IF NOT EXISTS MOTION_INTENT.CONNECTOR_TEST;

-- A role that can only use the warehouse and create objects in the test schema.
-- CREATE STAGE / FILE FORMAT: both connectors upload data through a temporary internal stage.
CREATE ROLE IF NOT EXISTS DATABRICKS_TEST_ROLE;
GRANT USAGE ON WAREHOUSE COMPUTE_WH TO ROLE DATABRICKS_TEST_ROLE;
GRANT USAGE ON DATABASE MOTION_INTENT TO ROLE DATABRICKS_TEST_ROLE;
GRANT USAGE, CREATE TABLE, CREATE STAGE, CREATE FILE FORMAT ON SCHEMA MOTION_INTENT.CONNECTOR_TEST TO ROLE DATABRICKS_TEST_ROLE;
GRANT ROLE DATABRICKS_TEST_ROLE TO ROLE ACCOUNTADMIN;  -- so you can look at the test tables in Snowsight

-- A service user (no password, no MFA prompt) that signs in with a key pair.
-- Paste the PUBLIC key body (no BEGIN/END lines, one line) in place of the placeholder; see the guide for the command.
CREATE USER IF NOT EXISTS DATABRICKS_SVC
    TYPE = SERVICE
    DEFAULT_ROLE = DATABRICKS_TEST_ROLE
    DEFAULT_WAREHOUSE = COMPUTE_WH
    DEFAULT_NAMESPACE = MOTION_INTENT.CONNECTOR_TEST
    COMMENT = 'Databricks -> Snowflake connector test (motion intent project)';
ALTER USER DATABRICKS_SVC SET RSA_PUBLIC_KEY = 'PASTE_PUBLIC_KEY_BODY_HERE';
GRANT ROLE DATABRICKS_TEST_ROLE TO USER DATABRICKS_SVC;

-- Check 1: RSA_PUBLIC_KEY_FP must equal the "SHA256:..." fingerprint the notebook prints.
DESC USER DATABRICKS_SVC;

-- Check 2: the hosts a client must reach. Type SNOWFLAKE_DEPLOYMENT is the server; STAGE is the cloud storage
-- the connectors upload data through. Save this output; the notebook also tries to read it.
SELECT SYSTEM$ALLOWLIST();

-- Cleanup when the test is finished (uncomment and run):
-- DROP SCHEMA IF EXISTS MOTION_INTENT.CONNECTOR_TEST;
-- DROP USER IF EXISTS DATABRICKS_SVC;
-- DROP ROLE IF EXISTS DATABRICKS_TEST_ROLE;
