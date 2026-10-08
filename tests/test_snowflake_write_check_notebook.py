"""Static safety checks for the Databricks -> Snowflake write test (databricks/snowflake_write_check.py).
It cannot run here (it needs Databricks and Snowflake), so these tests pin what must hold before anyone runs it:
it is valid Python, it can only write to the CONNECTOR_TEST schema, and no credential lives in the repo."""
import ast
import re

from config import ROOT

NOTEBOOK = ROOT / "databricks" / "snowflake_write_check.py"
SETUP_SQL = ROOT / "snowflake" / "10_connector_test_setup.sql"


def _code_lines(path, comment):
    """Lines with comments removed (SQL: whole comment lines; Python: everything from '#', which no string here uses)."""
    lines = [line.split(comment, 1)[0] if comment == "#" else line for line in path.read_text().splitlines()]
    return [line for line in lines if line.strip() and not line.lstrip().startswith(comment)]


def test_notebook_is_valid_python_in_databricks_source_format():
    source = NOTEBOOK.read_text()
    assert source.startswith("# Databricks notebook source\n")
    ast.parse(source)


def test_notebook_targets_only_the_test_schema():
    source = NOTEBOOK.read_text()
    assert re.search(r'^SF_SCHEMA = "CONNECTOR_TEST"', source, re.M)
    assert "PIPELINE" not in "\n".join(_code_lines(NOTEBOOK, "#"))
    # every table the notebook writes or drops is one of its own test tables
    written = set(re.findall(r'"dbtable", "(\w+)"', source)) | set(re.findall(r'write_pandas\(conn, pdf, "(\w+)"', source))
    assert written == {"SPARK_WRITE_TEST", "EVENTS_WRITE_TEST", "PYTHON_WRITE_TEST"}
    assert set(re.findall(r'for table in \(([^)]*)\)', source)[0].replace('"', "").replace(" ", "").split(",")) == written


def test_credentials_come_only_from_a_databricks_secret():
    source = NOTEBOOK.read_text()
    assert "dbutils.secrets.get(SECRET_SCOPE, SECRET_KEY)" in source
    code = "\n".join(_code_lines(NOTEBOOK, "#")).lower()
    assert "sfpassword" not in code and "password=" not in code.replace("password=none", "")
    assert "BEGIN PRIVATE KEY" not in source and "BEGIN RSA PRIVATE KEY" not in source


def test_setup_sql_never_touches_the_verified_schema():
    code = "\n".join(_code_lines(SETUP_SQL, "--")).upper()
    assert "PIPELINE" not in code
    assert "CREATE OR REPLACE" not in code  # nothing existing is replaced
    assert "PASTE_PUBLIC_KEY_BODY_HERE" in code  # the repo copy carries a placeholder, not a real key
    assert "BEGIN PUBLIC KEY" not in code and "PASSWORD" not in code


def test_private_keys_are_gitignored():
    assert "*.p8" in (ROOT / ".gitignore").read_text().splitlines()
