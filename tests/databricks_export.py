"""Read a Databricks notebook HTML export (File -> Export -> HTML, or `databricks jobs export-run`), as saved in evidence/.

Shared by the tests that check saved Databricks runs, so a claim like "it passed on Databricks" rests on saved output."""
import base64
import json
import re
import urllib.parse
from pathlib import Path

EVIDENCE = Path(__file__).resolve().parents[1] / "evidence"


def read_export(name: str) -> tuple[dict, str]:
    """(notebook model with every cell and its result, all printed text output joined) for evidence/<name>."""
    s = (EVIDENCE / name).read_text(encoding="utf-8")
    m = re.search(r'__DATABRICKS_NOTEBOOK_MODEL\s*=\s*[\'"]([A-Za-z0-9+/=]+)[\'"]', s)
    assert m, f"{name} is not a Databricks notebook HTML export"
    nb = json.loads(urllib.parse.unquote(base64.b64decode(m.group(1)).decode()))
    text = []
    for c in nb["commands"]:
        data = (c.get("results") or {}).get("data")
        if isinstance(data, str):
            text.append(data)
        for item in data if isinstance(data, list) else []:
            if isinstance(item, dict) and item.get("type") == "ansi":
                text.append(item["data"])
    return nb, "\n".join(text)
