"""Produce the Tableau CSVs (data/tableau/) by running the SAME views as snowflake/05_tableau_views.sql in DuckDB, and save the
expected results for comparing a Snowflake export (snowflake/expected_results/tableau_*.csv)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import local_sql as L

OUT = L.ROOT / "data" / "tableau"
VIEWS = {
    "tableau_phases": ("v_tableau_phases", "take, source, start_s"),
    "tableau_signals": ("v_tableau_signals", "take, t_s"),
    "tableau_accuracy": ("v_tableau_accuracy", "take"),
}


def build_tables(con=None) -> dict:
    con = con or L.connect()
    L.create_tableau_views(con)
    return {name: con.execute(f"SELECT * FROM {view} ORDER BY {order}").fetchdf() for name, (view, order) in VIEWS.items()}


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for name, df in build_tables().items():
        df.to_csv(OUT / f"{name}.csv", index=False)
        df.to_csv(L.SF / "expected_results" / f"{name}.csv", index=False)
        print(f"{name}: {len(df)} rows, {len(df.columns)} columns")
