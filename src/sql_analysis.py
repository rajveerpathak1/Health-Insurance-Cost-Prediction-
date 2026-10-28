"""Load cleaned data into SQLite and run the business KPI queries in sql/kpi_queries.sql."""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "insurance.db"
QUERY_FILE = ROOT / "sql" / "kpi_queries.sql"


def build_db(df: pd.DataFrame, db_path: Path = DB_PATH) -> Path:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["age", "sex", "bmi", "children", "smoker", "region", "charges",
            "bmi_category", "age_group", "risk_tier"]
    out = df[cols].copy()
    for c in ["bmi_category", "age_group"]:
        out[c] = out[c].astype(str)
    out.insert(0, "policy_id", range(1, len(out) + 1))
    with sqlite3.connect(db_path) as con:
        out.to_sql("policyholders", con, if_exists="replace", index=False)
    return db_path


def parse_queries(path: Path = QUERY_FILE) -> dict[str, str]:
    """Split the SQL file on `-- name: <query_name>` markers."""
    text = path.read_text(encoding="utf-8")
    parts = re.split(r"^--\s*name:\s*(\w+)\s*$", text, flags=re.M)
    return {parts[i]: parts[i + 1].strip().rstrip(";") for i in range(1, len(parts), 2)}


def run_queries(db_path: Path = DB_PATH) -> dict[str, pd.DataFrame]:
    with sqlite3.connect(db_path) as con:
        return {name: pd.read_sql_query(sql, con) for name, sql in parse_queries().items()}
