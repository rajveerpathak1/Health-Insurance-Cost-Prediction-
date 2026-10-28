"""Data loading, cleaning, quality checks and feature engineering."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "dataset" / "medical_insurance.csv"

EXPECTED_COLUMNS = ["age", "sex", "bmi", "children", "smoker", "region", "charges"]


def load_raw(path: Path | str = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing columns: {missing}")
    return df[EXPECTED_COLUMNS]


def quality_report(df: pd.DataFrame) -> dict:
    """Summarise data quality issues before cleaning."""
    q1, q3 = df["charges"].quantile([0.25, 0.75])
    iqr = q3 - q1
    return {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "charges_outliers_iqr": int(((df["charges"] < q1 - 1.5 * iqr) | (df["charges"] > q3 + 1.5 * iqr)).sum()),
        "invalid_bmi": int(((df["bmi"] < 10) | (df["bmi"] > 70)).sum()),
        "invalid_age": int(((df["age"] < 18) | (df["age"] > 100)).sum()),
    }


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["sex", "smoker", "region"]:
        df[col] = df[col].astype(str).str.strip().str.lower()
    df = df.dropna().drop_duplicates().reset_index(drop=True)
    df = df[(df["age"].between(18, 100)) & (df["bmi"].between(10, 70)) & (df["charges"] > 0)]
    df["children"] = df["children"].astype(int)
    return df.reset_index(drop=True)


def bmi_category(bmi: float) -> str:
    if bmi < 18.5:
        return "Underweight"
    if bmi < 25:
        return "Normal"
    if bmi < 30:
        return "Overweight"
    return "Obese"


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["bmi_category"] = pd.Categorical(
        df["bmi"].apply(bmi_category), ["Underweight", "Normal", "Overweight", "Obese"], ordered=True
    )
    df["age_group"] = pd.cut(
        df["age"], bins=[17, 29, 39, 49, 59, 100], labels=["18-29", "30-39", "40-49", "50-59", "60+"]
    )
    df["is_smoker"] = (df["smoker"] == "yes").astype(int)
    df["is_obese"] = (df["bmi"] >= 30).astype(int)
    df["has_children"] = (df["children"] > 0).astype(int)
    df["smoker_obese"] = df["is_smoker"] * df["is_obese"]
    df["log_charges"] = np.log(df["charges"])
    df["risk_tier"] = df.apply(rule_based_tier, axis=1)
    return df


def rule_based_tier(row) -> str:
    """Business-readable risk tier driven by the two strongest cost drivers."""
    smoker = row["smoker"] == "yes"
    obese = row["bmi"] >= 30
    if smoker and obese:
        return "Critical"
    if smoker:
        return "High"
    if obese or row["age"] >= 50:
        return "Medium"
    return "Low"


TIER_ORDER = ["Low", "Medium", "High", "Critical"]


def load_clean() -> pd.DataFrame:
    return add_features(clean(load_raw()))
