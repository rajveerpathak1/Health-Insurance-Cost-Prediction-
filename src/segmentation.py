"""Customer segmentation: rule-based risk tiers + K-Means behavioural clusters."""
from __future__ import annotations

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from .data_prep import TIER_ORDER

CLUSTER_FEATURES = ["age", "bmi", "children", "is_smoker"]


def tier_profile(df: pd.DataFrame) -> pd.DataFrame:
    total = df["charges"].sum()
    prof = (
        df.groupby("risk_tier")
        .agg(
            customers=("charges", "size"),
            avg_charges=("charges", "mean"),
            median_charges=("charges", "median"),
            total_charges=("charges", "sum"),
            avg_age=("age", "mean"),
            avg_bmi=("bmi", "mean"),
        )
        .reindex(TIER_ORDER)
        .dropna(how="all")
    )
    prof["customer_share_pct"] = prof["customers"] / len(df) * 100
    prof["cost_share_pct"] = prof["total_charges"] / total * 100
    prof["cost_to_customer_ratio"] = prof["cost_share_pct"] / prof["customer_share_pct"]
    return prof.round(2)


def choose_k(df: pd.DataFrame, k_range=range(2, 9), seed: int = 42) -> pd.DataFrame:
    X = StandardScaler().fit_transform(df[CLUSTER_FEATURES])
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
        rows.append({"k": k, "inertia": km.inertia_, "silhouette": silhouette_score(X, km.labels_)})
    return pd.DataFrame(rows)


def kmeans_segments(df: pd.DataFrame, k: int = 4, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    X = StandardScaler().fit_transform(df[CLUSTER_FEATURES])
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
    df["cluster"] = km.labels_
    prof = df.groupby("cluster").agg(
        customers=("charges", "size"),
        avg_age=("age", "mean"),
        avg_bmi=("bmi", "mean"),
        avg_children=("children", "mean"),
        smoker_rate=("is_smoker", "mean"),
        avg_charges=("charges", "mean"),
    )
    # Relabel clusters by avg cost so names are stable and readable
    order = prof["avg_charges"].sort_values().index
    names = {c: f"Segment {i + 1}" for i, c in enumerate(order)}
    df["segment"] = df["cluster"].map(names)
    prof.index = prof.index.map(names)
    prof = prof.sort_values("avg_charges")
    prof["persona"] = prof.apply(_persona, axis=1)
    return df, prof.round(2)


def _persona(r) -> str:
    smoke = "Smokers" if r["smoker_rate"] > 0.5 else "Non-smokers"
    age = "older" if r["avg_age"] >= 45 else "younger"
    fam = "families" if r["avg_children"] >= 1.5 else "individuals"
    return f"{age.title()} {smoke.lower()} / {fam}"
