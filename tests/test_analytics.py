import sqlite3

import pandas as pd
import pytest

from src import ab_test, modeling, segmentation, sql_analysis, stats_tests
from src.data_prep import add_features, clean, load_clean, load_raw, quality_report, rule_based_tier


@pytest.fixture(scope="session")
def df():
    return load_clean()


def test_raw_schema():
    raw = load_raw()
    assert list(raw.columns) == ["age", "sex", "bmi", "children", "smoker", "region", "charges"]
    assert len(raw) > 1000


def test_cleaning_removes_duplicates_and_invalid(df):
    assert df.duplicated(subset=["age", "sex", "bmi", "children", "smoker", "region", "charges"]).sum() == 0
    assert df.isna().sum().sum() == 0
    assert df["age"].between(18, 100).all() and df["bmi"].between(10, 70).all()


def test_clean_drops_bad_rows():
    bad = pd.DataFrame({"age": [30, 30, 5], "sex": ["Male ", "male", "female"], "bmi": [25, 25, 22],
                        "children": [0, 0, 0], "smoker": ["no", "no", "no"], "region": ["northeast"] * 3,
                        "charges": [1000.0, 1000.0, 500.0]})
    out = clean(bad)
    assert len(out) == 1 and out.loc[0, "sex"] == "male"


def test_quality_report_keys():
    q = quality_report(load_raw())
    assert {"rows", "duplicate_rows", "missing_values", "charges_outliers_iqr"} <= q.keys()


@pytest.mark.parametrize("smoker,bmi,age,expected", [
    ("yes", 35, 30, "Critical"), ("yes", 22, 30, "High"), ("no", 32, 30, "Medium"),
    ("no", 22, 55, "Medium"), ("no", 22, 30, "Low"),
])
def test_rule_based_tier(smoker, bmi, age, expected):
    assert rule_based_tier({"smoker": smoker, "bmi": bmi, "age": age}) == expected


def test_smoker_effect_significant(df):
    t = stats_tests.two_group_test(df, "smoker", "yes", "no")
    assert t["significant"] and t["ratio"] > 2 and t["ci95_low"] > 0


def test_all_tests_run(df):
    res = stats_tests.run_all(df)
    for r in res.values():
        assert 0 <= r.get("p_value_welch", r.get("p_value")) <= 1


def test_tier_shares_sum_to_100(df):
    prof = segmentation.tier_profile(df)
    assert abs(prof["customer_share_pct"].sum() - 100) < 0.1
    assert abs(prof["cost_share_pct"].sum() - 100) < 0.1
    assert prof.loc["Critical", "avg_charges"] > prof.loc["Low", "avg_charges"]


def test_kmeans_segments(df):
    seg_df, prof = segmentation.kmeans_segments(df, k=4)
    assert seg_df["segment"].nunique() == 4
    assert prof["avg_charges"].is_monotonic_increasing


def test_sample_size_monotonic():
    small = ab_test.required_sample_size(30000, 12000, 0.05)
    large = ab_test.required_sample_size(30000, 12000, 0.20)
    assert small > large > 0


def test_ab_detects_effect(df):
    res, exp = ab_test.run(df, mde_pct=0.10, true_effect_pct=0.15)
    assert res["significant"] and res["lift_pct"] < 0
    assert set(exp["group"]) == {"control", "treatment"}


def test_ab_no_effect_not_rolled_out(df):
    smokers = df[df.smoker == "yes"]
    exp = ab_test.simulate_experiment(smokers, 300, true_effect_pct=0.0, programme_cost=5000)
    assert ab_test.analyse(exp)["decision"] != "Roll out"


def test_sql_queries(df, tmp_path):
    db = sql_analysis.build_db(df, tmp_path / "t.db")
    with sqlite3.connect(db) as con:
        n = con.execute("SELECT COUNT(*) FROM policyholders").fetchone()[0]
    assert n == len(df)
    queries = sql_analysis.parse_queries()
    assert len(queries) >= 6
    with sqlite3.connect(db) as con:
        for name, q in queries.items():
            assert not pd.read_sql_query(q, con).empty, name
        dec = pd.read_sql_query(queries["top_decile_concentration"], con)
    assert abs(dec["cumulative_cost_share_pct"].iloc[-1] - 100) < 0.5


def test_model_quality_and_prediction(df, tmp_path):
    out = modeling.train_and_save(df, tmp_path / "m.pkl")
    best = out["results"].iloc[0]
    assert best["test_r2"] > 0.8
    m = modeling.load_model(tmp_path / "m.pkl")
    smoker = modeling.predict_one(m, 45, "male", 32, 1, "yes", "southeast")
    non = modeling.predict_one(m, 45, "male", 32, 1, "no", "southeast")
    assert smoker > non > 0
