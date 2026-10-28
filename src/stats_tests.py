"""Hypothesis testing with effect sizes and confidence intervals."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

ALPHA = 0.05


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    pooled = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / pooled)


def two_group_test(df: pd.DataFrame, group_col: str, a: str, b: str, value: str = "charges") -> dict:
    """Welch t-test + Mann-Whitney (charges are skewed) + effect size + 95% CI of mean diff."""
    x = df.loc[df[group_col] == a, value].to_numpy()
    y = df.loc[df[group_col] == b, value].to_numpy()
    t, p_t = stats.ttest_ind(x, y, equal_var=False)
    _, p_mw = stats.mannwhitneyu(x, y, alternative="two-sided")
    diff = x.mean() - y.mean()
    se = np.sqrt(x.var(ddof=1) / len(x) + y.var(ddof=1) / len(y))
    return {
        "test": f"{group_col}: {a} vs {b}",
        "mean_a": float(x.mean()),
        "mean_b": float(y.mean()),
        "ratio": float(x.mean() / y.mean()),
        "mean_diff": float(diff),
        "ci95_low": float(diff - 1.96 * se),
        "ci95_high": float(diff + 1.96 * se),
        "t_stat": float(t),
        "p_value_welch": float(p_t),
        "p_value_mannwhitney": float(p_mw),
        "cohens_d": _cohens_d(x, y),
        "significant": bool(p_t < ALPHA),
    }


def anova_test(df: pd.DataFrame, group_col: str, value: str = "charges") -> dict:
    groups = [g[value].to_numpy() for _, g in df.groupby(group_col, observed=True)]
    f, p = stats.f_oneway(*groups)
    h, p_kw = stats.kruskal(*groups)
    grand = df[value].mean()
    ss_between = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    ss_total = ((df[value] - grand) ** 2).sum()
    return {
        "test": f"ANOVA charges ~ {group_col}",
        "f_stat": float(f),
        "p_value": float(p),
        "p_value_kruskal": float(p_kw),
        "eta_squared": float(ss_between / ss_total),
        "significant": bool(p < ALPHA),
    }


def chi_square_test(df: pd.DataFrame, col_a: str, col_b: str) -> dict:
    table = pd.crosstab(df[col_a], df[col_b])
    chi2, p, dof, _ = stats.chi2_contingency(table)
    n = table.to_numpy().sum()
    cramers_v = np.sqrt(chi2 / (n * (min(table.shape) - 1)))
    return {
        "test": f"Chi-square {col_a} x {col_b}",
        "chi2": float(chi2),
        "dof": int(dof),
        "p_value": float(p),
        "cramers_v": float(cramers_v),
        "significant": bool(p < ALPHA),
    }


def correlation_test(df: pd.DataFrame, col: str, value: str = "charges") -> dict:
    r, p = stats.pearsonr(df[col], df[value])
    rho, p_s = stats.spearmanr(df[col], df[value])
    return {
        "test": f"Correlation {col} ~ {value}",
        "pearson_r": float(r),
        "p_value": float(p),
        "spearman_rho": float(rho),
        "p_value_spearman": float(p_s),
        "significant": bool(p < ALPHA),
    }


def interaction_effect(df: pd.DataFrame) -> pd.DataFrame:
    """Mean charges for smoker x obesity: tests whether BMI matters only for smokers."""
    return (
        df.pivot_table(index="smoker", columns="is_obese", values="charges", aggfunc="mean")
        .rename(columns={0: "not_obese", 1: "obese"})
        .round(0)
    )


def run_all(df: pd.DataFrame) -> dict:
    return {
        "smoker": two_group_test(df, "smoker", "yes", "no"),
        "sex": two_group_test(df, "sex", "male", "female"),
        "obesity_nonsmokers": two_group_test(df[df["smoker"] == "no"], "is_obese", 1, 0),
        "obesity_smokers": two_group_test(df[df["smoker"] == "yes"], "is_obese", 1, 0),
        "region": anova_test(df, "region"),
        "age_group": anova_test(df, "age_group"),
        "children": anova_test(df, "children"),
        "smoker_by_sex": chi_square_test(df, "smoker", "sex"),
        "smoker_by_region": chi_square_test(df, "smoker", "region"),
        "age_corr": correlation_test(df, "age"),
        "age_corr_nonsmokers": correlation_test(df[df["smoker"] == "no"], "age"),
        "bmi_corr": correlation_test(df, "bmi"),
    }
