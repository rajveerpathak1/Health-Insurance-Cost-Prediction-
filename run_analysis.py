"""End-to-end analytics pipeline.

    python run_analysis.py

Produces:
  data/insurance.db             SQLite database
  reports/figures/*.png         charts used in README / report
  reports/results.json          all computed metrics
  reports/sql_outputs/*.csv     KPI query results
  reports/INSIGHTS.md           executive summary with recommendations
  insurance_cost_model.pkl      trained pricing pipeline (used by app.py & dashboard)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src import ab_test, modeling, segmentation, sql_analysis, stats_tests
from src.data_prep import TIER_ORDER, add_features, clean, load_raw, quality_report

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"
FIG = REPORTS / "figures"
SQL_OUT = REPORTS / "sql_outputs"

PALETTE = {"yes": "#E4572E", "no": "#29335C"}
TIER_COLORS = ["#4CB944", "#F3A712", "#E4572E", "#A8201A"]
sns.set_theme(style="whitegrid", font_scale=1.05)


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=130)
    plt.close(fig)


def make_figures(df, tier_prof, ab_exp, model_out):
    # 1. Charges distribution by smoker
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(df, x="charges", hue="smoker", bins=50, palette=PALETTE, alpha=0.7, ax=ax)
    ax.set(title="Charges distribution: smokers form a separate high-cost population", xlabel="Annual charges ($)")
    _save(fig, "01_charges_distribution.png")

    # 2. Age vs charges coloured by smoker (three visible bands)
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.scatterplot(df, x="age", y="charges", hue="smoker", style="bmi_category", palette=PALETTE, alpha=0.6, ax=ax)
    ax.set(title="Age vs charges: cost rises with age; smoking shifts the whole curve up")
    _save(fig, "02_age_vs_charges.png")

    # 3. BMI x smoker interaction
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(df, x="bmi_category", y="charges", hue="smoker", palette=PALETTE, errorbar=("ci", 95), ax=ax)
    ax.set(title="Obesity only drives cost for smokers (interaction effect)", ylabel="Mean charges ($)")
    _save(fig, "03_bmi_smoker_interaction.png")

    # 4. Risk tier: customer share vs cost share
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(tier_prof))
    ax.bar(x - 0.2, tier_prof["customer_share_pct"], 0.4, label="% of customers", color="#8D99AE")
    ax.bar(x + 0.2, tier_prof["cost_share_pct"], 0.4, label="% of total cost", color="#E4572E")
    ax.set_xticks(x, tier_prof.index)
    ax.set(title="Risk tiers: High + Critical tiers drive a disproportionate share of cost", ylabel="%")
    ax.legend()
    _save(fig, "04_risk_tier_share.png")

    # 5. Region heatmap
    fig, ax = plt.subplots(figsize=(8, 4.5))
    pivot = df.pivot_table(index="region", columns="smoker", values="charges", aggfunc="mean")
    sns.heatmap(pivot, annot=True, fmt=",.0f", cmap="Reds", ax=ax)
    ax.set(title="Mean charges by region x smoker")
    _save(fig, "05_region_heatmap.png")

    # 6. Correlation matrix
    fig, ax = plt.subplots(figsize=(7, 5.5))
    corr = df[["age", "bmi", "children", "is_smoker", "is_obese", "charges"]].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set(title="Correlation matrix")
    _save(fig, "06_correlation.png")

    # 7. A/B test
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.boxplot(ab_exp, x="group", y="charges", hue="group", palette=["#8D99AE", "#4CB944"], legend=False, ax=ax)
    ax.set(title="Simulated A/B test: wellness programme vs control (smokers)")
    _save(fig, "07_ab_test.png")

    # 8. Feature importance
    fig, ax = plt.subplots(figsize=(8, 4.5))
    imp = model_out["importance"]
    sns.barplot(imp, x="importance", y="feature", color="#29335C", ax=ax)
    ax.set(title=f"Cost drivers ({model_out['best_model']} permutation importance)", xlabel="Drop in R² when shuffled")
    _save(fig, "08_feature_importance.png")

    # 9. Actual vs predicted
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(model_out["y_test"], model_out["test_pred"], alpha=0.5, color="#29335C")
    lim = [0, max(model_out["y_test"].max(), model_out["test_pred"].max())]
    ax.plot(lim, lim, "--", color="#E4572E")
    ax.set(title="Actual vs predicted (hold-out set)", xlabel="Actual ($)", ylabel="Predicted ($)")
    _save(fig, "09_actual_vs_predicted.png")


def _fmt_p(p):
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def write_insights(q, tests, tier_prof, seg_prof, ab, model_out, sql):
    s = tests["smoker"]
    on, os_ = tests["obesity_nonsmokers"], tests["obesity_smokers"]
    decile = sql["top_decile_concentration"]
    top10 = float(decile.loc[decile["decile"] == 1, "cost_share_pct"].iloc[0])
    top20 = float(decile.loc[decile["decile"] == 2, "cumulative_cost_share_pct"].iloc[0])
    smk = sql["cost_by_smoker"].set_index("smoker")
    hc = tier_prof.loc[["High", "Critical"]]
    best = model_out["results"].iloc[0]
    imp = model_out["importance"]

    md = f"""# Executive Summary: Health Insurance Cost Drivers

> Auto-generated by `run_analysis.py`. All figures are computed from the data.

## Business problem
An insurer needs to understand **who its costliest policyholders are, why they cost more,
and what it can do about it**, both for fair pricing and for targeted cost-reduction programmes.

## Data
- {q['rows']:,} raw records, {q['duplicate_rows']:,} duplicates removed, {q['missing_values']} missing values.
- {sum(tier_prof['customers']):,.0f} unique policyholders analysed.

## Key insights

1. **Smoking is the #1 cost driver.** Smokers pay **{s['ratio']:.1f}x** more on average
   (${s['mean_a']:,.0f} vs ${s['mean_b']:,.0f}; Welch t-test p {_fmt_p(s['p_value_welch'])}, Cohen's d = {s['cohens_d']:.2f}).
   They are **{smk.loc['yes', 'customer_share_pct']:.0f}% of customers but {smk.loc['yes', 'cost_share_pct']:.0f}% of total cost**.

2. **Obesity matters mainly for smokers (interaction effect).** Among smokers, obese members cost
   **${os_['mean_diff']:,.0f} more** (p {_fmt_p(os_['p_value_welch'])}). Among non-smokers the gap is only
   **${on['mean_diff']:,.0f}** (d = {on['cohens_d']:.2f}). So a "BMI surcharge" for everyone would be unfair and poorly targeted.

3. **Costs are highly concentrated.** The top 10% of policyholders generate **{top10:.0f}%** of total cost,
   and the top 20% generate **{top20:.0f}%**.

4. **High + Critical risk tiers** make up **{hc['customer_share_pct'].sum():.0f}% of customers** but
   **{hc['cost_share_pct'].sum():.0f}% of cost**. Critical-tier members (smoker + obese) average
   **${tier_prof.loc['Critical', 'avg_charges']:,.0f}**.

5. **Age is the main driver for non-smokers** (Pearson r = {tests['age_corr_nonsmokers']['pearson_r']:.2f} within non-smokers,
   p {_fmt_p(tests['age_corr_nonsmokers']['p_value'])}). **Sex** (p = {tests['sex']['p_value_welch']:.3f}, but Cohen's d = {tests['sex']['cohens_d']:.2f})
   and **region** (eta² = {tests['region']['eta_squared']:.3f}) are statistically weak or practically negligible.

## A/B test: smoking-cessation wellness programme (simulated)
- Power analysis: **{ab['sample_size_per_group']} smokers per group** to detect a {ab['mde_pct']:.0f}% cost reduction (alpha = 0.05, power = 0.80).
- Result: treatment changed mean cost by **{ab['lift_pct']:.1f}%** (95% CI ${ab['ci95_low']:,.0f} to ${ab['ci95_high']:,.0f}, p {_fmt_p(ab['p_value'])}).
- Saving per member **${ab['saving_per_member']:,.0f}** vs programme cost **${ab['programme_cost_per_member']:,.0f}**,
  so net **${ab['net_benefit_per_member']:,.0f}/member (ROI {ab['roi_pct']:.0f}%)**. Guardrail: engagement {ab['engagement_rate']*100:.0f}%.
- **Decision: {ab['decision']}.**

> Outcomes are simulated from the real smoker cost distribution with an assumed 10% effect; the
> methodology (power analysis, randomisation, Welch test, CI, ROI decision) is the deliverable.

## Pricing model (ML)
- Best model: **{best['model']}**. 5-fold CV R² = **{best['cv_r2_mean']:.3f} ± {best['cv_r2_std']:.3f}**,
  hold-out R² = **{best['test_r2']:.3f}**, MAE = **${best['test_mae']:,.0f}**.
- Top drivers by permutation importance: **{', '.join(imp['feature'].head(3))}**, which matches the hypothesis tests.

## Recommendations
| # | Recommendation | Evidence |
|---|---|---|
| 1 | Prioritise smokers for a cessation / wellness programme, starting with the Critical tier | Insight 1, 2, A/B ROI |
| 2 | Price on smoker x BMI **interaction**, not BMI alone | Insight 2 |
| 3 | Set up case management for the top-decile members | Insight 3 |
| 4 | Drop sex/region as pricing levers (low effect, regulatory risk) | Insight 5 |
| 5 | Run the wellness programme as a real randomised pilot with n = {ab['sample_size_per_group']}/arm before full rollout | A/B design |
"""
    (REPORTS / "INSIGHTS.md").write_text(md, encoding="utf-8")


def _to_jsonable(o):
    if isinstance(o, dict):
        return {k: _to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def main():
    for d in (FIG, SQL_OUT):
        d.mkdir(parents=True, exist_ok=True)

    raw = load_raw()
    q = quality_report(raw)
    df = add_features(clean(raw))
    print(f"[1/6] Data: {q['rows']} raw -> {len(df)} clean rows")

    sql_analysis.build_db(df)
    sql = sql_analysis.run_queries()
    for name, res in sql.items():
        res.to_csv(SQL_OUT / f"{name}.csv", index=False)
    print(f"[2/6] SQL: {len(sql)} KPI queries executed")

    tests = stats_tests.run_all(df)
    print(f"[3/6] Stats: {len(tests)} hypothesis tests")

    tier_prof = segmentation.tier_profile(df)
    k_scores = segmentation.choose_k(df)
    _, seg_prof = segmentation.kmeans_segments(df, k=4)
    print("[4/6] Segmentation done")

    ab, ab_exp = ab_test.run(df)
    print(f"[5/6] A/B test: n={ab['sample_size_per_group']}/group, decision={ab['decision']}")

    model_out = modeling.train_and_save(df)
    print(f"[6/6] Model: best={model_out['best_model']}")
    print(model_out["results"].round(3).to_string(index=False))

    make_figures(df, tier_prof, ab_exp, model_out)
    write_insights(q, tests, tier_prof, seg_prof, ab, model_out, sql)

    results = {
        "data_quality": q,
        "hypothesis_tests": tests,
        "interaction_table": stats_tests.interaction_effect(df).to_dict(),
        "risk_tiers": tier_prof.reset_index().to_dict(orient="records"),
        "kmeans_k_selection": k_scores.to_dict(orient="records"),
        "kmeans_segments": seg_prof.reset_index().to_dict(orient="records"),
        "ab_test": ab,
        "model_comparison": model_out["results"].to_dict(orient="records"),
        "best_model": model_out["best_model"],
        "feature_importance": model_out["importance"].to_dict(orient="records"),
    }
    (REPORTS / "results.json").write_text(json.dumps(_to_jsonable(results), indent=2, default=str), encoding="utf-8")
    print("Done. See reports/INSIGHTS.md")


if __name__ == "__main__":
    main()
