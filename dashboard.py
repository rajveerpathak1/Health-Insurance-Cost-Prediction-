"""Interactive analytics dashboard.

    streamlit run dashboard.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import ab_test, modeling, segmentation, stats_tests
from src.data_prep import TIER_ORDER, load_clean

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="Health Insurance Cost Analytics", page_icon="ðŸ“Š", layout="wide")

COLORS = {"yes": "#E4572E", "no": "#29335C"}
TIER_COLORS = dict(zip(TIER_ORDER, ["#4CB944", "#F3A712", "#E4572E", "#A8201A"]))


@st.cache_data
def get_data():
    return load_clean()


@st.cache_resource
def get_model():
    path = modeling.MODEL_PATH
    if not path.exists():
        modeling.train_and_save(get_data())
    return modeling.load_model(path)


df_all = get_data()

# ---------------- Sidebar filters ----------------
st.sidebar.header("Filters")
regions = st.sidebar.multiselect("Region", sorted(df_all["region"].unique()), default=sorted(df_all["region"].unique()))
smoker = st.sidebar.multiselect("Smoker", ["yes", "no"], default=["yes", "no"])
sex = st.sidebar.multiselect("Sex", ["male", "female"], default=["male", "female"])
age = st.sidebar.slider("Age range", int(df_all.age.min()), int(df_all.age.max()), (int(df_all.age.min()), int(df_all.age.max())))

df = df_all[
    df_all.region.isin(regions) & df_all.smoker.isin(smoker) & df_all.sex.isin(sex) & df_all.age.between(*age)
]
if df.empty:
    st.warning("No data for the selected filters.")
    st.stop()

st.title("ðŸ“Š Health Insurance Cost Analytics")
st.caption("Who are the costliest policyholders, why do they cost more, and what should the business do about it?")

tabs = st.tabs(["Overview", "Cost Drivers", "Segments", "Hypothesis Tests", "A/B Test", "Pricing Model"])

# ---------------- Overview ----------------
with tabs[0]:
    c = st.columns(5)
    c[0].metric("Policyholders", f"{len(df):,}")
    c[1].metric("Total charges", f"${df.charges.sum() / 1e6:,.2f}M")
    c[2].metric("Avg charges", f"${df.charges.mean():,.0f}")
    c[3].metric("Smoker rate", f"{df.is_smoker.mean() * 100:.1f}%")
    c[4].metric("Obesity rate", f"{df.is_obese.mean() * 100:.1f}%")

    l, r = st.columns(2)
    l.plotly_chart(px.histogram(df, x="charges", color="smoker", nbins=50, barmode="overlay",
                                color_discrete_map=COLORS, title="Charges distribution"), width="stretch")
    sorted_c = df.charges.sort_values(ascending=False).reset_index(drop=True)
    pareto = pd.DataFrame({"pct_customers": (sorted_c.index + 1) / len(sorted_c) * 100,
                           "pct_cost": sorted_c.cumsum() / sorted_c.sum() * 100})
    fig = px.line(pareto, x="pct_customers", y="pct_cost", title="Cost concentration (Pareto curve)")
    fig.add_shape(type="line", x0=0, y0=0, x1=100, y1=100, line=dict(dash="dash", color="grey"))
    r.plotly_chart(fig, width="stretch")
    top10 = pareto.loc[pareto.pct_customers <= 10, "pct_cost"].max()
    st.info(f"ðŸ’¡ The top 10% of policyholders account for **{top10:.0f}%** of total cost.")

# ---------------- Cost Drivers ----------------
with tabs[1]:
    l, r = st.columns(2)
    l.plotly_chart(px.scatter(df, x="age", y="charges", color="smoker", symbol="bmi_category", opacity=0.6,
                              color_discrete_map=COLORS, title="Age vs charges"), width="stretch")
    inter = df.groupby(["bmi_category", "smoker"], observed=True).charges.mean().reset_index()
    r.plotly_chart(px.bar(inter, x="bmi_category", y="charges", color="smoker", barmode="group",
                          color_discrete_map=COLORS, title="BMI x smoker interaction"), width="stretch")
    l, r = st.columns(2)
    heat = df.pivot_table(index="region", columns="smoker", values="charges", aggfunc="mean")
    l.plotly_chart(px.imshow(heat, text_auto=",.0f", color_continuous_scale="Reds", title="Region x smoker"),
                   width="stretch")
    ag = df.groupby(["age_group", "smoker"], observed=True).charges.mean().reset_index()
    r.plotly_chart(px.line(ag, x="age_group", y="charges", color="smoker", markers=True,
                           color_discrete_map=COLORS, title="Avg charges by age group"), width="stretch")

# ---------------- Segments ----------------
with tabs[2]:
    prof = segmentation.tier_profile(df)
    st.subheader("Rule-based risk tiers")
    fig = go.Figure([
        go.Bar(name="% customers", x=prof.index, y=prof.customer_share_pct, marker_color="#8D99AE"),
        go.Bar(name="% cost", x=prof.index, y=prof.cost_share_pct, marker_color="#E4572E"),
    ])
    fig.update_layout(barmode="group", title="Customer share vs cost share by tier")
    st.plotly_chart(fig, width="stretch")
    st.dataframe(prof, width="stretch")
    st.caption("Tiers: Critical = smoker + obese Â· High = smoker Â· Medium = obese or age 50+ Â· Low = rest")

    if len(df) >= 50:
        st.subheader("K-Means behavioural segments")
        k = st.slider("Number of clusters", 2, 6, 4)
        seg_df, seg_prof = segmentation.kmeans_segments(df, k=k)
        l, r = st.columns([3, 2])
        l.plotly_chart(px.scatter(seg_df, x="age", y="charges", color="segment", title="Segments"),
                       width="stretch")
        r.dataframe(seg_prof, width="stretch")

# ---------------- Hypothesis tests ----------------
with tabs[3]:
    try:
        tests = stats_tests.run_all(df)
        rows = []
        for key, t in tests.items():
            p = t.get("p_value_welch", t.get("p_value"))
            effect = next((f"{n}={t[n]:.2f}" for n in ("cohens_d", "eta_squared", "cramers_v", "pearson_r") if n in t), "")
            rows.append({"Test": t["test"], "p-value": p, "Effect size": effect,
                         "Significant (Î±=0.05)": "âœ…" if t["significant"] else "âŒ"})
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        st.markdown("**Smoker x obesity mean charges**")
        st.dataframe(stats_tests.interaction_effect(df), width="stretch")
    except Exception as e:  # filters can leave a group empty
        st.warning(f"Not enough data in every group for the current filters ({e}). Widen the filters.")

# ---------------- A/B ----------------
with tabs[4]:
    st.markdown("**Experiment:** offer smokers a smoking-cessation wellness programme. Primary metric: annual charges. "
                "Guardrail: engagement rate. *Outcomes simulated from the real smoker cost distribution.*")
    c = st.columns(3)
    mde = c[0].slider("Minimum detectable effect (%)", 3, 25, 10) / 100
    eff = c[1].slider("Assumed true effect (%)", 0, 25, 10) / 100
    cost = c[2].number_input("Programme cost / member ($)", 0, 10000, 1500, step=100)
    smokers = df_all[df_all.smoker == "yes"]
    n = ab_test.required_sample_size(smokers.charges.mean(), smokers.charges.std(), mde)
    exp = ab_test.simulate_experiment(smokers, n, eff, programme_cost=cost)
    res = ab_test.analyse(exp)
    m = st.columns(5)
    m[0].metric("Sample / group", f"{n:,}")
    m[1].metric("Lift", f"{res['lift_pct']:.1f}%")
    m[2].metric("p-value", f"{res['p_value']:.4f}")
    m[3].metric("Net benefit / member", f"${res['net_benefit_per_member']:,.0f}")
    m[4].metric("Decision", res["decision"])
    st.plotly_chart(px.box(exp, x="group", y="charges", color="group", title="Control vs treatment"),
                    width="stretch")

# ---------------- Model ----------------
with tabs[5]:
    res_path = ROOT / "reports" / "results.json"
    if res_path.exists():
        r = json.loads(res_path.read_text())
        st.subheader(f"Model comparison (best: {r['best_model']})")
        st.dataframe(pd.DataFrame(r["model_comparison"]).round(3), width="stretch", hide_index=True)
        st.plotly_chart(px.bar(pd.DataFrame(r["feature_importance"]), x="importance", y="feature", orientation="h",
                               title="Permutation importance"), width="stretch")
    st.subheader("ðŸ’° Premium estimator")
    model = get_model()
    c = st.columns(6)
    a = c[0].number_input("Age", 18, 100, 35)
    s = c[1].selectbox("Sex", ["male", "female"])
    b = c[2].number_input("BMI", 10.0, 70.0, 27.0)
    ch = c[3].number_input("Children", 0, 10, 1)
    sm = c[4].selectbox("Smoker", ["no", "yes"])
    rg = c[5].selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])
    pred = modeling.predict_one(model, a, s, b, ch, sm, rg)
    alt = modeling.predict_one(model, a, s, b, ch, "no" if sm == "yes" else "yes", rg)
    st.metric("Estimated annual charges", f"${pred:,.0f}", delta=f"{pred - alt:+,.0f} vs {'non-smoker' if sm == 'yes' else 'smoker'}",
              delta_color="inverse")
