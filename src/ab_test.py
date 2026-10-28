"""A/B experiment design & analysis for a smoking-cessation wellness programme.

Business question: if the insurer offers smokers a wellness programme, does it
reduce claim costs enough to justify the spend?

Because the source dataset is observational, outcomes for the experiment are
*simulated* from the real smoker cost distribution with an assumed treatment
effect. The value here is the methodology: power analysis -> randomisation ->
test -> guardrail check -> business decision.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.power import TTestIndPower


def required_sample_size(baseline_mean: float, baseline_std: float, mde_pct: float,
                         alpha: float = 0.05, power: float = 0.8) -> int:
    """Per-group sample size to detect a relative reduction of `mde_pct` in mean cost."""
    effect = (baseline_mean * mde_pct) / baseline_std
    n = TTestIndPower().solve_power(effect_size=effect, alpha=alpha, power=power, alternative="two-sided")
    return int(np.ceil(n))


def simulate_experiment(smokers: pd.DataFrame, n_per_group: int, true_effect_pct: float = 0.10,
                        programme_cost: float = 1500.0, seed: int = 7) -> pd.DataFrame:
    """Bootstrap smoker records into control/treatment and apply the treatment effect with noise."""
    rng = np.random.default_rng(seed)
    base = smokers["charges"].to_numpy()
    control = rng.choice(base, n_per_group, replace=True)
    treated_base = rng.choice(base, n_per_group, replace=True)
    effect = rng.normal(true_effect_pct, 0.05, n_per_group).clip(-0.1, 0.4)
    treatment = treated_base * (1 - effect)
    # Guardrail metric: programme drop-out / engagement
    engaged = rng.random(n_per_group) < 0.72
    return pd.DataFrame({
        "group": ["control"] * n_per_group + ["treatment"] * n_per_group,
        "charges": np.concatenate([control, treatment]),
        "engaged": np.concatenate([np.zeros(n_per_group, bool), engaged]),
        "programme_cost": np.concatenate([np.zeros(n_per_group), np.full(n_per_group, programme_cost)]),
    })


def analyse(exp: pd.DataFrame, alpha: float = 0.05) -> dict:
    c = exp.loc[exp["group"] == "control", "charges"].to_numpy()
    t = exp.loc[exp["group"] == "treatment", "charges"].to_numpy()
    tstat, p = stats.ttest_ind(t, c, equal_var=False)
    _, p_mw = stats.mannwhitneyu(t, c)
    diff = t.mean() - c.mean()
    se = np.sqrt(t.var(ddof=1) / len(t) + c.var(ddof=1) / len(c))
    cost = exp.loc[exp["group"] == "treatment", "programme_cost"].mean()
    saving = -diff
    return {
        "n_control": int(len(c)),
        "n_treatment": int(len(t)),
        "mean_control": float(c.mean()),
        "mean_treatment": float(t.mean()),
        "lift_pct": float(diff / c.mean() * 100),
        "ci95_low": float(diff - 1.96 * se),
        "ci95_high": float(diff + 1.96 * se),
        "p_value": float(p),
        "p_value_mannwhitney": float(p_mw),
        "significant": bool(p < alpha),
        "saving_per_member": float(saving),
        "programme_cost_per_member": float(cost),
        "net_benefit_per_member": float(saving - cost),
        "roi_pct": float((saving - cost) / cost * 100) if cost else float("nan"),
        "engagement_rate": float(exp.loc[exp["group"] == "treatment", "engaged"].mean()),
        "decision": "Roll out" if (p < alpha and saving > cost) else "Do not roll out / iterate",
    }


def run(df: pd.DataFrame, mde_pct: float = 0.10, true_effect_pct: float = 0.10) -> dict:
    smokers = df[df["smoker"] == "yes"]
    n = required_sample_size(smokers["charges"].mean(), smokers["charges"].std(), mde_pct)
    exp = simulate_experiment(smokers, n, true_effect_pct)
    return {"sample_size_per_group": n, "mde_pct": mde_pct * 100, **analyse(exp)}, exp
