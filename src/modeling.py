"""ML pricing model: compare regressors, cross-validate, explain drivers, persist pipeline."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "insurance_cost_model.pkl"

FEATURES = ["age", "sex", "bmi", "children", "smoker", "region"]
NUMERIC = ["age", "bmi", "children"]
CATEGORICAL = ["sex", "smoker", "region"]


def _preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CATEGORICAL),
    ])


def candidate_models(seed: int = 42) -> dict:
    return {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(n_estimators=300, max_depth=8, min_samples_leaf=3, random_state=seed, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, random_state=seed),
    }


def make_pipeline(model) -> Pipeline:
    return Pipeline([("prep", _preprocessor()), ("model", model)])


def evaluate(df: pd.DataFrame, seed: int = 42) -> tuple[pd.DataFrame, dict, tuple]:
    X, y = df[FEATURES], df["charges"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=seed)
    cv = KFold(5, shuffle=True, random_state=seed)
    rows, fitted = [], {}
    for name, model in candidate_models(seed).items():
        pipe = make_pipeline(model)
        cv_r2 = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="r2")
        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_te)
        rows.append({
            "model": name,
            "cv_r2_mean": cv_r2.mean(),
            "cv_r2_std": cv_r2.std(),
            "test_r2": r2_score(y_te, pred),
            "test_mae": mean_absolute_error(y_te, pred),
            "test_rmse": float(np.sqrt(mean_squared_error(y_te, pred))),
        })
        fitted[name] = pipe
    results = pd.DataFrame(rows).sort_values("cv_r2_mean", ascending=False).reset_index(drop=True)
    return results, fitted, (X_tr, X_te, y_tr, y_te)


def feature_importance(pipe: Pipeline, X_te: pd.DataFrame, y_te: pd.Series, seed: int = 42) -> pd.DataFrame:
    """Permutation importance on raw input columns (model-agnostic, business readable)."""
    r = permutation_importance(pipe, X_te, y_te, n_repeats=10, random_state=seed, scoring="r2")
    return (pd.DataFrame({"feature": FEATURES, "importance": r.importances_mean, "std": r.importances_std})
            .sort_values("importance", ascending=False).reset_index(drop=True))


def train_and_save(df: pd.DataFrame, path: Path = MODEL_PATH) -> dict:
    results, fitted, (X_tr, X_te, y_tr, y_te) = evaluate(df)
    best_name = results.iloc[0]["model"]
    best = fitted[best_name]
    importance = feature_importance(best, X_te, y_te)
    # Refit on all data for deployment
    final = make_pipeline(candidate_models()[best_name]).fit(df[FEATURES], df["charges"])
    joblib.dump(final, path)
    return {"results": results, "best_model": best_name, "importance": importance,
            "test_pred": best.predict(X_te), "y_test": y_te.to_numpy()}


def load_model(path: Path = MODEL_PATH) -> Pipeline:
    return joblib.load(path)


def predict_one(model: Pipeline, age, sex, bmi, children, smoker, region) -> float:
    row = pd.DataFrame([{"age": float(age), "sex": str(sex).lower(), "bmi": float(bmi),
                         "children": int(children), "smoker": str(smoker).lower(), "region": str(region).lower()}])
    return float(model.predict(row)[0])
