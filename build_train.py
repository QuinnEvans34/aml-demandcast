"""
build_train.py — Run all model training experiments for DemandCast
===================================================================
Trains three regression models against the engineered feature set and
logs every run to MLflow under the experiment "DemandCast".

Usage (from project root with .venv active):
    python build_train.py

Prerequisites:
    1. MLflow UI running:    mlflow ui
    2. Features built:       python build_features.py
"""

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor

from src.train import train_and_log


# ---------------------------------------------------------------------------
# Model 1 — Linear Regression (required baseline)
# Always start with the simplest model. Sets the performance floor.
# ---------------------------------------------------------------------------
run_id_lr = train_and_log(
    model=LinearRegression(),
    run_name="linear_regression_baseline",
    params={"model": "LinearRegression"},
)
print(f"Linear Regression run ID: {run_id_lr}\n")


# ---------------------------------------------------------------------------
# Model 2 — Random Forest Regressor
# Handles non-linear relationships and feature interactions (e.g. zone ×
# hour_of_day effects) that Linear Regression cannot capture.
# ---------------------------------------------------------------------------
run_id_rf = train_and_log(
    model=RandomForestRegressor(n_estimators=100, random_state=42),
    run_name="random_forest_100_estimators",
    params={
        "model":        "RandomForestRegressor",
        "n_estimators": 100,
        "random_state": 42,
    },
)
print(f"Random Forest run ID: {run_id_rf}\n")


# ---------------------------------------------------------------------------
# Model 3 — Ridge Regression (alpha=1.0)
# Regularized linear model. Chosen because the feature set includes highly
# correlated lag columns (demand_lag_1h, demand_lag_24h, demand_lag_168h)
# and rolling averages — multicollinearity that Ridge's L2 penalty handles
# better than plain OLS. Comparable to the baseline but more robust.
# ---------------------------------------------------------------------------
run_id_ridge = train_and_log(
    model=Ridge(alpha=1.0),
    run_name="ridge_alpha_1",
    params={
        "model": "Ridge",
        "alpha": 1.0,
    },
)
print(f"Ridge run ID: {run_id_ridge}\n")
