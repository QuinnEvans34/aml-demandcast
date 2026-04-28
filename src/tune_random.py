"""
src/tune_random.py — Random-split hyperparameter tuning for DemandCast
======================================================================

This is the canonical tuning script for the project. It uses random
splitting throughout (no temporal cutoffs) for methodological consistency
between the outer train/val/test split and the inner cross-validation:

  - 60 / 20 / 20 random train / val / test split (random_state=42)
  - Inside each Optuna trial: KFold(n_splits=5, shuffle=True) on `train`
  - Optuna minimizes mean CV MAE across the 5 folds
  - Per-trial: the trial's val_* metrics are also logged so trials can be
    spot-checked; selection is driven by mean_cv_mae.

After the study:
  - Fit an "eval model" on train + val with the best params, evaluate on
    the held-out test set → these are the HONEST test metrics that get
    logged and reported (test_mae, test_rmse, test_r2, test_mape, test_mbe).
  - Fit a "final deployment model" on train + val + test (full dataset)
    with the best params and register it as DemandCast Production. The
    deployed model has seen everything; the metrics in the run come from
    the eval model so they remain honest.
  - archive_existing_versions=False → the Week 3 baseline (v1) stays in
    Staging untouched. Whichever version gets created here is promoted to
    Production and the previous Production version (if any) is left alone
    for cleanup outside this script.

Run from the project root with the .venv active:

    python -m src.tune_random

Prerequisites:
  - MLflow server running at http://localhost:8080 with backend
    sqlite:///mlflow.db and artifact root ./mlartifacts.
  - data/features.parquet exists.
"""
import datetime
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import KFold, train_test_split

from src.features import FEATURE_COLS, TARGET

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MLFLOW_TRACKING_URI = "http://localhost:8080"
EXPERIMENT_NAME = "DemandCast"
MODEL_REGISTRY_NAME = "DemandCast"

DATA_PATH = Path(__file__).parent.parent / "data" / "features.parquet"
RANDOM_STATE = 42
N_TRIALS = 15

# 60/20/20 train/val/test. Achieved with two successive splits:
#   first carve off 20% for test, then carve off 25% of the remaining 80%
#   for val (0.25 * 0.80 = 0.20). Final fractions: 0.60 / 0.20 / 0.20.
TEST_FRACTION = 0.20
VAL_FRACTION_OF_REMAINING = 0.25

# Module-level cache so we read parquet exactly once per `python -m src.tune_random` invocation.
_SPLITS_CACHE = None


# ---------------------------------------------------------------------------
# Splits + metrics helpers
# ---------------------------------------------------------------------------

def _load_random_splits():
    """Load features.parquet and return a 60/20/20 random train/val/test split.

    Note on the `hour` column: FEATURE_COLS already references the integer
    `hour_of_day` feature, NOT the datetime `hour` column. So we just select
    FEATURE_COLS and TARGET — no datetime conversion needed.

    Returns
    -------
    X_train, y_train, X_val, y_val, X_test, y_test
    """
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"features.parquet not found at {DATA_PATH}. Run build_features.py first."
        )
    df = pd.read_parquet(DATA_PATH)
    X = df[FEATURE_COLS]
    y = df[TARGET]

    # First split: hold out 20% as test.
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=TEST_FRACTION,
        random_state=RANDOM_STATE,
        shuffle=True,
    )
    # Second split: 25% of the remaining 80% becomes val (= 20% overall).
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=VAL_FRACTION_OF_REMAINING,
        random_state=RANDOM_STATE,
        shuffle=True,
    )
    return X_train, y_train, X_val, y_val, X_test, y_test


def _get_splits():
    """Cached accessor — splits are deterministic so we only need to load once."""
    global _SPLITS_CACHE
    if _SPLITS_CACHE is None:
        _SPLITS_CACHE = _load_random_splits()
    return _SPLITS_CACHE


def _filtered_mape(y_true, y_pred):
    """MAPE filtered to y_true > 0 (matches evaluate.py for the baseline)."""
    nonzero = y_true > 0
    if nonzero.sum() == 0:
        return float("nan")
    y_nz = y_true[nonzero]
    p_nz = y_pred[nonzero.to_numpy()]
    return float(np.mean(np.abs((y_nz - p_nz) / y_nz)) * 100)


def _full_metric_suite(y_true, y_pred):
    """Compute the 5-metric suite used throughout the project.

    Returns
    -------
    dict with keys: mae, rmse, r2, mape, mbe
    """
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "mape": _filtered_mape(y_true, y_pred),
        "mbe": float(np.mean(y_pred - y_true)),
    }


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------

def objective(trial: optuna.Trial) -> float:
    """Optuna objective — KFold(shuffle=True) CV on train, return mean CV MAE.

    Each trial also fits a fresh model on full `train` and evaluates on
    `val` so per-trial val_* metrics get logged for spot-checking. Optuna's
    selection is driven by the returned `mean_cv_mae` (CV on train only).
    """
    params = {
        # 100 is RF default; above ~500 gives diminishing MAE returns on this
        # dataset size and adds significant compute time per trial.
        "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),

        # Shallow trees underfit zone-level demand patterns; very deep trees
        # overfit the 46% zero-demand rows. 5–30 brackets the useful range.
        "max_depth": trial.suggest_int("max_depth", 5, 30),

        # Controls smoothing of leaf predictions; higher values prevent the
        # model from memorizing individual zone-hour noise.
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),

        # Minimum samples required to split a node; prevents splits driven
        # by statistical noise in sparse outer-borough zones.
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),

        # sqrt is RF default and works well on structured tabular data;
        # log2 is more aggressive feature subsampling; 0.5 tries half the
        # features per split for more diverse trees in the ensemble.
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5]),

        "random_state": RANDOM_STATE,  # fixed for reproducibility — not tuned
        "n_jobs": -1,                   # use all cores — not tuned
    }

    X_train, y_train, X_val, y_val, _X_test, _y_test = _get_splits()

    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    run_name = (
        f"optuna_random_trial_{trial.number}_"
        f"{datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}"
    )

    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("logged_at_utc", datetime.datetime.utcnow().isoformat())
        mlflow.log_params(params)
        mlflow.log_param("objective", "kfold_shuffle_train")
        mlflow.log_param("split_strategy", "random_60_20_20_train_val_test")

        # ---- Inner CV: KFold(shuffle=True) on train, return mean MAE ----
        fold_maes = []
        for fold, (tr_idx, te_idx) in enumerate(kf.split(X_train)):
            X_tr = X_train.iloc[tr_idx]
            y_tr = y_train.iloc[tr_idx]
            X_te = X_train.iloc[te_idx]
            y_te = y_train.iloc[te_idx]

            fold_model = RandomForestRegressor(**params)
            fold_model.fit(X_tr, y_tr)
            preds = fold_model.predict(X_te)
            fold_mae = float(mean_absolute_error(y_te, preds))
            fold_maes.append(fold_mae)
            mlflow.log_metric(f"fold_{fold}_mae", fold_mae, step=fold)

        mean_cv_mae = float(np.mean(fold_maes))
        cv_mae_std = float(np.std(fold_maes, ddof=1))
        mlflow.log_metric("mean_cv_mae", mean_cv_mae)
        mlflow.log_metric("cv_mae_std", cv_mae_std)

        # ---- Per-trial val metrics: fit on train, eval on val ----
        # Honest because val is held out from the KFold CV above.
        trial_model = RandomForestRegressor(**params)
        trial_model.fit(X_train, y_train)
        val_preds = trial_model.predict(X_val)
        val_metrics = _full_metric_suite(y_val, val_preds)
        for k, v in val_metrics.items():
            mlflow.log_metric(f"val_{k}", v)

        mlflow.sklearn.log_model(trial_model, name="model")

    return mean_cv_mae


# ---------------------------------------------------------------------------
# Final retrain + register
# ---------------------------------------------------------------------------

def retrain_and_register(best_params: dict, stage: str = "Production") -> None:
    """Fit + register the final model.

    Two models are involved:
      1. eval_model — fit on train + val, evaluated on the held-out test set.
         Its metrics (test_*) are the HONEST numbers we report and log.
      2. final_model — fit on train + val + test (the entire dataset).
         This is the model artifact registered to the MLflow Model Registry
         and used by the dashboard for live predictions.

    The MLflow run logs:
      - best_params (the hyperparameters Optuna selected)
      - test_* metrics from eval_model (honest test eval)
      - val_* metrics from a model fit on train, eval on val (also honest)
      - the final_model artifact (fit on all data, the deployed model)
      - explicit params describing what each metric was computed against,
        so anyone auditing the run knows the metrics ≠ training set of the
        deployed artifact.

    archive_existing_versions=False so the Week 3 baseline in Staging is
    not touched. Registry cleanup (archiving prior tuned versions) is
    handled by a separate script.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    X_train, y_train, X_val, y_val, X_test, y_test = _get_splits()

    # 1) Honest val metrics: fit on train, eval on val.
    val_eval_model = RandomForestRegressor(**best_params)
    val_eval_model.fit(X_train, y_train)
    val_preds = val_eval_model.predict(X_val)
    val_metrics = _full_metric_suite(y_val, val_preds)

    # 2) Honest test metrics: fit on train+val, eval on the held-out test.
    X_trainval = pd.concat([X_train, X_val])
    y_trainval = pd.concat([y_train, y_val])
    test_eval_model = RandomForestRegressor(**best_params)
    test_eval_model.fit(X_trainval, y_trainval)
    test_preds = test_eval_model.predict(X_test)
    test_metrics = _full_metric_suite(y_test, test_preds)

    # 3) Final deployment model: fit on all data (train + val + test).
    X_all = pd.concat([X_train, X_val, X_test])
    y_all = pd.concat([y_train, y_val, y_test])
    final_model = RandomForestRegressor(**best_params)
    final_model.fit(X_all, y_all)

    run_name = (
        f"final_retrain_random_split_"
        f"{datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}"
    )
    with mlflow.start_run(run_name=run_name) as run:
        # --- Run-level params: hyperparams + provenance ---
        mlflow.log_param("logged_at_utc", datetime.datetime.utcnow().isoformat())
        mlflow.log_params(best_params)
        mlflow.log_param("split_strategy", "random_60_20_20_train_val_test")
        mlflow.log_param("registered_model_fit_on", "train+val+test_full_data")
        mlflow.log_param("val_metrics_fit_on", "train_only")
        mlflow.log_param("test_metrics_fit_on", "train+val")

        # --- val_* metrics (from val_eval_model: fit on train, eval on val) ---
        for k, v in val_metrics.items():
            mlflow.log_metric(f"val_{k}", v)

        # --- test_* metrics (from test_eval_model: fit on train+val, eval on test) ---
        for k, v in test_metrics.items():
            mlflow.log_metric(f"test_{k}", v)

        # --- Log + register the FINAL deployment model (fit on all data) ---
        mlflow.sklearn.log_model(final_model, name="model")
        model_uri = f"runs:/{run.info.run_id}/model"
        result = mlflow.register_model(model_uri=model_uri, name=MODEL_REGISTRY_NAME)

        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=MODEL_REGISTRY_NAME,
            version=result.version,
            stage=stage,
            archive_existing_versions=False,  # leave v1 (baseline) alone
        )

    # --- Console summary ---
    print()
    print("=" * 72)
    print("  Final retrain + register (random-split methodology)")
    print("=" * 72)
    print(f"  Registered:   {MODEL_REGISTRY_NAME} v{result.version} → {stage}")
    print(f"  Source run:   {run.info.run_id}")
    print()
    print("  Honest val metrics  (fit on train, eval on val):")
    print(f"    val_mae   = {val_metrics['mae']:.4f}")
    print(f"    val_rmse  = {val_metrics['rmse']:.4f}")
    print(f"    val_r2    = {val_metrics['r2']:.4f}")
    print(f"    val_mape  = {val_metrics['mape']:.2f}%")
    print(f"    val_mbe   = {val_metrics['mbe']:.4f}")
    print()
    print("  Honest test metrics (fit on train+val, eval on test):")
    print(f"    test_mae  = {test_metrics['mae']:.4f}")
    print(f"    test_rmse = {test_metrics['rmse']:.4f}")
    print(f"    test_r2   = {test_metrics['r2']:.4f}")
    print(f"    test_mape = {test_metrics['mape']:.2f}%")
    print(f"    test_mbe  = {test_metrics['mbe']:.4f}")
    print()
    print("  Deployed model artifact: fit on train+val+test (all data)")
    print("=" * 72)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Trigger the split once up front so we fail fast on a missing parquet.
    _ = _get_splits()

    print(f"Starting random-split Optuna study ({N_TRIALS} trials, KFold(shuffle=True))...")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

    print()
    print("=" * 72)
    print("  Optuna study complete")
    print("=" * 72)
    print(f"  Best trial number:  #{study.best_trial.number}")
    print(f"  Best mean CV MAE:   {study.best_value:.4f}")
    print(f"  Best hyperparams:")
    for k, v in study.best_trial.params.items():
        print(f"    {k:20} = {v}")
    print()
    print(f"  All trial CV MAEs (in order):")
    for t in study.trials:
        print(f"    Trial #{t.number:>2}  CV_MAE = {t.value:.4f}")
    print("=" * 72)

    retrain_and_register(study.best_trial.params, stage="Production")

    print()
    print("Done. Restart uvicorn so the dashboard backend reloads the new")
    print("Production model:")
    print("    Ctrl+C the uvicorn terminal, then:")
    print("    uvicorn app.api.main:app --port 8000")
