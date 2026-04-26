"""
tune.py — Hyperparameter tuning for DemandCast (Optuna + MLflow)
===============================================================
Runs an Optuna study to tune a RandomForestRegressor on the train/val
split. Each trial is logged to MLflow; the best run can be registered
to the MLflow Model Registry.

Run from project root with the `.venv` active:
    python tune.py
"""
from pathlib import Path
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import optuna
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import datetime

from src.features import FEATURE_COLS, TARGET


# ---------------------------------------------------------------------------
# Configuration — keep in sync with train.py and cv.py
# ---------------------------------------------------------------------------

MLFLOW_TRACKING_URI = "http://localhost:8080"
EXPERIMENT_NAME = "DemandCast"
MODEL_REGISTRY_NAME = "DemandCast"

DATA_PATH = Path(__file__).parent.parent / "data" / "features.parquet"
VAL_CUTOFF = "2025-01-22"
TEST_CUTOFF = "2025-02-01"

N_TRIALS = 15


def load_splits():
    """Load features.parquet and return train and validation splits.

    Returns
    -------
    X_train, y_train, X_val, y_val
    """
    # Step 1: Raise a FileNotFoundError if DATA_PATH does not exist.

    # Step 2: Read the parquet file at DATA_PATH into a DataFrame.

    # Step 3: Parse the "hour" column as datetime using pd.to_datetime().

    # Step 4: Filter rows where "hour" < VAL_CUTOFF into a `train` DataFrame.
    #         Filter rows where VAL_CUTOFF <= "hour" < TEST_CUTOFF into a `val` DataFrame.
    #         Use pd.to_datetime() when comparing against the cutoff strings.
    #         Call .copy() on each slice to avoid SettingWithCopyWarning.

    # Step 5: Convert the "hour" column in both `train` and `val` to integer
    #         hour-of-day using the .dt.hour accessor (consistent with train.py preprocessing).

    # Step 6: Return X_train, y_train, X_val, y_val.
    #         Select features with FEATURE_COLS and the target with TARGET.

    # Step 1: Raise FileNotFoundError if DATA_PATH does not exist.
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"features.parquet not found at {DATA_PATH}. Run build_features.py first."
        )

    # Step 2: Read the parquet file.
    df = pd.read_parquet(DATA_PATH)

    # Step 3: Parse "hour" as datetime.
    df["hour"] = pd.to_datetime(df["hour"])

    # Step 4: Temporal split on "hour".
    train = df[df["hour"] < pd.to_datetime(VAL_CUTOFF)].copy()
    val = df[
        (df["hour"] >= pd.to_datetime(VAL_CUTOFF))
        & (df["hour"] < pd.to_datetime(TEST_CUTOFF))
    ].copy()

    # Step 5: Convert "hour" datetime → integer hour-of-day in both splits.
    # Skeleton carry-over: kept verbatim for consistency with upstream
    # scripts. Note that FEATURE_COLS selects the separate `hour_of_day`
    # column (the integer time feature), not `hour`; this conversion is a
    # no-op for model input but leaves `hour` as a plain int instead of a
    # datetime, which avoids surprises if the column is referenced later.
    train["hour"] = train["hour"].dt.hour
    val["hour"] = val["hour"].dt.hour

    # Step 6: Return features and target for each split.
    X_train, y_train = train[FEATURE_COLS], train[TARGET]
    X_val, y_val = val[FEATURE_COLS], val[TARGET]
    return X_train, y_train, X_val, y_val


def objective(trial: optuna.Trial) -> float:
    """Optuna objective: suggest hyperparams, run TimeSeriesSplit CV on `train`,
    log per-fold metrics to MLflow, and return the mean CV MAE (minimize).
    """
    # --- Part 1: Search space ---
    # Build a `params` dict by sampling hyperparameters from the trial:
    #   - "n_estimators":      int in [lower_limit, higher_limit], step 50
    #   - "max_depth":         int in [lower_limit, higher_limit]
    #   - "min_samples_leaf":  int in [lower_limit, higher_limit]
    #   - "min_samples_split": int in [lower_limit, higher_limit]
    #   - "max_features":      categorical choice among ["sqrt", "log2", 0.5]
    # Also fix "random_state" to 42 and "n_jobs" to -1 (not tuned).
    # Use trial.suggest_int() and trial.suggest_categorical().

    # --- Part 2: Load data and prepare for cross-validation ---
    # Call load_splits() to get X_train, y_train, X_val, y_val.
    # TimeSeriesSplit requires rows to be in chronological order.
    # Sort X_train and y_train by their DatetimeIndex (use .argsort() on the index
    # and .iloc[] to reorder both arrays consistently).

    # Step 3: Create a TimeSeriesSplit object with n_splits=5.

    # Step 4: Configure MLflow — call mlflow.set_tracking_uri() and mlflow.set_experiment().

    # Step 5: Build a unique run name using the trial number and a UTC timestamp string,
    #         e.g. "optuna_trial_<number>_<YYYYMMDDTHHMMSSz>".

    # Step 6: Start an MLflow run using mlflow.start_run(run_name=...).
    #         Inside the run context:

    #   Step 6a: Log a "logged_at_utc" param with the current UTC ISO timestamp.
    #            Log all params from the `params` dict with mlflow.log_params().
    #            Log an "objective" param with value "tscv_train".

    #   Step 6b: Iterate over the folds produced by tscv.split(X_train).
    #            For each fold:
    #              - Slice X_train and y_train with the provided train/test indices.
    #              - Instantiate and fit a RandomForestRegressor using **params.
    #              - Predict on the fold's test slice.
    #              - Compute MAE with mean_absolute_error() and append to a list.
    #              - Log the fold MAE to MLflow as "fold_<n>_mae" at step=fold number.

    #   Step 6c: Compute the mean of all fold MAEs and log it as "mean_cv_mae".

    #   Step 6d: Train a fresh RandomForestRegressor(**params) on the full X_train/y_train.
    #            Predict on X_val, compute val MAE, and log it as "val_mae".

    #   Step 6e: Log the final model artifact with mlflow.sklearn.log_model(model, "model").

    # Primary objective: mean CV MAE on train (minimize)
    # Step 7: Return the mean CV MAE so Optuna can minimize it.

    # --- Part 1: Search space ---
    params = {
        "n_estimators": trial.suggest_int(
            "n_estimators", 100, 500, step=50
        ),  # 100 is RF default; above 500 gives diminishing returns on MAE
            # for this dataset size and adds significant compute time

        "max_depth": trial.suggest_int(
            "max_depth", 5, 30
        ),  # shallow trees underfit zone-level demand patterns;
            # deep trees overfit the 46% zero-demand rows in training data

        "min_samples_leaf": trial.suggest_int(
            "min_samples_leaf", 1, 20
        ),  # controls smoothing of leaf predictions; higher values prevent
            # the model from memorizing individual zone-hour noise

        "min_samples_split": trial.suggest_int(
            "min_samples_split", 2, 20
        ),  # minimum samples required to split a node; prevents splits
            # on statistical noise in sparse outer-borough zones

        "max_features": trial.suggest_categorical(
            "max_features", ["sqrt", "log2", 0.5]
        ),  # sqrt is RF default and performs well on structured tabular data;
            # log2 is more aggressive feature subsampling;
            # 0.5 tries half the features per split for more diversity

        "random_state": 42,    # fixed for reproducibility — not tuned
        "n_jobs": -1,          # use all cores — not tuned
    }

    # --- Part 2: Load data and prepare for CV ---
    X_train, y_train, X_val, y_val = load_splits()

    # TimeSeriesSplit requires rows to be in chronological order.
    # Sort by index defensively — features.parquet is already time-ordered,
    # so for a default RangeIndex this is a no-op, but the sort makes the
    # contract explicit and safe against upstream reordering.
    order = X_train.index.argsort()
    X_train = X_train.iloc[order]
    y_train = y_train.iloc[order]

    # Step 3: TimeSeriesSplit.
    tscv = TimeSeriesSplit(n_splits=5)

    # Step 4: MLflow configuration.
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Step 5: Unique run name. Import is `import datetime`, so the
    # professor's literal `datetime.utcnow()` spec maps to
    # `datetime.datetime.utcnow()` at the call site.
    run_name = (
        f"optuna_trial_{trial.number}_"
        f"{datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}"
    )

    # Step 6: MLflow run.
    with mlflow.start_run(run_name=run_name) as run:
        # Step 6a
        mlflow.log_param("logged_at_utc", datetime.datetime.utcnow().isoformat())
        mlflow.log_params(params)
        mlflow.log_param("objective", "tscv_train")

        # Step 6b — fold loop
        fold_maes = []
        for fold, (tr_idx, te_idx) in enumerate(tscv.split(X_train)):
            X_tr, X_te = X_train.iloc[tr_idx], X_train.iloc[te_idx]
            y_tr, y_te = y_train.iloc[tr_idx], y_train.iloc[te_idx]

            fold_model = RandomForestRegressor(**params)
            fold_model.fit(X_tr, y_tr)
            fold_preds = fold_model.predict(X_te)

            fold_mae = float(mean_absolute_error(y_te, fold_preds))
            fold_maes.append(fold_mae)
            mlflow.log_metric(f"fold_{fold}_mae", fold_mae, step=fold)

        # Step 6c — mean CV MAE
        mean_cv_mae = float(np.mean(fold_maes))
        mlflow.log_metric("mean_cv_mae", mean_cv_mae)

        # Step 6d — fit on full train, evaluate on validation.
        # Logs the five-metric suite established in Part 1 so every trial
        # is directly comparable to the baseline (val_mae, val_rmse,
        # val_r2, val_mape filtered to non-zero rows, val_mbe).
        model = RandomForestRegressor(**params)
        model.fit(X_train, y_train)
        val_preds = model.predict(X_val)

        val_mae = float(mean_absolute_error(y_val, val_preds))
        val_rmse = float(root_mean_squared_error(y_val, val_preds))
        val_r2 = float(r2_score(y_val, val_preds))

        # MAPE — filter to y_val > 0 to avoid division-by-zero blow-up
        # (same handling as evaluate.py for the baseline).
        nonzero = y_val > 0
        y_val_nz = y_val[nonzero]
        val_preds_nz = val_preds[nonzero.to_numpy()]
        val_mape = float(np.mean(np.abs((y_val_nz - val_preds_nz) / y_val_nz)) * 100)
        val_mbe = float(np.mean(val_preds - y_val))

        mlflow.log_metric("val_mae", val_mae)
        mlflow.log_metric("val_rmse", val_rmse)
        mlflow.log_metric("val_r2", val_r2)
        mlflow.log_metric("val_mape", val_mape)
        mlflow.log_metric("val_mbe", val_mbe)

        # Step 6e — log the final model artifact.
        mlflow.sklearn.log_model(model, "model")

    # Step 7: return mean CV MAE (Optuna will minimize it).
    return mean_cv_mae


def retrain_and_register(best_params: dict, stage: str = "Production") -> None:
    """Retrain the chosen hyperparameters on train+val, evaluate on test,
    log test metrics, and register the final model to the Model Registry.
    """
    # Step 1: Configure MLflow — call mlflow.set_tracking_uri() and mlflow.set_experiment().

    # Step 2: Load the full DataFrame from DATA_PATH and parse the "hour" column as datetime.

    # Step 3: Split into two DataFrames:
    #         - `trainval`: rows where "hour" < TEST_CUTOFF  (train + validation combined)
    #         - `test`:     rows where "hour" >= TEST_CUTOFF
    #         Call .copy() on each slice. Raise a ValueError if `trainval` is empty.

    # Step 4: Convert "hour" to integer hour-of-day in `trainval`.
    #         Build X_trainval (FEATURE_COLS) and y_trainval (TARGET).
    #         If `test` is not empty, do the same to get X_test and y_test.
    #         If `test` is empty, print a warning and set X_test and y_test to None.

    # Step 5: Instantiate a RandomForestRegressor(**best_params) and fit it on
    #         X_trainval / y_trainval.

    # Step 6: Build a unique run name using a UTC timestamp,
    #         e.g. "final_retrain_and_register_<YYYYMMDDTHHMMSSz>".
    #         Start an MLflow run with mlflow.start_run(run_name=...).
    #         Inside the run context:

    #   Step 6a: Log a "logged_at_utc" param with the current UTC ISO timestamp.
    #            Log all best_params with mlflow.log_params().

    #   Step 6b: If X_test is not None, predict on X_test, compute test MAE with
    #            mean_absolute_error(), log it as "test_mae", and print it.

    #   Step 6c: Log the final model artifact with mlflow.sklearn.log_model(model, "model").
    #            Register the model with mlflow.register_model() using uri
    #            "runs:/<run_id>/model" and name MODEL_REGISTRY_NAME.

    #   Step 6d: Use mlflow.tracking.MlflowClient() to transition the registered model
    #            version to the given `stage` via client.transition_model_version_stage().

    #   Step 6e: Print the registered model name, version, stage, and run ID.

    # Step 1: MLflow configuration.
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Step 2: Load and parse "hour".
    df = pd.read_parquet(DATA_PATH)
    df["hour"] = pd.to_datetime(df["hour"])

    # Step 3: Split into trainval (train+val combined) and test.
    trainval = df[df["hour"] < pd.to_datetime(TEST_CUTOFF)].copy()
    test = df[df["hour"] >= pd.to_datetime(TEST_CUTOFF)].copy()
    if trainval.empty:
        raise ValueError(
            "trainval split is empty — check DATA_PATH and TEST_CUTOFF."
        )

    # Step 4: Feature/target prep.
    trainval["hour"] = trainval["hour"].dt.hour
    X_trainval = trainval[FEATURE_COLS]
    y_trainval = trainval[TARGET]

    if not test.empty:
        test["hour"] = test["hour"].dt.hour
        X_test = test[FEATURE_COLS]
        y_test = test[TARGET]
    else:
        print("  ! test split is empty — skipping test evaluation.")
        X_test, y_test = None, None

    # Step 5: Fit on trainval with the tuned hyperparameters.
    model = RandomForestRegressor(**best_params)
    model.fit(X_trainval, y_trainval)

    # Step 6: Run, log, register, promote.
    run_name = (
        f"final_retrain_and_register_"
        f"{datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}"
    )
    with mlflow.start_run(run_name=run_name) as run:
        # Step 6a
        mlflow.log_param("logged_at_utc", datetime.datetime.utcnow().isoformat())
        mlflow.log_params(best_params)

        # Step 6b — test evaluation, if we have a non-empty test slice.
        if X_test is not None:
            test_preds = model.predict(X_test)
            test_mae = float(mean_absolute_error(y_test, test_preds))
            mlflow.log_metric("test_mae", test_mae)
            print(f"  test_mae: {test_mae:.4f}")

        # Step 6c — log artifact and register.
        mlflow.sklearn.log_model(model, "model")
        model_uri = f"runs:/{run.info.run_id}/model"
        result = mlflow.register_model(model_uri=model_uri, name=MODEL_REGISTRY_NAME)

        # Step 6d — transition to the requested stage. Leave other versions
        # untouched so the Week 3 baseline stays in Staging.
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=MODEL_REGISTRY_NAME,
            version=result.version,
            stage=stage,
            archive_existing_versions=False,
        )

        # Step 6e — print registration summary.
        print(f"  Registered: {MODEL_REGISTRY_NAME} v{result.version} → {stage}")
        print(f"  Run ID:     {run.info.run_id}")


if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=N_TRIALS)

    print(f"\nBest trial: {study.best_trial.number}")
    print(f"Best CV MAE: {study.best_value:.4f}")
    print(f"Best params: {study.best_trial.params}")

    retrain_and_register(study.best_trial.params, stage="Production")