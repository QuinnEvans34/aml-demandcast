"""
evaluate_tuned.py — Compute the five held-out validation metrics for the
tuned Random Forest configuration (Optuna trial #1) so the Part 2
comparison table in notebooks/04_evaluation.md can show metric parity
with the baseline.

Loads the artifact from the Optuna trial #1 MLflow run, which was trained
on the train split only (hour < 2025-01-22) and evaluated on the val
split (2025-01-22 <= hour < 2025-02-01). The registered Production model
(DemandCast v2) is intentionally a different artifact — `retrain_and_register`
re-trains the same hyperparameters on train+val combined before deployment,
so its "val" metrics would be in-sample training metrics, not generalization
estimates.

This script is read-only:
    - it does NOT create a new MLflow run
    - it does NOT modify the Model Registry

Run from the project root with .venv active:
    python evaluate_tuned.py

Prerequisites:
    - MLflow server running at http://localhost:8080
    - At least one FINISHED Optuna trial run (run name starting with
      "optuna_trial_1_") present in the DemandCast experiment
"""
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)

from src.features import FEATURE_COLS, TARGET

MLFLOW_TRACKING_URI = "http://localhost:8080"
EXPERIMENT_NAME = "DemandCast"

DATA_PATH = Path(__file__).parent / "data" / "features.parquet"
VAL_CUTOFF = "2025-01-22"
TEST_CUTOFF = "2025-02-01"


def _find_trial_one_run() -> "mlflow.entities.Run":
    """Locate the FINISHED Optuna trial #1 run in the DemandCast experiment.

    Trial #1 was trained only on the train split (hour < 2025-01-22) and is
    therefore the right artifact for held-out validation metrics. The
    registered Production model trained on train+val combined, so its
    val-window metrics would be in-sample, not generalization estimates.

    Implementation note: MLflow's filter_string uses SQL LIKE syntax, where
    underscores are single-char wildcards. A pattern like 'optuna_trial_1_%'
    therefore also matches 'optuna_trial_10_…' through 'optuna_trial_14_…'.
    To avoid that, we filter only on FINISHED status in the SQL query and
    then match the trial number exactly in Python on the run name's
    underscore-split parts.
    """
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise RuntimeError(
            f"Experiment '{EXPERIMENT_NAME}' not found at {MLFLOW_TRACKING_URI}"
        )

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["attributes.start_time ASC"],
        max_results=10000,
    )

    trial_one_runs = []
    for r in runs:
        name = r.data.tags.get("mlflow.runName", "")
        parts = name.split("_")
        # Expected format: "optuna_trial_<n>_<UTC timestamp>"
        if (
            len(parts) >= 4
            and parts[0] == "optuna"
            and parts[1] == "trial"
            and parts[2] == "1"   # exact — excludes "10", "11", "12", "13", "14"
        ):
            trial_one_runs.append(r)

    if not trial_one_runs:
        raise RuntimeError(
            "No FINISHED run with name 'optuna_trial_1_*' found in the "
            f"'{EXPERIMENT_NAME}' experiment. Re-run `python src/tune.py` "
            "or inspect the MLflow UI."
        )

    # If multiple matches exist (multiple tune.py runs over time), pick the
    # most recent one. order_by ASC above means the last element is latest.
    return trial_one_runs[-1]


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"features.parquet not found at {DATA_PATH}. Run build_features.py first."
        )

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    trial_run = _find_trial_one_run()
    model_uri = f"runs:/{trial_run.info.run_id}/model"
    model = mlflow.sklearn.load_model(model_uri)
    print(f"Loaded trial #1 model from {model_uri}")
    print(f"  (run name: {trial_run.data.tags.get('mlflow.runName', '?')})")

    df = pd.read_parquet(DATA_PATH)
    df["hour"] = pd.to_datetime(df["hour"])
    val = df[(df["hour"] >= VAL_CUTOFF) & (df["hour"] < TEST_CUTOFF)].copy()

    X_val, y_val = val[FEATURE_COLS], val[TARGET]
    val_preds = model.predict(X_val)

    mae = float(mean_absolute_error(y_val, val_preds))
    rmse = float(root_mean_squared_error(y_val, val_preds))
    r2 = float(r2_score(y_val, val_preds))

    nonzero = y_val > 0
    y_val_nz = y_val[nonzero]
    val_preds_nz = val_preds[nonzero.to_numpy()]
    mape = float(np.mean(np.abs((y_val_nz - val_preds_nz) / y_val_nz)) * 100)
    mbe = float(np.mean(val_preds - y_val))

    n_nonzero = int(nonzero.sum())
    n_total = int(len(y_val))
    zero_pct = 100.0 * (1 - n_nonzero / n_total)

    print("=" * 56)
    print("  DemandCast v2 (Production) — Validation Metrics")
    print("=" * 56)
    print(f"  val_mae   : {mae:>12.4f}")
    print(f"  val_rmse  : {rmse:>12.4f}")
    print(f"  val_r2    : {r2:>12.4f}")
    print(f"  val_mape  : {mape:>12.4f}   (% on {n_nonzero:,} of {n_total:,} non-zero rows)")
    print(f"  val_mbe   : {mbe:>12.4f}")
    print("-" * 56)
    print(f"  zero-demand validation rows: {zero_pct:.1f}%")
    print("=" * 56)


if __name__ == "__main__":
    main()
