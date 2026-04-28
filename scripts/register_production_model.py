"""
scripts/register_production_model.py — Register the tuned random-split model.

This is the production version of the script — it does NOT call
`mlflow.sklearn.log_model()` because that path proved to hang reliably
on the ~2 GB pickle upload through the MLflow tracking server's HTTP
artifact proxy on localhost. Instead it does the same thing in a way
that does not depend on a large HTTP body transfer:

    1. Save the model locally with `mlflow.sklearn.save_model(...)`.
       This writes the exact same MLmodel manifest, model.pkl,
       requirements.txt, python_env.yaml, conda.yaml that
       `log_model` would have produced — just to a local temp
       directory on disk. No network involved. ~5 seconds for a 2 GB
       random forest.

    2. Move that local directory into ./mlartifacts/1/models/m-<uuid>/artifacts
       so it lives under the MLflow server's --default-artifact-root,
       which is what the server already serves to clients. The model
       is now reachable via a `mlflow-artifacts:/...` URI without
       anything ever being uploaded over HTTP.

    3. Create an MLflow run via the server (metadata only; tiny HTTP
       request). Log all params + metrics through the API.

    4. Register the model version with `source` pointing at the
       artifact location we placed in step 2.

    5. Transition the new version to Production with
       archive_existing_versions=True so v3 (currently broken
       Production) gets cleanly demoted to Archived. v1 stays in
       Staging untouched. v3 + its run + its model.pkl all stay on
       disk for audit history.

The dashboard loads `models:/DemandCast/Production` via
`mlflow.sklearn.load_model` exactly as before — it sees a Production
version with a valid `source` URI and the artifact files at the
expected location. No code change required on the dashboard side.

Memory management is preserved: each eval model is `del`'d and
gc'd before the next fit, so only one ~2 GB random forest is alive
at a time.

PREREQUISITE: the MLflow tracking server must be running:
    mlflow server --backend-store-uri sqlite:///mlflow.db \\
        --default-artifact-root ./mlartifacts \\
        --host 127.0.0.1 --port 8080

Run from project root with .venv active:
    python -m scripts.register_production_model
"""
import datetime
import gc
import os
import shutil
import sys
import tempfile
import uuid
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import train_test_split

from src.features import FEATURE_COLS, TARGET

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "features.parquet"
ARTIFACT_ROOT_DIR = PROJECT_ROOT / "mlartifacts"

MLFLOW_TRACKING_URI = "http://localhost:8080"
EXPERIMENT_NAME = "DemandCast"
MODEL_REGISTRY_NAME = "DemandCast"
RANDOM_STATE = 42

# Hyperparameters Optuna selected in the most recent study (best trial #12,
# mean_cv_mae = 3.77). Deterministic given random_state=42 and the 60/20/20
# random split.
BEST_PARAMS = {
    "n_estimators": 350,
    "max_depth": 30,
    "min_samples_leaf": 1,
    "min_samples_split": 2,
    "max_features": "sqrt",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_utc() -> datetime.datetime:
    """Timezone-aware UTC now (datetime.utcnow() is deprecated in 3.12)."""
    return datetime.datetime.now(datetime.timezone.utc)


def _filtered_mape(y_true, y_pred):
    nonzero = y_true > 0
    if nonzero.sum() == 0:
        return float("nan")
    y_nz = y_true[nonzero]
    p_nz = y_pred[nonzero.to_numpy()]
    return float(np.mean(np.abs((y_nz - p_nz) / y_nz)) * 100)


def _full_metric_suite(y_true, y_pred):
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "mape": _filtered_mape(y_true, y_pred),
        "mbe": float(np.mean(y_pred - y_true)),
    }


def _load_random_splits():
    df = pd.read_parquet(DATA_PATH)
    X = df[FEATURE_COLS]
    y = df[TARGET]
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, shuffle=True
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=RANDOM_STATE, shuffle=True
    )
    return X_train, y_train, X_val, y_val, X_test, y_test


def _close_stuck_runs(client: MlflowClient, experiment_id: str) -> int:
    """Status-correct any RUNNING final_retrain_random_split_* runs."""
    candidates = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string="tags.mlflow.runName LIKE 'final_retrain_random_split%'",
        max_results=200,
    )
    stuck = [r for r in candidates if r.info.status == "RUNNING"]
    for r in stuck:
        client.set_tag(
            r.info.run_id, "stuck_run_recovery",
            "marked FAILED by register_production_model.py",
        )
        client.set_terminated(r.info.run_id, status="FAILED")
        print(f"    closed {r.info.run_id} ({r.info.run_name}) → FAILED")
    return len(stuck)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    print(f"Tracking URI: {MLFLOW_TRACKING_URI}")
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

    # Sanity: server reachable?
    try:
        exp = client.get_experiment_by_name(EXPERIMENT_NAME)
    except Exception as e:
        print(f"ERROR: cannot reach MLflow server. {type(e).__name__}: {e}")
        return 2
    if exp is None:
        print(f"ERROR: experiment '{EXPERIMENT_NAME}' not found.")
        return 2
    exp_id = exp.experiment_id
    mlflow.set_experiment(EXPERIMENT_NAME)

    # ---- 1. Close stuck RUNNING runs from previous attempts ----
    print()
    print("Cleaning up stuck RUNNING runs...")
    closed = _close_stuck_runs(client, exp_id)
    if closed == 0:
        print("    (none found)")

    # ---- 2. Refit (val_eval, test_eval, final) with memory management ----
    print()
    print("Refitting models with Optuna's best params...")
    print(f"    best_params = {BEST_PARAMS}")
    X_train, y_train, X_val, y_val, X_test, y_test = _load_random_splits()
    print(f"    splits: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    print("    fitting val_eval_model (train → predict on val)...")
    val_eval_model = RandomForestRegressor(**BEST_PARAMS)
    val_eval_model.fit(X_train, y_train)
    val_metrics = _full_metric_suite(y_val, val_eval_model.predict(X_val))
    del val_eval_model
    gc.collect()
    print(f"      val:  mae={val_metrics['mae']:.4f} rmse={val_metrics['rmse']:.4f} r2={val_metrics['r2']:.4f}")

    print("    fitting test_eval_model (train+val → predict on test)...")
    X_trainval = pd.concat([X_train, X_val])
    y_trainval = pd.concat([y_train, y_val])
    test_eval_model = RandomForestRegressor(**BEST_PARAMS)
    test_eval_model.fit(X_trainval, y_trainval)
    test_metrics = _full_metric_suite(y_test, test_eval_model.predict(X_test))
    del test_eval_model
    gc.collect()
    print(f"      test: mae={test_metrics['mae']:.4f} rmse={test_metrics['rmse']:.4f} r2={test_metrics['r2']:.4f}")

    print("    fitting final_model (train+val+test, deployed artifact)...")
    X_all = pd.concat([X_train, X_val, X_test])
    y_all = pd.concat([y_train, y_val, y_test])
    final_model = RandomForestRegressor(**BEST_PARAMS)
    final_model.fit(X_all, y_all)
    print("      final_model fit complete")

    # ---- 3. Create the MLflow run FIRST so we know its run_id ----
    # We need the run_id to compute the run's artifact directory on disk;
    # MLflow's create_model_version security check requires the source
    # local path to live inside that directory.
    print()
    print("Creating MLflow run (so we have a run_id to pin the artifact under)...")
    run_name = f"final_retrain_random_split_{_now_utc().strftime('%Y%m%dT%H%M%SZ')}"
    run = mlflow.start_run(run_name=run_name)
    try:
        run_id = run.info.run_id
        # The run's artifact_uri is mlflow-artifacts:/<exp_id>/<run_id>/artifacts
        # which the MLflow server resolves on disk to:
        #     <artifact_root>/<exp_id>/<run_id>/artifacts/
        # The model goes one level deeper, under a "model" subdirectory, so
        # the registry source URI becomes runs:/<run_id>/model — the standard
        # MLflow convention.
        run_artifact_dir = ARTIFACT_ROOT_DIR / "1" / run_id / "artifacts"
        model_dir_on_disk = run_artifact_dir / "model"
        run_artifact_dir.mkdir(parents=True, exist_ok=True)
        print(f"    run_id:                 {run_id}")
        print(f"    run artifact dir:       {run_artifact_dir}")
        print(f"    model artifact target:  {model_dir_on_disk}")

        # ---- 4. Save model LOCALLY (no HTTP body, just file I/O) ----
        print()
        print("Saving model artifact locally (no HTTP body upload)...")
        with tempfile.TemporaryDirectory() as td:
            local_save_path = Path(td) / "model"
            # save_model writes the same MLmodel manifest + model.pkl
            # (cloudpickle) + requirements.txt + python_env.yaml + conda.yaml
            # that log_model would have written — just to a local directory.
            mlflow.sklearn.save_model(final_model, str(local_save_path))
            print(f"    saved to temp: {local_save_path}")

            # Free the model from memory before the copy.
            del final_model
            gc.collect()

            # Move into the run's artifact dir. shutil.move is fast — within
            # the same filesystem it's a metadata rename, not a byte copy.
            shutil.move(str(local_save_path), str(model_dir_on_disk))
            print(f"    moved to:      {model_dir_on_disk}")

        # ---- 5. Log params + metrics (HTTP metadata only, tiny payload) ----
        print()
        print("Logging params + metrics to the run...")
        mlflow.log_param("logged_at_utc", _now_utc().isoformat())
        mlflow.log_params(BEST_PARAMS)
        mlflow.log_param("split_strategy", "random_60_20_20_train_val_test")
        mlflow.log_param("registered_model_fit_on", "train+val+test_full_data")
        mlflow.log_param("val_metrics_fit_on", "train_only")
        mlflow.log_param("test_metrics_fit_on", "train+val")
        mlflow.log_param("artifact_storage", "direct_filesystem_save_no_http_upload")

        for k, v in val_metrics.items():
            mlflow.log_metric(f"val_{k}", v)
        for k, v in test_metrics.items():
            mlflow.log_metric(f"test_{k}", v)

        # ---- 6. Register the model version using the runs:/ URI ----
        # runs:/<run_id>/model resolves to the run's artifact dir + /model,
        # which is exactly where we just placed the files. MLflow's security
        # check accepts this because the source is contained within the run's
        # artifact directory and run_id is supplied.
        source_uri = f"runs:/{run_id}/model"
        result = client.create_model_version(
            name=MODEL_REGISTRY_NAME,
            source=source_uri,
            run_id=run_id,
        )
        print(f"    registered:   {MODEL_REGISTRY_NAME} v{result.version}  source={source_uri}")

        client.transition_model_version_stage(
            name=MODEL_REGISTRY_NAME,
            version=result.version,
            stage="Production",
            archive_existing_versions=True,
        )
        print(f"    promoted:     v{result.version} → Production (prior Production demoted to Archived)")

        mlflow.end_run(status="FINISHED")
    except Exception as e:
        print(f"    ! exception: {type(e).__name__}: {e}")
        mlflow.end_run(status="FAILED")
        raise

    # ---- 6. Summary ----
    print()
    print("=" * 72)
    print(f"  Registered:   {MODEL_REGISTRY_NAME} v{result.version} → Production")
    print(f"  Source run:   {run.info.run_id}")
    print(f"  Artifact:     {model_dir_on_disk}")
    print()
    print("  Honest val metrics (train-fit, eval on val):")
    for k in ("mae", "rmse", "r2", "mape", "mbe"):
        print(f"    val_{k:<5} = {val_metrics[k]:.4f}")
    print()
    print("  Honest test metrics (train+val-fit, eval on test):")
    for k in ("mae", "rmse", "r2", "mape", "mbe"):
        print(f"    test_{k:<5} = {test_metrics[k]:.4f}")
    print()
    print("  Registry state:")
    for v in sorted(client.search_model_versions(f"name='{MODEL_REGISTRY_NAME}'"),
                    key=lambda v: int(v.version)):
        print(f"    v{v.version}  stage={v.current_stage:<10}  run_id={v.run_id}")
    print("=" * 72)
    print()
    print("Restart uvicorn so the dashboard loads the new Production model:")
    print("    Ctrl+C the running uvicorn, then:")
    print("    uvicorn app.api.main:app --port 8000")
    return 0


if __name__ == "__main__":
    sys.exit(main())
