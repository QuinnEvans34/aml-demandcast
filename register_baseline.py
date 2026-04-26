"""
register_baseline.py — Seed the MLflow Model Registry with the Week 3
RandomForest baseline as DemandCast v1 → Staging.

This is a one-time setup step run once before `python src/tune.py`. tune.py
registers the tuned model as DemandCast v2 → Production; this script seeds
v1 in Staging so the final registry state matches the Part 2 rubric.

Run from the project root with `.venv` active:
    python register_baseline.py

Prerequisites:
    - MLflow server running at http://localhost:8080
    - At least one `rf_baseline_full_evaluation` run present in the
      DemandCast experiment (produced by `python evaluate.py`)
"""
import mlflow
from mlflow.tracking import MlflowClient

MLFLOW_TRACKING_URI = "http://localhost:8080"
EXPERIMENT_NAME = "DemandCast"
MODEL_REGISTRY_NAME = "DemandCast"
BASELINE_RUN_NAME = "rf_baseline_full_evaluation"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient()

experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
if experiment is None:
    raise RuntimeError(f"Experiment '{EXPERIMENT_NAME}' not found at {MLFLOW_TRACKING_URI}")

runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string=f"tags.mlflow.runName = '{BASELINE_RUN_NAME}'",
    order_by=["attributes.start_time DESC"],
    max_results=1,
)
if not runs:
    raise RuntimeError(
        f"No '{BASELINE_RUN_NAME}' run found in experiment '{EXPERIMENT_NAME}'. "
        "Run `python evaluate.py` first to produce it."
    )

baseline_run = runs[0]
model_uri = f"runs:/{baseline_run.info.run_id}/model"

result = mlflow.register_model(model_uri=model_uri, name=MODEL_REGISTRY_NAME)

client.transition_model_version_stage(
    name=MODEL_REGISTRY_NAME,
    version=result.version,
    stage="Staging",
    archive_existing_versions=False,
)

print(
    f"Registered {MODEL_REGISTRY_NAME} v{result.version} → Staging "
    f"(run_id={baseline_run.info.run_id[:8]})"
)
