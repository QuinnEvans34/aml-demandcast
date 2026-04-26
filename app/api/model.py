"""
Loads the DemandCast Production model from the MLflow Model Registry once at
startup and exposes a singleton accessor.
"""
import mlflow
import mlflow.sklearn

MLFLOW_TRACKING_URI = "http://localhost:8080"
MODEL_URI = "models:/DemandCast/Production"

_model = None


def get_model():
    global _model
    if _model is None:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        _model = mlflow.sklearn.load_model(MODEL_URI)
    return _model
