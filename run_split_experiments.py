"""
run_split_experiments.py — Diagnostic tests of split methodology.

Compares the established temporal-split results against two alternatives,
all using the Week 3 baseline (RandomForestRegressor(n_estimators=100,
random_state=42)):

  1. Random-split 5-fold KFold CV — answers "does shuffling produce
     a dramatically lower MAE because train and val end up as
     near-duplicate samples in feature space?"
  2. Temporal single-shot split with an alternate val window
     (train Jan 1–14, val Jan 15–21, which includes MLK Day) —
     answers "is the original val window (Jan 22–31) unusually
     easy or hard? How sensitive is val MAE to which week we picked?"

Logs each run to MLflow under experiment "DemandCast" with descriptive
run names. Does NOT register, retrain, or modify any other artifact.

Usage from project root with .venv active:
    python run_split_experiments.py
"""
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import KFold

from src.features import FEATURE_COLS, TARGET

MLFLOW_TRACKING_URI = "http://localhost:8080"
EXPERIMENT_NAME = "DemandCast"

DATA_PATH = Path(__file__).parent / "data" / "features.parquet"
VAL_CUTOFF = "2025-01-22"
TEST_CUTOFF = "2025-02-01"

# Alternate temporal window for the second experiment.
ALT_VAL_START = "2025-01-15"
ALT_VAL_END = "2025-01-22"   # exclusive — gives Jan 15–21 inclusive


def _filtered_mape(y_true: pd.Series, y_pred: np.ndarray) -> float:
    """MAPE on rows where y_true > 0 only — same handling as evaluate.py."""
    nonzero = y_true > 0
    y_nz = y_true[nonzero]
    p_nz = y_pred[nonzero.to_numpy()]
    return float(np.mean(np.abs((y_nz - p_nz) / y_nz)) * 100)


def _five_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae":  float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        "r2":   float(r2_score(y_true, y_pred)),
        "mape": _filtered_mape(y_true, y_pred),
        "mbe":  float(np.mean(y_pred - y_true)),
    }


def load_trainval() -> pd.DataFrame:
    df = pd.read_parquet(DATA_PATH)
    df["hour"] = pd.to_datetime(df["hour"])
    return df[df["hour"] < pd.to_datetime(TEST_CUTOFF)].sort_values("hour").reset_index(drop=True)


def experiment_random_kfold(df: pd.DataFrame) -> dict:
    """5-fold KFold(shuffle=True) CV. Random-split methodology."""
    X = df[FEATURE_COLS]
    y = df[TARGET]

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_maes = []

    with mlflow.start_run(run_name="cv_random_split_kfold") as run:
        mlflow.log_param("model", "RandomForestRegressor")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("split_strategy", "KFold(n_splits=5, shuffle=True)")
        mlflow.log_param("features", FEATURE_COLS)

        for fold, (tr_idx, te_idx) in enumerate(kf.split(X)):
            X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
            y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]

            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            model.fit(X_tr, y_tr)
            preds = model.predict(X_te)

            fold_mae = float(mean_absolute_error(y_te, preds))
            fold_maes.append(fold_mae)
            mlflow.log_metric("fold_mae", fold_mae, step=fold)
            print(f"  KFold fold {fold}: MAE={fold_mae:.4f}")

        mean_mae = float(np.mean(fold_maes))
        std_mae = float(np.std(fold_maes, ddof=1))
        mlflow.log_metric("cv_mae_mean", mean_mae)
        mlflow.log_metric("cv_mae_std", std_mae)
        run_id = run.info.run_id

    return {
        "fold_maes": fold_maes,
        "mean_mae": mean_mae,
        "std_mae": std_mae,
        "run_id": run_id,
    }


def experiment_alt_temporal_window(df: pd.DataFrame) -> dict:
    """Single-shot temporal split with val Jan 15–21 (includes MLK Day Jan 20)."""
    train = df[df["hour"] < pd.to_datetime(ALT_VAL_START)]
    val = df[
        (df["hour"] >= pd.to_datetime(ALT_VAL_START))
        & (df["hour"] < pd.to_datetime(ALT_VAL_END))
    ]
    if train.empty or val.empty:
        raise RuntimeError(
            f"Empty split for alt window train<{ALT_VAL_START} val<{ALT_VAL_END}: "
            f"train={len(train)}, val={len(val)}"
        )

    X_train, y_train = train[FEATURE_COLS], train[TARGET]
    X_val, y_val = val[FEATURE_COLS], val[TARGET]

    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    val_preds = model.predict(X_val)
    metrics = _five_metrics(y_val, val_preds)

    with mlflow.start_run(
        run_name=f"single_temporal_alt_window_{ALT_VAL_START}_to_{ALT_VAL_END}"
    ) as run:
        mlflow.log_param("model", "RandomForestRegressor")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("split_strategy", "temporal_single_shot")
        mlflow.log_param("alt_val_start", ALT_VAL_START)
        mlflow.log_param("alt_val_end_exclusive", ALT_VAL_END)
        mlflow.log_param("features", FEATURE_COLS)

        mlflow.log_metric("val_mae",  metrics["mae"])
        mlflow.log_metric("val_rmse", metrics["rmse"])
        mlflow.log_metric("val_r2",   metrics["r2"])
        mlflow.log_metric("val_mape", metrics["mape"])
        mlflow.log_metric("val_mbe",  metrics["mbe"])
        run_id = run.info.run_id

    return {
        "metrics": metrics,
        "n_train_rows": len(train),
        "n_val_rows": len(val),
        "run_id": run_id,
    }


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"features.parquet not found at {DATA_PATH}. Run build_features.py first."
        )

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = load_trainval()
    print(f"Loaded {len(df):,} rows for trainval span ({df['hour'].min()} → {df['hour'].max()})")

    print("\n[1/2] Random-split KFold CV...")
    rand = experiment_random_kfold(df)
    print(
        f"  → CV MAE mean={rand['mean_mae']:.4f}  std={rand['std_mae']:.4f}"
    )

    print("\n[2/2] Alternate temporal window (Jan 15–21) single-shot val...")
    alt = experiment_alt_temporal_window(df)
    m = alt["metrics"]
    print(
        f"  → val_mae={m['mae']:.4f}  val_rmse={m['rmse']:.4f}  val_r2={m['r2']:.4f}  "
        f"val_mape={m['mape']:.2f}%  val_mbe={m['mbe']:.4f}"
    )
    print(f"  → train rows: {alt['n_train_rows']:,}, val rows: {alt['n_val_rows']:,}")

    print("\nDone. Two new MLflow runs logged in experiment 'DemandCast'.")
    print(f"  cv_random_split_kfold → run_id {rand['run_id'][:8]}")
    print(f"  single_temporal_alt_window_… → run_id {alt['run_id'][:8]}")


if __name__ == "__main__":
    main()
