"""
DemandCast dashboard backend — FastAPI service that wraps the Production model
in a small REST API used by the Next.js frontend (Phase 09b).

Run from the project root:
    uvicorn app.api.main:app --reload --port 8000
"""
from typing import Iterable
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from app.api.lookups import active_zone_ids, get_lookup, get_zone_meta
from app.api.model import get_model
from app.api.schemas import (
    HeatmapCell,
    HourPrediction,
    ModelCard,
    ModelCardMetric,
    PredictResponse,
    ZoneMeta,
    ZonePrediction,
)

# ---------------------------------------------------------------------------
# Model-input order — must match src.features.FEATURE_COLS exactly.
# Imported from there to stay in sync.
# ---------------------------------------------------------------------------
from src.features import FEATURE_COLS

# Validation MAE from notebooks/03_evaluation.md (random-split tuned model
# — Optuna best trial #12, mean_cv_mae 3.77; honest val_mae from the train-
# only fit evaluated on the held-out val split is 3.62). Used as the "± error"
# band in the dashboard's hero prediction.
TUNED_VAL_MAE = 3.62

# Plain-language metric copy mirrors the Tuned Model — Plain-Language Metrics
# table in notebooks/03_evaluation.md. Values come from the random-split
# methodology (60/20/20 train/val/test, KFold CV inside trials).
#   - val_* metrics: model fit on train only, evaluated on the held-out val
#     split — these are the honest generalization numbers shown to users.
#   - The deployed Production artifact is a separate fit on train+val+test
#     (full data) so deployment uses every available row; the metrics here
#     describe generalization behavior on data the model never saw.
TUNED_METRICS: list[ModelCardMetric] = [
    ModelCardMetric(
        name="MAE",
        value="3.62",
        interpretation=(
            "After tuning, predictions are off by 3.62 trips per hour per zone "
            "on average — about a quarter-trip tighter than the 3.88 baseline. "
            "For a dispatcher, the typical staffing decision now lands within "
            "roughly four trips of actual demand."
        ),
    ),
    ModelCardMetric(
        name="RMSE",
        value="10.46",
        interpretation=(
            "The typical error on the biggest misses is 10.46 trips, down "
            "almost a full trip from the baseline's 11.39. Peak-hour busy-zone "
            "forecasting is meaningfully more reliable than the baseline, "
            "though it still drives the largest single-prediction risks."
        ),
    ),
    ModelCardMetric(
        name="R²",
        value="0.9662",
        interpretation=(
            "The tuned model explains 96.6% of the variation in demand across "
            "all zones and hours, slightly above the baseline's 96.5%. The "
            "model captures essentially all of the predictable signal in zone, "
            "time-of-day, day-of-week, and recent-history patterns."
        ),
    ),
    ModelCardMetric(
        name="MAPE",
        value="42.7%",
        interpretation=(
            "On validation rows with strictly positive demand, predictions are "
            "off by 42.7% on average. Zero-demand rows are excluded because a "
            "percentage error against zero is undefined and would blow up the "
            "mean. Read alongside MAE — relative error is naturally noisy on "
            "low-count zones where missing by a few trips is a large percentage."
        ),
    ),
    ModelCardMetric(
        name="MBE",
        value="-0.06",
        interpretation=(
            "The tuned model under-predicts by an average of 0.06 trips per "
            "zone per hour — effectively zero bias. A near-zero MBE means the "
            "model is neither systematically over- nor under-staffing; "
            "directional risk in scheduling decisions is symmetric."
        ),
    ),
]

app = FastAPI(title="DemandCast", version="0.1.0")

# CORS — allow the Next.js dev server (localhost:3000) to call us during dev.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_rush_hour(hour: int, dow: int) -> int:
    return int(hour in {7, 8, 17, 18} and dow < 5)


def _build_feature_row(
    zone: int, hour: int, dow: int, weekend: int, lookup: pd.DataFrame
) -> pd.DataFrame:
    """Construct one row of model input in FEATURE_COLS order."""
    try:
        lag_row = lookup.loc[(zone, hour, dow)]
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail=f"No historical data for zone={zone} hour={hour} dow={dow}",
        )

    row = {
        "hour_of_day": hour,
        "day_of_week": dow,
        "is_weekend": weekend,
        "is_rush_hour": _is_rush_hour(hour, dow),
        "is_holiday": 0,
        "PULocationID": zone,
        "demand_lag_1h": float(lag_row["mean_demand_lag_1h"]),
        "demand_lag_24h": float(lag_row["mean_demand_lag_24h"]),
        "demand_lag_168h": float(lag_row["mean_demand_lag_168h"]),
        "demand_rolling_3h": float(lag_row["mean_demand_rolling_3h"]),
        "demand_rolling_168h": float(lag_row["mean_demand_rolling_168h"]),
    }
    return pd.DataFrame([row])[FEATURE_COLS]


def _predict_one(zone: int, hour: int, dow: int, weekend: int) -> tuple[float, float]:
    """Returns (prediction, historical_mean_demand) for a single (zone, hour, dow)."""
    lookup = get_lookup()
    X = _build_feature_row(zone, hour, dow, weekend, lookup)
    yhat = float(get_model().predict(X)[0])
    historical_mean = float(lookup.loc[(zone, hour, dow), "mean_demand"])
    return yhat, historical_mean


def _predict_batch(
    triples: Iterable[tuple[int, int, int, int]],
) -> list[tuple[int, int, int, int, float, float]]:
    """Vectorized batch predict. Each triple is (zone, hour, dow, weekend).
    Returns the same triples with (prediction, historical_mean) appended.
    """
    lookup = get_lookup()
    rows = []
    keys = []
    for zone, hour, dow, weekend in triples:
        try:
            lag_row = lookup.loc[(zone, hour, dow)]
        except KeyError:
            continue
        rows.append({
            "hour_of_day": hour,
            "day_of_week": dow,
            "is_weekend": weekend,
            "is_rush_hour": _is_rush_hour(hour, dow),
            "is_holiday": 0,
            "PULocationID": zone,
            "demand_lag_1h": float(lag_row["mean_demand_lag_1h"]),
            "demand_lag_24h": float(lag_row["mean_demand_lag_24h"]),
            "demand_lag_168h": float(lag_row["mean_demand_lag_168h"]),
            "demand_rolling_3h": float(lag_row["mean_demand_rolling_3h"]),
            "demand_rolling_168h": float(lag_row["mean_demand_rolling_168h"]),
        })
        keys.append((zone, hour, dow, weekend, float(lag_row["mean_demand"])))

    if not rows:
        return []

    X = pd.DataFrame(rows)[FEATURE_COLS]
    preds = get_model().predict(X)
    return [(z, h, d, w, float(p), m) for (z, h, d, w, m), p in zip(keys, preds)]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health() -> dict:
    """Cheap liveness check; also warms the model on first hit."""
    _ = get_model()
    return {"status": "ok"}


@app.get("/api/zones", response_model=list[ZoneMeta])
def list_zones() -> list[ZoneMeta]:
    """All zones the model has historical data for, with their TLC names."""
    meta = get_zone_meta()
    return [
        ZoneMeta(id=zid, name=meta.get(zid, {}).get("name", f"Zone {zid}"),
                 borough=meta.get(zid, {}).get("borough", "Unknown"))
        for zid in active_zone_ids()
    ]


@app.get("/api/predict", response_model=PredictResponse)
def predict(
    zone: int = Query(..., description="PULocationID"),
    hour: int = Query(..., ge=0, le=23),
    dow: int = Query(..., ge=0, le=6, description="0=Monday, 6=Sunday"),
    weekend: int = Query(..., ge=0, le=1),
) -> PredictResponse:
    yhat, hist = _predict_one(zone, hour, dow, weekend)
    vs_avg = ((yhat - hist) / hist * 100.0) if hist > 0 else 0.0
    name = get_zone_meta().get(zone, {}).get("name", f"Zone {zone}")
    return PredictResponse(
        zone_id=zone,
        zone_name=name,
        hour_of_day=hour,
        day_of_week=dow,
        is_weekend=weekend,
        prediction=yhat,
        error_band=TUNED_VAL_MAE,
        historical_mean=hist,
        vs_avg_pct=vs_avg,
    )


@app.get("/api/predict/all", response_model=list[ZonePrediction])
def predict_all(
    hour: int = Query(..., ge=0, le=23),
    dow: int = Query(..., ge=0, le=6),
    weekend: int = Query(..., ge=0, le=1),
) -> list[ZonePrediction]:
    """Map view — predictions for every active zone at a single time slot."""
    triples = [(z, hour, dow, weekend) for z in active_zone_ids()]
    out = _predict_batch(triples)
    return [ZonePrediction(zone_id=z, prediction=p, historical_mean=m)
            for (z, _, _, _, p, m) in out]


@app.get("/api/predict/timeline", response_model=list[HourPrediction])
def predict_timeline(
    zone: int = Query(..., description="PULocationID"),
    dow: int = Query(..., ge=0, le=6),
    weekend: int = Query(..., ge=0, le=1),
) -> list[HourPrediction]:
    """Timeline view — 24-hour forecast for one zone on one day-of-week."""
    triples = [(zone, h, dow, weekend) for h in range(24)]
    out = _predict_batch(triples)
    return [HourPrediction(hour=h, prediction=p) for (_, h, _, _, p, _) in out]


@app.get("/api/predict/heatmap", response_model=list[HeatmapCell])
def predict_heatmap(
    zone: int = Query(..., description="PULocationID"),
) -> list[HeatmapCell]:
    """Heatmap view — 24×7 grid of predictions for one zone."""
    triples = []
    for dow in range(7):
        weekend = 1 if dow >= 5 else 0
        for hour in range(24):
            triples.append((zone, hour, dow, weekend))
    out = _predict_batch(triples)
    return [HeatmapCell(hour=h, dow=d, prediction=p) for (_, h, d, _, p, _) in out]


@app.get("/api/model", response_model=ModelCard)
def model_card() -> ModelCard:
    # v4 = the random-split tuned RandomForestRegressor (best Optuna trial #12,
    # n_estimators=350, max_depth=30, max_features='sqrt', min_samples_leaf=1,
    # min_samples_split=2). Earlier versions (v1 baseline, v2 temporal-tuned,
    # v3 interrupted retrain) are intentionally preserved in the registry as
    # evidence of the training history.
    return ModelCard(
        version="v4",
        stage="Production",
        metrics=TUNED_METRICS,
    )
