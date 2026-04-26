"""Pydantic response schemas for the dashboard API."""
from pydantic import BaseModel


class ZoneMeta(BaseModel):
    id: int
    name: str
    borough: str


class PredictResponse(BaseModel):
    zone_id: int
    zone_name: str
    hour_of_day: int
    day_of_week: int
    is_weekend: int
    prediction: float
    error_band: float           # val_mae from Part 1, used as ± error
    historical_mean: float      # from the lookup
    vs_avg_pct: float           # (prediction - historical_mean) / historical_mean * 100


class ZonePrediction(BaseModel):
    zone_id: int
    prediction: float
    historical_mean: float


class HourPrediction(BaseModel):
    hour: int
    prediction: float


class HeatmapCell(BaseModel):
    hour: int
    dow: int
    prediction: float


class ModelCardMetric(BaseModel):
    name: str
    value: str
    interpretation: str


class ModelCard(BaseModel):
    version: str
    stage: str
    metrics: list[ModelCardMetric]
