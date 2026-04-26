export interface ZoneMeta {
  id: number;
  name: string;
  borough: string;
}

export interface PredictResponse {
  zone_id: number;
  zone_name: string;
  hour_of_day: number;
  day_of_week: number;
  is_weekend: number;
  prediction: number;
  error_band: number;
  historical_mean: number;
  vs_avg_pct: number;
}

export interface ZonePrediction {
  zone_id: number;
  prediction: number;
  historical_mean: number;
}

export interface HourPrediction {
  hour: number;
  prediction: number;
}

export interface HeatmapCell {
  hour: number;
  dow: number;
  prediction: number;
}

export interface ModelCardMetric {
  name: string;
  value: string;
  interpretation: string;
}

export interface ModelCard {
  version: string;
  stage: string;
  metrics: ModelCardMetric[];
}

export interface DashboardSelection {
  zone: number;
  hour: number;
  dow: number;
  weekend: number;
}
