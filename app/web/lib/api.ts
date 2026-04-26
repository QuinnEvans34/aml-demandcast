import type {
  HeatmapCell,
  HourPrediction,
  ModelCard,
  PredictResponse,
  ZoneMeta,
  ZonePrediction,
} from "./types";

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${API}${path}`);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText} on ${path}`);
  return res.json() as Promise<T>;
}

export const api = {
  zones: () => get<ZoneMeta[]>("/api/zones"),
  predict: (zone: number, hour: number, dow: number, weekend: number) =>
    get<PredictResponse>(
      `/api/predict?zone=${zone}&hour=${hour}&dow=${dow}&weekend=${weekend}`
    ),
  predictAll: (hour: number, dow: number, weekend: number) =>
    get<ZonePrediction[]>(
      `/api/predict/all?hour=${hour}&dow=${dow}&weekend=${weekend}`
    ),
  timeline: (zone: number, dow: number, weekend: number) =>
    get<HourPrediction[]>(
      `/api/predict/timeline?zone=${zone}&dow=${dow}&weekend=${weekend}`
    ),
  heatmap: (zone: number) =>
    get<HeatmapCell[]>(`/api/predict/heatmap?zone=${zone}`),
  model: () => get<ModelCard>("/api/model"),
};
