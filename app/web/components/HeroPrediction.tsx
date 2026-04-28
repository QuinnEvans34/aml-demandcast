"use client";

import { Calendar, Info } from "lucide-react";
import { DOW_LABELS_FULL, formatForecastDate, formatHour } from "@/lib/format";
import type { PredictResponse } from "@/lib/types";

export function HeroPrediction({ prediction }: { prediction: PredictResponse | null }) {
  if (!prediction) {
    return (
      <div className="border border-slate-200 rounded-lg p-8 bg-white">
        <div className="text-sm text-slate-400">Loading prediction…</div>
      </div>
    );
  }
  return (
    <div className="border border-slate-200 rounded-lg p-8 bg-white">
      <div className="flex items-baseline gap-6">
        <div className="text-7xl font-semibold tabular-nums text-accent leading-none">
          {prediction.prediction.toFixed(0)}
        </div>
        <div>
          <div className="text-sm text-slate-500 uppercase tracking-wider">Predicted pickups</div>
          <div className="text-base text-slate-700 mt-1 tabular-nums flex items-center gap-2">
            ± {prediction.error_band.toFixed(0)}
            <span title="Typical model error (validation MAE 3.62 trips/hour). The actual demand will be within this band most of the time.">
              <Info className="h-3.5 w-3.5 text-slate-400" />
            </span>
          </div>
        </div>
      </div>
      <div className="text-sm text-slate-700 mt-4">
        <span className="font-medium">{prediction.zone_name}</span>
        {" · "}
        {DOW_LABELS_FULL[prediction.day_of_week]}
        {" · "}
        {formatHour(prediction.hour_of_day)}
      </div>
      <div className="text-xs text-slate-500 mt-1.5 flex items-center gap-1.5">
        <Calendar className="h-3 w-3" />
        Forecast for {formatForecastDate(prediction.day_of_week)}
      </div>
    </div>
  );
}
