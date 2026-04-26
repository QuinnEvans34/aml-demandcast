"use client";

import { useEffect, useState } from "react";
import { Activity, Calendar, Info } from "lucide-react";
import { api } from "@/lib/api";
import { FORECAST_WEEK_LABEL, TRAINING_PERIOD_LABEL } from "@/lib/format";
import type { ModelCard as ModelCardData } from "@/lib/types";

export function ModelCard() {
  const [card, setCard] = useState<ModelCardData | null>(null);
  useEffect(() => { api.model().then(setCard).catch(console.error); }, []);

  if (!card) return null;

  return (
    <div className="border border-slate-200 rounded-lg p-6 bg-white">
      <div className="flex items-center gap-2 mb-4">
        <Activity className="h-4 w-4 text-accent" />
        <span className="text-xs uppercase tracking-wider text-slate-500">
          Model · {card.version} · {card.stage}
        </span>
      </div>
      <div className="grid grid-cols-5 gap-4">
        {card.metrics.map((m) => (
          <div key={m.name} className="border-l border-slate-200 pl-4 first:border-l-0 first:pl-0">
            <div className="text-[10px] uppercase tracking-wider text-slate-500 flex items-center gap-1">
              {m.name}
              <span title={m.interpretation}>
                <Info className="h-3 w-3 text-slate-400" />
              </span>
            </div>
            <div className="text-2xl font-medium tabular-nums mt-1">{m.value}</div>
          </div>
        ))}
      </div>
      <div className="mt-6 pt-4 border-t border-slate-100 text-[11px] text-slate-500 flex items-center gap-1.5 flex-wrap">
        <Calendar className="h-3 w-3" />
        <span>
          Trained on <span className="text-slate-700">{TRAINING_PERIOD_LABEL}</span> NYC TLC Yellow Cab data.
          Forecast period: <span className="text-slate-700">{FORECAST_WEEK_LABEL}</span>.
        </span>
      </div>
    </div>
  );
}
