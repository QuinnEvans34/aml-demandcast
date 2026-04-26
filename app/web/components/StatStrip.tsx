"use client";

import { Trophy, TrendingDown, TrendingUp } from "lucide-react";
import { formatPercent } from "@/lib/format";
import type { PredictResponse } from "@/lib/types";

interface Props {
  prediction: PredictResponse | null;
  rank: number | null;
  totalZones: number;
}

export function StatStrip({ prediction, rank, totalZones }: Props) {
  if (!prediction) return null;
  const up = prediction.vs_avg_pct >= 0;

  return (
    <div className="grid grid-cols-3 gap-4">
      <Stat
        icon={up ? <TrendingUp className="h-4 w-4 text-emerald-600" /> : <TrendingDown className="h-4 w-4 text-rose-600" />}
        label="vs typical"
        value={formatPercent(prediction.vs_avg_pct)}
        sub={`historical ${prediction.historical_mean.toFixed(1)}`}
        valueClass={up ? "text-emerald-600" : "text-rose-600"}
      />
      <Stat
        icon={<Trophy className="h-4 w-4 text-slate-500" />}
        label="rank"
        value={rank ? `#${rank}` : "—"}
        sub={totalZones ? `of ${totalZones} zones` : ""}
      />
      <Stat
        label="error band"
        value={`± ${prediction.error_band.toFixed(0)}`}
        sub="validation MAE"
      />
    </div>
  );
}

function Stat({
  icon, label, value, sub, valueClass = "text-slate-900",
}: { icon?: React.ReactNode; label: string; value: string; sub: string; valueClass?: string }) {
  return (
    <div className="border border-slate-200 rounded-lg p-4 bg-white">
      <div className="flex items-center gap-1.5 text-[10px] uppercase tracking-wider text-slate-500 mb-2">
        {icon}<span>{label}</span>
      </div>
      <div className={`text-2xl font-medium tabular-nums ${valueClass}`}>{value}</div>
      <div className="text-xs text-slate-500 mt-1 tabular-nums">{sub}</div>
    </div>
  );
}
