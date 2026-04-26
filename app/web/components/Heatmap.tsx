"use client";

import { useEffect, useMemo, useState } from "react";
import { api } from "@/lib/api";
import { DOW_LABELS, FORECAST_WEEK_LABEL, formatHour } from "@/lib/format";
import type { HeatmapCell } from "@/lib/types";

export function Heatmap({ zone }: { zone: number }) {
  const [cells, setCells] = useState<HeatmapCell[]>([]);
  useEffect(() => {
    api.heatmap(zone).then(setCells).catch(console.error);
  }, [zone]);

  const max = useMemo(() => (cells.length ? Math.max(...cells.map((c) => c.prediction)) : 1), [cells]);
  const grid = useMemo(() => {
    const g: (number | null)[][] = Array.from({ length: 24 }, () => Array(7).fill(null));
    for (const c of cells) g[c.hour][c.dow] = c.prediction;
    return g;
  }, [cells]);

  function color(v: number | null): string {
    if (v == null) return "#f8fafc";
    const t = Math.min(1, v / max);
    const r = Math.round(255 - (255 - 0x25) * t);
    const g = Math.round(255 - (255 - 0x63) * t);
    const b = Math.round(255 - (255 - 0xeb) * t);
    return `rgb(${r}, ${g}, ${b})`;
  }

  return (
    <div className="h-full flex flex-col">
      <div className="text-[10px] uppercase tracking-wider text-slate-500 mb-2">
        Operational fingerprint · typical week ({FORECAST_WEEK_LABEL})
      </div>
      <div className="flex-1 min-h-0 overflow-auto">
        <div className="grid" style={{ gridTemplateColumns: "auto repeat(7, 1fr)", gap: "1px" }}>
          <div></div>
          {DOW_LABELS.map((d) => (
            <div key={d} className="text-[10px] text-slate-500 text-center py-1 uppercase tracking-wider">{d}</div>
          ))}
          {grid.map((row, hour) => (
            <>
              <div key={`l${hour}`} className="text-[10px] text-slate-500 pr-2 text-right tabular-nums self-center">{formatHour(hour)}</div>
              {row.map((v, dow) => (
                <div
                  key={`c${hour}-${dow}`}
                  className="aspect-[3/1] flex items-center justify-center text-[10px] tabular-nums border border-white"
                  style={{ background: color(v), color: v != null && v / max > 0.55 ? "white" : "#334155" }}
                  title={v == null ? "no data" : `${formatHour(hour)} ${DOW_LABELS[dow]} — ${v.toFixed(0)} pickups`}
                >
                  {v == null ? "" : v.toFixed(0)}
                </div>
              ))}
            </>
          ))}
        </div>
      </div>
    </div>
  );
}
