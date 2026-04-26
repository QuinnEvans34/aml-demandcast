"use client";

import { Trophy } from "lucide-react";
import type { ZoneMeta, ZonePrediction } from "@/lib/types";

interface Props {
  predictions: ZonePrediction[];
  zones: ZoneMeta[];
  selectedZone: number;
  onZoneSelect: (zone: number) => void;
  topN?: number;
}

export function TopZonesPanel({
  predictions,
  zones,
  selectedZone,
  onZoneSelect,
  topN = 5,
}: Props) {
  const zoneNameById = new Map(zones.map((z) => [z.id, z.name]));
  const sorted = predictions
    .slice()
    .sort((a, b) => b.prediction - a.prediction)
    .slice(0, topN);

  return (
    <div className="h-full flex flex-col gap-3 pl-4 border-l border-slate-200">
      <div className="flex items-center gap-1.5 text-[10px] uppercase tracking-wider text-slate-500">
        <Trophy className="h-3 w-3" />
        Top {topN} zones
      </div>
      <div className="space-y-1 overflow-auto flex-1">
        {sorted.length === 0 && (
          <div className="text-xs text-slate-400 px-2 py-1">Loading…</div>
        )}
        {sorted.map((p, i) => {
          const isSelected = p.zone_id === selectedZone;
          const name = zoneNameById.get(p.zone_id) ?? `Zone ${p.zone_id}`;
          return (
            <button
              key={p.zone_id}
              onClick={() => onZoneSelect(p.zone_id)}
              className={`w-full text-left px-2.5 py-1.5 rounded text-xs flex items-center justify-between gap-2 transition-colors ${
                isSelected
                  ? "bg-accent text-white"
                  : "hover:bg-slate-50 text-slate-700"
              }`}
            >
              <span className="flex items-center gap-2 min-w-0">
                <span
                  className={`tabular-nums text-[10px] shrink-0 ${
                    isSelected ? "text-white/70" : "text-slate-400"
                  }`}
                >
                  #{i + 1}
                </span>
                <span className="truncate">{name}</span>
              </span>
              <span className="tabular-nums shrink-0 font-medium">
                {p.prediction.toFixed(0)}
              </span>
            </button>
          );
        })}
      </div>
      <div className="text-[10px] text-slate-400 leading-tight">
        Click any zone to switch selection. Click on the map: select.
        Shift+click on the map: compare.
      </div>
    </div>
  );
}
