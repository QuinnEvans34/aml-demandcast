"use client";

import { useState } from "react";
import dynamic from "next/dynamic";
import { Grid3x3, LineChart as LineIcon, Map as MapIcon } from "lucide-react";
import { Timeline } from "./Timeline";
import { Heatmap } from "./Heatmap";
import { TopZonesPanel } from "./TopZonesPanel";
import type { DashboardSelection, ZoneMeta, ZonePrediction } from "@/lib/types";

const ZoneMap = dynamic(() => import("./ZoneMap").then((m) => m.ZoneMap), {
  ssr: false,
  loading: () => (
    <div className="h-full flex items-center justify-center text-sm text-slate-400">
      Loading map…
    </div>
  ),
});

interface Props {
  selection: DashboardSelection;
  onZoneSelect: (zone: number) => void;
  allPredictions: ZonePrediction[];
  zones: ZoneMeta[];
}

type Tab = "map" | "timeline" | "heatmap";

export function ContextTabs({ selection, onZoneSelect, allPredictions, zones }: Props) {
  const [tab, setTab] = useState<Tab>("map");
  return (
    <div className="border border-slate-200 rounded-lg bg-white">
      <div className="border-b border-slate-200 flex">
        <TabButton active={tab === "map"} onClick={() => setTab("map")} icon={<MapIcon className="h-3.5 w-3.5" />} label="Map" />
        <TabButton active={tab === "timeline"} onClick={() => setTab("timeline")} icon={<LineIcon className="h-3.5 w-3.5" />} label="Timeline" />
        <TabButton active={tab === "heatmap"} onClick={() => setTab("heatmap")} icon={<Grid3x3 className="h-3.5 w-3.5" />} label="Heatmap" />
      </div>
      <div className="p-4 h-[480px]">
        {tab === "map" && (
          <div className="grid grid-cols-[1fr_240px] gap-4 h-full">
            <ZoneMap
              selectedZone={selection.zone}
              onZoneSelect={onZoneSelect}
              predictions={allPredictions}
              selection={selection}
            />
            <TopZonesPanel
              predictions={allPredictions}
              zones={zones}
              selectedZone={selection.zone}
              onZoneSelect={onZoneSelect}
              topN={5}
            />
          </div>
        )}
        {tab === "timeline" && (
          <Timeline zone={selection.zone} dow={selection.dow} weekend={selection.weekend} hour={selection.hour} />
        )}
        {tab === "heatmap" && <Heatmap zone={selection.zone} />}
      </div>
    </div>
  );
}

function TabButton({ active, onClick, icon, label }: { active: boolean; onClick: () => void; icon: React.ReactNode; label: string }) {
  return (
    <button
      onClick={onClick}
      className={`flex items-center gap-1.5 px-4 py-2.5 text-xs uppercase tracking-wider border-b-2 ${
        active ? "border-accent text-accent" : "border-transparent text-slate-500 hover:text-slate-700"
      }`}
    >
      {icon}<span>{label}</span>
    </button>
  );
}
