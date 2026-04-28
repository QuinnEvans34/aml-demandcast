"use client";

import { useEffect, useState } from "react";
import { Activity } from "lucide-react";
import { Sidebar } from "@/components/Sidebar";
import { HeroPrediction } from "@/components/HeroPrediction";
import { StatStrip } from "@/components/StatStrip";
import { ContextTabs } from "@/components/ContextTabs";
import { ModelCard } from "@/components/ModelCard";
import { api } from "@/lib/api";
import { isWeekendFromDow } from "@/lib/format";
import type { DashboardSelection, PredictResponse, ZoneMeta, ZonePrediction } from "@/lib/types";

const DEFAULT_ZONE = 161;   // Midtown Center
const DEFAULT_HOUR = 17;
const DEFAULT_DOW = 4;

export default function Home() {
  const [zones, setZones] = useState<ZoneMeta[]>([]);
  const [sel, setSel] = useState<DashboardSelection>({
    zone: DEFAULT_ZONE,
    hour: DEFAULT_HOUR,
    dow: DEFAULT_DOW,
    weekend: isWeekendFromDow(DEFAULT_DOW),
  });
  const [prediction, setPrediction] = useState<PredictResponse | null>(null);
  const [allPredictions, setAllPredictions] = useState<ZonePrediction[]>([]);

  useEffect(() => {
    api.zones().then(setZones).catch(console.error);
  }, []);

  useEffect(() => {
    api.predict(sel.zone, sel.hour, sel.dow, sel.weekend)
      .then(setPrediction)
      .catch(console.error);
  }, [sel.zone, sel.hour, sel.dow, sel.weekend]);

  useEffect(() => {
    api.predictAll(sel.hour, sel.dow, sel.weekend)
      .then(setAllPredictions)
      .catch(console.error);
  }, [sel.hour, sel.dow, sel.weekend]);

  const rank = allPredictions.length
    ? allPredictions
        .slice()
        .sort((a, b) => b.prediction - a.prediction)
        .findIndex((p) => p.zone_id === sel.zone) + 1
    : null;

  return (
    <div className="min-h-screen bg-white text-slate-900">
      {/* Top bar */}
      <header className="border-b border-slate-200 px-8 py-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Activity className="h-5 w-5 text-accent" />
          <span className="font-semibold tracking-tight">DemandCast</span>
          <span className="text-xs text-slate-500 ml-2">
            Production model · DemandCast v4
          </span>
        </div>
        <a href="/compare" className="text-sm text-slate-700 hover:text-accent">
          Compare zones →
        </a>
      </header>

      <div className="grid grid-cols-[280px_1fr] min-h-[calc(100vh-65px)]">
        {/* Sidebar */}
        <Sidebar
          zones={zones}
          selection={sel}
          onChange={setSel}
        />

        {/* Main */}
        <main className="p-8 space-y-8">
          <HeroPrediction prediction={prediction} />
          <StatStrip
            prediction={prediction}
            rank={rank}
            totalZones={allPredictions.length}
          />
          <ContextTabs
            selection={sel}
            onZoneSelect={(zone) => setSel({ ...sel, zone })}
            allPredictions={allPredictions}
            zones={zones}
          />
          <ModelCard />
        </main>
      </div>
    </div>
  );
}
