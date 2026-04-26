"use client";

import { useEffect, useState } from "react";
import { useSearchParams } from "next/navigation";
import { Activity, ArrowLeft } from "lucide-react";
import Link from "next/link";
import { CompareCard } from "@/components/CompareCard";
import { api } from "@/lib/api";
import { isWeekendFromDow } from "@/lib/format";
import type { DashboardSelection, ZoneMeta } from "@/lib/types";

function clampInt(raw: string | null, fallback: number, min: number, max: number): number {
  if (raw == null) return fallback;
  const n = Number(raw);
  if (!Number.isFinite(n)) return fallback;
  return Math.min(Math.max(Math.round(n), min), max);
}

export default function ComparePage() {
  const searchParams = useSearchParams();

  const initialHour = clampInt(searchParams.get("hour"), 17, 0, 23);
  const initialDow = clampInt(searchParams.get("dow"), 4, 0, 6);
  const initialWeekend =
    searchParams.get("weekend") != null
      ? clampInt(searchParams.get("weekend"), 0, 0, 1)
      : isWeekendFromDow(initialDow);

  const [zones, setZones] = useState<ZoneMeta[]>([]);
  const [a, setA] = useState<DashboardSelection>({
    zone: clampInt(searchParams.get("a"), 161, 1, 999),
    hour: initialHour,
    dow: initialDow,
    weekend: initialWeekend,
  });
  const [b, setB] = useState<DashboardSelection>({
    zone: clampInt(searchParams.get("b"), 132, 1, 999),
    hour: initialHour,
    dow: initialDow,
    weekend: initialWeekend,
  });

  useEffect(() => { api.zones().then(setZones).catch(console.error); }, []);

  return (
    <div className="min-h-screen bg-white text-slate-900">
      <header className="border-b border-slate-200 px-8 py-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Activity className="h-5 w-5 text-accent" />
          <span className="font-semibold tracking-tight">DemandCast</span>
          <span className="text-xs text-slate-500 ml-2">Compare zones</span>
        </div>
        <Link href="/" className="text-sm text-slate-700 hover:text-accent inline-flex items-center gap-1">
          <ArrowLeft className="h-3.5 w-3.5" />
          Back to dashboard
        </Link>
      </header>

      <main className="p-8">
        <div className="grid grid-cols-2 gap-6">
          <CompareCard zones={zones} selection={a} onChange={setA} label="A" />
          <CompareCard zones={zones} selection={b} onChange={setB} label="B" />
        </div>
        <p className="text-xs text-slate-500 mt-4 max-w-2xl">
          Pick two (zone, hour, day) combinations to see how predictions diverge.
          Useful for "where should I send drivers next" decisions when two zones look
          comparable on paper.
        </p>
      </main>
    </div>
  );
}
