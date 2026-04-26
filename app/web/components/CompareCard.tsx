"use client";

import { useEffect, useState } from "react";
import { CalendarDays, Clock, MapPin, Moon, Sun } from "lucide-react";
import { LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, ResponsiveContainer, ReferenceDot } from "recharts";
import { api } from "@/lib/api";
import { DOW_LABELS, DOW_LABELS_FULL, formatHour, isWeekendFromDow, formatPercent } from "@/lib/format";
import type { DashboardSelection, HourPrediction, PredictResponse, ZoneMeta } from "@/lib/types";

interface Props {
  zones: ZoneMeta[];
  selection: DashboardSelection;
  onChange: (sel: DashboardSelection) => void;
  label: string;
}

export function CompareCard({ zones, selection, onChange, label }: Props) {
  const [pred, setPred] = useState<PredictResponse | null>(null);
  const [timeline, setTimeline] = useState<HourPrediction[]>([]);

  useEffect(() => {
    api.predict(selection.zone, selection.hour, selection.dow, selection.weekend)
      .then(setPred).catch(console.error);
    api.timeline(selection.zone, selection.dow, selection.weekend)
      .then(setTimeline).catch(console.error);
  }, [selection.zone, selection.hour, selection.dow, selection.weekend]);

  const up = pred && pred.vs_avg_pct >= 0;

  return (
    <section className="border border-slate-200 rounded-lg bg-white">
      <div className="px-6 py-3 border-b border-slate-200 flex items-center justify-between">
        <span className="text-xs uppercase tracking-wider text-slate-500">{label}</span>
        {pred && <span className="text-xs text-slate-400 tabular-nums">{pred.zone_name}</span>}
      </div>

      {/* Controls — compact horizontal version */}
      <div className="px-6 py-4 border-b border-slate-200 grid grid-cols-2 gap-x-6 gap-y-4">
        <div>
          <Lbl icon={<MapPin className="h-3.5 w-3.5" />}>Zone</Lbl>
          <select
            value={selection.zone}
            onChange={(e) => onChange({ ...selection, zone: Number(e.target.value) })}
            className="w-full rounded-md border border-slate-200 bg-white px-2 py-1.5 text-sm tabular-nums focus:border-accent focus:outline-none"
          >
            {zones.map((z) => (
              <option key={z.id} value={z.id}>{z.name}</option>
            ))}
          </select>
        </div>
        <div>
          <Lbl icon={<Clock className="h-3.5 w-3.5" />}>
            Hour <span className="text-slate-500 ml-1 tabular-nums">{formatHour(selection.hour)}</span>
          </Lbl>
          <input
            type="range"
            min={0}
            max={23}
            value={selection.hour}
            onChange={(e) => onChange({ ...selection, hour: Number(e.target.value) })}
            className="w-full accent-accent"
          />
        </div>
        <div className="col-span-2">
          <Lbl icon={<CalendarDays className="h-3.5 w-3.5" />}>Day</Lbl>
          <div className="grid grid-cols-7 gap-1">
            {DOW_LABELS.map((label, dow) => (
              <button
                key={dow}
                onClick={() => onChange({ ...selection, dow, weekend: isWeekendFromDow(dow) })}
                className={`text-xs py-1 rounded border ${
                  selection.dow === dow
                    ? "border-accent bg-accent text-white"
                    : "border-slate-200 text-slate-600 hover:border-slate-300"
                }`}
              >
                {label}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Hero */}
      <div className="px-6 py-6 border-b border-slate-200">
        {pred ? (
          <>
            <div className="flex items-baseline gap-3">
              <div className="text-5xl font-semibold tabular-nums text-accent leading-none">{pred.prediction.toFixed(0)}</div>
              <div className="text-sm text-slate-500 tabular-nums">± {pred.error_band.toFixed(0)}</div>
            </div>
            <div className="text-xs text-slate-700 mt-3">
              {DOW_LABELS_FULL[pred.day_of_week]} · {formatHour(pred.hour_of_day)}
              {selection.weekend ? <span className="ml-2 inline-flex items-center"><Moon className="h-3 w-3" /></span>
                                : <span className="ml-2 inline-flex items-center"><Sun className="h-3 w-3" /></span>}
            </div>
            <div className={`text-xs mt-3 tabular-nums ${up ? "text-emerald-600" : "text-rose-600"}`}>
              {formatPercent(pred.vs_avg_pct)} vs typical ({pred.historical_mean.toFixed(1)})
            </div>
          </>
        ) : (
          <div className="text-sm text-slate-400">Loading…</div>
        )}
      </div>

      {/* Mini timeline */}
      <div className="h-48 p-4">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={timeline} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
            <CartesianGrid strokeDasharray="2 4" stroke="#e2e8f0" />
            <XAxis dataKey="hour" tickFormatter={formatHour} tick={{ fontSize: 10, fill: "#64748b" }} axisLine={false} tickLine={false} />
            <YAxis tick={{ fontSize: 10, fill: "#64748b" }} axisLine={false} tickLine={false} />
            <Tooltip formatter={(v: number) => [`${v.toFixed(0)} pickups`, "predicted"]} labelFormatter={formatHour} />
            <Line type="monotone" dataKey="prediction" stroke="#2563eb" strokeWidth={2} dot={false} />
            {timeline.find((d) => d.hour === selection.hour) && (
              <ReferenceDot x={selection.hour} y={timeline.find((d) => d.hour === selection.hour)!.prediction} r={4} fill="#2563eb" stroke="white" strokeWidth={2} />
            )}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </section>
  );
}

function Lbl({ icon, children }: { icon: React.ReactNode; children: React.ReactNode }) {
  return (
    <div className="flex items-center gap-1.5 text-[10px] uppercase tracking-wider text-slate-500 mb-1">
      {icon}<span>{children}</span>
    </div>
  );
}
