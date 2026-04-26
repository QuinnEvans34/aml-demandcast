"use client";

import { CalendarDays, Clock, MapPin, Moon, Sun } from "lucide-react";
import { DOW_LABELS, isWeekendFromDow, formatHour } from "@/lib/format";
import type { DashboardSelection, ZoneMeta } from "@/lib/types";

interface Props {
  zones: ZoneMeta[];
  selection: DashboardSelection;
  onChange: (sel: DashboardSelection) => void;
}

export function Sidebar({ zones, selection, onChange }: Props) {
  return (
    <aside className="border-r border-slate-200 bg-white p-6 space-y-8">
      {/* Zone */}
      <div>
        <Label icon={<MapPin className="h-3.5 w-3.5" />}>Zone</Label>
        <select
          value={selection.zone}
          onChange={(e) => onChange({ ...selection, zone: Number(e.target.value) })}
          className="w-full rounded-md border border-slate-200 bg-white px-3 py-2 text-sm tabular-nums focus:border-accent focus:outline-none focus:ring-1 focus:ring-accent"
        >
          {zones.map((z) => (
            <option key={z.id} value={z.id}>
              {z.name} · {z.borough}
            </option>
          ))}
        </select>
      </div>

      {/* Hour */}
      <div>
        <Label icon={<Clock className="h-3.5 w-3.5" />}>
          Hour <span className="text-slate-500 ml-2 tabular-nums">{formatHour(selection.hour)}</span>
        </Label>
        <input
          type="range"
          min={0}
          max={23}
          value={selection.hour}
          onChange={(e) => onChange({ ...selection, hour: Number(e.target.value) })}
          className="w-full accent-accent"
        />
        <div className="flex justify-between text-[10px] text-slate-400 mt-1 tabular-nums">
          <span>0</span><span>6</span><span>12</span><span>18</span><span>23</span>
        </div>
      </div>

      {/* Day of week */}
      <div>
        <Label icon={<CalendarDays className="h-3.5 w-3.5" />}>Day</Label>
        <div className="grid grid-cols-7 gap-1">
          {DOW_LABELS.map((label, dow) => (
            <button
              key={dow}
              onClick={() =>
                onChange({ ...selection, dow, weekend: isWeekendFromDow(dow) })
              }
              className={`text-xs py-1.5 rounded border tabular-nums ${
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

      {/* Weekend (display-only — derives from day of week, but user can override) */}
      <div>
        <Label
          icon={selection.weekend ? <Moon className="h-3.5 w-3.5" /> : <Sun className="h-3.5 w-3.5" />}
        >
          Weekend
        </Label>
        <div className="flex gap-1">
          <button
            onClick={() => onChange({ ...selection, weekend: 0 })}
            className={`flex-1 text-xs py-1.5 rounded border ${
              selection.weekend === 0
                ? "border-accent bg-accent text-white"
                : "border-slate-200 text-slate-600 hover:border-slate-300"
            }`}
          >
            Weekday
          </button>
          <button
            onClick={() => onChange({ ...selection, weekend: 1 })}
            className={`flex-1 text-xs py-1.5 rounded border ${
              selection.weekend === 1
                ? "border-accent bg-accent text-white"
                : "border-slate-200 text-slate-600 hover:border-slate-300"
            }`}
          >
            Weekend
          </button>
        </div>
      </div>
    </aside>
  );
}

function Label({ icon, children }: { icon: React.ReactNode; children: React.ReactNode }) {
  return (
    <div className="flex items-center gap-1.5 text-[10px] uppercase tracking-wider text-slate-500 mb-2">
      {icon}
      <span>{children}</span>
    </div>
  );
}
