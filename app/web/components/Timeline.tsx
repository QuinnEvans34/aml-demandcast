"use client";

import { useEffect, useState } from "react";
import { LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, ReferenceDot, ResponsiveContainer } from "recharts";
import { api } from "@/lib/api";
import { DOW_LABELS_FULL, formatForecastDateShort, formatHour } from "@/lib/format";
import type { HourPrediction } from "@/lib/types";

interface Props {
  zone: number;
  dow: number;
  weekend: number;
  hour: number;
}

export function Timeline({ zone, dow, weekend, hour }: Props) {
  const [data, setData] = useState<HourPrediction[]>([]);
  useEffect(() => {
    api.timeline(zone, dow, weekend).then(setData).catch(console.error);
  }, [zone, dow, weekend]);

  return (
    <div className="h-full flex flex-col">
      <div className="text-[10px] uppercase tracking-wider text-slate-500 mb-2">
        24-hour forecast · {DOW_LABELS_FULL[dow]}, {formatForecastDateShort(dow)}
      </div>
      <div className="flex-1 min-h-0">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 12, right: 24, left: 0, bottom: 0 }}>
            <CartesianGrid strokeDasharray="2 4" stroke="#e2e8f0" />
            <XAxis
              dataKey="hour"
              tickFormatter={formatHour}
              tick={{ fontSize: 11, fill: "#64748b" }}
              axisLine={false}
              tickLine={false}
            />
            <YAxis
              tick={{ fontSize: 11, fill: "#64748b" }}
              axisLine={false}
              tickLine={false}
            />
            <Tooltip
              formatter={(v: number) => [`${v.toFixed(0)} pickups`, "predicted"]}
              labelFormatter={formatHour}
            />
            <Line
              type="monotone"
              dataKey="prediction"
              stroke="#2563eb"
              strokeWidth={2}
              dot={false}
            />
            {data.find((d) => d.hour === hour) && (
              <ReferenceDot
                x={hour}
                y={data.find((d) => d.hour === hour)!.prediction}
                r={4}
                fill="#2563eb"
                stroke="white"
                strokeWidth={2}
              />
            )}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
