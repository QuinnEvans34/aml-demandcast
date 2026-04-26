"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { MapContainer, TileLayer, GeoJSON } from "react-leaflet";
import L from "leaflet";
import type { Feature, FeatureCollection } from "geojson";
import type { DashboardSelection, ZonePrediction } from "@/lib/types";

interface Props {
  selectedZone: number;
  onZoneSelect: (zone: number) => void;
  predictions: ZonePrediction[];
  selection: DashboardSelection;
}

export function ZoneMap({ selectedZone, onZoneSelect, predictions, selection }: Props) {
  const [geo, setGeo] = useState<FeatureCollection | null>(null);
  const router = useRouter();

  useEffect(() => {
    fetch("/nyc_taxi_zones.geojson")
      .then((r) => r.json())
      .then((g: FeatureCollection) => setGeo(g))
      .catch(console.error);
  }, []);

  const predByZone = new Map(predictions.map((p) => [p.zone_id, p.prediction]));
  const max = predictions.length ? Math.max(...predictions.map((p) => p.prediction)) : 1;

  function colorFor(zoneId: number): string {
    const v = predByZone.get(zoneId);
    if (v == null) return "#e2e8f0"; // slate-200 for "no data"
    // sqrt scale — gives more visual separation in the lower range while
    // preserving rank order. Better than linear when demand has a long tail.
    const t = Math.min(1, Math.sqrt(v / max));
    const r = Math.round(255 - (255 - 0x25) * t);
    const g = Math.round(255 - (255 - 0x63) * t);
    const b = Math.round(255 - (255 - 0xeb) * t);
    return `rgb(${r}, ${g}, ${b})`;
  }

  function styleZone(feature: Feature | undefined) {
    const id = feature?.properties?.location_id
      ?? feature?.properties?.LocationID
      ?? feature?.properties?.locationid;
    const isSelected = Number(id) === selectedZone;
    return {
      fillColor: colorFor(Number(id)),
      fillOpacity: 0.75,
      color: isSelected ? "#2563eb" : "#cbd5e1",
      weight: isSelected ? 2.5 : 0.5,
      className: "zone-polygon",
    };
  }

  if (!geo) {
    return <div className="h-full flex items-center justify-center text-sm text-slate-400">Loading map…</div>;
  }

  return (
    <div className="relative h-full w-full">
      <MapContainer
        center={[40.7484, -73.9857]}
        zoom={11}
        scrollWheelZoom={true}
        style={{ height: "100%", width: "100%" }}
      >
        <TileLayer
          url="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png"
          attribution='&copy; OpenStreetMap &copy; CARTO'
        />
        <GeoJSON
          key={`${predictions.length}-${selectedZone}`}
          data={geo}
          style={styleZone as L.StyleFunction}
          onEachFeature={(feature, layer) => {
            const id = Number(
              feature.properties?.location_id
                ?? feature.properties?.LocationID
                ?? feature.properties?.locationid
            );
            const name = feature.properties?.zone ?? `Zone ${id}`;
            const v = predByZone.get(id);
            layer.bindTooltip(
              v == null
                ? `${name} — no data`
                : `${name} — ${v.toFixed(0)} predicted pickups`,
              { sticky: true }
            );
            layer.on({
              click: (e: L.LeafletMouseEvent) => {
                const native = e.originalEvent as MouseEvent;
                if (native.shiftKey) {
                  // Shift-click: deep-link to /compare with both zones + the
                  // current time selection pre-filled.
                  const params = new URLSearchParams({
                    a: String(selectedZone),
                    b: String(id),
                    hour: String(selection.hour),
                    dow: String(selection.dow),
                    weekend: String(selection.weekend),
                  });
                  router.push(`/compare?${params.toString()}`);
                } else {
                  onZoneSelect(id);
                }
              },
              mouseover: (e: L.LeafletMouseEvent) => {
                (e.target as L.Path).setStyle({
                  fillOpacity: 0.92,
                  weight: 1.5,
                  color: "#94a3b8",
                });
              },
              mouseout: (e: L.LeafletMouseEvent) => {
                const target = e.target as L.Path;
                const isSelected = id === selectedZone;
                target.setStyle({
                  fillColor: colorFor(id),
                  fillOpacity: 0.75,
                  color: isSelected ? "#2563eb" : "#cbd5e1",
                  weight: isSelected ? 2.5 : 0.5,
                });
              },
            });
          }}
        />
      </MapContainer>

      {/* Color legend */}
      <div className="absolute bottom-4 right-4 z-[1000] bg-white/95 border border-slate-200 rounded px-2.5 py-2 shadow-sm pointer-events-none">
        <div className="text-[10px] uppercase tracking-wider text-slate-500 mb-1.5">
          Predicted demand
        </div>
        <div className="flex items-center gap-2">
          <span className="text-[10px] text-slate-500 tabular-nums">low</span>
          <div
            className="w-20 h-2 rounded-sm border border-slate-200"
            style={{
              background: "linear-gradient(to right, white, #2563eb)",
            }}
          />
          <span className="text-[10px] text-slate-500 tabular-nums">high</span>
        </div>
      </div>
    </div>
  );
}
