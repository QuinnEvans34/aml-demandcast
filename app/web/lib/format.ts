export const DOW_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
export const DOW_LABELS_FULL = [
  "Monday",
  "Tuesday",
  "Wednesday",
  "Thursday",
  "Friday",
  "Saturday",
  "Sunday",
];

export function formatHour(hour: number): string {
  if (hour === 0) return "12 AM";
  if (hour === 12) return "12 PM";
  return hour < 12 ? `${hour} AM` : `${hour - 12} PM`;
}

export function formatPercent(value: number): string {
  const sign = value > 0 ? "+" : "";
  return `${sign}${value.toFixed(1)}%`;
}

export function isWeekendFromDow(dow: number): number {
  return dow >= 5 ? 1 : 0;
}

// Forecast week — the seven days immediately after the training data ends.
// Training data ends 2025-02-01 00:00; forecast covers 2025-02-02 → 2025-02-08.
// dow is Python convention: 0 = Mon, 6 = Sun.
const FORECAST_DATES_ISO: Record<number, string> = {
  0: "2025-02-03",  // Monday
  1: "2025-02-04",  // Tuesday
  2: "2025-02-05",  // Wednesday
  3: "2025-02-06",  // Thursday
  4: "2025-02-07",  // Friday
  5: "2025-02-08",  // Saturday
  6: "2025-02-02",  // Sunday
};

export const FORECAST_WEEK_LABEL = "Feb 2–8, 2025";
export const TRAINING_PERIOD_LABEL = "Dec 31, 2024 – Feb 1, 2025";

export function forecastDateFor(dow: number): Date {
  const iso = FORECAST_DATES_ISO[dow];
  // Anchor at noon UTC to avoid timezone-rollover off-by-one when the user's
  // local zone differs from UTC.
  return new Date(`${iso}T12:00:00Z`);
}

export function formatForecastDate(dow: number): string {
  return forecastDateFor(dow).toLocaleDateString("en-US", {
    weekday: "long",
    month: "short",
    day: "numeric",
    year: "numeric",
    timeZone: "UTC",
  });
}

export function formatForecastDateShort(dow: number): string {
  return forecastDateFor(dow).toLocaleDateString("en-US", {
    weekday: "short",
    month: "short",
    day: "numeric",
    timeZone: "UTC",
  });
}
