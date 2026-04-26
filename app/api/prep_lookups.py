"""
prep_lookups.py — Build the historical averages lookup the dashboard backend
uses to fill in lag and rolling features at predict-time.

The trained model needs 11 features. The dashboard only collects 4 from the
user (zone, hour, day_of_week, is_weekend). For the remaining 7, two are
deterministic (`is_rush_hour`, `is_holiday`), one is the zone id itself
(`PULocationID`), and four are historical (`demand_lag_1h`, `demand_lag_24h`,
`demand_lag_168h`, `demand_rolling_3h`, `demand_rolling_168h`). This script
pre-computes the means of those four-plus-demand for every
(zone, hour_of_day, day_of_week) cell so the API can do a single dict lookup
at predict-time.

Run from the project root with .venv active:
    python app/api/prep_lookups.py

Writes:
    data/dashboard/zone_hour_dow_averages.parquet
        Columns: PULocationID, hour_of_day, day_of_week,
                 mean_demand,
                 mean_demand_lag_1h, mean_demand_lag_24h, mean_demand_lag_168h,
                 mean_demand_rolling_3h, mean_demand_rolling_168h
"""
from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SOURCE = REPO_ROOT / "data" / "features.parquet"
OUT_DIR = REPO_ROOT / "data" / "dashboard"
OUT = OUT_DIR / "zone_hour_dow_averages.parquet"

LAG_COLS = [
    "demand_lag_1h",
    "demand_lag_24h",
    "demand_lag_168h",
    "demand_rolling_3h",
    "demand_rolling_168h",
]


def main() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(f"{SOURCE} not found. Run build_features.py first.")

    df = pd.read_parquet(SOURCE)

    grouped = (
        df.groupby(["PULocationID", "hour_of_day", "day_of_week"], as_index=False)
          .agg({col: "mean" for col in ["demand", *LAG_COLS]})
          .rename(columns={col: f"mean_{col}" for col in ["demand", *LAG_COLS]})
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    grouped.to_parquet(OUT, index=False)
    print(f"Wrote {OUT} with {len(grouped):,} (zone, hour, dow) rows.")


if __name__ == "__main__":
    main()
