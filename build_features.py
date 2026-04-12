"""Build features pipeline for DemandCast.

Loads raw parquet data, cleans it, engineers temporal features,
aggregates to hourly demand per zone, adds lag features, and
saves the result to data/features.parquet.
"""

from pathlib import Path
import pandas as pd
from src.features import (
    clean_data,
    create_temporal_features,
    aggregate_to_hourly_demand,
    add_temporal_features_to_hourly,
    fill_missing_hours,
    filter_inactive_zones,
    add_lag_features,
    fill_lag_nans,
    add_rolling_features,
)

DATA_DIR = Path("data")
RAW_FILE = DATA_DIR / "yellow_tripdata_2025-01.parquet"
OUTPUT_FILE = DATA_DIR / "features.parquet"


def main():
    # Step 1: Load raw data
    print(f"Loading raw data from {RAW_FILE} ...")
    df = pd.read_parquet(RAW_FILE)
    print(f"  Loaded {len(df):,} rows, {df.shape[1]} columns")

    # Step 2: Clean data
    print("Cleaning data (removing erroneous trips) ...")
    df = clean_data(df)
    print(f"  {len(df):,} rows after cleaning")

    # Step 3: Create temporal features
    print("Creating temporal features ...")
    df = create_temporal_features(df)
    print(f"  Added columns: hour, day_of_week, is_weekend, month, is_rush_hour, is_holiday")

    # Step 4: Aggregate to hourly demand
    print("Aggregating to hourly demand per zone ...")
    hourly_df = aggregate_to_hourly_demand(df)
    print(f"  {len(hourly_df):,} rows in hourly_df")

    # Step 5: Re-derive temporal features on hourly DataFrame
    print("Adding temporal features to hourly DataFrame ...")
    hourly_df = add_temporal_features_to_hourly(hourly_df)
    print(f"  Added columns: hour_of_day, day_of_week, is_weekend, is_rush_hour, is_holiday")

    # Step 5b: Fill missing hours so every zone has a contiguous hourly series
    print("Filling missing hours per zone (demand=0 for sparse hours) ...")
    hourly_df = fill_missing_hours(hourly_df)
    print(f"  {len(hourly_df):,} rows after fill")

    # Step 5c: Filter inactive zones
    print("Filtering inactive zones ...")
    hourly_df = filter_inactive_zones(hourly_df, min_active_hours=100)

    # Step 6: Add lag features
    print("Adding lag features (1h, 24h, 168h) ...")
    hourly_df = add_lag_features(hourly_df, zone_col='PULocationID', target_col='demand')
    print(f"  Added columns: demand_lag_1h, demand_lag_24h, demand_lag_168h")

    # Step 6b: Fill lag NaNs with zone + hour-of-day means
    print("Filling lag NaNs with zone + hour-of-day means ...")
    hourly_df = fill_lag_nans(hourly_df, zone_col='PULocationID', target_col='demand')

    # Step 7: Add rolling features (min_periods=1, no NaNs produced)
    print("Adding rolling features (3h, 168h) ...")
    hourly_df = add_rolling_features(hourly_df, zone_col='PULocationID', target_col='demand')
    print(f"  Added columns: demand_rolling_3h, demand_rolling_168h")

    # Step 8: Verify and save
    print(f"Final shape: {hourly_df.shape}")
    print(f"Active zones in output: {hourly_df['PULocationID'].nunique()}")
    print(f"Earliest hour: {hourly_df['hour'].min()}")
    print(f"Latest hour: {hourly_df['hour'].max()}")
    print(f"Total NaN values: {hourly_df.isna().sum().sum()}")
    print(f"Zero-demand rows: {(hourly_df['demand'] == 0).sum():,} "
          f"({100 * (hourly_df['demand'] == 0).mean():.1f}%)")
    print(f"NaNs per column:\n{hourly_df.isna().sum()}")
    hourly_df.to_parquet(OUTPUT_FILE, index=False)
    print(f"Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
