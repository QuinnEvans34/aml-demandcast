# AI prompt used: "Write four feature engineering functions for NYC Yellow Taxi demand forecasting:
# 1) clean_data - filter out erroneous trips using trip_distance (0,50], fare_amount [2.50,200],
#    passenger_count [1,6]. 2) create_temporal_features - extract hour, day_of_week, is_weekend,
#    month, is_rush_hour from tpep_pickup_datetime. 3) aggregate_to_hourly_demand - group trips
#    by PULocationID and hour-floored pickup time, counting trips as demand. 4) add_lag_features -
#    add 1h, 24h, and 168h lag features grouped by zone for the demand column."

import pandas as pd
import numpy as np


NYC_HOLIDAYS_JAN_2025 = {
    pd.Timestamp('2025-01-01').date(),   # New Year's Day
    pd.Timestamp('2025-01-20').date(),   # Martin Luther King Jr. Day
}


FEATURE_COLS: list[str] = [
    'hour_of_day',
    'day_of_week',
    'is_weekend',
    'is_rush_hour',
    'is_holiday',
    'PULocationID',
    'demand_lag_1h',
    'demand_lag_24h',
    'demand_lag_168h',
    'demand_rolling_3h',
    'demand_rolling_168h',
]

TARGET: str = 'demand'


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw trip-level rows before feature engineering.

    Use thresholds determined during EDA. The defaults below are reasonable
    starting points — override them if your EDA revealed different
    breakpoints for your data sample.

    Cleaning strategy (student exercise)
    -----------------------------------
    Implement the data cleaning strategies you determined during exploratory
    data analysis (EDA). Do not hard-code specific thresholds in this
    template; instead document and apply the rules you identified (for
    example: outlier detection, sensible missing-value handling, sensor-error
    filters, or domain-specific rules). Justify your choices in the
    accompanying notebook and use the methods you found appropriate for the
    dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Raw trip-level DataFrame loaded from the parquet file.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame. Index is reset so it is contiguous after row drops.

    Examples
    --------
    >>> clean_df = clean_data(df)
    >>> print(f"Rows removed: {len(df) - len(clean_df)}")
    """
    df = df[df['trip_distance'] > 0]
    df = df[df['trip_distance'] <= 50]
    df = df[df['fare_amount'] >= 2.50]
    df = df[df['fare_amount'] <= 200]
    df = df[df['passenger_count'] >= 1]
    df = df[df['passenger_count'] <= 6]
    df = df.reset_index(drop=True)
    return df


def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract time-based features from the tpep_pickup_datetime column.

    All features are derived from a single source column so there is no risk
    of data leakage — we are only decomposing information already present at
    prediction time.

    New columns added
    -----------------
    pickup_hour : datetime64
        The pickup datetime floored to the nearest hour.
        Used as the groupby key in aggregate_to_hourly_demand().
    hour : int
        Hour of day (0–23).
    day_of_week : int
        Day of week (0 = Monday, 6 = Sunday). Use dt.dayofweek.
    is_weekend : int
        1 if day_of_week >= 5, else 0.
    month : int
        Month of year (1–12).
    is_rush_hour : int
        1 if (hour is 7, 8 OR hour is 17, 18) AND day_of_week < 5, else 0.
        Morning rush: 7–9am. Evening rush: 5–7pm. Weekdays only.
    is_holiday : int
        1 if the pickup date is a federal holiday in January 2025
        (Jan 1 New Year's Day, Jan 20 MLK Day), else 0.
        Hardcoded for the January 2025 dataset. Extend NYC_HOLIDAYS_JAN_2025
        for additional months.

    Parameters
    ----------
    df : pd.DataFrame
        Trip-level DataFrame. Must contain column tpep_pickup_datetime.

    Returns
    -------
    pd.DataFrame
        Original DataFrame with new feature columns appended.

    Examples
    --------
    >>> df = create_temporal_features(df)
    >>> df[['hour', 'day_of_week', 'is_weekend', 'is_rush_hour']].head()
    """
    df = df.copy()
    df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
    df['pickup_hour'] = df['tpep_pickup_datetime'].dt.floor('h')
    df['hour'] = df['tpep_pickup_datetime'].dt.hour
    df['day_of_week'] = df['tpep_pickup_datetime'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['month'] = df['tpep_pickup_datetime'].dt.month
    df['is_rush_hour'] = (
        df['hour'].isin([7, 8, 17, 18]) & (df['day_of_week'] < 5)
    ).astype(int)
    df['is_holiday'] = df['tpep_pickup_datetime'].dt.date.isin(
        NYC_HOLIDAYS_JAN_2025
    ).astype(int)
    return df


def aggregate_to_hourly_demand(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate individual trips into hourly demand counts per pickup zone.

    This function performs the core transformation that converts the raw
    trip-level data (one row per trip) into the modeling target (one row per
    zone per hour, where the value is the number of pickups).

    Input shape  : (n_trips, many columns)  — e.g. 2.5M rows for January 2024
    Output shape : (n_zones × n_hours, 3)   — e.g. ~260 zones × 744 hours

    Output columns
    --------------
    PULocationID : int
        Pickup zone ID (1–265 in NYC TLC data).
    hour : datetime64
        The hour bucket (pickup_hour floored to the nearest hour).
    demand : int
        Number of taxi pickups in this zone during this hour.

    Parameters
    ----------
    df : pd.DataFrame
        Trip-level DataFrame after create_temporal_features() has been called.
        Must contain columns: PULocationID, pickup_hour.

    Returns
    -------
    pd.DataFrame
        Aggregated demand DataFrame with columns [PULocationID, hour, demand].

    Examples
    --------
    >>> hourly = aggregate_to_hourly_demand(df)
    >>> print(hourly.shape)   # expect (n_zones * n_hours, 3)
    >>> hourly.head()
    """
    hourly_df = (
        df.groupby([
            'PULocationID',
            pd.Grouper(key='pickup_hour', freq='h')
        ])
        .size()
        .reset_index(name='demand')
        .rename(columns={'pickup_hour': 'hour'})
    )
    return hourly_df


def add_temporal_features_to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Re-derive temporal features on the hourly demand DataFrame.

    After aggregation, the hourly DataFrame contains only [PULocationID, hour,
    demand]. This function re-derives all temporal features from the 'hour'
    datetime column so the full feature matrix is available for modeling.
    Must be called after aggregate_to_hourly_demand and before add_lag_features.

    New columns added
    -----------------
    hour_of_day : int
        Hour of day (0-23). Named hour_of_day to avoid collision with the
        'hour' datetime column.
    day_of_week : int
        Day of week (0 = Monday, 6 = Sunday).
    is_weekend : int
        1 if day_of_week >= 5, else 0.
    is_rush_hour : int
        1 if hour_of_day in [7, 8, 17, 18] AND day_of_week < 5, else 0.
    is_holiday : int
        1 if the date is in NYC_HOLIDAYS_JAN_2025, else 0.

    Parameters
    ----------
    df : pd.DataFrame
        Hourly demand DataFrame returned by aggregate_to_hourly_demand().
        Must contain column: hour (datetime64).

    Returns
    -------
    pd.DataFrame
        DataFrame with five new feature columns appended.

    Examples
    --------
    >>> hourly = aggregate_to_hourly_demand(df)
    >>> hourly = add_temporal_features_to_hourly(hourly)
    >>> hourly[['hour', 'hour_of_day', 'day_of_week', 'is_holiday']].head()
    """
    df = df.copy()
    df['hour_of_day'] = df['hour'].dt.hour
    df['day_of_week'] = df['hour'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_rush_hour'] = (
        df['hour_of_day'].isin([7, 8, 17, 18]) & (df['day_of_week'] < 5)
    ).astype(int)
    df['is_holiday'] = df['hour'].dt.date.isin(
        NYC_HOLIDAYS_JAN_2025
    ).astype(int)
    return df


def fill_missing_hours(
    df: pd.DataFrame, zone_col: str = 'PULocationID'
) -> pd.DataFrame:
    """Reindex every zone to a complete, contiguous hourly DatetimeIndex.

    After aggregate_to_hourly_demand, zones with no pickups in a given hour
    simply have no row for that hour — they are absent rather than zero.
    This creates gaps in the per-zone time series that silently corrupt
    lag and rolling features: shift(n) and rolling(n) operate on *row
    position*, not calendar time, so a zone missing 20 early hours will
    have its 168-row lag reach 20 hours further into the past than intended.

    This function ensures every zone has exactly one row per hour across the
    full observed range, filling demand with 0 for hours with no pickups.
    Temporal feature columns are re-derived on the newly inserted rows so
    the full feature matrix remains valid.

    Must be called after add_temporal_features_to_hourly and before
    add_lag_features.

    Parameters
    ----------
    df : pd.DataFrame
        Hourly demand DataFrame after add_temporal_features_to_hourly().
        Must contain columns: zone_col, 'hour' (datetime64), 'demand',
        and the temporal feature columns added by
        add_temporal_features_to_hourly.
    zone_col : str, optional
        Name of the zone identifier column. Default: 'PULocationID'.

    Returns
    -------
    pd.DataFrame
        DataFrame reindexed to a complete hourly grid for every zone.
        Rows added for missing hours have demand = 0 and fully populated
        temporal feature columns. The output is sorted by zone and hour.

    Examples
    --------
    >>> hourly = aggregate_to_hourly_demand(df)
    >>> hourly = add_temporal_features_to_hourly(hourly)
    >>> hourly = fill_missing_hours(hourly)
    >>> counts = hourly.groupby('PULocationID')['hour'].count()
    >>> assert counts.min() == counts.max()  # every zone has same row count
    """
    full_range = pd.date_range(df['hour'].min(), df['hour'].max(), freq='h')
    zones = df[zone_col].unique()

    idx = pd.MultiIndex.from_product(
        [zones, full_range], names=[zone_col, 'hour']
    )
    df = (
        df.set_index([zone_col, 'hour'])
        .reindex(idx)
        .reset_index()
    )

    df['demand'] = df['demand'].fillna(0).astype(int)

    # Re-derive temporal features for newly inserted rows
    df['hour_of_day'] = df['hour'].dt.hour
    df['day_of_week'] = df['hour'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_rush_hour'] = (
        df['hour_of_day'].isin([7, 8, 17, 18]) & (df['day_of_week'] < 5)
    ).astype(int)
    df['is_holiday'] = df['hour'].dt.date.isin(
        NYC_HOLIDAYS_JAN_2025
    ).astype(int)

    return df.sort_values([zone_col, 'hour']).reset_index(drop=True)


def filter_inactive_zones(
    df: pd.DataFrame,
    zone_col: str = 'PULocationID',
    min_active_hours: int = 100,
) -> pd.DataFrame:
    """Remove zones where yellow taxis operate too infrequently to model reliably.

    After fill_missing_hours, every zone has a complete hourly time series with
    demand = 0 for hours with no pickups. However, some zones in the NYC TLC
    zone map are locations where yellow taxis essentially never operate —
    remote Staten Island areas, restricted zones, or administrative boundaries
    that appear in the zone list but generate fewer than a handful of real
    trips per month.

    Keeping these zones creates two problems:
        1. Their time series is almost entirely zeros, providing no meaningful
           signal for demand forecasting.
        2. They inflate the dataset with low-quality rows that can push the
           model toward predicting near-zero demand everywhere.

    This function filters to zones that had at least min_active_hours hours
    of real demand (demand > 0) across the full time series. Zero-demand rows
    are intentionally KEPT for zones that pass the threshold — knowing that
    an active zone had zero pickups at 3am on a Tuesday is valid and important
    signal. Only entire zones that never meaningfully participated in yellow
    taxi activity are removed.

    The default threshold of 100 active hours out of ~749 possible hours
    (~13% activity rate) is a conservative minimum. A zone clearing this bar
    had real taxi activity on at least 4 days of the month, which is enough
    to make a lag-based demand model meaningful.

    Parameters
    ----------
    df : pd.DataFrame
        Hourly demand DataFrame after fill_missing_hours has been called.
        Must contain columns: zone_col, 'demand'.
    zone_col : str, optional
        Name of the zone identifier column. Default: 'PULocationID'.
    min_active_hours : int, optional
        Minimum number of hours with demand > 0 for a zone to be retained.
        Default: 100. Zones below this threshold are dropped entirely,
        including their zero-demand rows.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame with inactive zones removed. Index is reset.

    Examples
    --------
    >>> df = filter_inactive_zones(df, min_active_hours=100)
    >>> df['PULocationID'].nunique()   # fewer zones than before
    >>> (df['demand'] == 0).mean()     # zero-demand pct, lower than before
    """
    original_count = len(df)

    active_hours = (
        df[df['demand'] > 0]
        .groupby(zone_col)
        .size()
    )
    active_zones = active_hours[active_hours >= min_active_hours].index

    dropped = df[~df[zone_col].isin(active_zones)][zone_col].unique()
    df = df[df[zone_col].isin(active_zones)].reset_index(drop=True)

    print(f"  Zones before filter: {df[zone_col].nunique() + len(dropped)}")
    print(f"  Zones after filter:  {df[zone_col].nunique()}")
    print(f"  Zones dropped (< {min_active_hours} active hours): {len(dropped)}")
    print(f"  Rows before: {original_count:,}")
    print(f"  Rows after:  {len(df):,}")
    print(f"  Zero-demand rows retained: {(df['demand'] == 0).sum():,}")
    print(f"  Zero-demand pct: {100 * (df['demand'] == 0).mean():.1f}%")

    return df


def add_lag_features(df: pd.DataFrame, zone_col: str = 'PULocationID',
                     target_col: str = 'demand') -> pd.DataFrame:
    """Add lagged demand features, computed separately for each zone.

    ⚠️  COMMON BUG WARNING ⚠️
    Lag features MUST be computed per zone using groupby. If you call
    df[target_col].shift(n) without groupby, you will bleed one zone's demand
    into the previous/next zone's lag column. This is a subtle data quality
    bug — the model will train without errors, but the features are wrong.

    Correct pattern:
        df[target_col].shift(n)                          ← WRONG
        df.groupby(zone_col)[target_col].shift(n)        ← CORRECT

    New columns added
    -----------------
    demand_lag_1h : float
        Demand for this zone 1 time-step ago (= 1 hour in the hourly table).
    demand_lag_24h : float
        Demand for this zone 24 time-steps ago (= same hour yesterday).
    demand_lag_168h : float
        Demand for this zone 168 time-steps ago (= same hour last week).

    Note: The first n rows for each zone will be NaN for a lag of n.
    Drop these rows after calling this function, or handle them in your
    training pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Hourly demand DataFrame returned by aggregate_to_hourly_demand().
        Must be sorted by zone and hour before calling this function.
        Must contain columns: zone_col, target_col.
    zone_col : str, optional
        Name of the zone identifier column. Default: 'PULocationID'.
    target_col : str, optional
        Name of the demand column to lag. Default: 'demand'.

    Returns
    -------
    pd.DataFrame
        DataFrame with three new lag columns appended.

    Examples
    --------
    >>> hourly = hourly.sort_values(['PULocationID', 'hour'])
    >>> hourly = add_lag_features(hourly, zone_col='PULocationID', target_col='demand')
    >>> hourly[['PULocationID', 'hour', 'demand', 'demand_lag_1h']].head(10)
    """
    df = df.sort_values([zone_col, 'hour']).copy()
    for lag_hours, suffix in [(1, '1h'), (24, '24h'), (168, '168h')]:
        df[f'{target_col}_lag_{suffix}'] = (
            df.groupby(zone_col)[target_col].shift(lag_hours)
        )
    return df


def fill_lag_nans(
    df: pd.DataFrame,
    zone_col: str = 'PULocationID',
    target_col: str = 'demand',
) -> pd.DataFrame:
    """Fill NaN values in lag feature columns using zone and hour-of-day means.

    Lag features produced by add_lag_features() are NaN for the first n rows
    of each zone, where n is the lag size. The binding constraint is
    demand_lag_168h, which has no real historical data for the first 168 hours
    (7 days) of each zone's time series.

    Rather than dropping these rows, NaNs are filled with the mean demand for
    that zone at that specific hour of day, computed across all hours in the
    full time series. This is semantically meaningful: in the absence of last
    week's actual data, the best estimate of what a zone looked like at 8am
    is what that zone typically looks like at 8am.

    This approach is standard practice for time series burn-in periods. The
    filled rows carry less predictive signal than post-week-1 rows where true
    lag values are available, but they are valid training examples rather than
    wasted data.

    Fill strategy per column
    ------------------------
    demand_lag_1h : float
        NaN only in the first row per zone. Filled with zone + hour_of_day
        mean. Rare in practice since most zones have data at hour 0.
    demand_lag_24h : float
        NaN in the first 24 rows per zone (first calendar day). Filled with
        zone + hour_of_day mean, which is a reasonable estimate of the prior
        day's same-hour demand.
    demand_lag_168h : float
        NaN in the first 168 rows per zone (first 7 days). Filled with
        zone + hour_of_day mean — the best available estimate of last week's
        same-hour demand when true historical data does not exist.

    Parameters
    ----------
    df : pd.DataFrame
        Hourly demand DataFrame returned by add_lag_features().
        Must contain columns: zone_col, 'hour_of_day', and all three lag
        columns (demand_lag_1h, demand_lag_24h, demand_lag_168h).
    zone_col : str, optional
        Name of the zone identifier column. Default: 'PULocationID'.
    target_col : str, optional
        Name of the demand column used to compute fill means. Default: 'demand'.

    Returns
    -------
    pd.DataFrame
        DataFrame with no NaN values in the three lag columns.

    Examples
    --------
    >>> hourly = add_lag_features(hourly)
    >>> hourly = fill_lag_nans(hourly)
    >>> hourly[['demand_lag_1h', 'demand_lag_24h', 'demand_lag_168h']].isna().sum()
    demand_lag_1h      0
    demand_lag_24h     0
    demand_lag_168h    0
    dtype: int64
    """
    fill_values = df.groupby([zone_col, 'hour_of_day'])[target_col].transform('mean')

    for col in ['demand_lag_1h', 'demand_lag_24h', 'demand_lag_168h']:
        df[col] = df[col].fillna(fill_values)

    return df


def add_rolling_features(
    df: pd.DataFrame,
    zone_col: str = 'PULocationID',
    target_col: str = 'demand',
) -> pd.DataFrame:
    """Add rolling mean demand features, computed separately for each zone.

    ⚠️  COMMON BUG WARNING ⚠️
    Rolling features MUST be computed per zone using groupby, identical to the
    pattern required for lag features. Computing rolling means without groupby
    will bleed demand from one zone into adjacent zones in the sorted DataFrame.

    Correct pattern:
        df[target_col].rolling(n).mean()                              ← WRONG
        df.groupby(zone_col)[target_col].transform(
            lambda x: x.shift(1).rolling(n).mean()
        )                                                             ← CORRECT

    The shift(1) inside the lambda is intentional: it excludes the current
    hour from its own rolling window, preventing the model from seeing the
    value it is trying to predict.

    New columns added
    -----------------
    demand_rolling_3h : float
        Rolling mean of the past 3 hours of demand for this zone.
        Uses min_periods=1 so no NaN values are produced. The first 1-2 rows
        per zone use partial windows of whatever history exists.
    demand_rolling_168h : float
        Rolling mean of demand over the past 168 hours (7 days) for this zone.
        Uses min_periods=1 so no NaN values are produced. Rows before hour
        168 of each zone use partial windows — the mean grows more
        representative as history accumulates.

    Parameters
    ----------
    df : pd.DataFrame
        Hourly demand DataFrame after add_lag_features() has been called.
        Must be sorted by zone and hour.
        Must contain columns: zone_col, target_col.
    zone_col : str, optional
        Name of the zone identifier column. Default: 'PULocationID'.
    target_col : str, optional
        Name of the demand column. Default: 'demand'.

    Returns
    -------
    pd.DataFrame
        DataFrame with two new rolling feature columns appended.

    Examples
    --------
    >>> hourly = add_rolling_features(hourly, zone_col='PULocationID',
    ...                                target_col='demand')
    >>> hourly[['PULocationID', 'hour', 'demand',
    ...         'demand_rolling_3h', 'demand_rolling_168h']].head(10)
    """
    df = df.sort_values([zone_col, 'hour']).copy()
    # min_periods=1 allows rolling means to compute on partial windows.
    # During the first 168 hours of each zone's time series, there is no
    # full week of history available. Rather than returning NaN and losing
    # that data, min_periods=1 uses however many rows exist — 1 hour in,
    # it averages 1 row; 50 hours in, it averages 50 rows; hour 168+
    # produces a true 7-day rolling mean. This is standard practice for
    # time series burn-in periods and produces no NaN values.
    df[f'{target_col}_rolling_3h'] = df.groupby(zone_col)[target_col].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    ).fillna(0)
    df[f'{target_col}_rolling_168h'] = df.groupby(zone_col)[target_col].transform(
        lambda x: x.shift(1).rolling(168, min_periods=1).mean()
    ).fillna(0)
    return df
