"""
Runtime accessors for the historical averages lookup table built by
prep_lookups.py, plus the zone metadata pulled from the GeoJSON.
"""
import json
from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
LOOKUP_PATH = REPO_ROOT / "data" / "dashboard" / "zone_hour_dow_averages.parquet"
GEOJSON_PATH = REPO_ROOT / "data" / "dashboard" / "nyc_taxi_zones.geojson"

_lookup: pd.DataFrame | None = None
_zone_meta: dict[int, dict] | None = None


def get_lookup() -> pd.DataFrame:
    global _lookup
    if _lookup is None:
        if not LOOKUP_PATH.exists():
            raise FileNotFoundError(
                f"{LOOKUP_PATH} not found. Run `python app/api/prep_lookups.py` first."
            )
        _lookup = pd.read_parquet(LOOKUP_PATH)
        _lookup = _lookup.set_index(
            ["PULocationID", "hour_of_day", "day_of_week"]
        ).sort_index()
    return _lookup


def get_zone_meta() -> dict[int, dict]:
    """Returns {LocationID: {name, borough}} for every zone in the GeoJSON."""
    global _zone_meta
    if _zone_meta is None:
        if not GEOJSON_PATH.exists():
            raise FileNotFoundError(
                f"{GEOJSON_PATH} not found. Run `python app/api/prep_geojson.py` first."
            )
        with open(GEOJSON_PATH) as f:
            gj = json.load(f)
        meta = {}
        for feat in gj["features"]:
            props = feat["properties"]
            # Current NYC OpenData dataset (8meu-9t5y) uses `locationid`;
            # older exports used `location_id` or `LocationID`. Try all.
            loc_id = int(
                props.get("location_id")
                or props.get("LocationID")
                or props.get("locationid")
            )
            meta[loc_id] = {
                "name": props.get("zone", f"Zone {loc_id}"),
                "borough": props.get("borough", "Unknown"),
            }
        _zone_meta = meta
    return _zone_meta


def active_zone_ids() -> list[int]:
    """Returns the sorted list of zone ids the model has historical data for."""
    return sorted(get_lookup().index.get_level_values("PULocationID").unique().tolist())
