"""
prep_geojson.py — Download the NYC TLC taxi zone GeoJSON for the dashboard map.

Pulls from NYC OpenData (dataset id d3c5-ddgc, "NYC Taxi Zones") and writes
to data/dashboard/nyc_taxi_zones.geojson. Each feature has a LocationID
property that matches the model's PULocationID column.

Run from the project root with .venv active:
    python app/api/prep_geojson.py

Writes:
    data/dashboard/nyc_taxi_zones.geojson
"""
from pathlib import Path
import requests

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = REPO_ROOT / "data" / "dashboard"
OUT = OUT_DIR / "nyc_taxi_zones.geojson"

URL = "https://data.cityofnewyork.us/api/geospatial/8meu-9t5y?method=export&format=GeoJSON"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Fetching {URL} ...")
    response = requests.get(URL, timeout=60)
    response.raise_for_status()
    OUT.write_bytes(response.content)
    print(f"Wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
