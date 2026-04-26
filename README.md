# DemandCast

Hourly NYC Yellow Cab pickup demand forecasting per pickup zone, end-to-end:
EDA → feature engineering → temporal-split training → MLflow tracking →
Optuna hyperparameter tuning → MLflow Model Registry → live React + FastAPI
dashboard.

Built for AML Project 1, Weeks 1–4. Random Forest model trained on
January 2025 NYC TLC data; production model registered as
`DemandCast v2 → Production`.

---

## Quickstart

You need three terminals running for the dashboard.

### Prereqs

- Python 3.10+ with the project `.venv` active (`source .venv/bin/activate`)
- Node.js 20+ and `npm`
- The repo cloned with the parquet data files present in `data/`

### Terminal 1 — MLflow tracking server

```bash
mlflow server --host 127.0.0.1 --port 8080
```

(If you've never run training locally before, you'll also need to seed the
registry by running, in order: `python build_features.py`, then
`python build_train.py`, then `python evaluate.py`, then
`python register_baseline.py`, then `python src/tune.py`. Most graders
will already have these from prior phases.)

### Terminal 2 — FastAPI backend

```bash
pip install -r app/api/requirements.txt
python app/api/prep_lookups.py     # one-time — builds the historical averages table
python app/api/prep_geojson.py     # one-time — downloads the NYC taxi zone GeoJSON
cp data/dashboard/nyc_taxi_zones.geojson app/web/public/nyc_taxi_zones.geojson
uvicorn app.api.main:app --port 8000
```

The backend loads `models:/DemandCast/Production` from MLflow at startup.
It exposes seven endpoints (zones, single predict, all-zones predict,
timeline, heatmap, model card, health) — see `app/api/main.py` for the
full surface.

### Terminal 3 — Next.js frontend

```bash
cd app/web
npm install
npm run dev
```

Open <http://localhost:3000>.

---

## What's in the dashboard

- **Sidebar controls** — zone (searchable), hour-of-day (slider), day-of-week
  (pill selector), weekend toggle. The four user inputs the rubric requires.
- **Hero prediction** — predicted pickups for the selection, plus the
  validation MAE as an explicit `± error band` (3.86 trips/hour).
- **Stat strip** — three small cards: percent vs typical demand for that
  (zone, hour, day), rank against all 171 active zones at the same time slot,
  and the error-band reminder.
- **Map tab** — NYC zones rendered from the TLC GeoJSON, color-graded by
  predicted demand for the current hour/day/weekend. Click a zone to switch
  the selection. The visual hook for live demos.
- **Timeline tab** — 24-hour forecast for the selected zone on the selected
  day-of-week, with the current hour marked.
- **Heatmap tab** — 24-row × 7-column "operational fingerprint" of the
  selected zone, colored by predicted demand.
- **Model card** — model version, registry stage, and the five validation
  metrics (MAE / RMSE / R² / MAPE / MBE) each with a plain-language tooltip
  for a non-technical audience.
- **`/compare` page** — pin two (zone, hour, day, weekend) combinations
  side-by-side with their own controls and timelines.

---

## How predictions are made

The Production model expects 11 features. The dashboard's user inputs cover
four of them (zone, hour-of-day, day-of-week, weekend). The remaining seven
are filled at predict-time by the backend:

- `is_rush_hour`: derived (`1` if hour is 7, 8, 17, or 18 AND day-of-week
  is Mon–Fri, else `0`)
- `is_holiday`: defaulted to `0` (the dashboard does not surface a holiday
  picker — see `notebooks/04_evaluation.md` for the design rationale)
- `PULocationID`: copied from the zone input
- `demand_lag_1h`, `demand_lag_24h`, `demand_lag_168h`,
  `demand_rolling_3h`, `demand_rolling_168h`: looked up from a per-(zone,
  hour, day-of-week) historical-average table built once by
  `app/api/prep_lookups.py` from `data/features.parquet`

Predictions therefore describe **typical demand for this combination of
inputs** rather than "demand right now." That tradeoff is documented in
`notebooks/04_evaluation.md` and called out in the dashboard's model card.

---

## Stack

- **Model**: scikit-learn `RandomForestRegressor`, tuned with Optuna; logged
  to MLflow with the full five-metric evaluation suite per Part 1
- **Data**: NYC TLC Yellow Taxi public records, January 2025, aggregated to
  hourly demand per pickup zone
- **Backend**: FastAPI + Pydantic, loads the Production model from the
  MLflow Model Registry once at startup, holds the historical averages
  parquet in memory
- **Frontend**: Next.js 14 (App Router) + TypeScript + Tailwind, with
  `react-leaflet` for the map, `recharts` for the timeline, and
  `lucide-react` for icons
- **Note on Streamlit**: this project uses Next.js + FastAPI in lieu of the
  default Streamlit dashboard, with explicit instructor permission. The
  rubric requirements (Production model loaded from MLflow Registry, four
  sidebar inputs, prominent prediction display, plain-language metric
  context, at least one visualization) are all satisfied by the dashboard
  above.

---

## Project structure

```
aml-demandcast/
├── app/
│   ├── api/              FastAPI backend
│   └── web/              Next.js frontend
├── data/
│   ├── features.parquet           feature matrix produced by build_features.py
│   ├── yellow_tripdata_2025-01.parquet   raw TLC data
│   └── dashboard/                 generated artifacts (gitignored)
├── docs/
│   └── presentation_outline.md    Part 4 deliverable
├── notebooks/
│   └── 04_evaluation.md           Part 1 + Part 2 deliverable
├── src/
│   ├── features.py       feature engineering + FEATURE_COLS
│   ├── train.py          single-model training + train_and_log
│   ├── cv.py             time-series cross-validation
│   └── tune.py           Optuna study + retrain_and_register
├── build_features.py     pipeline runner
├── build_train.py        runs all three Week 3 baseline models
├── build_cv.py           runs cross-validation on the best model
├── evaluate.py           Part 1 — five-metric baseline evaluation
├── evaluate_tuned.py     Part 1 — five-metric tuned model evaluation (held-out)
├── register_baseline.py  one-time MLflow Registry seed (DemandCast v1 → Staging)
├── run_split_experiments.py   appendix — random-split methodology diagnostic
└── README.md             this file
```

---

## Where the rubric items live

| Rubric item | File / location |
|---|---|
| Five validation metrics + plain-language interpretations + MAPE edge case | `notebooks/04_evaluation.md` (Part 1 section) |
| `objective()` implementation, ≥15 trials, search space justification | `src/tune.py` lines 150–180 (search space with per-line comments); 15 trials at line 37 |
| Tuning comparison & reflection | `notebooks/04_evaluation.md` (Part 2 section, including Compute Cost and Test Set Note subsections) |
| Model registered as `DemandCast v2 → Production`, v1 in Staging | MLflow Model Registry — verify with the MLflow UI at `:8080` |
| Dashboard: Production model load + 4 sidebar inputs + `st.metric()`-equivalent + plain-language metrics + visualization | `app/api/main.py` (model load, endpoints) and `app/web/app/page.tsx` + `app/web/components/*` (UI) |
| 5-section presentation outline | `docs/presentation_outline.md` |

---

## Methodology appendix

`notebooks/04_evaluation.md` includes a one-time **Split Methodology
Diagnostic** appendix comparing temporal-split CV (the approach used
throughout) against random KFold and an alternate temporal validation
window. Documents that random-split val MAE drops by only ~0.56 (not the
order of magnitude initially feared), but that the per-fold std is ~13×
tighter under random splitting — which understates the deployment-relevant
uncertainty. Decision: temporal split remains primary; random split is on
file as evidence the methodology choice is not artificially penalizing the
model.
