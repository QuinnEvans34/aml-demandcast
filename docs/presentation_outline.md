# DemandCast — Project 1 Presentation Outline

This is a preparation tool, not slides. Each numbered section is one talking-point block for the live presentation.

---

## 1. Problem

- **DemandCast forecasts hourly taxi demand for each NYC Yellow Cab pickup zone.** Given a (zone, hour) pair plus calendar context, the model predicts how many pickups will happen.
- **Audience: a taxi operations manager** who decides where to position drivers ahead of demand spikes.
- **Why it matters:** under-staffed zones lose revenue (riders wait, switch to Uber/Lyft); over-staffed zones waste fuel and driver time. Even a small forecasting edge over historical averages translates to real dollars across thousands of zone-hours per day.
- **Scope of this prototype:** January 2025 NYC Yellow Cab data — 128,079 hourly observations across 171 active pickup zones. The model predicts one zone-hour at a time.

---

## 2. Data & Features

- **Source:** NYC TLC Yellow Taxi public trip records, January 2025 (`yellow_tripdata_2025-01.parquet`).
- **Aggregation:** raw trip-level rows are rolled up to **hourly demand per pickup zone** (count of trips for each `PULocationID × hour` cell).
- **11 model features:**
  - Time signals: `hour_of_day`, `day_of_week`, `is_weekend`, `is_rush_hour`, `is_holiday`
  - Zone identity: `PULocationID`
  - Demand history (lags): `demand_lag_1h`, `demand_lag_24h`, `demand_lag_168h`
  - Recent trend (rolling means): `demand_rolling_3h`, `demand_rolling_168h`
- **Most important EDA finding:** **46% of zone-hours have zero demand.** Demand is concentrated in Manhattan, JFK, and LGA; the long tail of outer-borough low-volume zones drives most of the row count but very little of the actual trips. This shaped two downstream decisions: (a) MAE is the primary metric (every row counts equally), and (b) MAPE is computed only on non-zero rows because dividing percentage error by zero would blow up the mean and tell a dispatcher nothing useful.

---

## 3. Model

- **Models tried:** Linear Regression, Ridge (α=1), Random Forest baseline (n_estimators=100), Optuna-tuned Random Forest (15-trial study over `n_estimators`, `max_depth`, `min_samples_leaf`, `min_samples_split`, `max_features`).
- **Winner:** the tuned Random Forest, registered as **DemandCast v2 → Production** in the MLflow Model Registry.
- **Best validation MAE: 3.86 trips per hour per zone.**
- **Plain-language version:** "On any given hour at any given pickup zone, our forecast is on average within about 4 pickups of the real number. That's tight enough to make confident staffing decisions for most zones."
- **Honest framing on the tuning gain:** tuning lowered val MAE by 0.025 (3.88 → 3.86). That sits inside the cross-validation's natural fold-to-fold std of 0.66, so the gain is real but not statistically meaningful — the tuned model is operationally identical to the baseline. What tuning actually bought was confidence that the search space had been explored. The binding constraint on this pipeline is feature richness, not hyperparameter precision.

---

## 4. Demo (planned for the live presentation)

Five concrete things to show in the Next.js + FastAPI dashboard. Start it with `uvicorn app.api.main:app --port 8000` (backend) and `npm run dev` from `app/web/` (frontend), then open <http://localhost:3000>:

1. **Set sidebar inputs** to a known busy hour: zone `Midtown Center`, hour slider at `5 PM`, day pill `Fri`, weekday selected. The hero card shows ~496 predicted pickups with the `± 4` validation MAE band next to it. The "vs typical" stat below shows it tracking the historical mean for that slot.
2. **Switch the hour slider to 4 AM and the day pill to Sun** for the same zone. The hero number drops sharply — the model has learned the time-of-day and weekday-vs-weekend rhythm. The "vs typical" stat flips direction and color.
3. **Open the Map tab** and show NYC zones color-graded white → blue by predicted demand for the current selection. Click any other zone on the map to switch the selection live; the hero, stat strip, and timeline all update together. The map is the strongest visual hook for a non-technical audience.
4. **Open the Heatmap tab** to show the 24×7 "operational fingerprint" for the selected zone — every hour-of-day × day-of-week cell colored by predicted demand. Switch to an outer-borough zone to contrast a dense Manhattan pattern with a sparse one. Makes the 46% zero-demand finding from EDA visible at a glance.
5. **Hover the model card metrics at the bottom** to surface the plain-language tooltips for MAE / RMSE / R² / MAPE / MBE — the same five sentences from `notebooks/04_evaluation.md`. Optionally jump to `/compare` to pin two zones side-by-side, which makes the "where should I send drivers next" question concrete with two live predictions.

---

## 5. Reflection

- **One thing that surprised me:** [CHOOSE ONE / EDIT]
  - "I expected a random train/val split to leak data dramatically. I tested it directly: random-split CV mean MAE was only 0.56 lower than temporal CV mean (3.77 vs 4.33). The bigger lesson was in the std — random CV had 0.05 std vs temporal CV's 0.66. Random isn't more reliable; it's just hiding the real week-to-week variance, not eliminating it."
  - "The tuning gain (0.025 MAE) was inside the CV's natural noise floor (0.66 std). I expected tuning to be a clear win and it wasn't — confirming that the binding constraint on this model is feature richness, not hyperparameter precision."

- **One thing I would do differently:** [CHOOSE ONE / EDIT]
  - "Add a weather signal early in the project. Demand swings hard with rain and cold; the lag features partly proxy for this but a clean weather-by-zone-hour join would probably move val MAE more than tuning did, with much less compute."
  - "Report mean ± std from the start, not single-shot validation MAE. The CV std (0.66) is the most important number for this dataset — it tells the operator the natural error bar around any headline metric — and I didn't surface it consistently until late."
