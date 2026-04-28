# DemandCast — Project 1 Presentation Outline
# Better notes will be provided during the presentation. 

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
- **Winner:** the tuned Random Forest, registered as **DemandCast v4 → Production** in the MLflow Model Registry. Earlier versions (v1 baseline in Staging, v2 first temporal-split tuning attempt, v3 interrupted retrain) are intentionally preserved in the registry as evidence of the training history.
- **Best validation MAE: 3.62 trips per hour per zone.** Held-out test MAE on a never-seen 20% slice is 3.65 — val and test agree closely, which is the signature of a model that generalizes rather than overfits.
- **Plain-language version:** "On any given hour at any given pickup zone, our forecast is on average within about 4 pickups of the real number. That's tight enough to make confident staffing decisions for most zones."
- **Honest framing on the tuning gain:** tuning lowered val MAE by 0.26 (3.88 → 3.62), a ~6.8% improvement, with parallel gains on RMSE (11.39 → 10.46) and MBE (0.13 → -0.06 — effectively zero bias). The improvement is several times larger than the random-CV fold-to-fold std (~0.05), so it's a real win, not noise. The methodology change that unlocked this — switching from a temporal split + TimeSeriesSplit pairing to a fully random split + KFold pairing for internal consistency — was as important as the hyperparameter search itself.

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

- **One thing that surprised me:** 
I was really suprised with how good my dashboard turned out. I ended up using next.js instead of streamlit, which was something I asked if I could do on thursday, and I was really happy with the results. I have gotten really used to making streamlit apps, so it was cool to sort of push myself and try to make a quality final product. I was suprised with how it turned out, and how easy it ended up being. With the help of claude and codex I was able to finish it up in a couple hours. I have worked with .next before, so I was familiar with it, but I was really happy with the final outcome.

- **One thing I would do differently:** 
If I were to do one thing differently, I think that I would have really thoroughly worked on the ML and anything else in part 1 and 2 before moving on. I did everything that was required, but then jumped directly into the dashboard, because of this I had to spend some extra time and commits on github. I had different work on two branches, which ended up being fine because I just merged the work to main. But I also had files that were being changed inside my stashed files, so when I got into the final main, I had to do some work in git to make sure I was not committing files that were not necessary, and that I did not loose anything that was in my stashed changes.