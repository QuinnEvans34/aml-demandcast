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

Five concrete things to show in the Streamlit dashboard at `streamlit run app/dashboard.py`:

1. **Set sidebar inputs** to a known busy hour: `PULocationID=161` (Midtown Center), `hour_of_day=17`, `day_of_week=4` (Friday), `is_weekend=No`. Read the predicted demand from the `st.metric()` card.
2. **Switch to a known quiet hour** for the same zone: `hour_of_day=4` (4 AM), `day_of_week=6` (Sunday). Show the prediction drops sharply — the model has learned the time-of-day rhythm.
3. **Toggle the weekend flag** for the busy-hour case (Midtown Center 5 PM, weekend = Yes). Show how the prediction changes when the only thing that moves is the calendar context.
4. **Hover over each metric tooltip** so the audience sees the plain-language interpretation behind MAE, RMSE, R², MAPE, and MBE — the same five sentences from `notebooks/04_evaluation.md`.
5. **Show the bar chart of average hourly demand by hour-of-day.** This is the daily rhythm the model is learning from — a 6 AM build-up, a midday plateau, the 5–7 PM peak, and the late-night fall-off. The chart contextualizes why the model's predictions move the way they do.

---

## 5. Reflection

- **One thing that surprised me:** [CHOOSE ONE / EDIT]
  - "I expected a random train/val split to leak data dramatically. I tested it directly: random-split CV mean MAE was only 0.56 lower than temporal CV mean (3.77 vs 4.33). The bigger lesson was in the std — random CV had 0.05 std vs temporal CV's 0.66. Random isn't more reliable; it's just hiding the real week-to-week variance, not eliminating it."
  - "The tuning gain (0.025 MAE) was inside the CV's natural noise floor (0.66 std). I expected tuning to be a clear win and it wasn't — confirming that the binding constraint on this model is feature richness, not hyperparameter precision."

- **One thing I would do differently:** [CHOOSE ONE / EDIT]
  - "Add a weather signal early in the project. Demand swings hard with rain and cold; the lag features partly proxy for this but a clean weather-by-zone-hour join would probably move val MAE more than tuning did, with much less compute."
  - "Report mean ± std from the start, not single-shot validation MAE. The CV std (0.66) is the most important number for this dataset — it tells the operator the natural error bar around any headline metric — and I didn't surface it consistently until late."
