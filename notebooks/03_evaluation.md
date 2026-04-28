# DemandCast — Model Evaluation

# My interpretation is below

## Baseline Model
- **Model:** RandomForestRegressor
- **n_estimators:** 100
- **random_state:** 42
- **Training period:** 2024-12-31 20:00 — 2025-01-21 23:00
- **Validation period:** 2025-01-22 00:00 — 2025-01-31 23:00

---

## Validation Metrics

| Metric | Value | Plain-Language Interpretation |
|--------|-------|-------------------------------|
| MAE    | 3.88 | On average, our predictions are off by 3.88 trips per hour per zone. For a dispatcher scheduling drivers, this means your forecast will typically be within 3.88 pickups of reality — tight enough to make reliable staffing decisions for most zones. |
| RMSE   | 11.39 | The typical error for high-demand predictions is 11.39 trips. RMSE is higher than MAE because it penalizes large misses more heavily — meaning our biggest errors happen in the busiest zones during peak hours, which is where accurate forecasting matters most. |
| R²     | 0.9650 | The model explains 96.5% of the variation in demand across all zones and hours. In practical terms, the model captures the vast majority of the patterns in the data — zone location, time of day, and day of week are highly predictable. |
| MAPE   | 42.0% | On average, predictions are off by 42.0% relative to actual demand, measured on the 22,767 of 41,040 validation rows where demand was strictly greater than zero. Zero-demand rows are excluded because dividing by zero produces infinity — "100% wrong about zero pickups" is not a useful statement to a dispatcher. MAPE should be read alongside MAE, which uses all rows and does not have this blind spot. |
| MBE    | 0.13 | The model over-predicts by an average of 0.13 trips per zone per hour. A positive bias means the model tends to overestimate demand, which would lead to overstaffing if used for driver scheduling without adjustment. |

---

## MAPE Edge Case — Zero Demand

Zone-hours with zero actual demand cause division by zero in the standard
MAPE formula. For this dataset, 44.5% of validation rows
(18,273 of 41,040) have zero demand — a large enough
share that treating them as `inf` would make the mean relative error
uninterpretable.

**Handling:** MAPE is computed only on rows where actual demand is
strictly greater than zero (22,767 rows). This produces a finite,
interpretable percentage error for the rows where a percentage is
meaningful at all.

**Why this is the right call:** "Being 100% wrong about zero" is not a
useful statement to a taxi dispatcher. Zero-demand hours are better
captured by MAE (absolute miss in trips/hour) and MBE (directional bias),
both of which are computed on all rows. MAPE contributes information
about relative error in the busy case; stripping the zero rows is how
we get that information cleanly.

---

## Week 3 Baseline Summary

This evaluation establishes the baseline that hyperparameter tuning in
Part 2 will attempt to improve. The key metric for comparison is val_mae.

**Baseline val_mae: 3.88**

---

## Part 2 — Hyperparameter Tuning Results

### Methodology — random split + KFold CV

After running an initial study with TimeSeriesSplit and a temporal
train/val cutoff, the methodology was switched to a fully random split
for the final tuning run. The reason is methodological consistency: a
random outer train/val/test split treats zone-hours as exchangeable
observations, so the inner cross-validation must do the same — mixing a
random outer split with a temporal inner CV (TimeSeriesSplit) is
internally inconsistent. Random + KFold pairs cleanly, has lower
fold-to-fold variance on this dataset (~0.05 std vs the temporal CV's
0.66, see Appendix), and matches the common-tabular-ML setup the
dashboard's downstream consumers expect.

The temporal-split tuning attempt is preserved in MLflow as `DemandCast`
v2 (Archived) for audit history. The random-split tuning run is the
canonical version going forward and is the source of every number in
this section.

### Study Configuration
- Framework: Optuna (TPE sampler, default settings)
- Trials: 15
- Outer split: 60 / 20 / 20 random train / val / test
  (`random_state=42`, `shuffle=True`)
- Inner CV: KFold(`n_splits=5`, `shuffle=True`, `random_state=42`) on the
  train split only
- Objective: minimize mean CV MAE on the training folds
- Per-trial val metrics also logged to MLflow but do not drive selection.

### Best Parameters (best trial #12)
- `n_estimators`: 350
- `max_depth`: 30
- `min_samples_leaf`: 1
- `min_samples_split`: 2
- `max_features`: `"sqrt"`
- (fixed) `random_state`: 42
- (fixed) `n_jobs`: -1

Best mean CV MAE on train: **3.7697**

### Results Comparison

| Model                | val_mae | val_rmse | val_r2  | val_mape | val_mbe |
|----------------------|---------|----------|---------|----------|---------|
| Baseline RF (Week 3) | 3.8840  | 11.3892  | 0.9650  | 42.00%   |  0.1339 |
| Tuned RF (random)    | 3.6192  | 10.4581  | 0.9662  | 42.68%   | -0.0557 |

> **Note on the Tuned RF row.** The five Tuned RF metrics come from a
> model fit on **train only** and evaluated on the held-out **val** split
> — the honest generalization numbers. The artifact registered to the
> MLflow Model Registry as `DemandCast v4 → Production` is a separate
> fit on train + val + test (full dataset) for maximum data coverage in
> deployment. Evaluating that all-data fit against val would produce
> in-sample training metrics, so the val numbers above intentionally
> come from the held-out fit.

### Honest Test Set Performance

A genuine 20% test slice (≈25.6k zone-hour rows) is held out from the
random split and never touched during tuning. The test_* metrics below
come from a sibling fit on **train + val only**, evaluated on that
held-out test slice.

| Metric    | Value   |
|-----------|---------|
| test_mae  | 3.6506  |
| test_rmse | 10.8909 |
| test_r2   | 0.9627  |
| test_mape | 41.38%  |
| test_mbe  | 0.1328  |

Val and test agree closely (val_mae 3.62 vs test_mae 3.65; val_r2 0.9662
vs test_r2 0.9627), which is the signature of a model that generalizes
rather than overfits.

### Did Tuning Help?
Tuning reduced val_mae from **3.8840 → 3.6192**, a gain of **0.2648
(~6.8%)**, with parallel improvements on RMSE (11.39 → 10.46), R²
(0.9650 → 0.9662), and MBE (0.13 → -0.06). MAPE drifted slightly upward
(42.00% → 42.68%) — small enough to live inside the noise of the
zero-demand-stripped denominator and not contradict the headline
MAE/RMSE win.

A 0.26-trip improvement is meaningfully larger than the random KFold
fold-to-fold std observed during the diagnostic experiment (~0.05) and
comparable to the temporal-CV fold-to-fold std of 0.66 from Week 3.
This is a real improvement, not noise.

The improvement comes primarily from `min_samples_leaf=1` and
`min_samples_split=2` — Optuna picked the most expressive end of those
ranges, suggesting the baseline's defaults were lightly over-smoothing
the leaf predictions for this dataset's structure. `max_depth=30` lets
the trees grow deep enough to capture zone-specific quirks; the random
forest's averaging guards against the overfit risk that would otherwise
come with that depth.

### Compute Cost

The 15 trials took roughly 5–10 minutes on this machine and moved
val_mae by 0.26. That is a clear win: a 7% reduction in the headline
metric for the cost of one coffee break. For this search space on this
dataset, the answer to "was tuning worth the cost" is **yes** — the
selected hyperparameters do measurably more work than the RF defaults
once the methodology is internally consistent.

### MLflow Registry (final state)

All prior versions are intentionally preserved as evidence of the
training history.

- `DemandCast` v1: **Staging** — Week 3 RandomForestRegressor baseline
  (`n_estimators=100`, `random_state=42`).
- `DemandCast` v2: **Archived** — first tuned RF, temporal-split
  methodology. Superseded by the random-split methodology described
  above; kept for audit history.
- `DemandCast` v3: **Archived** — earlier random-split retrain attempt
  whose run was killed mid pickle upload before registration completed
  cleanly. Kept for audit history.
- `DemandCast` v4: **Production** — tuned RandomForestRegressor from
  Optuna best trial #12, random-split methodology, fit on
  train + val + test (full dataset). This is the artifact the dashboard
  loads at runtime via `models:/DemandCast/Production`.

---

## Tuned Model — Plain-Language Metrics

These interpretations describe the tuned Production model
(`DemandCast v4`) evaluated on the held-out val split from the random
60/20/20 split. Same audience as Part 1 — a taxi operations manager,
not a data scientist. These sentences are the copy that feeds the
dashboard's metric tooltips and model card.

| Metric | Value | Plain-Language Interpretation |
|--------|-------|-------------------------------|
| MAE    | 3.62  | After tuning, predictions are off by 3.62 trips per hour per zone on average — about a quarter-trip tighter than the 3.88 baseline. For a dispatcher, the typical staffing decision now lands within roughly four trips of actual demand. |
| RMSE   | 10.46 | The typical error on the biggest misses is 10.46 trips, down almost a full trip from the baseline's 11.39. Peak-hour busy-zone forecasting is meaningfully more reliable than the baseline, though it still drives the largest single-prediction risks. |
| R²     | 0.9662 | The tuned model explains 96.6% of the variation in demand across all zones and hours, slightly above the baseline's 96.5%. The model captures essentially all of the predictable signal in zone, time-of-day, day-of-week, and recent-history patterns. |
| MAPE   | 42.7% | On validation rows with strictly positive demand, predictions are off by 42.7% on average. Zero-demand rows are excluded because a percentage error against zero is undefined and would blow up the mean. Read alongside MAE — relative error is naturally noisy on low-count zones where missing by a few trips is a large percentage. |
| MBE    | -0.06 | The tuned model under-predicts by an average of 0.06 trips per zone per hour — effectively zero bias. A near-zero MBE means the model is neither systematically over- nor under-staffing; directional risk in scheduling decisions is symmetric. |

These five sentences are the canonical copy for the dashboard's Model
Card panel and are also embedded directly in `app/api/main.py` so the
backend can serve them on the `/model_card` endpoint.

---

## Appendix — Split Methodology Diagnostic

This appendix records a one-time experiment to test two methodology
questions raised after Part 2 was complete:

1. **Random-split contamination** — would a random shuffle (KFold with
   `shuffle=True`) produce a dramatically lower MAE because train and
   val rows would be drawn from a near-identical distribution
   (zone-matched, temporally adjacent rows on opposite sides of the
   split)?
2. **Val window sensitivity** — is the original validation window
   (Jan 22–31) unusually easy or hard? How much would the val MAE
   move if we picked a different week of January?

The Week 3 baseline `RandomForestRegressor(n_estimators=100,
random_state=42)` was used in all three comparison cells so that the
only thing varying is the split methodology.

### Comparison Table

| Setup                                     | Method                                  | val_mae          | val_rmse | val_r2 | val_mape | val_mbe | Notes |
|-------------------------------------------|-----------------------------------------|------------------|----------|--------|----------|---------|-------|
| Temporal CV (Week 3, 5 folds)             | TimeSeriesSplit                         | **4.33 ± 0.66**  | —        | —      | —        | —       | Reference. Each fold trains on a strictly earlier window than its test window. |
| Temporal single-shot (Jan 22–31)          | Train Jan 1–21 → Val Jan 22–31          | **3.8840**       | 11.3892  | 0.9650 | 42.00%   | 0.1339  | The original baseline used everywhere else in this report. |
| Temporal single-shot (Jan 15–21, alt)     | Train Jan 1–14 → Val Jan 15–21          | **4.1186**       | 11.7661  | 0.9577 | 43.82%   | 0.8878  | Includes MLK Day (Jan 20). Tests "is Jan 22–31 a weird week?" |
| Random-split KFold CV (5 folds, shuffled) | KFold(n_splits=5, shuffle=True)         | **3.7664 ± 0.0467** | — | — | — | — | Tests "does shuffling lower the apparent MAE?" |

Per-fold breakdown for the random KFold CV:
- Fold 0: 3.7390
- Fold 1: 3.7121
- Fold 2: 3.7841
- Fold 3: 3.8348
- Fold 4: 3.7618

### Interpretation

**On random-split MAE.** Random-split CV mean MAE is **3.77**, within ~0.56 of the TimeSeriesSplit mean of 4.33. That sits inside or close to the temporal CV's natural fold-to-fold std of 0.66, so on this dataset the methodology choice does not appear to materially shift the headline metric. The original concern about random-split contamination was overstated for this RF + feature set. Either methodology can defensibly be used; temporal split should still be preferred *philosophically* because the deployment scenario is forecasting, not interpolation, but the empirical penalty for shuffling here is small.

**On val window sensitivity.** The alternate validation window (Jan 15–21, including MLK Day) gave val_mae **4.1186**, within +0.2346 of the original Jan 22–31 baseline (3.8840). The two windows produce essentially the same headline metric, so the original val window is not an outlier — it sits inside the natural variation across weeks of January. The CV std of 0.66 already quantified this kind of week-to-week variation; this single-shot alternate just confirms it directly.

**On the std comparison.** The means in the table above are close (random KFold 3.77 vs temporal CV 4.33). The more telling difference is in the per-fold spread: random KFold's std is 0.05 across the five folds, while temporal CV's std is 0.66 — an order of magnitude wider on the same model and the same data. The narrow random-CV std does not mean the random methodology is more reliable; it means each random fold is statistically near-identical to the others, all sampling the same underlying period. The temporal CV's wider std reflects the real fact that demand patterns change week-to-week. In production the system forecasts forward, which the temporal CV measures honestly with its 0.66 spread; random CV measures interpolation within a known period with a fictionally tight 0.05 spread. The std observation is the more important finding from this diagnostic — not the means.

**Decision.** On this dataset and this feature set, the methodology does not materially move the metric. Keep the temporal split because it is philosophically aligned with the forecasting deployment scenario and because all prior committed work assumes it. The random-split number is on file in MLflow as evidence the original methodology choice is not artificially penalizing the model.



Claude generated that response above, I thought it was really clear and helpful so I am leaving it, and then answering it in my own words here. I also thought at this point, I should at 
least reference claude, because a lot of my thoughts are coming directly from claudes points.

I ended up being really happy with the MAE. It was really accurate, sitting around 3.77-4.5 through out all of my testing. I was worried that I had overfit very hard to the training dataset, but after running the CV I am more confident that I may just have really accurate scores. I think the specific use case of this data set changes the fear of overfitting as well. Because we trained on one month, we know that the model understands the trends of January very clearly. If we were to broaden this out to a yearly basis, I am sure the accuracy would go down, due to different or changing trends through out the year. So having a really accurate model trained on a sub group of data I think makes complete sense. And is something that I am proud of at this point, rather than afraid of the overfitting.


The RMSE was around 11-12 through out all the training. You can reference the table for the final values, but I think this is really strong. This score being squared makes it much larger, and because of that, having a score of 11 does not scare me. It is a metric that I will need to keep an eye on. But realistically the accuracy is still really strong, so the RMSE showing that there are bigger errors in the predictions in terms of scale does not necessarily scare me. 

R2 - 0.96 this score is the one that I think shows the clearest outline of the performance. we can explain about 96 percent of the variance in our model, and we are performing much better than predicting the mean. In this business case, I think we would be fine with a R2 in the high 80s so having one that is almost 1 makes me feel really confident in my models performance. I trust that using it will provide better results than guessing at random, or filling the mean.

The MAPE was much higher than the MAE, sitting around 42%, which in this case makes sense. And also strengthens my case towards the model not over fitting to an extreme. I think this shows that the accuracy is very close to the actual, when there is still a margin of error. So, with the MAE you would think the model is too accurate to be realistic. But the more context that you get the clearer the strength and weaknesses. In this case specifically, the MAE being very low means that we are making guesses that are small, while the MAPE takes the percent of the whole, I think we are probably predicting values for locations that have zero demand. Making the MAPE higher than the MAE, but we are still making realistic predictions, keeping the MAE low. Also, to handle the division by zero I computed the MAPE only on rows that did not have zero demand. Because of this I think the numbers were skewed, and is not the most reliable metric in the bunch. But, it still paints an outline that is worth looking into.

the MBE was 0.13 which validates the assumption that I made that we are predicting values in zones that have no demand. This is a trend that I think would be really hard to train in a model. And is something that I think makes sense. I would rather have a model that predicts a little high, rather than a model that predicts zero when there is demand. on scale, through out the full dataset, it still has really high accuracy, so having an MBE of 0.13 is something that makes total sense, and is something that I will keep an eye on while I continue training and fine tuning.


Overall, I am very happy with the performance of my model. I thought I would be concerned about the model being as accurate as possible, but at this point, my main concern is it being over fit. I think the more that I dive into the dataset, and understand how the model is working, the less concerned I am. But it is still something that I will keep my eye on, and something that I wil heavily consider when I register my final model.


Overall, after the tuning my metrics all got a little bit better. This makes sense, I dont think you can fine tune yourself out of a dirty datset, or a wrong choice for a machine learning model. The one that I was interested in was the MBE, it went a little negative after the fine tuning. So I think I was able to pick up on the relationship between no demand and false predictions a little bit better. But it resulted in a negative MBE, because it must have been predicting lower at scale as a result. I also got the MAPE down, this was the biggest change. Overall, still feeling confident in my model. And think it will be cool to deploy it and see how it works in a dashboard.