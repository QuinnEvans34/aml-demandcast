# DemandCast — Model Evaluation

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

### Study Configuration
- Framework: Optuna
- Trials: 15
- CV strategy: TimeSeriesSplit (n_splits=5)
- Objective: minimize mean CV MAE on the training set

### Best Parameters
- `n_estimators`: 400
- `max_depth`: 25
- `min_samples_leaf`: 8
- `min_samples_split`: 10
- `max_features`: 0.5
- (fixed) `random_state`: 42
- (fixed) `n_jobs`: -1

Best trial: #1
Best mean CV MAE: 4.8536

### Results Comparison

| Model                | val_mae | val_rmse | val_r2 | val_mape       | val_mbe |
|----------------------|---------|----------|--------|----------------|---------|
| Baseline RF (Week 3) | 3.8840  | 11.3892  | 0.9650 | 42.00%         | 0.1339  |
| Tuned RF             | 3.8593  | 11.3976  | 0.9649 | 41.17%         | -0.0095 |

> **Note on the Tuned RF row.** The five Tuned RF metrics above come from
> the **Optuna trial #1 MLflow run**, which was trained on the train split
> only (`hour < 2025-01-22`) and evaluated on the held-out val split
> (`2025-01-22 ≤ hour < 2025-02-01`). The model registered as
> `DemandCast v2 → Production` is a different artifact: `retrain_and_register`
> re-fits the same tuned hyperparameters on **train + val combined** before
> deployment, which is the standard practice for shipping a production model.
> Evaluating that retrained-on-everything artifact against the val split
> would produce in-sample training metrics rather than generalization
> estimates, so the val numbers reported here intentionally come from the
> trial-time held-out fit.

### Did Tuning Help?
Tuning reduced val_mae from 3.8840 → 3.8593, a gain of 0.0247 (≈0.64%),
while val_rmse and val_r2 essentially did not move (11.39 → 11.40 and
0.9650 → 0.9649). That 0.0247 improvement is an order of magnitude
smaller than the Week 3 CV std of 0.66, so it sits well inside the CV
noise floor — in other words, the tuned model is not meaningfully
better than the baseline on this validation slice; the difference is
within the run-to-run variance we already know this pipeline has.
The convergence trajectory supports this reading: the best trial (#1)
landed its CV MAE at 4.85 after two samples and no later trial beat it
across 13 more attempts. Running more trials of the same search would
most likely keep bouncing around this plateau; a meaningful next step
would be a different lever (different features, a different model
class, or widening the search beyond RF hyperparameters) rather than
more Optuna budget.

### Compute Cost

The 15 trials took roughly 5–10 minutes on this machine and moved val_mae by 0.0247. That is a poor return: a comparably-sized improvement could have come from a single feature-engineering iteration (e.g. adding a weather signal or a holiday-neighborhood feature) at a fraction of the compute. For this search space on this dataset, the honest answer to "was tuning worth the cost" is **no** — not because tuning is pointless, but because RF hyperparameters are not the binding constraint on this pipeline's performance.

### Test Set Note

The `retrain_and_register` step evaluated the tuned model on the sealed test window and logged `test_mae = 23.6083`. This number is **not** a reliable test-set performance estimate: `data/features.parquet` ends at `2025-02-01 00:00`, so the test window (`hour >= 2025-02-01`) contains exactly one timestamp — 171 zones × 1 hour of data. That single off-peak hour happens to include a few high-outlier zones, which inflates MAE relative to the validation error on the full 10-day val window. A proper test-set evaluation requires extending the raw data to cover the full February 1–7 span; that is Week 5+ work and explicitly sealed until then. The number is kept for auditability but should not be interpreted as the tuned model's generalization performance.

### MLflow Registry
- `DemandCast` v1: Staging — Week 3 RandomForestRegressor baseline (n_estimators=100, random_state=42)
- `DemandCast` v2: Production — tuned RandomForestRegressor from Optuna trial #1

---

## Tuned Model — Plain-Language Metrics

These interpretations describe the tuned Production model (DemandCast v2)
on the same validation window as the baseline. Same audience as Part 1 —
a taxi operations manager, not a data scientist. These sentences are the
copy that will feed the Streamlit dashboard's metric tooltips in Part 3.

| Metric | Value | Plain-Language Interpretation |
|--------|-------|-------------------------------|
| MAE    | 3.86 | After tuning, predictions are off by 3.86 trips per hour per zone on average — a 0.0247-trip improvement over the 3.88 baseline. For a dispatcher, the tuned model and the baseline are operationally interchangeable: the improvement is smaller than the run-to-run variance the CV study already measured (std 0.66), so driver scheduling decisions should not be rebalanced on the strength of this change alone. |
| RMSE   | 11.40 | The typical error on the biggest misses is 11.40 trips, essentially unchanged from the baseline's 11.39. Tuning did not materially shrink the penalty on large busy-zone misses — peak-hour busy-zone forecasting remains the biggest operational risk to plan around, regardless of which model version is live. |
| R²     | 0.9649 | The tuned model explains 96.5% of the variation in demand across all zones and hours — indistinguishable from the baseline's 96.5%. Tuning did not unlock new signal; the fraction of demand variability the model cannot capture is the same before and after. |
| MAPE   | 41.2% | On the 22,767 of 41,040 validation rows with strictly positive demand, predictions are off by 41.2% on average, lower than the baseline's 42.0%. Zero-demand rows are excluded for the same reason as in Part 1 — a percentage error against zero is undefined and would blow up the mean. |
| MBE    | -0.0095 | The tuned model under-predicts by an average of 0.0095 trips per zone per hour, closer to zero than the baseline's positive bias of 0.13. A negative bias means the model would lead to understaffing if used for driver scheduling without adjustment. |

These five sentences are designed to be droppable directly into Streamlit
`st.metric()` tooltips or `help=` text in Part 3.

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