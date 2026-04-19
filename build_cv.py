"""
build_cv.py — Time-series cross-validation for DemandCast
==========================================================
Runs TimeSeriesSplit cross-validation on the best-performing model from
Part 2 (RandomForestRegressor, n_estimators=100, random_state=42) and logs
per-fold + summary metrics to MLflow under the experiment "DemandCast".

Usage (from project root with .venv active):
    python build_cv.py

Prerequisites:
    1. MLflow server running:   mlflow server --host 0.0.0.0 --port 8080
    2. Features built:          python build_features.py
    3. Training completed:      python build_train.py
"""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.cv import time_series_cv, DATA_PATH, TEST_CUTOFF, TARGET
from src.features import FEATURE_COLS


# ---------------------------------------------------------------------------
# Load features and restrict to train + val (test set stays sealed)
# ---------------------------------------------------------------------------
df = pd.read_parquet(DATA_PATH)
df = df[df["hour"] < TEST_CUTOFF].sort_values("hour").reset_index(drop=True)

X = df[FEATURE_COLS]
y = df[TARGET]


# ---------------------------------------------------------------------------
# Cross-validate the best model from Part 2 — Random Forest won on val_mae.
# We reuse the exact hyperparameters logged in the train.py run so the CV
# numbers are directly comparable to the validation-set numbers in MLflow.
# ---------------------------------------------------------------------------
results = time_series_cv(
    model=RandomForestRegressor(n_estimators=100, random_state=42),
    X=X,
    y=y,
    n_splits=5,
    run_name="cv_random_forest_100est",
)

mean_mae = results["mae"].mean()
std_mae  = results["mae"].std()
print(f"\nCV MAE: {mean_mae:.2f} ± {std_mae:.2f}")


# ---------------------------------------------------------------------------
# Interpretation — what the std of MAE across folds tells us about stability
# ---------------------------------------------------------------------------
# (Professor requested this live as a code comment rather than a markdown
# cell — same expectation as the split-rationale comment in train.py.)
#
# A LOW standard deviation relative to the mean (roughly < ~10% of mean MAE)
# means the model's error is consistent across time windows. Whether we
# train on the first week or the first four, the next block's error looks
# similar. That signals a stable model: it is not overly dependent on which
# slice of history happened to be in the training fold, which is what we
# want before committing to hyperparameter tuning in Week 4.
#
# A HIGH standard deviation relative to the mean means per-fold error is
# swinging around. Possible causes: the model is over-fitting to short-term
# patterns that don't recur, or the underlying demand distribution is
# shifting over the month (holiday week vs. regular week, weather events,
# etc.). In that case a single validation-split MAE would be misleading —
# the true uncertainty around the reported number is wider than one split
# implies, and tuning decisions made from a single split could be noise.
#
# Observed: CV MAE 4.33 ± 0.66 — std is ~15% of mean, moderately stable
# (Fold 0 MAE 5.49 is the outlier; Folds 1–4 cluster tightly at ~3.9–4.3).

# the previous answer was AI generated, I liked what it said, so I kept it as is.
# the MAE for all of our folds were very similar. the first was 5.49, the second was 4.02,
# the third was 4.29, the fourth was 3.89 and the fifth was 3.98. Because of this, I think
# the data was prepared very well, and the sample does not necessarily dictate how accurate
# the model is. This is different than what we were seeing in class. So I think I may have
# separated the data in a more coherent way. I added more features, and I think this could
# Be contributing to the accuracy. I feel confident about the random forest that I trained,
# and after the CV I feel even more confident. The distrubution of the MAE between folds
# shows that the model is stable for the most part. And that anything beyond this should
# perform in a similar way. We can also see that it performs stronger with each training,
# As it gets more data to train on, the better it does. Which is also a sign of our model
# learning the patterns in the data, not just overfitting.