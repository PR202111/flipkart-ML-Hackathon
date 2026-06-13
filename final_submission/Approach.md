# Traffic Demand Prediction - Solution Architecture

**Final Evaluation Metric:** $R^2$ Score: **0.915409**

## Overview

This Zip contains the solution for predicting continuous traffic demand across various geographic zones and times. The final architecture relies on an ensemble of **LightGBM** and **CatBoost** regressors, heavily supported by robust spatial-temporal feature engineering and Out-Of-Fold (OOF) target encoding.

The approach successfully normalizes the heavily right-skewed target variable, extracts hierarchical spatial features from encrypted geohashes, and optimizes high-cardinality categorical data natively within the gradient-boosting frameworks.

---

## 1. Feature Engineering

The raw data lacked explicit continuous temporal and spatial variables. The `preprocess` function was constructed to uncover these latent relationships.

### A. Geospatial Processing

* **Geohash Decoding:** The base32 `geohash` strings were decoded into exact float `latitude` and `longitude` coordinates using a custom binary interval mapping function. This localized the data into a physical coordinate space.
* **Geohash Hierarchy:** Extracted variable-length prefixes (`geo3`, `geo4`, `geo5`, `geo6`) from the original geohash. This allowed the models to learn traffic patterns at different levels of "zoom" (from macro-regions down to micro-neighborhoods).

### B. Temporal Extraction

* **Cyclic Time Variables:** Raw `timestamp` (e.g., "14:30") was split into `hour`, `minute`, and `total_minutes`. To maintain the continuous loop of time (where 23:59 is next to 00:00), `total_minutes` was transformed using `Sine` and `Cosine` waves.
* **Peak Hour Flags:** Created binary flags for `is_morning_peak` (07:00 - 09:00) and `is_evening_peak` (17:00 - 20:00) to isolate rush hour dynamics.
* **Day of Week:** Extracted a cyclical `day_of_week` by computing `day % 7`.

### C. Imputation & Categorical Handling

* `Temperature` NaNs were filled smartly using the median temperature grouped by the specific `day` and `hour`. Remaining missing values defaulted to the global median.
* `RoadType` and `Weather` missing values were filled with the explicit class `'Unknown'`.
* `LargeVehicles` and `Landmarks` were mapped to binary integers (`1`/`0`).

---

## 2. Out-of-Fold (OOF) Target Encoding

Because the geohash hierarchy (`geo3` through `geo6`) contained high-cardinality data, standard One-Hot Encoding would have created a massive, sparse matrix. Instead, **Target Encoding** was used.

To strictly prevent data leakage, a **5-Fold Cross-Validation** approach was implemented during the encoding phase:

1. The training data was split into 5 folds.
2. For each validation fold, the `mean` and `median` demand for every geohash prefix was calculated using *only* the remaining 4 training folds.
3. This was done for all levels of the hierarchy (`geo3_mean`, `geo4_median`, etc.), alongside a feature that counted the frequency of appearances (`_count`).

---

## 3. Target Transformation (Log-Space)

Exploratory Data Analysis revealed that the raw `demand` target was heavily right-skewed. To ensure the RMSE-based loss function optimized effectively without over-penalizing outliers, the target was log-transformed prior to training:

```python
y_train_log = np.log1p(train_df['demand'])
```

During the prediction phase, the outputs were converted back to the original space using `np.expm1()`.

---

## 4. Modeling Strategy

The data was modeled using two state-of-the-art Gradient Boosted Decision Tree (GBDT) frameworks. Both models were trained inside a 5-Fold CV loop to generate robust predictions. Categorical variables were cast to the `category` datatype to utilize the algorithms' native categorical splitting logic.

### Model 1: LightGBM (`LGBMRegressor`)

* **Objective:** Regression (RMSE)
* **Hyperparameters:** `learning_rate`: 0.05, `num_leaves`: 127, `max_depth`: 10, `feature_fraction`: 0.8
* **Training:** Used early stopping at 150 rounds with up to 4000 estimators.

### Model 2: CatBoost (`CatBoostRegressor`)

* **Objective:** Regression (RMSE)
* **Hyperparameters:** `learning_rate`: 0.03, `depth`: 8, `iterations`: 5000
* **Training:** Explicitly passed the `cat_features` list to leverage CatBoost's superior handling of high-cardinality categorical data.

---

## 5. Ensembling & Blending

Rather than relying on a single algorithm, the final predictions were a weighted average of the LightGBM and CatBoost out-of-fold test predictions.

A dynamic blend search loop tested weights from `0.0` to `1.0` in increments of `0.05`. By evaluating against the holdout set, the optimal blend was discovered. The final submission was calculated as:

$$Final Prediction = (0.585 \times LightGBM) + (0.415 \times CatBoost)$$
