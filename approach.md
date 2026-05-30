# Gridlock Hackathon: Spatial-Temporal Traffic Demand Forecasting
**Final Leaderboard Score:** ~90.0% R²  
**Local CV Score:** ~95.48% R²

## 1. The Core Intuition
The initial instinct for this problem was to treat it as a pure time-series forecasting task using rolling window lags (e.g., `lag_1`, `ewm_alpha_04`). However, this introduced a fatal **train-serve skew**. Because the test set requires predicting 87 consecutive timestamps in the future where intermediate ground-truth demand is `NaN`, any intra-day lag features would cascade into zeroes during inference, causing catastrophic performance drops.

To solve this, I pivoted the architecture from a sequential time-series model to a **Cross-Sectional Spatial-Temporal Regression Model**. Instead of asking the model "What happened 15 minutes ago?", I asked it "What is the historical expectation for this specific location, at this specific time, under these specific conditions?"

## 2. Data Architecture & Preprocessing
To ensure categorical encoding and statistical distributions matched perfectly between training and inference, I merged the `train` and `test` datasets during the preprocessing pipeline. 

* **Geohash Decoding:** Instead of treating geohashes as opaque categorical strings, I mathematically decoded them into exact `lat` and `lon` floating-point coordinates. This allowed the tree-based model to learn continuous spatial boundaries and geographic clusters natively.
* **Target Transformation (`np.log1p`):** Traffic demand data is heavily right-skewed (mostly quiet periods with sudden, massive rush-hour spikes). Training directly on raw values caused the model to over-index on outliers. Applying `np.log1p(demand)` stabilized the gradient updates and resulted in a massive boost to the R² score.

## 3. Feature Engineering (The Secret Sauce)
The success of this model came down to explicitly spoon-feeding mathematical expectations to the LightGBM trees so they could focus entirely on predicting the residuals.

### A. The Golden Feature: `demand_yesterday`
Since intra-day lags were off the table, I mapped the demand from exactly 24 hours prior (`day - 1`) for the exact same geohash and timestamp. This provided the model with a safe, highly correlated momentum feature that did not leak into the future.

### B. Spatial Spillover (The Neighbor Map)
Traffic congestion does not happen in isolated cells; it bleeds into adjacent areas. I wrote a custom geohash shifting algorithm that queried the exact geographical neighbors (North, South, East, West) for every geohash and calculated the `neighbor_mean_demand`. This taught the model how traffic flows spatially.

### C. Cyclical Time & Interaction Ratios
* **Time cyclicality:** Encoded `mins_of_day` into `time_sin` and `time_cos` to preserve the continuous nature of time (knowing that 23:59 and 00:01 are only 2 minutes apart, not 23 hours).
* **Relative Busyness:** Created division interaction features like `momentum_ratio = demand_yesterday / loc_mean` and `ts_vs_loc = ts_mean / loc_mean` to explicitly tell the tree if the current condition was "hotter" or "colder" than the historical baseline.

### D. Leak-Proof Out-Of-Fold (OOF) Target Encoding
Early iterations suffered from extreme target leakage when grouping by both `geohash` and `timestamp`. The model achieved a fake 99.6% local CV but failed on the leaderboard. 

To fix this, I implemented strict **OOF Geohash Target Encoding**. Inside the 5-Fold CV loop, the geohash historical mean was calculated *only* on the training fold and projected onto the validation fold with a smoothing factor (`SMOOTH = 10`). This prevented the model from memorizing the answer key while still extracting powerful categorical signals.

## 4. Model Architecture & Validation
The final model is an ensemble of 5 LightGBM regressors trained via **5-Fold Cross-Validation**. 

**Hyperparameter Strategy:**
* **Learning Rate (`0.015`):** Lowered to allow the model to slowly build up to ~4000 trees without overfitting.
* **Regularization (`lambda_l1=1.0`, `lambda_l2=2.0`):** Increased L1 and L2 penalties to force the model to ignore noisy data and drop useless features.
* **Tree Constraints (`num_leaves=48`, `max_depth=7`):** Kept the trees relatively shallow to prevent the model from memorizing exact training coordinates, forcing it to generalize the underlying traffic curves.

## 5. What Didn't Work
Just like tracking down edge cases in a complex competitive programming algorithm, a lot of trial and error went into finding what *didn't* work:
1. **Deep Log Transforms (`np.log(x + 0.005)`):** Attempting to stretch the lower bounds of the target variable resulted in gradients exploding on near-zero values. The model wasted thousands of trees trying to predict noise. `log1p` was the sweet spot.
2. **Absolute Time Features (`hour`, `minute`):** Feeding raw hours to a tree model caused Extrapolation Failure. If it only trained on morning data, it didn't know how to predict evening data. Converting time to cyclic sine/cosine waves fixed this.

## 6. Conclusion
By treating the problem as a cross-sectional spatial regression, ruthlessly eliminating data leaks, and utilizing strictly stable features (like `demand_yesterday` and OOF Target Encoding), the model achieved a robust and highly generalized 90% R² on unseen data.