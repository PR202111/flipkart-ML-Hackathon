# 🚦 Flipkart Gridlock Hackathon 2.0 - Traffic Demand Prediction

Welcome to our repository for the **Flipkart Gridlock Hackathon**! 

This repository isn't just a final codebase; it is a log of our entire journey, tracking our various attempts, iterations, and architectural pivots as we tackled the spatial-temporal traffic demand forecasting problem.

## 🧠 About This Repository

We built this repo to document our process. Predicting hyper-local traffic demand across different geohashes and timestamps is a notoriously noisy problem. Here, we are uploading all our tries—from our initial flawed time-series models to our final, leak-proof spatial regression architecture.

## 🚀 The Journey (Our Iterations)

* **Attempt 1: The Time-Series Trap** * *Approach:* Standard rolling windows and intra-day lag features (`lag_1`, `ewm`). 
  * *Result:* Suffered from massive train-serve skew. Lags collapsed during the 87-step future inference.
* **Attempt 2: The Golden Feature & The Leak**
  * *Approach:* Pivoted to a cross-sectional model using `demand_yesterday` and cyclical time encoding. Tried grouping by `[geohash, timestamp]` for historical means.
  * *Result:* Hit an artificial 99.6% Local CV R² due to target leakage, which subsequently bombed on the leaderboard.
* **Attempt 3: The Jackpot Architecture (Final)**
  * *Approach:* Purged the target leak and implemented strict **Out-Of-Fold (OOF) Geohash Target Encoding**, spatial neighbor mapping, and `RoadType` time-curves. Used a standard `np.log1p` target transform with LightGBM.
  * *Result:* Stable convergence and our highest leaderboard score.

## 🏗️ Final Model Architecture

Our final solution uses **LightGBM** inside a 5-Fold Cross-Validation loop, treating the problem as a cross-sectional spatial-temporal regression rather than a pure sequential time series.

**Key Features Engineered:**
- **Spatial Spillover:** Calculated demand from neighboring geohashes (N, S, E, W).
- **Safe Lags:** Mapped exact `demand_yesterday` as a momentum baseline.
- **Road-Time Curves:** Captured how different `RoadTypes` behave at specific 15-minute `time_buckets`.
- **Cyclical Time:** Transformed absolute minutes into `time_sin` and `time_cos` to prevent tree extrapolation failures.
- **OOF Target Encoding:** Encoded geohash historical demand strictly out-of-fold to extract signal without leaking the target.

## 📊 Results

* **Local 5-Fold CV R²:** ~0.955 (95.5%)
* **Leaderboard R²:** ~90.0%+

## 🛠️ Tech Stack

* **Language:** Python
* **Libraries:** Pandas, NumPy, LightGBM, Scikit-Learn
* **Techniques:** K-Fold CV, Geohash Decoding, OOF Encoding, Log Target Transformation

---
*Built for the Flipkart Gridlock Hackathon.*