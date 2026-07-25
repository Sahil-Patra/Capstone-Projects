# 🏀 NBA Shot Selection — Kobe Bryant Shot Prediction

A binary classification project that predicts whether a shot attempted by Kobe Bryant will be **made or missed**, based on in-game contextual features such as shot location, distance, shot type, game time, and opponent.

## 📌 Problem Statement

Build a shot prediction model that classifies whether a player will score or not, based on the circumstances of the shot. The goal is to identify the model with the best accuracy and F1-score, providing insights that can help coaching staff and analysts develop more effective game strategies.

## 📂 Dataset

The dataset (`data.csv`) contains individual shot attempts by Kobe Bryant across his career, with features including:

| Category | Example Columns | Description |
|---|---|---|
| Spatial | `loc_x`, `loc_y`, `lat`, `lon`, `shot_distance`, `shot_zone_area`, `shot_zone_basic` | Where the shot was taken on the court |
| Shot Technique | `action_type`, `combined_shot_type` | How the shot was taken (e.g. Jump Shot, Dunk, Layup) |
| Game Context | `period`, `minutes_remaining`, `seconds_remaining`, `season` | When in the game the shot occurred |
| Matchup | `opponent`, `matchup`, `game_date` | Who the shot was taken against |
| Target | `shot_made_flag` | 1 = Made, 0 = Missed |

## 🔍 Exploratory Data Analysis

Key findings from univariate and bivariate analysis:

- **Class balance:** Roughly a 45% (made) / 55% (missed) split — no aggressive resampling (e.g. SMOTE) required.
- **Shot distance is bimodal:** peaks near the rim (0–5 ft) and beyond the 3-point line (23–25 ft), with a "dead zone" of low-frequency long-2s (10–18 ft).
- **Accuracy decreases with distance:** ~65% near the basket down to ~30–35% beyond 23 ft, with a slight stabilization at the 3-point line.
- **Action type matters more than distance for short shots:** Dunks convert at >90% while Jump Shots convert at ~30–40% even from similar ranges.
- **Court zone bias:** Center-court shots are most efficient; slightly higher accuracy from the right side, consistent with Kobe being right-handed.
- **Clutch-time effect:** Shooting accuracy becomes more volatile and trends downward in the final 2 minutes of the 4th quarter, likely due to contested, low-quality shot attempts under pressure.

A custom court-drawing function was built with `matplotlib` to overlay shot charts and hexbin frequency heatmaps on a to-scale NBA court.

## 🧹 Data Preprocessing

- Dropped redundant/low-signal columns: `lat`, `lon`, `team_id`, `team_name`, `shot_type`, `shot_zone_range`, `game_id`, `matchup`, `game_date`, `game_event_id`, `shot_id`.
- Engineered features:
  - `seconds_to_end` — continuous time-remaining variable combining minutes and seconds.
  - `is_home_game` — derived from the `matchup` string.
  - `season_code` — numeric encoding of the season's starting year.
  - `opponent_freq` — frequency encoding of opponent to avoid high-cardinality one-hot columns.
- Grouped rare `action_type` values (long-tail of 50+ categories) into an `Other` bucket, keeping the top 20 most frequent types, then one-hot encoded.
- One-hot encoded `combined_shot_type`, `shot_zone_area`, and `shot_zone_basic`.
- Capped outliers in `loc_y` and `shot_distance` at ±3 standard deviations.
- **Chronological train/test split (80/20)** instead of a random split, to prevent data leakage from future games into training.
- Standardized features with `StandardScaler`.

## 🤖 Modeling

### Baseline Models
Seven classifiers were trained with default hyperparameters to establish a baseline: Logistic Regression, K-Nearest Neighbors, Decision Tree, Random Forest, Gradient Boosting, SVM, and XGBoost.

**Key baseline findings:**
- Decision Tree and Random Forest overfit severely (perfect 1.00 training accuracy, ~0.60–0.66 test accuracy).
- Linear models (SVM, Logistic Regression) generalized best out of the box (~67.7–67.8% test accuracy) despite being simpler.

### Hyperparameter Tuning
`RandomizedSearchCV` (100 iterations, 3-fold CV) was used to tune the top 3 candidates: **Random Forest**, **Decision Tree**, and **XGBoost**.

### Final Results (Test Set)

| Model | Accuracy | F1 Score | Precision | Recall | ROC AUC |
|---|---|---|---|---|---|
| **Random Forest** | **0.6772** | **0.6612** | **0.7283** | 0.4525 | **0.6570** |
| XGBoost | 0.6768 | 0.6608 | 0.7276 | 0.4521 | 0.6566 |
| Decision Tree | 0.6724 | 0.6587 | 0.7073 | **0.4655** | 0.6538 |

**Random Forest** was selected as the final model, with a train/test accuracy gap of under 0.5%, indicating good generalization after tuning.

## 💡 Business Strategy & Conclusion

- The final model achieves ~67.7% accuracy, which is a strong benchmark given the inherent randomness of individual shot outcomes in basketball.
- With ~73% precision, when the model predicts a "Make," it is right roughly 3 out of 4 times — useful for real-time shot-quality signaling.
- Lower recall suggests the model conservatively flags contested or low-percentage shots as misses, which aligns with sound shot-selection strategy even when such shots occasionally go in.

## ⚠️ Challenges & Solutions

| Challenge | Solution |
|---|---|
| Feature redundancy (`lat`/`lon` vs `loc_x`/`loc_y`) | Dropped redundant coordinates to avoid multicollinearity |
| High cardinality in `action_type` | Grouped rare categories into "Other"; frequency-encoded `opponent` |
| Data leakage risk | Used a chronological (not random) train/test split |

## 🛠️ Tech Stack

- **Language:** Python 3.11
- **Data handling:** pandas, numpy
- **Visualization:** matplotlib, seaborn
- **Modeling:** scikit-learn, XGBoost

## 📁 Project Structure

```
├── NBA_shot_prediction.ipynb   # Full analysis: EDA, preprocessing, modeling, evaluation
├── data.csv                     # Raw dataset (not included — see Setup)
└── README.md
```

## 📈 Future Improvements

- Incorporate defender proximity / shot clock data if available, to capture true shot difficulty.
- Try probability calibration (e.g. Platt scaling) since precision/recall trade-offs matter for the "trust the shot" use case.
- Explore ensembling Random Forest and XGBoost predictions.
