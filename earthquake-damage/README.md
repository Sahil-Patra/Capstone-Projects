# 🌍 Earthquake Damage Prediction

Predicting building-level earthquake damage grade (Low / Medium / Severe) using structural, geographic, and demographic features, based on the 2015 Gorkha earthquake dataset (Nepal).

## 📌 Problem Statement

Create a predictive model to estimate the **damage grade** of a building after an earthquake, using information about its structure, location, and construction. Accurate predictions can help seismologists and disaster-response teams prioritize inspections, reinforcement, and relief resources.

## 🎯 Target Variable

`damage_grade` — an ordinal category describing the severity of damage to a building:

| Grade | Meaning | Description |
|:---:|---|---|
| 1 | Low damage | Minor cracks, plaster damage, building safe to occupy |
| 2 | Medium damage | Large cracks, partial wall failure, cracked but standing columns/beams |
| 3 | Severe damage / collapse | Partial or full collapse, foundation failure, building unsafe/destroyed |

## 🗂️ Dataset

The dataset contains building-level records with:
- **Geographic identifiers**: `geo_level_1_id` (province), `geo_level_2_id` (district), `geo_level_3_id` (municipality)
- **Building geometry & age**: `count_floors_pre_eq`, `age`, `area_percentage`, `height_percentage`
- **Structural attributes**: foundation type, roof type, ground/other floor type, position, plan configuration, land surface condition, legal ownership status
- **Superstructure materials** (binary flags): adobe/mud, mud-mortar stone, cement-mortar stone/brick, timber, bamboo, RC engineered/non-engineered, etc.
- **Secondary building use** (binary flags): agriculture, hotel, rental, institution, school, industry, health post, government office, police, other
- **Target**: `damage_grade` (1, 2, or 3)

> Data files (`train_values.csv`, `train_labels.csv`) are **not included** in this repository. Download them from the [Richter's Predictor: Modeling Earthquake Damage](https://www.drivendata.org/competitions/57/nepal-earthquake/) competition on DrivenData and place them in a local `Data/` folder, then update the file paths in the notebook.

## 🔍 Project Workflow

### 1. Exploratory Data Analysis (EDA)
- **Univariate analysis**: target class distribution, numerical feature histograms/box plots, categorical feature frequency plots, outlier detection (IQR method), skewness/kurtosis
- **Bivariate analysis**: numerical features vs. damage grade (violin plots), categorical features vs. damage grade (normalized stacked bar charts), superstructure/secondary-use features vs. damage grade
- **Multivariate analysis**: correlation heatmap, pairplots, and interaction plots (e.g., foundation type × age group, position × floor count)

Key findings:
- The target is **imbalanced**, with Grade 2 (medium damage) dominating.
- Most buildings use **mud-mortar stone**, a seismically vulnerable material.
- `age`, `count_floors_pre_eq`, `area_percentage`, and `height_percentage` are right-skewed with notable outliers (e.g., `age = 995` likely encodes "unknown/ancient").
- Geographic features (`geo_level_1/2/3_id`) show highly clustered, high-cardinality distributions with strong predictive potential.

### 2. Data Preprocessing & Feature Engineering
- Dropped the `building_id` identifier column
- Created **risk-score mappings** for foundation, roof, and plan configuration types
- **One-hot encoded** low-cardinality categorical features
- Engineered **building geometry features**: `height_per_floor`, `area_per_floor`, `aspect_ratio`, `volume_proxy`, `families_per_floor`, `is_high_rise`
- Engineered **material scores**: `weak_material_score`, `strong_material_score`, `total_superstructure_types`, `total_secondary_uses`
- Created **interaction features**: `old_weak_interaction`, `height_weak_interaction`, `age_floors_interaction`
- Applied **log1p transformation** to reduce skewness in numerical features
- Applied **Empirical Bayes (multiclass) target smoothing** to high-cardinality geographic IDs (`geo_level_1/2/3_id`), producing per-grade probability features while avoiding overfitting on sparse locations
- Removed highly correlated features (correlation > 0.9) to reduce multicollinearity

### 3. Model Building
Three baseline models were trained and compared:
- **Random Forest**
- **XGBoost**
- **LightGBM**

Hyperparameters were tuned using `RandomizedSearchCV` with stratified 3-fold cross-validation, optimizing for **Micro-F1 score** (well-suited to imbalanced, multi-class problems).

### 4. Results

| Model | Micro F1-Score | Notes |
|---|:---:|---|
| **XGBoost (Selected)** | **0.7651** | Best performer; handles high cardinality & non-linear interactions natively |
| Random Forest | 0.7522 | Strong, robust baseline |
| LightGBM | 0.7326 | Fast, but weaker on this feature set |

**XGBoost** was selected as the final model based on Micro-F1 score and its ability to capture complex geographic-structural interactions.

### 5. Feature Importance
Top drivers of predicted damage grade:
- `weak_material_score` (weak construction materials)
- Empirical-Bayes geographic risk probabilities (especially `geo_level_3`)
- `ground_floor_type` (particularly Type V, associated with lower damage)

A confusion matrix was also generated to evaluate per-class prediction performance.

## 💡 Recommendations for Seismologists

- Prioritize reinforcement/replacement of **mud-mortar and non-engineered stone** structures, the strongest predictor of damage.
- Adopt **micro-zoning** (village/neighborhood-level risk mapping) using `geo_level_3` risk scores rather than relying on broad regional assessments.
- Encourage **reinforced concrete or modern ground-floor construction** in new builds, as it correlates with significantly lower damage.
- Use model-predicted damage probabilities to build **predictive heatmaps** for pre-positioning emergency response resources.

## ⚠️ Challenges & Solutions

| Challenge | Solution |
|---|---|
| High-cardinality geo-location features | Kept as numeric / applied target encoding rather than one-hot encoding |
| Class imbalance (Grade 2 majority) | Stratified train/test split + Micro-F1 as primary metric |
| Multicollinearity (e.g., floors vs. height) | Correlation filtering (threshold 0.9); tree models also handle this natively |

## 🛠️ Tech Stack

- **Language**: Python 3
- **Data handling**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Modeling**: scikit-learn, XGBoost, LightGBM
- **Statistics**: scipy

## 📁 Repository Structure

```
.
├── Earth_Damage_Prediction.ipynb   # Main analysis & modeling notebook
├── README.md                       # Project documentation
└── requirements.txt                # Python dependencies
```

## 🚀 Getting Started

1. Clone the repository
   ```bash
   git clone https://github.com/<your-username>/earthquake-damage-prediction.git
   cd earthquake-damage-prediction
   ```
2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```
3. Download the dataset from [DrivenData](https://www.drivendata.org/competitions/57/nepal-earthquake/) and place `train_values.csv` and `train_labels.csv` in a `Data/` folder.
4. Update the file paths in the notebook's data-loading cell to match your local setup.
5. Run the notebook:
   ```bash
   jupyter notebook Earth_Damage_Prediction.ipynb
   ```

## 📄 License

This project is open source. Add a license file (e.g., MIT) if you plan to distribute it publicly.

## 🙏 Acknowledgements

Dataset provided by [DrivenData — Richter's Predictor: Modeling Earthquake Damage](https://www.drivendata.org/competitions/57/nepal-earthquake/), based on the 2015 Gorkha earthquake in Nepal.
