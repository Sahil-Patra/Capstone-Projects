# PRCP-1006 Home Loan Default Prediction
## Complete Data Analysis & Predictive Modeling Project

---

## 📋 Project Overview

**Domain:** Banking - Credit Risk Assessment

**Objective:**
1. Perform comprehensive Exploratory Data Analysis (EDA)
2. Create a predictive model to identify customer segments eligible for loans
3. Compare multiple machine learning models and recommend the best for production

**Target Variable:**
- 1: Defaulter (customer who defaulted on loan)
- 0: Non-Defaulter (customer who repaid loan successfully)

---

## 📁 Dataset Description

The project uses 7 CSV files:

1. **application_train.csv** - Main training data with target variable
2. **bureau.csv** - Previous credits from other financial institutions
3. **bureau_balance.csv** - Monthly balances of previous credits
4. **POS_CASH_balance.csv** - Monthly POS and cash loan balances
5. **credit_card_balance.csv** - Monthly credit card balances
6. **previous_application.csv** - Previous Home Credit loan applications
7. **installments_payments.csv** - Repayment history

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm imbalanced-learn scipy
```

### Required Libraries
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost
- lightgbm
- imbalanced-learn
- scipy

### Data Setup

1. Download the dataset from: https://d3ilbtxij3aepc.cloudfront.net/projects/CDS-Capstone-Projects/PRCP-1006-HomeLoanDef.zip
2. Extract all CSV files to the project directory
3. Ensure all 7 CSV files are in the same folder as the analysis script

---

## 📊 Analysis Pipeline

### 1. Data Loading
- Load all 7 CSV files
- Verify data integrity and structure

### 2. Exploratory Data Analysis (EDA)
- **Target Variable Analysis**: Distribution, imbalance ratio
- **Missing Values Analysis**: Identify and categorize missing data
- **Numerical Features**: Distributions, outliers, correlations
- **Categorical Features**: Cardinality, default rates by category
- **Correlation Analysis**: Feature relationships with target

### 3. Feature Engineering

#### Application Features
- Age-related: `AGE_YEARS`, `AGE_GROUP`
- Employment: `EMPLOYMENT_YEARS`, `EMPLOYED_TO_AGE_RATIO`
- Income ratios: `INCOME_TO_CREDIT_RATIO`, `INCOME_TO_ANNUITY_RATIO`
- Family features: `INCOME_PER_PERSON`, `CHILDREN_RATIO`
- External sources: `EXT_SOURCE_MEAN`, `EXT_SOURCE_STD`, `EXT_SOURCE_MAX`, `EXT_SOURCE_MIN`
- Document count aggregation
- Phone and region features

#### Bureau Features
- Credit history aggregations (min, max, mean)
- Overdue amounts and counts
- Debt ratios
- Credit prolongation statistics

#### Previous Application Features
- Number of previous applications
- Approval rates
- Amount statistics
- Payment counts

#### POS Cash Features
- Installment counts
- Balance snapshots
- Days past due (DPD) statistics

#### Credit Card Features
- Balance and limit ratios
- Drawing patterns (ATM, POS)
- Payment history

#### Installments Features
- Payment differences and ratios
- Late payment indicators
- Days difference statistics

### 4. Data Preprocessing

- **Missing Value Handling:**
  - Drop columns with >70% missing values
  - Median imputation for numerical features
  - Mode imputation for categorical features

- **Encoding:**
  - Label encoding for categorical variables

- **Scaling:**
  - RobustScaler for handling outliers

- **Feature Selection:**
  - Mutual information for selecting top 100 features

- **Class Imbalance:**
  - SMOTE (Synthetic Minority Over-sampling Technique)

### 5. Model Training & Evaluation

**Models Compared:**
1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Gradient Boosting
5. XGBoost
6. LightGBM

**Evaluation Metrics:**
- Accuracy (Train & Test)
- Precision
- Recall
- F1 Score
- ROC-AUC Score

---

## 💻 Usage

### Running the Analysis

```bash
python home_loan_default_analysis.py
```

### Output Files

1. **model_comparison_results.csv** - Performance metrics for all models
2. **Console output** - Detailed analysis results and progress

---

## 🎯 Key Findings

### Data Characteristics
- **Class Imbalance**: ~92% Non-Defaulters, ~8% Defaulters (11.5:1 ratio)
- **High Dimensionality**: 400+ features after engineering
- **Missing Data**: Many columns have >50% missing values

### Important Features
1. External source scores (EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3)
2. Credit history from bureau
3. Age and employment duration
4. Previous application behavior
5. Income-to-credit ratios

### Model Performance
- Best model typically: XGBoost or LightGBM
- ROC-AUC score: ~0.75-0.78 (expected range)
- Important to balance Precision and Recall for credit risk

---

## 🔍 Challenges & Solutions

### 1. Class Imbalance
- **Challenge:** 92% non-defaulters, models biased toward majority class
- **Solution:** SMOTE for oversampling, class_weight='balanced', ROC-AUC metric

### 2. Missing Values
- **Challenge:** >50% missing in many columns
- **Solution:** Drop >70% missing columns, median/mode imputation

### 3. High Dimensionality
- **Challenge:** 400+ features after engineering
- **Solution:** Mutual information for feature selection, select top 100

### 4. Outliers
- **Challenge:** Extreme values in financial data
- **Solution:** RobustScaler, ratio features instead of absolute values

### 5. Multiple Data Sources
- **Challenge:** Complex relationships across 7 files
- **Solution:** Careful aggregation at customer level, meaningful statistics

### 6. Temporal Features
- **Challenge:** Negative days format (DAYS_BIRTH, DAYS_EMPLOYED)
- **Solution:** Convert to positive years, handle anomalies

---

## 📈 Business Recommendations

### 1. Automated Screening
- Use model for initial loan application screening
- Automatically approve low-risk customers
- Flag high-risk for manual review

### 2. Risk-Based Pricing
- Low risk: Standard interest rates
- Medium risk: Higher rates with additional requirements
- High risk: Reject or require collateral

### 3. Customer Segmentation
- **Low Risk (0-30% default probability):**
  - Fast-track approval
  - Offer premium products
  
- **Medium Risk (30-60% default probability):**
  - Manual underwriting
  - Additional documentation required
  
- **High Risk (60-100% default probability):**
  - Reject or special programs with high rates

### 4. Model Deployment
- Deploy as REST API for real-time predictions
- Monitor performance metrics monthly
- Retrain model quarterly with new data
- A/B test against current system

### 5. Explainability
- Use feature importances to explain rejections
- Provide actionable feedback to applicants
- Ensure regulatory compliance

---

## 📝 Files Included

1. **home_loan_default_analysis.py** - Main analysis script
2. **README.md** - This file
3. **CHALLENGES_REPORT.md** - Detailed challenges and solutions
4. **model_comparison_results.csv** - Model performance comparison (generated after running)

---

## 🔄 Next Steps

1. **Hyperparameter Tuning:**
   - Grid search for optimal parameters
   - Cross-validation for robust estimates

2. **Advanced Feature Engineering:**
   - Interaction features
   - Time-based features
   - Domain-specific ratios

3. **Model Interpretability:**
   - SHAP values for feature importance
   - LIME for local explanations

4. **Production Deployment:**
   - Create REST API
   - Implement monitoring dashboard
   - Set up automated retraining

5. **Business Integration:**
   - Integrate with loan origination system
   - Create decision rules framework
   - Develop risk-based pricing engine

---

## 👥 Authors

Data Science Team

---

## 📄 License

This project is for educational purposes as part of the CDS Capstone Project.

---

## 📞 Support

For questions or issues, please refer to the project documentation or contact the data science team.

---

## 🙏 Acknowledgments

- Dataset provided by Home Credit Group via Kaggle
- CDS Capstone Project framework
- Open-source Python community for excellent libraries
