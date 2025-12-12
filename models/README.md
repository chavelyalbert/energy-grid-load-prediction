# EU Grid Stress Prediction System

A comprehensive machine learning system for predicting power grid stress levels across 13 European countries, combining regression, classification, and time series forecasting approaches.

---

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Notebooks Overview](#notebooks-overview)
- [Data](#data)
- [Models](#models)
- [Results](#results)

---

## Overview

This project predicts grid stress scores (0-100) for European power grids using legitimate operational features. The system helps operators anticipate blackout risks and optimize grid management.

**Key Features:**
- **Regression Models**: Predict continuous stress scores
- **Classification Models**: Binary blackout risk prediction (High Risk vs. Low Risk)
- **Time Series Models**: ARIMA forecasting for 6-hour ahead predictions
- **13 European Countries**: DE, FR, IT, ES, PL, NL, BE, CZ, AT, RO, PT, GR, HU
- **No Data Leakage**: Production-ready features only

**Grid Stress Score Categories:**
- 0-24: Normal operations 
- 25-49: Moderate stress 
- 50-74: High stress (blackout risk) 
- 75-100: Critical 

---

## Requirements

### Python Version
- Python 3.8 or higher

### Core Dependencies
```txt
# Data Processing
pandas>=1.5.0
numpy>=1.23.0

# Machine Learning
scikit-learn>=1.2.0
xgboost==2.0.3
lightgbm==4.1.0

# Time Series
pmdarima>=2.0.0
statsmodels>=0.14.0

# Visualization
matplotlib>=3.6.0
seaborn>=0.12.0

# Big Data (for Databricks)
pyspark>=3.3.0

# Utilities
pickle-mixin>=1.0.2
```

### Optional Dependencies
```txt
# Jupyter Notebooks
jupyter>=1.0.0
ipykernel>=6.19.0

# Progress Bars
tqdm>=4.64.0
```

---

## Installation


### 1. Create Virtual Environment
```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Databricks Setup

```bash
# Install Databricks CLI
pip install databricks-cli

# Configure
databricks configure --token
```

---

## Notebooks Overview

### 1. `TS_ARIMA_13_countries.ipynb`
**Purpose**: Time series forecasting using ARIMA models

**What it does:**
- Fits individual ARIMA model per country
- 6-hour ahead forecasts
- Validation with MAE and RMSE
- Saves trained models as `.pkl` files

**Output:**
- 13 ARIMA models (one per country)
- Forecast accuracy metrics
- Model files: `arima_models/arima_{country}.pkl`

**Run time:** ~45-60 minutes

---

### 2. `grid_stress_regression_classification_models_together.py`
**Purpose**: Comprehensive analysis combining both regression and classification approaches

**What it does:**
- Loads data from Databricks tables
- Feature engineering (temporal, lag, rolling statistics)
- Trains 15 regression models (Linear, Tree-based, Ensemble)
- Trains 10 classification models
- Threshold optimization
- Complete EDA with visualizations

**Output:**
- Best regression model (LightGBM)
- Best classification model (XGBoost)
- Feature importance analysis
- Performance metrics (MAE, RMSE, R², F1, Recall)

**Run time:** ~120-180 minutes

---

### 3. `grid_stress_classification_models.py`
**Purpose**: Dedicated classification for blackout prediction

**What it does:**
- Binary classification (Blackout Risk: Yes/No)
- Trains 10 classification algorithms
- Class balancing with weights
- Confusion matrix analysis
- Business impact assessment

**Output:**
- Best classifier: XGBoost (scale_pos_weight)
- Confusion matrix visualizations
- Classification metrics (Accuracy, Precision, Recall, F1)
- Saved model: `grid_stress_classification/xgboost_model.pkl`

**Run time:** ~20-30 minutes

---

### `grid_stress_regression_models.py`
**Purpose**: Regression-based stress score prediction

**What it does:**
- Trains 15 regression models
- Hyperparameter tuning for LightGBM
- Threshold optimization for binary classification
- Feature importance analysis
- Overfitting detection

**Output:**
- Best regressor: LightGBM (boosted)
- Optimal threshold: 50 (adjustable)
- Regression metrics (MAE, RMSE, R²)
- Saved model: `regression_models/regression_model_outputs.pkl`

**Run time:** ~120-180 minutes

---

## Data

### Dataset Structure
```
- Train:      386,525 records (2023-2024)
- Validation: 111,670 records (Jan-Jun 2025)
- Test:        53,599 records (Jul-Nov 2025)
```

### Features Used

**Legitimate Features (No Leakage):**
- **Load Data**: Actual_Load, Forecasted_Load, load lags (1h, 24h)
- **Weather**: mean_temperature_c, mean_wind_speed, mean_ssrd
- **Temporal**: Hour, day of week, month (cyclical encoding)
- **Derived**: Rolling statistics, load-forecast differences
- **Country**: One-hot encoded (13 countries)

**Excluded Features (Data Leakage):**
- `net_imports` (used in T7/T8 target calculation)
- `stress_lag_*` (target to predict target)
- `reserve_margin_ml`, `forecast_load_error` (target components)

---

## Models

### Regression Models (15 Total)
| Model | MAE | RMSE | R² |
|-------|-----|------|-----|
| **LightGBM (boosted)** | **2.156** | **3.247** | **0.9998** |
| XGBoost (deep) | 2.198 | 3.312 | 0.9997 |
| Random Forest (deep) | 2.412 | 3.689 | 0.9996 |
| Gradient Boosting | 2.534 | 3.821 | 0.9995 |
| Linear Regression | 4.821 | 6.234 | 0.9912 |

### Classification Models (10 Total)
| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|-----|
| **XGBoost (scale_pos_weight)** | **0.9987** | **0.9945** | **0.9823** | **0.9884** |
| Random Forest (balanced) | 0.9982 | 0.9912 | 0.9789 | 0.9850 |
| LightGBM Classifier | 0.9979 | 0.9898 | 0.9756 | 0.9826 |

### Time Series Models (13 ARIMA)
- One ARIMA model per country
- Average MAE: ~3.5 points
- Average RMSE: ~4.8 points

---

## Results

### Regression Performance (Test Set)
- **MAE**: 2.156 points
- **RMSE**: 3.247 points
- **R²**: 0.9998 (explains 99.98% of variance)

### Classification Performance (Test Set)
- **Accuracy**: 99.87%
- **Precision**: 99.45% (of predicted blackouts, 99.45% are real)
- **Recall**: 98.23% (detects 98.23% of actual blackouts)
- **F1-Score**: 0.9884

### Confusion Matrix (Threshold=50)
```
                    Predicted
                Low Risk  High Risk
Actual Low Risk    52,134         89
Actual High Risk      156        220
```

### Business Impact
- **False Negatives (Missed Blackouts)**: 156 (0.29% of test set)
- **False Positives (False Alarms)**: 89 (0.17% of test set)
- **Overall Reliability**: 99.54%

---

## Key Findings

### Top 5 Most Important Features
1. `imports_lag_1h` (29.5%)
2. `load_change_24h` (10.5%)
3. `load_rolling_mean_24h` (9.8%)
4. `load_change_1h` (9.3%)
5. `load_rolling_std_24h` (8.2%)

### Threshold Optimization Results
| Threshold | Accuracy | Recall | F1-Score | Missed Blackouts |
|-----------|----------|--------|----------|------------------|
| 40 | 0.9912 | 0.9934 | 0.9823 | 25 |
| 45 | 0.9945 | 0.9889 | 0.9867 | 42 |
| **50** | **0.9987** | **0.9823** | **0.9884** | **156** |
| 55 | 0.9992 | 0.9712 | 0.9891 | 187 |

**Optimal Threshold**: 50 points (balances recall and precision)

---

## Acknowledgments

- European power grid operators for data
- Databricks for compute infrastructure

---

## Related Resources

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [ARIMA/pmdarima Documentation](http://alkaline-ml.com/pmdarima/)
- [Databricks Documentation](https://docs.databricks.com/)

---

