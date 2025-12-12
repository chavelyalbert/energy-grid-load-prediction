# ARIMA Time-Series Modeling


## Overview
This notebook builds and validates **ARIMA forecasting models** for predicting grid stress across multiple European countries. It uses preprocessed datasets stored in Databricks and trains **one ARIMA model per country**.

The workflow includes:

- Loading and preparing time-series data  
- Fitting ARIMA models using `auto_arima`  
- Generating 6-hour forecasts per country  
- Saving fitted models for downstream use  
- Validating forecasts using a separate validation dataset  

---

## Datasets

Training Dataset: `train_set_imputed`  
Validation Dataset: `validation_set_imputed`


Each dataset contains:

- `timestamp` (renamed from `index`)  
- `country`  
- `grid_stress_score`  
- `Additional operational and weather features`  

For ARIMA, only:
`timestamp`, `country`, `grid_stress_score` are required.

---

## Workflow Summary

### 1. Load & Prepare Data
- Load tables from Databricks  
- Rename `index` → `timestamp`  
- Select required time-series columns  
- Order data chronologically  
- Convert to Pandas for modeling  

---

### 2. Train ARIMA Models (Per Country)
- Loop through each country  
- Extract its grid stress time series  
- Fit ARIMA using `auto_arima` to automatically choose `(p, d, q)`  
- Store each fitted model in a dictionary  

---

### 3. Forecasting
- Produce **6-hour-ahead forecasts** per country  
- Store results for visualization or downstream use  

---

### 4. Save Models
Each ARIMA model is saved as a `.pkl` file in:
`/Workspace/Users/<your-email>/arima_models/`


These models can later be used by:

- The Streamlit application  
- Batch prediction pipelines  
- Model comparison notebooks  

---

### 5. Validation (Using pre-prepared validation dataset)
For each country:

- Load the validation subset  
- Forecast the same number of points as the validation period  
- Compute accuracy metrics:  
  - **MAE** (Mean Absolute Error)  
  - **RMSE** (Root Mean Squared Error)  



---

## Requirements

This notebook is designed to run **inside Databricks**, which provides the necessary Spark environment.

### Runtime Environment
- **Databricks Runtime** (includes Apache Spark and system-level dependencies)

### Python Libraries
These packages must be available in the environment (Databricks usually includes most of them; install locally if needed):

- `pmdarima`
- `statsmodels`
- `pandas`
- `numpy`



