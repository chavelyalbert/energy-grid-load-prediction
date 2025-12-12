# GridWatch - European Power Grid Stress Predictor

A real-time interactive Streamlit dashboard for predicting and monitoring electricity grid stress across 13 European countries. Uses machine learning to forecast grid instability and enable proactive decision-making by grid operators.

## Overview

GridWatch integrates weather, generation, load, and cross-border flow data to predict grid stress levels and help prevent blackouts. The underlying XGBoost model was trained on over 550,000 hourly records from the ENTSOE Transparency Platform and Weather Reanalysis Data from Copernicus (2023-2025).

## The dashboard provides:

- **Real-time Stress Prediction**: Instant stress score calculation (0-100 scale)
- **Interactive Controls**: Adjust load, imports, weather, and temporal parameters
- **Scenario Presets**: Quickly simulate heat waves, cold snaps, import crises, and more
- **6-Hour Projections**: Forecast stress levels 6 hours ahead using ARIMA models
- **Interactive EU heatmap visualizations**
- **Target Breakdown**: Visualize individual stress components
- **Multi-Country Support**: Coverage of 13 European nations

## Model Performance

| Metric | Value |
|--------|-------|
| R² Score | 0.999878 |
| AUC-ROC (Binary) | 0.826 |
| Training Records | 550,000+ |
| Countries | 24 |
| Features | 100 |

## Stress Score Interpretation

| Score Range | Status | Action |
|-------------|--------|--------|
| 0-24 | Normal | Standard operations |
| 25-49 | Moderate | Increased monitoring |
| 50-74 | High Risk | Immediate intervention |
| 75 | Critical | Emergency protocols |

## Installation

1. Clone or download this repository

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Linux/Mac
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Ensure all data files are in the same directory:
   - `xgboost_model.pkl` - Trained XGBoost model
   - `feature_names.pkl` - Model feature names
   - `country_stats.csv` - Country-level statistics

5. Create a .env file with Databricks credentials
```python
   DATABRICKS_TOKEN=your_token_here
   DATABRICKS_HOSTNAME=your_hostname.cloud.databricks.com
   DATABRICKS_HTTP_PATH=/sql/1.0/warehouses/your_warehouse_id
```

## Usage

Run the Streamlit application:

```bash
streamlit run app_eu_grid_stress.py
```

The dashboard will open in your default web browser at `http://localhost:8501`.

## Dashboard Components

### Sidebar Controls

- **Country Selection**: Choose from 13 European countries
- **Scenario Presets**: Quick-load predefined scenarios
- **Load Parameters**: Adjust actual and forecasted load
- **Cross-Border Flow**: Set net imports/exports
- **Weather**: Configure temperature, wind, and solar
- **Time**: Set hour and day of week

### Main Dashboard

- **Top Metrics**: Stress score, current load, imports, targets triggered
- **Stress Gauge**: Visual representation of current stress level
- **Target Breakdown**: Individual component contributions (underlying cause)
- **6-Hour Projection**: Forecasted stress fro the next 6 hours
- **Feature Importance**: Key model drivers

## Scenario Presets

| Preset | Description |
|--------|-------------|
| Normal Operations | Baseline conditions |
| Heat Wave | High temperature, increased cooling demand |
| Cold Snap | Extreme cold, heating demand surge |
| Wind Drought | Low wind generation |
| Import Crisis | High import dependency |
| Forecast Error | Significant load forecasting miss |
| Peak Hour Stress | High demand period |

## Countries Covered

AT (Austria), BE (Belgium), BG (Bulgaria), CH (Switzerland), DE (Germany), DK (Denmark), EE (Estonia), ES (Spain), FI (Finland), FR (France), GR (Greece), HR (Croatia), HU (Hungary), IT (Italy), LT (Lithuania), LV (Latvia), NL (Netherlands), NO (Norway), PL (Poland), PT (Portugal), RO (Romania), SE (Sweden), SK (Slovakia)

## Technical Details

### Model Architecture
- **Algorithm**: XGBoost Classifier
- **Features**: 100 engineered features including:
  - Load metrics (actual, forecast, lags, rolling statistics)
  - Import/export flows and dependencies
  - Weather data (temperature, wind, solar)
  - Temporal patterns (cyclical encoding)
  - Country indicators (one-hot encoded)

### Data Sources
- ENTSOE Transparency Platform
- Weather data from Reanalysis data produced by ECMWF as part of the Copernicus Program
- Live Weather data from OpenMeteo

## File Structure

```
streamlit_capstone/
├── app_eu_grid_stress.py     # Main Streamlit application
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── xgboost_model.pkl         # Trained model
├── feature_names.pkl         # Feature names list
├── country_stats.csv         # Country statistics
```

## Dependencies

- Check requirements.txt file.

## Author

Capstone Project - European Power Grid Stress Prediction  
December 2025
Team 6 - GridWatch: 
Chavely Albert Fernandez
Pedro Miguel
Ya-Chi Hsiao
Maria Sokotushchenko


## Acknowledgments

- ENTSOE Transparency Platform for grid data
- ECMWF for Climate reanalyses
- OpenMeteo for the live weather data
- Academic advisors and mentors