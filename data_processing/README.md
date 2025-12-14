# Data processing

## Overview

These notebooks process and integrates multiple data sources across European countries to create a machine learning-ready dataset for grid stress prediction. The pipeline:

- Aggregates hourly weather metrics for 27 European countries
- Normalizes electricity generation data from various sources
- Combines generation, load, and cross-border flow data
- Calculates grid stress scores based on multiple indicators
- Prepares training, validation, and test datasets with handled missing values

## Data Pipeline Architecture

### Stage 1: Data Processing

**01_weather_data_processing.py**
- Loads hourly weather data from the raw European grid load database
- Uses reverse geocoding to map coordinates (latitude/longitude) to country codes
- Computes country-level hourly averages for key metrics:
  - Solar Radiation (SSRD)
  - Wind Speed
  - Temperature (°C)
- Filters data to include only 27 European countries
- Output: `weather_europe` table

**02_generation_data_processing.py**
- Loads raw electricity generation data by type and country
- Computes hourly averages across all generation sources
- Standardizes timestamps to hourly intervals
- Output: `workspace.schema_capstone.generation_clean` table

### Stage 2: Data Integration

**03_all_tables_processing.py**
- Consolidates 7 data sources:
  - Weather data
  - Generation (actual)
  - Load (actual)
  - Cross-border flows
  - Load forecasts
  - Solar generation forecasts
  - Wind generation forecasts
- Normalizes all timestamps to hourly intervals
- Calculates net imports/exports per country by aggregating cross-border flows
- Performs inner joins to keep only rows with complete data across all sources
- Output: `electricity_and_weather_europe` table

### Stage 3: Target Variable Engineering

**04_define_target_variable.py**
- Removes redundant columns representing duplicate generation types
- Calculates three grid stress indicators (100-point scoring system):
  - **Reserve Margin**: Compares current load vs. 24-hour historical average (0-25 points)
  - **Load Forecast Error**: Measures prediction accuracy relative to actual load (0-25 points)
  - **Cross-Border Flow Stress**: Flags unusually high imports/exports using 90th/10th percentiles (0-50 points)
- Output: `electricity_and_weather_europe_with_target` table

### Stage 4: Train/Validation/Test Split

**05_train_val_test_split.py**
- Temporal train-validation-test split:
  - **Training**: Data up to 2024-12-31
  - **Validation**: 2024-01-01 to 2025-07-31
  - **Test**: After 2025-07-31
- Output: `train_set`, `validation_set`, `test_set` tables

### Stage 5: Missing Value Imputation

**06a_filling_nans_train.py**, **06b_filling_nans_validation.py**, **06c_filling_nans_test.py**

These notebooks handle null values in generation data using an optimized approach:

1. **Data Filtering**: Removes countries with missing generation data (DK, FI, LV, SE, EE, GR, RO, SI, NO, CH, BG)
2. **Column Selection**: Targets generation columns ending in `__Actual_Aggregated` or `__Actual_Consumption`
3. **Evaluation Framework**: Tests three imputation methods on artificially masked data (10% held-out per country):
   - Mean imputation (country-level average)
   - Forward-fill + Backward-fill (temporal interpolation)
   - Median imputation (country-level median)
4. **Method Selection**: Chooses best method per column based on Mean Absolute Error (MAE)
5. **Final Imputation**: Applies selected methods, then fills remaining nulls with zero (appropriate for unreported energy sources like nuclear)

Output: `train_set_imputed`, `validation_set_imputed`, `test_set_imputed` tables

## Data Schema

### Key Tables

**weather_europe**
- `country` (string): ISO country code
- `timestamp` (timestamp): Hourly timestamp
- `mean_ssrd` (double): Mean solar radiation (W/m²)
- `mean_wind_speed` (double): Mean wind speed (m/s)
- `mean_temperature_c` (double): Mean temperature (°C)

**generation_clean**
- `country` (string): ISO country code
- `index` (timestamp): Hourly timestamp
- Generation columns by type: Nuclear, Hydro variants, Solar, Wind (Onshore/Offshore), Biomass, Fossil types, etc.

**electricity_and_weather_europe**
- `country` (string): ISO country code
- `index` (timestamp): Hourly timestamp
- Weather metrics (mean_*)
- Generation data by type
- Load actual and forecast data
- Solar and wind generation forecasts
- `net_imports` (double): Net electricity flow in MW (positive = importing, negative = exporting)

**electricity_and_weather_europe_with_target**
- All columns from above, plus:
- `reserve_margin_ml` (double): Reserve margin ratio
- `load_rel_error` (double): Relative load forecast error
- `grid_stress_score` (double): Total stress score (0-100)
- `score_reserve_margin`, `score_load_error`, `score_T7`, `score_T8` (double): Component scores

## Supported Countries

Spain, Portugal, France, Germany, Italy, Great Britain, Netherlands, Belgium, Austria, Switzerland, Poland, Czech Republic, Denmark, Sweden, Norway, Finland, Greece, Ireland, Romania, Bulgaria, Hungary, Slovakia, Slovenia, Croatia, Estonia, Lithuania, Latvia

## Technologies

- **Apache Spark**: Distributed data processing
- **Databricks**: Platform and execution environment
- **Python**: PySpark for data transformations
- **Libraries**: pyspark.sql, pyspark.sql.functions, reverse_geocode

## Usage

Run the notebooks in order:

```
01_weather_data_processing.py
02_generation_data_processing.py
03_all_tables_processing.py
04_define_target_variable.py
05_train_val_test_split.py
06a_filling_nans_train.py
06b_filling_nans_validation.py
06c_filling_nans_test.py
```
Each notebook produces a Spark table ready for the next stage.

**Automated Pipeline (Hourly)**<br>
For live/real-time data processing, use the automated pipeline:<br>
**Location**: /data_processing_pipeline<br>
**Execution Schedule**: Every 1 hour (configurable) <br>
**Trigger**: Databricks Job Scheduler (after new ENTSO-E data arrives)<br>
The automated pipeline follows the same logic as the manual notebooks but:
- Reads live data from source APIs (ENTSO-E, Copernicus)
- Applies transformations incrementally (append-only)
- Updates live tables: workspace.live_data.*
- Runs as scheduled Databricks jobs

## Target Variable: Grid Stress Score

The grid stress score (0-100) combines three independent indicators:

| Indicator | Low (0 pts) | Medium (12.5 pts) | High (25 pts) | Binary (0/25 pts) |
|-----------|-------------|-------------------|---------------|-------------------|
| Reserve Margin | \|RM\| ≥ 20% | 10% ≤ \|RM\| < 20% | \|RM\| < 10% | — |
| Load Error | Error ≤ 3% | 3% < Error ≤ 10% | Error > 10% | — |
| High Exports | — | — | — | net_imports < P10 |
| High Imports | — | — | — | net_imports > P90 |

## Output Datasets

- `train_set_imputed`: Training data with imputed values (ready for model training)
- `validation_set_imputed`: Validation data for hyperparameter tuning
- `test_set_imputed`: Test data for final model evaluation

Each contains the complete feature set plus `grid_stress_score` target variable.

## Next Steps

These imputed datasets are ready for:
- Feature engineering and selection
- Time-series modeling
- Classification or regression for grid stress prediction
- Anomaly detection during high-stress periods
- Forecasting of critical grid events

## Notes

- Reverse geocoding maps all unique coordinates to country codes
- Countries without generation reporting are excluded after filtering
- Imputation methods are evaluated per country to account for regional patterns
- Remaining nulls after imputation are filled with zeros (appropriate for unreported energy types)
- Temporal windows for reserve margin and fills are country-specific to handle data asynchronicity

## Authors

Capstone Project - European Power Grid Stress Prediction  
December 2025<br> 
Team 6 - GridWatch:<br> 
Chavely Albert Fernandez<br>
Pedro Miguel<br>
Ya-Chi Hsiao<br>
Maria Sokotushchenko


## Acknowledgments

- ENTSOE Transparency Platform for grid data
- ECMWF for Climate reanalyses
- OpenMeteo for the live weather data