# Databricks notebook source
# Install packages
%pip install xgboost==2.0.3 lightgbm==4.1.0

# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
from xgboost import XGBRegressor
import lightgbm as lgb

pd.set_option('display.max_columns', None)
sns.set_style('whitegrid')

# COMMAND ----------

# Load datasets
data_df = spark.table("workspace.live_data.electricity_and_weather_europe_imputed").toPandas()

# COMMAND ----------

# FEATURE ENGINEERING

def create_clean_features(df):
    """
    Create features WITHOUT any data leakage.
    Excludes: net_imports, stress_lag_*, reserve_margin_ml, forecast_load_error
    """

    df = df.sort_values(['country', 'index']).reset_index(drop=True)
    
    # ========================================================================
    # TEMPORAL FEATURES
    # ========================================================================
    
    df['hour'] = df['index'].dt.hour
    df['month'] = df['index'].dt.month
    df['day_of_week'] = df['index'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # Cyclical encoding
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    # Peak hours
    df['is_morning_peak'] = df['hour'].isin([7, 8, 9]).astype(int)
    df['is_evening_peak'] = df['hour'].isin([18, 19, 20, 21]).astype(int)
    df['is_peak_hour'] = (df['is_morning_peak'] | df['is_evening_peak']).astype(int)
    
    # ========================================================================
    # LAG FEATURES (Using past values only - NO stress lags!)
    # ========================================================================
    
    # Load lags
    for lag in [1, 24]:
        df[f'load_lag_{lag}h'] = df.groupby('country')['Actual_Load'].shift(lag)
    
    # Import lags (using past net_imports - legitimate!)
    df['imports_lag_1h'] = df.groupby('country')['net_imports'].shift(1)
    
    # Temperature lags
    df['temp_lag_1h'] = df.groupby('country')['mean_temperature_c'].shift(1)
    
    # ========================================================================
    # ROLLING STATISTICS
    # ========================================================================
    
    df['load_rolling_mean_24h'] = df.groupby('country')['Actual_Load'].transform(
        lambda x: x.rolling(window=24, min_periods=1).mean()
    )
    df['load_rolling_std_24h'] = df.groupby('country')['Actual_Load'].transform(
        lambda x: x.rolling(window=24, min_periods=1).std()
    )
    
    df['imports_rolling_mean_24h'] = df.groupby('country')['net_imports'].transform(
        lambda x: x.shift(1).rolling(window=24, min_periods=1).mean()
    )
    
    # Change features
    df['load_change_1h'] = df.groupby('country')['Actual_Load'].diff(1)
    df['load_change_24h'] = df.groupby('country')['Actual_Load'].diff(24)
    
    # ========================================================================
    # INTERACTION FEATURES
    # ========================================================================
    
    # Load-forecast interactions
    df['load_forecast_diff'] = df['Actual_Load'] - df['Forecasted_Load']
    df['load_forecast_ratio'] = df['Actual_Load'] / (df['Forecasted_Load'] + 1e-6)
    df['load_forecast_error_pct'] = np.abs(df['load_forecast_diff']) / (df['Forecasted_Load'] + 1e-6) * 100
    
    # Weather-load interactions
    df['load_per_temp'] = df['Actual_Load'] / (df['mean_temperature_c'] + 20)
    df['temp_load_product'] = df['mean_temperature_c'] * df['Actual_Load'] / 10000
    
    # Weather extremes
    df['is_very_cold'] = (df['mean_temperature_c'] < 0).astype(int)
    df['temp_extreme'] = df['is_very_cold'].astype(int)
    
    # Wind power potential
    df['wind_power_index'] = df['mean_wind_speed'] ** 3 / 100
    
    # ========================================================================
    # SEASONALITY
    
    df['hourly_avg_load'] = df.groupby(['country', 'hour'])['Actual_Load'].transform('mean')
    df['load_deviation_from_hourly_avg'] = df['Actual_Load'] - df['hourly_avg_load']
    
    df['daily_avg_load'] = df.groupby(['country', 'day_of_week'])['Actual_Load'].transform('mean')
    df['load_deviation_from_daily_avg'] = df['Actual_Load'] - df['daily_avg_load']
    
    print("✓ Feature engineering complete\n")
    return df

# Apply to datasets
data_df = create_clean_features(data_df)

# COMMAND ----------

# DATA PREPARATION

# ==========================================================================
# Select final features (keep best performers, remove redundant)
# ==========================================================================

features_to_keep = [
    'Actual_Load', 'Forecasted_Load',
    'load_lag_1h', 'load_lag_24h',
    'load_rolling_mean_24h', 'load_rolling_std_24h',
    'load_change_1h', 'load_change_24h',
    'load_forecast_diff', 'load_forecast_ratio', 'load_forecast_error_pct',
    'load_deviation_from_hourly_avg', 'load_deviation_from_daily_avg',
    'mean_temperature_c', 'mean_wind_speed', 'mean_ssrd',
    'solar_forecast', 'wind_forecast', 'temp_lag_1h',
    'load_per_temp', 'temp_load_product', 'is_very_cold', 'temp_extreme',
    'wind_power_index', 'imports_lag_1h', 'imports_rolling_mean_24h',
    'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
    'day_of_week_sin', 'day_of_week_cos', 'is_weekend', 'is_morning_peak', 'is_evening_peak', 'is_peak_hour',
    'country', 'grid_stress_score','index','net_imports'
]

# grid_stress_score stays as a columns for plotting purposes as 24h previous values in the time series plot.
# it was excluded from the file features_names.pkl saved when training the model to avoid leakage
# similar happens to net_imports, which is used to determine the stress score using real data, and it's visualized in the time series plot

LEAKAGE_COLS = [
    'grid_stress_score', 'reserve_margin_ml', 'forecast_load_error', 'load_rel_error',
    'net_imports', 'P10_net', 'P90_net', 'score_reserve_margin', 'score_load_error', 'score_T7', 'score_T8',
    'T7_high_exports', 'T8_high_imports', 'hour', 'month', 'day_of_week'
]

all_cols = data_df.columns.tolist()
feature_candidates = [col for col in all_cols if col not in LEAKAGE_COLS]
generation_features = [f for f in feature_candidates if 'Actual_Aggregated' in f and data_df[f].isnull().sum() / len(data_df) < 0.80]
final_features = features_to_keep + generation_features
final_features = [f for f in final_features if f in data_df.columns]

data_with_features = data_df[final_features].copy()
data_with_features = data_with_features.fillna(0)

# One-hot encode country
if 'country' in data_with_features.columns:
    data_with_features = pd.get_dummies(data_with_features, columns=['country'], prefix='country', drop_first=False)

# Reconstruct country column from one-hot encoding
country_cols = [col for col in data_with_features.columns if col.startswith('country_')]
if country_cols:
    data_with_features['country'] = data_with_features[country_cols].idxmax(axis=1).str.replace('country_', '')


# COMMAND ----------

schema_name = "live_data"

# Save data
spark_df = spark.createDataFrame(data_with_features)
spark_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{schema_name}.electricity_and_weather_europe_imputed_with_features")
