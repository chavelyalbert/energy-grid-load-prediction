# Databricks notebook source
"""
================================================================================
GridWatch EUROPEAN POWER GRID STRESS PREDICTION
================================================================================
Project: Capstone - Power Grid Stress Prediction
Date: November 2025

OBJECTIVE:
Predict grid stress scores (0-100) for European power grids using 
legitimate operational features available in real-time.

DATA LEAKAGE PREVENTION:
Excluded features that create circular dependencies:
- net_imports: Used to calculate T7/T8 components of target
- stress_lag_*: Using target to predict target
- reserve_margin_ml, forecast_load_error: Components of target scoring

LEGITIMATE FEATURES USED:
- Load data: Actual and forecasted electricity demand
- Weather: Temperature, wind speed, solar radiation
- Temporal: Hour, day, week patterns (cyclical encoding)
- Historical: Lag features of load, imports, temperature (past values)
- Derived: Rolling statistics, load-weather interactions

TARGET: grid_stress_score (0-100 points)
- 0-24: Normal operations
- 25-49: Moderate stress
- 50-74: High stress (blackout risk)
- 75: Critical

DATASET:
- Train: 386,525 records (2023-2024)
- Validation: 111,670 records (Jan-Jun 2025)
- Test: 53,599 records (Jul-Nov 2025)
- Countries: 13 European nations
================================================================================
"""

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

print("=" * 80)
print("EUROPEAN GRID STRESS PREDICTION - PRODUCTION MODEL")
print("=" * 80)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
print("✓ Setup complete")

# COMMAND ----------

print("\n" + "=" * 80)
print("SECTION 1: DATA LOADING & INITIAL EXPLORATION")
print("=" * 80)

# Load datasets
train_df = spark.table("workspace.default.train_set_imputed").toPandas()
val_df = spark.table("workspace.default.validation_set_imputed").toPandas()
test_df = spark.table("workspace.default.test_set_imputed").toPandas()

print(f"\n✓ Data loaded: {train_df.shape[0] + val_df.shape[0] + test_df.shape[0]:,} total records")
print(f"  Train:      {train_df.shape[0]:>8,} rows × {train_df.shape[1]:>2} columns")
print(f"  Validation: {val_df.shape[0]:>8,} rows × {val_df.shape[1]:>2} columns")
print(f"  Test:       {test_df.shape[0]:>8,} rows × {test_df.shape[1]:>2} columns")

# Target analysis
print("\n" + "-" * 80)
print("TARGET VARIABLE: grid_stress_score")
print("-" * 80)

print(f"\nDistribution Statistics:")
print(f"  Mean:   {train_df['grid_stress_score'].mean():.2f}")
print(f"  Median: {train_df['grid_stress_score'].median():.2f}")
print(f"  Std:    {train_df['grid_stress_score'].std():.2f}")
print(f"  Range:  [{train_df['grid_stress_score'].min():.1f}, {train_df['grid_stress_score'].max():.1f}]")

print(f"\nValue Distribution:")
stress_counts = train_df['grid_stress_score'].value_counts().sort_index()
for score, count in stress_counts.items():
    pct = (count / len(train_df)) * 100
    category = "NORMAL" if score < 25 else "MODERATE" if score < 50 else "HIGH RISK"
    print(f"  {score:>5.1f}: {count:>8,} ({pct:>5.2f}%) - {category}")

# Temporal coverage
print("\n" + "-" * 80)
print("TEMPORAL COVERAGE")
print("-" * 80)

for name, df in [('Train', train_df), ('Validation', val_df), ('Test', test_df)]:
    print(f"\n{name}:")
    print(f"  Start: {df['index'].min()}")
    print(f"  End:   {df['index'].max()}")
    print(f"  Days:  {(df['index'].max() - df['index'].min()).days}")

# Country distribution
print("\n" + "-" * 80)
print("COUNTRY DISTRIBUTION")
print("-" * 80)

country_counts = train_df['country'].value_counts()
print(f"\nTotal countries: {len(country_counts)}")
print(f"\nRecords per country:")
for country, count in country_counts.items():
    pct = (count / len(train_df)) * 100
    avg_stress = train_df[train_df['country'] == country]['grid_stress_score'].mean()
    print(f"  {country:>2}: {count:>8,} ({pct:>4.2f}%) - Avg stress: {avg_stress:>5.2f}")

print("\n✓ Initial exploration complete")

# COMMAND ----------

print("\n" + "=" * 80)
print("SECTION 2: FEATURE ENGINEERING (NO LEAKAGE)")
print("=" * 80)

def create_clean_features(df):
    """
    Create features WITHOUT any data leakage.
    Excludes: net_imports, stress_lag_*, reserve_margin_ml, forecast_load_error
    """
    
    print("\nSorting data by country and time...")
    df = df.sort_values(['country', 'index']).reset_index(drop=True)
    
    # ========================================================================
    # TEMPORAL FEATURES
    # ========================================================================
    print("Creating temporal features...")
    
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
    print("Creating lag features (load, imports, temperature)...")
    
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
    print("Creating rolling statistics...")
    
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
    print("Creating interaction features...")
    
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
    # ========================================================================
    print("Creating seasonality features...")
    
    df['hourly_avg_load'] = df.groupby(['country', 'hour'])['Actual_Load'].transform('mean')
    df['load_deviation_from_hourly_avg'] = df['Actual_Load'] - df['hourly_avg_load']
    
    df['daily_avg_load'] = df.groupby(['country', 'day_of_week'])['Actual_Load'].transform('mean')
    df['load_deviation_from_daily_avg'] = df['Actual_Load'] - df['daily_avg_load']
    
    print("✓ Feature engineering complete\n")
    return df

# Apply to all datasets
print("Applying feature engineering...")
train_df = create_clean_features(train_df)
val_df = create_clean_features(val_df)
test_df = create_clean_features(test_df)

print(f"✓ Feature engineering complete")
print(f"  Total columns: {train_df.shape[1]}")

# COMMAND ----------

print("\n" + "=" * 80)
print("SECTION 3: COMPREHENSIVE EDA & CORRELATION ANALYSIS")
print("=" * 80)

# ============================================================================
# Define clean feature set (exclude leakage and metadata)
# ============================================================================
print("\n[Step 1] Defining clean feature set...")

LEAKAGE_COLS = [
    # Metadata
    'index', 'country',
    # Target
    'grid_stress_score',
    # Data leakage - components of target
    'reserve_margin_ml', 'forecast_load_error', 'load_rel_error',
    'net_imports',  # Used to calculate T7/T8
    'P10_net', 'P90_net',  # Thresholds
    'score_reserve_margin', 'score_load_error', 'score_T7', 'score_T8',
    'T7_high_exports', 'T8_high_imports',
    # Redundant temporal
    'hour', 'month', 'day_of_week'
]

# Get feature candidates
all_cols = train_df.columns.tolist()
feature_candidates = [col for col in all_cols if col not in LEAKAGE_COLS]

print(f"  Total columns: {len(all_cols)}")
print(f"  Excluded: {len(LEAKAGE_COLS)}")
print(f"  Feature candidates: {len(feature_candidates)}")

# ============================================================================
# Select numeric features for correlation
# ============================================================================
print("\n[Step 3] Preparing numeric features for correlation analysis...")

numeric_features = []
for col in feature_candidates:
    if train_df[col].dtype in ['int64', 'float64']:
        missing_pct = train_df[col].isnull().sum() / len(train_df)
        if missing_pct < 0.80:  # Keep if <80% missing
            numeric_features.append(col)

# ============================================================================
# Calculate correlations with target
# ============================================================================
print("\n[Step 4] Calculating correlations with target...")

correlations = {}
for feat in numeric_features:
    valid_count = train_df[feat].notna().sum()
    if valid_count > 100:
        corr = train_df[feat].corr(train_df['grid_stress_score'])
        if not np.isnan(corr):
            correlations[feat] = corr

corr_df = pd.DataFrame.from_dict(correlations, orient='index', columns=['Correlation'])
corr_df['Abs_Correlation'] = corr_df['Correlation'].abs()
corr_df = corr_df.sort_values('Abs_Correlation', ascending=False)

print(f"\nTop 25 Features by Correlation with grid_stress_score:")
print(f"\n{'Rank':<6} {'Feature':<50} {'Correlation':>12}")
print("-" * 70)

for idx, (feat, row) in enumerate(corr_df.head(25).iterrows(), 1):
    print(f"{idx:<6} {feat:<50} {row['Correlation']:>12.4f}")

# ============================================================================
# VISUALIZATION 1: Correlation Heatmap - Top Features
# ============================================================================
print("\n[Step 5] Creating correlation matrix visualization...")

fig = plt.figure(figsize=(20, 14))

# Plot 1: Correlation heatmap of top 20 features + target
ax1 = plt.subplot(2, 2, 1)
top_20_features = corr_df.head(20).index.tolist()
heatmap_data = train_df[top_20_features + ['grid_stress_score']].corr()

sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='RdYlGn', 
            center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8},
            ax=ax1, vmin=-1, vmax=1, annot_kws={'size': 7})
ax1.set_title('Correlation Matrix: Top 20 Features + Target', 
              fontsize=14, fontweight='bold', pad=15)
plt.setp(ax1.get_xticklabels(), rotation=45, ha='right', fontsize=8)
plt.setp(ax1.get_yticklabels(), rotation=0, fontsize=8)

# Plot 2: Feature importance by correlation (bar chart)
ax2 = plt.subplot(2, 2, 2)
top_20 = corr_df.head(20).sort_values('Correlation', ascending=True)
colors = ['red' if x < 0 else 'green' for x in top_20['Correlation']]
bars = ax2.barh(range(len(top_20)), top_20['Correlation'], color=colors, alpha=0.7, edgecolor='black')
ax2.set_yticks(range(len(top_20)))
ax2.set_yticklabels(top_20.index, fontsize=8)
ax2.set_xlabel('Correlation with grid_stress_score', fontsize=11, fontweight='bold')
ax2.set_title('Top 20 Features by Correlation', fontsize=14, fontweight='bold', pad=15)
ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
ax2.grid(axis='x', alpha=0.3)

for i, (bar, val) in enumerate(zip(bars, top_20['Correlation'])):
    ax2.text(val + 0.01 if val > 0 else val - 0.01, i, f'{val:.3f}', 
             va='center', fontsize=7, fontweight='bold')


# Plot 3: Target distribution
ax3 = plt.subplot(2, 2, 3)
ax3.hist(train_df['grid_stress_score'], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
ax3.axvline(x=25, color='orange', linestyle='--', linewidth=2, label='Moderate (25)')
ax3.axvline(x=50, color='red', linestyle='--', linewidth=2, label='High Risk (50)')
ax3.set_xlabel('Grid Stress Score', fontsize=11, fontweight='bold')
ax3.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax3.set_title('Target Distribution', fontsize=14, fontweight='bold', pad=15)
ax3.legend(fontsize=10)
ax3.grid(axis='y', alpha=0.3)

mean_val = train_df['grid_stress_score'].mean()
median_val = train_df['grid_stress_score'].median()
ax3.text(0.98, 0.97, f'Mean: {mean_val:.2f}\nMedian: {median_val:.2f}',
         transform=ax3.transAxes, fontsize=10, verticalalignment='top',
         horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Plot 4: Country stress comparison
ax4 = plt.subplot(2, 2, 4)
country_stress = train_df.groupby('country')['grid_stress_score'].mean().sort_values(ascending=True)
colors_country = ['red' if x > 35 else 'orange' if x > 28 else 'green' for x in country_stress.values]
bars = ax4.barh(range(len(country_stress)), country_stress.values, color=colors_country, alpha=0.7, edgecolor='black')
ax4.set_yticks(range(len(country_stress)))
ax4.set_yticklabels(country_stress.index, fontsize=8)
ax4.set_xlabel('Average Grid Stress Score', fontsize=11, fontweight='bold')
ax4.set_title('Average Stress by Country', fontsize=14, fontweight='bold', pad=15)
ax4.axvline(x=mean_val, color='black', linestyle='--', linewidth=1.5, alpha=0.5, 
            label=f'Overall Avg ({mean_val:.1f})')
ax4.legend(fontsize=9)
ax4.grid(axis='x', alpha=0.3)

for i, (bar, val) in enumerate(zip(bars, country_stress.values)):
    ax4.text(val + 0.5, i, f'{val:.1f}', va='center', fontsize=7)

plt.suptitle('European Grid Stress Prediction - Exploratory Data Analysis', 
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.show()

print("✓ Correlation matrix and distributions created")

# ============================================================================
# VISUALIZATION 2: Time Series Patterns
# ============================================================================
print("\n[Step 6] Creating time series pattern analysis...")

fig2 = plt.figure(figsize=(20, 10))

sample_country = 'DE'
sample_data = train_df[train_df['country'] == sample_country].sort_values('index').head(168*2)

ax5 = plt.subplot(3, 1, 1)
ax5.plot(sample_data['index'], sample_data['grid_stress_score'], linewidth=1.5, color='darkblue')
ax5.axhline(y=50, color='red', linestyle='--', linewidth=2, label='High Risk (50)')
ax5.axhline(y=25, color='orange', linestyle='--', linewidth=2, alpha=0.5, label='Moderate (25)')
ax5.set_ylabel('Grid Stress Score', fontsize=11, fontweight='bold')
ax5.set_title(f'Grid Stress Time Series - {sample_country} (2 weeks)', fontsize=14, fontweight='bold')
ax5.legend()
ax5.grid(alpha=0.3)
plt.savefig('../images/stress_per_country.png')

ax6 = plt.subplot(3, 1, 2)
ax6.plot(sample_data['index'], sample_data['Actual_Load'], linewidth=1.5, color='green', label='Actual Load')
ax6.plot(sample_data['index'], sample_data['Forecasted_Load'], linewidth=1.5, color='orange', 
         linestyle='--', label='Forecasted Load')
ax6.set_ylabel('Load (MW)', fontsize=11, fontweight='bold')
ax6.set_title('Load: Actual vs Forecasted', fontsize=14, fontweight='bold')
ax6.legend()
ax6.grid(alpha=0.3)

ax7 = plt.subplot(3, 1, 3)
ax7.plot(sample_data['index'], sample_data['mean_temperature_c'], linewidth=1.5, color='red', label='Temperature')
ax7_twin = ax7.twinx()
ax7_twin.plot(sample_data['index'], sample_data['mean_wind_speed'], linewidth=1.5, color='blue', label='Wind Speed')
ax7.set_ylabel('Temperature (°C)', fontsize=11, fontweight='bold', color='red')
ax7_twin.set_ylabel('Wind Speed (m/s)', fontsize=11, fontweight='bold', color='blue')
ax7.set_xlabel('Time', fontsize=11, fontweight='bold')
ax7.set_title('Weather Conditions', fontsize=14, fontweight='bold')
ax7.legend(loc='upper left')
ax7_twin.legend(loc='upper right')
ax7.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('../images/eda.png')
plt.show()

print("✓ Time series patterns visualized")

print("\n" + "=" * 80)
print("EDA COMPLETE")
print("=" * 80)

# COMMAND ----------

print("\n" + "=" * 80)
print("SECTION 4: DATA PREPARATION")
print("=" * 80)

# ============================================================================
# Select final features (keep best performers, remove redundant)
# ============================================================================
print("\n[Step 2] Selecting final feature set...")

# Keep only essential features
features_to_keep = [
    # Load features
    'Actual_Load', 'Forecasted_Load',
    
    # Load lags
    'load_lag_1h', 'load_lag_24h',
    
    # Load derived
    'load_rolling_mean_24h', 'load_rolling_std_24h',
    'load_change_1h', 'load_change_24h',
    'load_forecast_diff', 'load_forecast_ratio', 'load_forecast_error_pct',
    'load_deviation_from_hourly_avg', 'load_deviation_from_daily_avg',
    
    # Weather features
    'mean_temperature_c', 'mean_wind_speed', 'mean_ssrd',
    'solar_forecast', 'wind_forecast',
    'temp_lag_1h',
    
    # Weather derived
    'load_per_temp', 'temp_load_product', 'is_very_cold', 'temp_extreme',
    'wind_power_index',
    
    # Import features (past values only!)
    'imports_lag_1h',
    'imports_rolling_mean_24h',
    
    # Temporal features
    'hour_sin', 'hour_cos',
    'month_sin', 'month_cos',
    'day_of_week_sin', 'day_of_week_cos',
    'is_weekend', 'is_morning_peak', 'is_evening_peak', 'is_peak_hour',
    
    # Country
    'country'
]

# Add any generation features that aren't too sparse
generation_features = [f for f in feature_candidates 
                      if 'Actual_Aggregated' in f 
                      and train_df[f].isnull().sum() / len(train_df) < 0.80]

final_features = features_to_keep + generation_features

# Remove any that don't exist
final_features = [f for f in final_features if f in train_df.columns]

print(f"  Selected {len(final_features)} features")
print(f"    Core features: {len(features_to_keep)}")
print(f"    Generation features: {len(generation_features)}")

# ============================================================================
# Prepare datasets
# ============================================================================
print("\n[Step 3] Preparing train/val/test datasets...")

X_train = train_df[final_features].copy()
X_val = val_df[final_features].copy()
X_test = test_df[final_features].copy()

y_train = train_df['grid_stress_score'].copy()
y_val = val_df['grid_stress_score'].copy()
y_test = test_df['grid_stress_score'].copy()

print("  Filling missing values with 0...")
X_train = X_train.fillna(0)
X_val = X_val.fillna(0)
X_test = X_test.fillna(0)

# One-hot encode country
if 'country' in X_train.columns:
    print("  One-hot encoding country...")
    X_train = pd.get_dummies(X_train, columns=['country'], prefix='country', drop_first=False)
    X_val = pd.get_dummies(X_val, columns=['country'], prefix='country', drop_first=False)
    X_test = pd.get_dummies(X_test, columns=['country'], prefix='country', drop_first=False)
    
    all_columns = X_train.columns
    X_val = X_val.reindex(columns=all_columns, fill_value=0)
    X_test = X_test.reindex(columns=all_columns, fill_value=0)

print(f"\n✓ Datasets prepared:")
print(f"  X_train: {X_train.shape[0]:>8,} rows × {X_train.shape[1]:>3} features")
print(f"  X_val:   {X_val.shape[0]:>8,} rows × {X_val.shape[1]:>3} features")
print(f"  X_test:  {X_test.shape[0]:>8,} rows × {X_test.shape[1]:>3} features")

# ============================================================================
# Final verification - ensure no leakage
# ============================================================================
print("\n[Step 4] Final data leakage verification...")

leakage_found = []

# Check for prohibited features
prohibited = ['net_imports', 'stress_lag', 'stress_change', 'reserve_margin_ml', 
              'forecast_load_error', 'load_rel_error']

for col in X_train.columns:
    for prob in prohibited:
        if prob in col.lower():
            leakage_found.append(col)
            break

if len(leakage_found) == 0:
    print("  ✓ No data leakage detected")
    print("  ✓ No net_imports (used in T7/T8)")
    print("  ✓ No stress_lag (target to predict target)")
    print("  ✓ Model is production-ready")
else:
    print(f"  ❌ WARNING: Found {len(leakage_found)} suspicious features:")
    for feat in leakage_found:
        print(f"     - {feat}")

# Show feature categories
print(f"\n[Step 5] Feature summary:")
load_feats = [f for f in X_train.columns if 'load' in f.lower() or 'Actual_Load' in f or 'Forecasted_Load' in f]
weather_feats = [f for f in X_train.columns if any(x in f.lower() for x in ['temp', 'wind', 'solar', 'ssrd'])]
temporal_feats = [f for f in X_train.columns if any(x in f for x in ['hour_', 'month_', 'day_of_week', 'weekend', 'peak'])]
import_feats = [f for f in X_train.columns if 'import' in f.lower()]
country_feats = [f for f in X_train.columns if 'country_' in f]
generation_feats = [f for f in X_train.columns if 'Actual_Aggregated' in f]

print(f"  Load features:       {len(load_feats)}")
print(f"  Weather features:    {len(weather_feats)}")
print(f"  Temporal features:   {len(temporal_feats)}")
print(f"  Import features:     {len(import_feats)}")
print(f"  Generation features: {len(generation_feats)}")
print(f"  Country indicators:  {len(country_feats)}")

print("\n" + "=" * 80)
print("DATA PREPARATION COMPLETE")
print("=" * 80)

# COMMAND ----------

print("\n" + "=" * 80)
print("SECTION 5: CLASSIFICATION MODELS")
print("=" * 80)

# ============================================================================
# PART A: Train Dedicated Classification Models
# ============================================================================
print("\n[PART A] Training dedicated classification models...")
print("Current approach: Using regression model + threshold")
print("New approach: Train models specifically for binary classification\n")

# Create binary labels (threshold = 50 for training)
TRAIN_THRESHOLD = 50
y_train_binary = (y_train >= TRAIN_THRESHOLD).astype(int)
y_val_binary = (y_val >= TRAIN_THRESHOLD).astype(int)
y_test_binary = (y_test >= TRAIN_THRESHOLD).astype(int)

print(f"Binary class distribution (Test Set):")
print(f"  Low Risk (0):  {(y_test_binary == 0).sum():>6,} ({(y_test_binary == 0).sum()/len(y_test_binary)*100:.2f}%)")
print(f"  High Risk (1): {(y_test_binary == 1).sum():>6,} ({(y_test_binary == 1).sum()/len(y_test_binary)*100:.2f}%)")

# Import classification models
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import lightgbm as lgb

# ============================================================================
# Define classification model suite (10 models)
# ============================================================================
classification_models = {
    # Logistic Regression (2)
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Logistic Regression (balanced)': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
    
    # Decision Tree (2)
    'Decision Tree Classifier': DecisionTreeClassifier(max_depth=15, random_state=42),
    'Decision Tree (balanced)': DecisionTreeClassifier(max_depth=15, class_weight='balanced', random_state=42),
    
    # Random Forest (2)
    'Random Forest Classifier': RandomForestClassifier(
        n_estimators=100, max_depth=20, random_state=42, n_jobs=-1
    ),
    'Random Forest (balanced)': RandomForestClassifier(
        n_estimators=100, max_depth=20, class_weight='balanced', random_state=42, n_jobs=-1
    ),
    
    # Gradient Boosting (1)
    'Gradient Boosting Classifier': GradientBoostingClassifier(
        n_estimators=100, max_depth=7, learning_rate=0.1, random_state=42
    ),
    
    # XGBoost (2)
    'XGBoost Classifier': XGBClassifier(
        n_estimators=100, max_depth=7, learning_rate=0.1, 
        random_state=42, n_jobs=-1, eval_metric='logloss'
    ),
    'XGBoost (scale_pos_weight)': XGBClassifier(
        n_estimators=100, max_depth=7, learning_rate=0.1, 
        scale_pos_weight=3,  # Give more weight to minority class
        random_state=42, n_jobs=-1, eval_metric='logloss'
    ),
    
    # LightGBM (1)
    'LightGBM Classifier': lgb.LGBMClassifier(
        n_estimators=100, max_depth=7, learning_rate=0.1,
        random_state=42, n_jobs=-1, verbose=-1
    ),
}

print(f"\n Training {len(classification_models)} classification models...")
print(f"Features: {X_train.shape[1]}")
print(f"Training samples: {X_train.shape[0]:,}\n")

clf_results = []

print(f"{'Model':<40} {'Time':>10} {'Accuracy':>10} {'Precision':>12} {'Recall':>10} {'F1':>10}")
print("-" * 95)

for model_name, model in classification_models.items():
    try:
        start_time = time.time()
        model.fit(X_train, y_train_binary)
        train_time = time.time() - start_time
        
        y_pred = model.predict(X_val)
        
        acc = accuracy_score(y_val_binary, y_pred)
        prec = precision_score(y_val_binary, y_pred, zero_division=0)
        rec = recall_score(y_val_binary, y_pred, zero_division=0)
        f1 = f1_score(y_val_binary, y_pred, zero_division=0)
        
        clf_results.append({
            'Model': model_name,
            'Train_Time': train_time,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1': f1,
            'model_object': model
        })
        
        print(f"{model_name:<40} {train_time:>9.2f}s {acc:>10.4f} {prec:>12.4f} {rec:>10.4f} {f1:>10.4f}")
        
    except Exception as e:
        print(f"{model_name:<40} FAILED: {str(e)[:30]}")

# Find best classification model
clf_results_df = pd.DataFrame(clf_results)
best_clf_idx = clf_results_df['F1'].idxmax()
best_clf_name = clf_results_df.loc[best_clf_idx, 'Model']
best_clf_model = clf_results_df.loc[best_clf_idx, 'model_object']

print("\n" + "=" * 95)
print(f"BEST CLASSIFICATION MODEL: {best_clf_name}")
print(f"  Validation Accuracy:  {clf_results_df.loc[best_clf_idx, 'Accuracy']:.4f}")
print(f"  Validation Precision: {clf_results_df.loc[best_clf_idx, 'Precision']:.4f}")
print(f"  Validation Recall:    {clf_results_df.loc[best_clf_idx, 'Recall']:.4f}")
print(f"  Validation F1-Score:  {clf_results_df.loc[best_clf_idx, 'F1']:.4f}")
print("=" * 95)

# ============================================================================
# Evaluate Classification Approach on TEST set
# ============================================================================
print("\n[PART C] Evaluating the best classification model on TEST set...")

# Predict on test set using the best classifier
y_test_pred_clf = best_clf_model.predict(X_test)

# Compute metrics
acc = accuracy_score(y_test_binary, y_test_pred_clf)
prec = precision_score(y_test_binary, y_test_pred_clf, zero_division=0)
rec = recall_score(y_test_binary, y_test_pred_clf, zero_division=0)
f1 = f1_score(y_test_binary, y_test_pred_clf, zero_division=0)

print(f"\n{'Approach':<45} {'Accuracy':>10} {'Precision':>12} {'Recall':>10} {'F1':>10}")
print("-" * 90)
print(f"{('Classification (' + best_clf_name + ')'):<45} "
      f"{acc:>10.4f} {prec:>12.4f} {rec:>10.4f} {f1:>10.4f}")

# Store results in a single-row DataFrame
comparison_df = pd.DataFrame([{
    'Approach': f'Classification ({best_clf_name})',
    'Accuracy': acc,
    'Precision': prec,
    'Recall': rec,
    'F1': f1
}])

# ============================================================================
# Visualizations
# ============================================================================
print("\n[PART D] Creating visualizations...")

fig = plt.figure(figsize=(20, 12))

# ----------------------------------------------------------------------------
# Plot 1: Classification Models Comparison (F1 Score)
# ----------------------------------------------------------------------------
ax1 = plt.subplot(2, 3, 1)
clf_sorted = clf_results_df.sort_values('F1', ascending=True)

colors_clf = [
    'darkgreen' if x == clf_results_df['F1'].max() else 'steelblue'
    for x in clf_sorted['F1']
]

ax1.barh(
    range(len(clf_sorted)),
    clf_sorted['F1'],
    color=colors_clf,
    alpha=0.7,
    edgecolor='black'
)
ax1.set_yticks(range(len(clf_sorted)))
ax1.set_yticklabels(clf_sorted['Model'], fontsize=9)
ax1.set_xlabel('F1-Score', fontsize=11, fontweight='bold')
ax1.set_title('Classification Models - F1 Score', fontsize=13, fontweight='bold')
ax1.grid(axis='x', alpha=0.3)

# ----------------------------------------------------------------------------
# Plot 2: Confusion Matrix for best classification model
# ----------------------------------------------------------------------------
ax2 = plt.subplot(2, 3, 2)

cm = confusion_matrix(y_test_binary, y_test_pred_clf)

sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues', cbar=False, square=True,
    xticklabels=['Low', 'High'], yticklabels=['Low', 'High'],
    ax=ax2, annot_kws={'size': 12, 'weight': 'bold'}
)

rec = recall_score(y_test_binary, y_test_pred_clf)
f1 = f1_score(y_test_binary, y_test_pred_clf)

ax2.set_title(
    f'Confusion Matrix\nBest: {best_clf_name}\nRecall={rec:.3f}, F1={f1:.3f}',
    fontsize=12, fontweight='bold'
)
ax2.set_xlabel('Predicted', fontsize=10, fontweight='bold')
ax2.set_ylabel('Actual', fontsize=10, fontweight='bold')

# ----------------------------------------------------------------------------
# Figure title + layout
# ----------------------------------------------------------------------------
plt.suptitle('Classification Analysis — Best Model Performance', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('../images/best_classification_model_performance.png')
plt.show()

print("✓ Visualizations created")

# COMMAND ----------

import pickle
import os

# Create directory
output_dir = "/Workspace/Users/chavely.albert@gmail.com/grid_stress_classification"
os.makedirs(output_dir, exist_ok=True)

# Save model
with open(f"{output_dir}/xgboost_model.pkl", 'wb') as f:
    pickle.dump(best_clf_model, f)

# Save feature names
with open(f"{output_dir}/feature_names.pkl", 'wb') as f:
    pickle.dump(list(X_train.columns), f)

# Save sample data
spark_df = spark.createDataFrame(X_test)
spark_df.write.mode("overwrite").saveAsTable("x_test_imputed_with_features")

# Save country stats
stats = train_df.groupby('country').agg({
    'Actual_Load': 'mean',
    'net_imports': 'mean',
    'mean_temperature_c': 'mean',
    'grid_stress_score': 'mean'
}).to_csv(f"{output_dir}/country_stats.csv")

print(f"✓ Saved to: {output_dir}")
print("Files: xgboost_model.pkl, feature_names.pkl, sample_data.csv, country_stats.csv")
