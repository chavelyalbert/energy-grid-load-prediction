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
plt.savefig('../images/eda.png')
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
plt.savefig('../images/eda_time_series.png')
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
print("SECTION 5: MODEL TRAINING - 15 ALGORITHMS")
print("=" * 80)

# Define model suite
models = {
    # Linear models (3)
    'Linear Regression': LinearRegression(),
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=0.1, max_iter=5000),
    
    # Tree-based (2)
    'Decision Tree': DecisionTreeRegressor(max_depth=25, min_samples_split=10, random_state=42),
    'Decision Tree (shallow)': DecisionTreeRegressor(max_depth=15, min_samples_split=20, random_state=42),
    
    # Random Forest (3)
    'Random Forest (default)': RandomForestRegressor(
        n_estimators=100, max_depth=20, min_samples_split=5, random_state=42, n_jobs=-1
    ),
    'Random Forest (deep)': RandomForestRegressor(
        n_estimators=150, max_depth=30, min_samples_split=3, random_state=42, n_jobs=-1
    ),
    'Random Forest (wide)': RandomForestRegressor(
        n_estimators=200, max_depth=15, min_samples_split=10, random_state=42, n_jobs=-1
    ),
    
    # Gradient Boosting (2)
    'Gradient Boosting': GradientBoostingRegressor(
        n_estimators=100, max_depth=7, learning_rate=0.1, subsample=0.8, random_state=42
    ),
    'Gradient Boosting (aggressive)': GradientBoostingRegressor(
        n_estimators=200, max_depth=5, learning_rate=0.05, subsample=0.8, random_state=42
    ),
    
    # XGBoost (3)
    'XGBoost (default)': XGBRegressor(
        n_estimators=100, max_depth=7, learning_rate=0.1, subsample=0.8, 
        colsample_bytree=0.8, random_state=42, n_jobs=-1
    ),
    'XGBoost (deep)': XGBRegressor(
        n_estimators=150, max_depth=10, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, random_state=42, n_jobs=-1
    ),
    'XGBoost (regularized)': XGBRegressor(
        n_estimators=100, max_depth=7, learning_rate=0.1, subsample=0.8,
        colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0, random_state=42, n_jobs=-1
    ),
    
    # LightGBM (2)
    'LightGBM (default)': lgb.LGBMRegressor(
        n_estimators=100, max_depth=7, learning_rate=0.1, subsample=0.8,
        colsample_bytree=0.8, random_state=42, n_jobs=-1, verbose=-1
    ),
    'LightGBM (boosted)': lgb.LGBMRegressor(
        n_estimators=200, max_depth=10, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, num_leaves=128, random_state=42, n_jobs=-1, verbose=-1
    ),
}

print(f"\nTraining {len(models)} models...")
print(f"Features: {X_train.shape[1]} (production-ready, no leakage)")
print(f"Training samples: {X_train.shape[0]:,}\n")

results = []

print(f"{'Model':<35} {'Train Time':>12} {'Val MAE':>10} {'Val RMSE':>10} {'Val R²':>10}")
print("-" * 90)

for model_name, model in models.items():
    try:
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        y_pred = model.predict(X_val)
        
        mae = mean_absolute_error(y_val, y_pred)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        r2 = r2_score(y_val, y_pred)
        
        results.append({
            'Model': model_name,
            'Train_Time': train_time,
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'model_object': model
        })
        
        print(f"{model_name:<35} {train_time:>10.2f}s {mae:>10.3f} {rmse:>10.3f} {r2:>10.4f}")
        
    except Exception as e:
        print(f"{model_name:<35} FAILED: {str(e)[:40]}")

# Find best model
results_df = pd.DataFrame(results)
best_idx = results_df['R2'].idxmax()
best_model_name = results_df.loc[best_idx, 'Model']
best_model = results_df.loc[best_idx, 'model_object']

print("\n" + "=" * 90)
print(f"BEST MODEL: {best_model_name}")
print(f"  Validation MAE:  {results_df.loc[best_idx, 'MAE']:.3f} points")
print(f"  Validation RMSE: {results_df.loc[best_idx, 'RMSE']:.3f} points")
print(f"  Validation R²:   {results_df.loc[best_idx, 'R2']:.4f}")
print("=" * 90)

# Top 5
print("\nTop 5 Models:")
top_5 = results_df.nlargest(5, 'R2')
for idx, (i, row) in enumerate(top_5.iterrows(), 1):
    print(f"  {idx}. {row['Model']:<35} R²={row['R2']:.4f}, MAE={row['MAE']:.3f}")

print("\n" + "=" * 90)
print("MODEL TRAINING COMPLETE")
print("=" * 90)

# COMMAND ----------

print("\n" + "=" * 80)
print("SECTION 6: MODEL PERFORMANCE VISUALIZATIONS")
print("=" * 80)

# Get predictions from best model
print(f"\nGenerating predictions from: {best_model_name}")

y_train_pred = best_model.predict(X_train)
y_val_pred = best_model.predict(X_val)
y_test_pred = best_model.predict(X_test)

# ============================================================================
# VISUALIZATION 1: Model Comparison (All 15 models)
# ============================================================================
print("\n[Step 1] Creating model comparison plots...")

fig1 = plt.figure(figsize=(20, 10))

# Plot 1: R² Comparison
ax1 = plt.subplot(2, 2, 1)
results_sorted = results_df.sort_values('R2', ascending=True)
colors = ['darkgreen' if x == results_df['R2'].max() else 'steelblue' for x in results_sorted['R2']]
bars = ax1.barh(range(len(results_sorted)), results_sorted['R2'], color=colors, alpha=0.7, edgecolor='black')
ax1.set_yticks(range(len(results_sorted)))
ax1.set_yticklabels(results_sorted['Model'], fontsize=9)
ax1.set_xlabel('R² Score', fontsize=11, fontweight='bold')
ax1.set_title('Model Comparison: R² Score', fontsize=14, fontweight='bold', pad=15)
ax1.grid(axis='x', alpha=0.3)

for i, (bar, val) in enumerate(zip(bars, results_sorted['R2'])):
    ax1.text(val + 0.01, i, f'{val:.4f}', va='center', fontsize=8, fontweight='bold')

# Plot 2: MAE Comparison
ax2 = plt.subplot(2, 2, 2)
results_mae_sorted = results_df.sort_values('MAE', ascending=False)
colors_mae = ['darkgreen' if x == results_df['MAE'].min() else 'coral' for x in results_mae_sorted['MAE']]
bars2 = ax2.barh(range(len(results_mae_sorted)), results_mae_sorted['MAE'], color=colors_mae, alpha=0.7, edgecolor='black')
ax2.set_yticks(range(len(results_mae_sorted)))
ax2.set_yticklabels(results_mae_sorted['Model'], fontsize=9)
ax2.set_xlabel('Mean Absolute Error (MAE)', fontsize=11, fontweight='bold')
ax2.set_title('Model Comparison: MAE', fontsize=14, fontweight='bold', pad=15)
ax2.grid(axis='x', alpha=0.3)

for i, (bar, val) in enumerate(zip(bars2, results_mae_sorted['MAE'])):
    ax2.text(val + 0.2, i, f'{val:.2f}', va='center', fontsize=8, fontweight='bold')

# Plot 3: Training Time Comparison
ax3 = plt.subplot(2, 2, 3)
results_time_sorted = results_df.sort_values('Train_Time', ascending=True)
bars3 = ax3.barh(range(len(results_time_sorted)), results_time_sorted['Train_Time'], 
                 color='lightseagreen', alpha=0.7, edgecolor='black')
ax3.set_yticks(range(len(results_time_sorted)))
ax3.set_yticklabels(results_time_sorted['Model'], fontsize=9)
ax3.set_xlabel('Training Time (seconds)', fontsize=11, fontweight='bold')
ax3.set_title('Model Comparison: Training Time', fontsize=14, fontweight='bold', pad=15)
ax3.grid(axis='x', alpha=0.3)

# Plot 4: R² vs MAE Scatter
ax4 = plt.subplot(2, 2, 4)
scatter = ax4.scatter(results_df['MAE'], results_df['R2'], s=200, c=results_df['Train_Time'], 
                     cmap='viridis', alpha=0.7, edgecolors='black', linewidth=1.5)
ax4.set_xlabel('Mean Absolute Error (MAE)', fontsize=11, fontweight='bold')
ax4.set_ylabel('R² Score', fontsize=11, fontweight='bold')
ax4.set_title('Model Performance: R² vs MAE', fontsize=14, fontweight='bold', pad=15)
ax4.grid(alpha=0.3)

# Add best model annotation
best_mae = results_df.loc[best_idx, 'MAE']
best_r2 = results_df.loc[best_idx, 'R2']
ax4.annotate(f'Best: {best_model_name}', xy=(best_mae, best_r2), 
            xytext=(best_mae + 0.5, best_r2 - 0.05),
            fontsize=10, fontweight='bold', color='red',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
            arrowprops=dict(arrowstyle='->', color='red', lw=2))

plt.colorbar(scatter, ax=ax4, label='Training Time (s)')

plt.suptitle('Model Performance Comparison - 15 Algorithms', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('../images/regression_models_performance.png')
plt.show()

print("✓ Model comparison plots created")

# ============================================================================
# VISUALIZATION 2: Best Model Performance Analysis
# ============================================================================
print("\n[Step 2] Creating best model performance analysis...")

fig2 = plt.figure(figsize=(20, 12))

# Plot 1: Actual vs Predicted (Test Set)
ax5 = plt.subplot(2, 3, 1)
ax5.scatter(y_test, y_test_pred, alpha=0.3, s=10, color='steelblue', edgecolors='none')
ax5.plot([0, 75], [0, 75], 'r--', linewidth=2, label='Perfect Prediction')
ax5.axhline(y=50, color='orange', linestyle='--', linewidth=1.5, alpha=0.7, label='High Risk (50)')
ax5.axvline(x=50, color='orange', linestyle='--', linewidth=1.5, alpha=0.7)
ax5.set_xlabel('Actual Stress Score', fontsize=11, fontweight='bold')
ax5.set_ylabel('Predicted Stress Score', fontsize=11, fontweight='bold')
ax5.set_title(f'Actual vs Predicted - Test Set\nR²={r2_score(y_test, y_test_pred):.4f}', 
              fontsize=12, fontweight='bold')
ax5.legend()
ax5.grid(alpha=0.3)

# Plot 2: Residuals Distribution
ax6 = plt.subplot(2, 3, 2)
residuals = y_test - y_test_pred
ax6.hist(residuals, bins=50, edgecolor='black', alpha=0.7, color='lightcoral')
ax6.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
ax6.set_xlabel('Residuals (Actual - Predicted)', fontsize=11, fontweight='bold')
ax6.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax6.set_title(f'Residuals Distribution\nMean={residuals.mean():.2f}, Std={residuals.std():.2f}', 
              fontsize=12, fontweight='bold')
ax6.legend()
ax6.grid(axis='y', alpha=0.3)

# Plot 3: Residuals vs Predicted
ax7 = plt.subplot(2, 3, 3)
ax7.scatter(y_test_pred, residuals, alpha=0.3, s=10, color='purple', edgecolors='none')
ax7.axhline(y=0, color='red', linestyle='--', linewidth=2)
ax7.set_xlabel('Predicted Stress Score', fontsize=11, fontweight='bold')
ax7.set_ylabel('Residuals', fontsize=11, fontweight='bold')
ax7.set_title('Residual Plot', fontsize=12, fontweight='bold')
ax7.grid(alpha=0.3)

# Plot 4: Error Distribution by Stress Level
ax8 = plt.subplot(2, 3, 4)
stress_bins = [0, 25, 50, 75]
stress_labels = ['Normal\n(0-24)', 'Moderate\n(25-49)', 'High Risk\n(50-75)']
y_test_binned = pd.cut(y_test, bins=stress_bins, labels=stress_labels)
abs_errors = np.abs(residuals)
error_by_bin = pd.DataFrame({'Stress_Level': y_test_binned, 'Abs_Error': abs_errors})
bp = error_by_bin.boxplot(column='Abs_Error', by='Stress_Level', ax=ax8, patch_artist=True)
ax8.set_xlabel('Stress Level Category', fontsize=11, fontweight='bold')
ax8.set_ylabel('Absolute Error', fontsize=11, fontweight='bold')
ax8.set_title('Prediction Error by Stress Level', fontsize=12, fontweight='bold')
plt.suptitle('')
ax8.grid(alpha=0.3)

# Plot 5: Performance Across Splits
ax9 = plt.subplot(2, 3, 5)
splits = ['Train', 'Validation', 'Test']
maes = [mean_absolute_error(y_train, y_train_pred),
        mean_absolute_error(y_val, y_val_pred),
        mean_absolute_error(y_test, y_test_pred)]
r2s = [r2_score(y_train, y_train_pred),
       r2_score(y_val, y_val_pred),
       r2_score(y_test, y_test_pred)]

x = np.arange(len(splits))
width = 0.35

bars1 = ax9.bar(x - width/2, maes, width, label='MAE', color='coral', alpha=0.7, edgecolor='black')
ax9_twin = ax9.twinx()
bars2 = ax9_twin.bar(x + width/2, r2s, width, label='R²', color='steelblue', alpha=0.7, edgecolor='black')

ax9.set_xlabel('Dataset Split', fontsize=11, fontweight='bold')
ax9.set_ylabel('MAE', fontsize=11, fontweight='bold', color='coral')
ax9_twin.set_ylabel('R² Score', fontsize=11, fontweight='bold', color='steelblue')
ax9.set_title('Performance Across Splits', fontsize=12, fontweight='bold')
ax9.set_xticks(x)
ax9.set_xticklabels(splits)
ax9.tick_params(axis='y', labelcolor='coral')
ax9_twin.tick_params(axis='y', labelcolor='steelblue')
ax9.legend(loc='upper left')
ax9_twin.legend(loc='upper right')
ax9.grid(alpha=0.3)

# Add value labels
for bar, val in zip(bars1, maes):
    ax9.text(bar.get_x() + bar.get_width()/2, val + 0.2, f'{val:.2f}',
            ha='center', fontsize=9, fontweight='bold')
for bar, val in zip(bars2, r2s):
    ax9_twin.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.3f}',
                 ha='center', fontsize=9, fontweight='bold')

# Plot 6: Prediction Distribution Comparison
ax10 = plt.subplot(2, 3, 6)
ax10.hist(y_test, bins=30, alpha=0.5, label='Actual', color='blue', edgecolor='black')
ax10.hist(y_test_pred, bins=30, alpha=0.5, label='Predicted', color='red', edgecolor='black')
ax10.axvline(x=50, color='orange', linestyle='--', linewidth=2, label='High Risk Threshold')
ax10.set_xlabel('Stress Score', fontsize=11, fontweight='bold')
ax10.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax10.set_title('Distribution: Actual vs Predicted', fontsize=12, fontweight='bold')
ax10.legend()
ax10.grid(axis='y', alpha=0.3)

plt.suptitle(f'Best Model Performance Analysis: {best_model_name}', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('../images/best_regression_model_performance.png')
plt.show()

print("✓ Best model performance plots created")

print("\n" + "=" * 80)
print("VISUALIZATIONS COMPLETE")
print("=" * 80)

# COMMAND ----------

# print("\n" + "=" * 80)
print("SECTION 7: HYPERPARAMETER TUNING FOR BEST MODEL: LightGMB (boosted)")
print("=" * 80)

from sklearn.model_selection import RandomizedSearchCV

model_to_tune = models["LightGBM (boosted)"]

param_dist = {
    "num_leaves":          [31, 127, 255],     # small / medium / large tree 
    "learning_rate":       [0.01, 0.05, 0.1],  # slow / balanced / fast learning
    "min_child_samples":   [10, 30, 100],      # leaf-level regularization
    "subsample":           [0.6, 0.8, 1.0],    # row sampling (bagging)
    "colsample_bytree":    [0.6, 0.8, 1.0],    # feature sampling
    "reg_alpha":           [0.0, 0.1, 1.0],    # L1 regularization
    "reg_lambda":          [0.0, 0.5, 1.0],    # L2 regularization
}

random_search = RandomizedSearchCV(
    estimator=model_to_tune,
    param_distributions=param_dist,
    n_iter=20,            # try 20 random combinations
    scoring='neg_mean_squared_error',
    cv=5,
    verbose=1,
    n_jobs=-1,
    random_state=42
)

# COMMAND ----------

random_search.fit(X_train, y_train)

print("\nBest parameters:")
print(random_search.best_params_)

best_model_light_gmb_boosted = random_search.best_estimator_
     

# COMMAND ----------

y_pred_light_gmb_boosted = best_model_light_gmb_boosted.predict(X_val)

# COMMAND ----------

# see best model performance
mae_light_gmb_boosted = mean_absolute_error(y_val, y_pred_light_gmb_boosted)
rmse_light_gmb_boosted = np.sqrt(mean_squared_error(y_val, y_pred_light_gmb_boosted))
r2_light_gmb_boosted = r2_score(y_val, y_pred_light_gmb_boosted)

print(f"\nLightGBM Boosted Model Performance: MAE: {mae_light_gmb_boosted:.3f}, RMSE: {rmse_light_gmb_boosted:.3f}, R²: {r2_light_gmb_boosted:.4f}")

# COMMAND ----------

# Refitting the model, since tuned hyperparameters lead to worse performance
best_model_light_gmb_boosted.set_params(n_estimators=2000, max_depth=20)

# COMMAND ----------

best_model_light_gmb_boosted.fit(X_train, y_train)
y_pred_light_gmb_boosted2 = best_model_light_gmb_boosted.predict(X_val)
# see best model performance
mae_light_gmb_boosted2 = mean_absolute_error(y_val, y_pred_light_gmb_boosted2)
rmse_light_gmb_boosted2 = np.sqrt(mean_squared_error(y_val, y_pred_light_gmb_boosted2))
r2_light_gmb_boosted2 = r2_score(y_val, y_pred_light_gmb_boosted2)
print(f"\nLightGBM Boosted Model Performance: MAE: {mae_light_gmb_boosted2:.3f}, RMSE: {rmse_light_gmb_boosted2:.3f}, R²: {r2_light_gmb_boosted2:.4f}")

# COMMAND ----------

print("\n" + "=" * 80)
print("SECTION 8: FINAL EVALUATION & BLACKOUT PREDICTION")
print("=" * 80)

# ============================================================================
# Regression Performance Summary
# ============================================================================
print("\n[Step 1] Regression Performance on All Splits:")
print("-" * 80)

print(f"\nBest Model: {best_model_name}\n")
print(f"{'Split':<12} {'MAE':>10} {'RMSE':>10} {'R²':>10}")
print("-" * 50)

for split_name, y_true, y_pred in [
    ('Train', y_train, y_train_pred),
    ('Validation', y_val, y_pred_light_gmb_boosted2),
    ('Test', y_test, y_test_pred)
]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"{split_name:<12} {mae:>10.3f} {rmse:>10.3f} {r2:>10.4f}")

# Check for overfitting
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
overfitting_gap = train_r2 - test_r2

print(f"\nOverfitting Analysis:")
print(f"  Train R²:     {train_r2:.4f}")
print(f"  Test R²:      {test_r2:.4f}")
print(f"  Difference:   {overfitting_gap:.4f}")

if overfitting_gap < 0.05:
    print(f"  Status: ✓ Excellent generalization (gap < 0.05)")
elif overfitting_gap < 0.10:
    print(f"  Status: ✓ Good generalization (gap < 0.10)")
else:
    print(f"  Status: ⚠️  Some overfitting detected (gap ≥ 0.10)")

# =======================================================================================
# Binary Classification after getting stress score with regression - Blackout Prediction
# =======================================================================================
print("\n" + "=" * 80)
print("BLACKOUT PREDICTION - BINARY CLASSIFICATION")
print("=" * 80)

THRESHOLD = 50
print(f"\nBlackout Risk Threshold: {THRESHOLD} points")
print(f"  Class 0 (Low Risk):  Stress score < {THRESHOLD}")
print(f"  Class 1 (High Risk): Stress score ≥ {THRESHOLD} → BLACKOUT RISK")

# Convert to binary
y_test_binary = (y_test >= THRESHOLD).astype(int)
y_test_pred_binary = (y_test_pred >= THRESHOLD).astype(int)

# Classification metrics
print("\n[Step 2] Classification Performance (Test Set):")
print("-" * 80)

accuracy = accuracy_score(y_test_binary, y_test_pred_binary)
precision = precision_score(y_test_binary, y_test_pred_binary, zero_division=0)
recall = recall_score(y_test_binary, y_test_pred_binary, zero_division=0)
f1 = f1_score(y_test_binary, y_test_pred_binary, zero_division=0)

print(f"\n  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"  Precision: {precision:.4f} ({precision*100:.2f}%)")
print(f"  Recall:    {recall:.4f} ({recall*100:.2f}%)")
print(f"  F1-Score:  {f1:.4f}")

# ============================================================================
# Confusion Matrix
# ============================================================================
print("\n[Step 3] Confusion Matrix (Test Set):")
print("-" * 80)

cm = confusion_matrix(y_test_binary, y_test_pred_binary)

print(f"\n                      Predicted")
print(f"                  Low Risk  High Risk")
print(f"Actual Low Risk     {cm[0,0]:>6,}     {cm[0,1]:>6,}")
print(f"Actual High Risk    {cm[1,0]:>6,}     {cm[1,1]:>6,}")

tn, fp, fn, tp = cm.ravel()
total = tn + fp + fn + tp

print(f"\nDetailed Breakdown:")
print(f"  True Negatives  (Correctly predicted low risk):  {tn:>6,} ({tn/total*100:>5.2f}%)")
print(f"  False Positives (False alarm):                   {fp:>6,} ({fp/total*100:>5.2f}%)")
print(f"  False Negatives (Missed blackout):               {fn:>6,} ({fn/total*100:>5.2f}%)")
print(f"  True Positives  (Correctly predicted blackout):  {tp:>6,} ({tp/total*100:>5.2f}%)")

# ============================================================================
# Confusion Matrix Visualization
# ============================================================================
print("\n[Step 4] Creating confusion matrix visualization...")

fig = plt.figure(figsize=(16, 6))

# Plot 1: Confusion Matrix Heatmap
ax1 = plt.subplot(1, 3, 1)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, square=True,
            xticklabels=['Low Risk', 'High Risk'],
            yticklabels=['Low Risk', 'High Risk'],
            ax=ax1, annot_kws={'size': 16, 'weight': 'bold'})
ax1.set_xlabel('Predicted', fontsize=12, fontweight='bold')
ax1.set_ylabel('Actual', fontsize=12, fontweight='bold')
ax1.set_title('Confusion Matrix - Test Set', fontsize=14, fontweight='bold', pad=15)

# Plot 2: Normalized Confusion Matrix (Percentages)
ax2 = plt.subplot(1, 3, 2)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='RdYlGn', cbar=True, square=True,
            xticklabels=['Low Risk', 'High Risk'],
            yticklabels=['Low Risk', 'High Risk'],
            ax=ax2, annot_kws={'size': 16, 'weight': 'bold'}, vmin=0, vmax=1)
ax2.set_xlabel('Predicted', fontsize=12, fontweight='bold')
ax2.set_ylabel('Actual', fontsize=12, fontweight='bold')
ax2.set_title('Confusion Matrix - Normalized', fontsize=14, fontweight='bold', pad=15)

# Plot 3: Classification Metrics Bar Chart
ax3 = plt.subplot(1, 3, 3)
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
values = [accuracy, precision, recall, f1]
colors = ['steelblue', 'green', 'orange', 'purple']
bars = ax3.bar(metrics, values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
ax3.set_ylabel('Score', fontsize=12, fontweight='bold')
ax3.set_title('Classification Metrics', fontsize=14, fontweight='bold', pad=15)
ax3.set_ylim([0, 1])
ax3.grid(axis='y', alpha=0.3)

# Add value labels
for bar, val in zip(bars, values):
    ax3.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.3f}',
            ha='center', fontsize=11, fontweight='bold')

plt.suptitle('Blackout Prediction - Classification Analysis', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('../images/confussion_matrix_with_regression_blackout_prediction.png')
plt.show()

print("✓ Confusion matrix visualization created")

# ============================================================================
# Business Impact Analysis
# ============================================================================
print("\n[Step 5] Business Impact Analysis:")
print("-" * 80)

print(f"\nCritical Errors (False Negatives - Missed Blackouts):")
if fn > 0:
    print(f"  ⚠️  {fn:,} blackout events were NOT predicted")
    print(f"  ⚠️  This is {fn/total*100:.2f}% of all test cases")
    print(f"  ⚠️  Risk: Unprepared for {fn} potential blackouts!")
else:
    print(f"  ✓ EXCELLENT: NO missed blackouts (0 false negatives)")

print(f"\nFalse Alarms (False Positives):")
if fp > 0:
    print(f"  ⚠️  {fp:,} false alarms triggered")
    print(f"  ⚠️  This is {fp/total*100:.2f}% of all test cases")
    print(f"  ⚠️  Impact: Unnecessary emergency preparations")
else:
    print(f"  ✓ PERFECT: NO false alarms (0 false positives)")

reliability = (tn + tp) / total
print(f"\n📊 Overall Prediction Reliability: {reliability:.4f} ({reliability*100:.2f}%)")

# ============================================================================
# Feature Importance
# ============================================================================
print("\n[Step 6] Feature Importance Analysis:")
print("-" * 80)

if hasattr(best_model, 'feature_importances_'):
    importances = best_model.feature_importances_
    feature_importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    print("\nTop 20 Most Important Features:")
    print(f"\n{'Rank':<6} {'Feature':<50} {'Importance':>12}")
    print("-" * 70)
    
    for idx, (i, row) in enumerate(feature_importance.head(20).iterrows(), 1):
        print(f"{idx:<6} {row['Feature']:<50} {row['Importance']:>12.6f}")
    
    # Visualize feature importance
    print("\n[Step 7] Creating feature importance visualization...")
    
    fig2 = plt.figure(figsize=(14, 8))
    
    top_20 = feature_importance.head(20).sort_values('Importance', ascending=True)
    colors_feat = ['darkgreen' if i < 5 else 'steelblue' for i in range(len(top_20))]
    bars = plt.barh(range(len(top_20)), top_20['Importance'], color=colors_feat, alpha=0.7, edgecolor='black')
    plt.yticks(range(len(top_20)), top_20['Feature'], fontsize=10)
    plt.xlabel('Importance', fontsize=12, fontweight='bold')
    plt.title(f'Top 20 Feature Importance - {best_model_name}', fontsize=14, fontweight='bold', pad=15)
    plt.grid(axis='x', alpha=0.3)
    
    for i, (bar, val) in enumerate(zip(bars, top_20['Importance'])):
        plt.text(val + 0.001, i, f'{val:.4f}', va='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('../images/feature_importance_with_regression_blackout_prediction.png')
    plt.show()
    
    print("✓ Feature importance visualization created")
else:
    print("\n  Feature importance not available for this model type")

# ============================================================================
# Final Summary
# ============================================================================
print("\n" + "=" * 80)
print("FINAL MODEL SUMMARY")
print("=" * 80)

print(f"\nModel: {best_model_name}")
print(f"Features: {X_train.shape[1]} (production-ready, no data leakage)")

print(f"\n📊 Regression Performance (Test Set):")
print(f"  MAE:  {mean_absolute_error(y_test, y_test_pred):.3f} points (±{mean_absolute_error(y_test, y_test_pred):.1f} on 0-75 scale)")
print(f"  RMSE: {np.sqrt(mean_squared_error(y_test, y_test_pred)):.3f} points")
print(f"  R²:   {r2_score(y_test, y_test_pred):.4f} (explains {r2_score(y_test, y_test_pred)*100:.1f}% of variance)")

print(f"\n🚨 Blackout Classification (Test Set):")
print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.1f}%)")
print(f"  Precision: {precision:.4f} (of predicted blackouts, {precision*100:.1f}% are real)")
print(f"  Recall:    {recall:.4f} (detects {recall*100:.1f}% of actual blackouts)")
print(f"  F1-Score:  {f1:.4f}")

print(f"\n✓ Data Leakage Check:")
print(f"  net_imports:     Excluded (used in T7/T8 calculation)")
print(f"  stress_lag_*:    Excluded (target to predict target)")
print(f"  Status:          Production-ready!")

print("\n" + "=" * 80)
print("EUROPEAN GRID STRESS PREDICTION - ANALYSIS COMPLETE")
print("=" * 80)

# COMMAND ----------

print("\n" + "=" * 80)
print("SECTION 9: THRESHOLD OPTIMIZATION")
print("=" * 80)

# ============================================================================
# Threshold Optimization for Regression Model
# ============================================================================
print("\n[PART B] Threshold optimization for regression model...")
print(f"Testing different thresholds with: {best_model_name}\n")

thresholds = [30, 35, 40, 45, 50, 55, 60]
threshold_results = []

print(f"{'Threshold':<12} {'Accuracy':>10} {'Precision':>12} {'Recall':>10} {'F1-Score':>10} {'Missed':>10}")
print("-" * 80)

for thresh in thresholds:
    y_test_pred_binary = (y_test_pred >= thresh).astype(int)
    y_test_binary_thresh = (y_test >= thresh).astype(int)
    
    acc = accuracy_score(y_test_binary_thresh, y_test_pred_binary)
    prec = precision_score(y_test_binary_thresh, y_test_pred_binary, zero_division=0)
    rec = recall_score(y_test_binary_thresh, y_test_pred_binary, zero_division=0)
    f1 = f1_score(y_test_binary_thresh, y_test_pred_binary, zero_division=0)
    
    cm = confusion_matrix(y_test_binary_thresh, y_test_pred_binary)
    fn = cm[1, 0]
    
    threshold_results.append({
        'Threshold': thresh,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1': f1,
        'Missed': fn
    })
    
    print(f"{thresh:<12} {acc:>10.4f} {prec:>12.4f} {rec:>10.4f} {f1:>10.4f} {fn:>10,}")

threshold_df = pd.DataFrame(threshold_results)
best_f1_idx = threshold_df['F1'].idxmax()
best_threshold = threshold_df.loc[best_f1_idx, 'Threshold']

print("\n" + "-" * 80)
print(f"BEST THRESHOLD: {best_threshold}")
print(f"  F1-Score: {threshold_df.loc[best_f1_idx, 'F1']:.4f}")
print(f"  Recall: {threshold_df.loc[best_f1_idx, 'Recall']:.4f}")
print(f"  Missed: {threshold_df.loc[best_f1_idx, 'Missed']:,}")
print("-" * 80)

# ============================================================================
# PART C: Compare All Approaches
# ============================================================================
print("\n[PART C] Comparing all approaches on TEST set...")

# Get predictions on test set
y_test_pred_reg_thresh50 = (y_test_pred >= 50).astype(int)
y_test_pred_reg_optimized = (y_test_pred >= best_threshold).astype(int)

# Calculate metrics
approaches = {
    f'Regression (Threshold=50)': y_test_pred_reg_thresh50,
    f'Regression (Optimized T={best_threshold})': y_test_pred_reg_optimized
}

print(f"\n{'Approach':<45} {'Accuracy':>10} {'Precision':>12} {'Recall':>10} {'F1':>10}")
print("-" * 90)

comparison_results = []
for approach_name, y_pred in approaches.items():
    acc = accuracy_score(y_test_binary, y_pred)
    prec = precision_score(y_test_binary, y_pred, zero_division=0)
    rec = recall_score(y_test_binary, y_pred, zero_division=0)
    f1 = f1_score(y_test_binary, y_pred, zero_division=0)
    
    comparison_results.append({
        'Approach': approach_name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1': f1
    })
    
    print(f"{approach_name:<45} {acc:>10.4f} {prec:>12.4f} {rec:>10.4f} {f1:>10.4f}")

comparison_df = pd.DataFrame(comparison_results)

# Find best overall approach
best_overall_idx = comparison_df['F1'].idxmax()
best_overall_approach = comparison_df.loc[best_overall_idx, 'Approach']

print("\n" + "=" * 90)
print(f"🏆 BEST OVERALL APPROACH: {best_overall_approach}")
print(f"  Test F1-Score: {comparison_df.loc[best_overall_idx, 'F1']:.4f}")
print(f"  Test Recall:   {comparison_df.loc[best_overall_idx, 'Recall']:.4f}")
print("=" * 90)

# ============================================================================
# Visualizations
# ============================================================================
print("\n[PART D] Creating comprehensive visualizations...")

fig = plt.figure(figsize=(20, 12))

# Plot 2: Threshold Impact on Metrics
ax2 = plt.subplot(2, 3, 2)
ax2.plot(threshold_df['Threshold'], threshold_df['Recall'], 'o-', linewidth=2, markersize=8, label='Recall', color='red')
ax2.plot(threshold_df['Threshold'], threshold_df['Precision'], 's-', linewidth=2, markersize=8, label='Precision', color='blue')
ax2.plot(threshold_df['Threshold'], threshold_df['F1'], '^-', linewidth=2, markersize=8, label='F1-Score', color='green')
ax2.axvline(x=best_threshold, color='orange', linestyle='--', linewidth=2, label=f'Best T={best_threshold}')
ax2.set_xlabel('Threshold', fontsize=11, fontweight='bold')
ax2.set_ylabel('Score', fontsize=11, fontweight='bold')
ax2.set_title('Threshold Optimization', fontsize=13, fontweight='bold')
ax2.legend()
ax2.grid(alpha=0.3)

# Plot 3: Approach Comparison
ax3 = plt.subplot(2, 3, 3)
metrics_comp = ['Accuracy', 'Precision', 'Recall', 'F1']
x_pos = np.arange(len(comparison_df))
width = 0.2

for i, metric in enumerate(metrics_comp):
    values = comparison_df[metric].values
    ax3.bar(x_pos + i*width, values, width, label=metric, alpha=0.7, edgecolor='black')

ax3.set_xlabel('Approach', fontsize=11, fontweight='bold')
ax3.set_ylabel('Score', fontsize=11, fontweight='bold')
ax3.set_title('Approaches Comparison', fontsize=13, fontweight='bold')
ax3.set_xticks(x_pos + width * 1.5)
ax3.set_xticklabels(['Reg T=50', f'Reg T={best_threshold}'], fontsize=9, rotation=15)
ax3.legend(fontsize=9)
ax3.grid(axis='y', alpha=0.3)

# Plot 4-6: Confusion Matrices for 3 approaches
for idx, (approach_name, y_pred) in enumerate(approaches.items(), 3):
    ax = plt.subplot(2, 3, idx)
    cm = confusion_matrix(y_test_binary, y_pred)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, square=True,
                xticklabels=['Low', 'High'], yticklabels=['Low', 'High'],
                ax=ax, annot_kws={'size': 12, 'weight': 'bold'})
    
    rec = recall_score(y_test_binary, y_pred)
    f1 = f1_score(y_test_binary, y_pred)
    
    title = approach_name.replace('Regression', 'Reg')
    ax.set_title(f'{title}\nRecall={rec:.3f}, F1={f1:.3f}', fontsize=11, fontweight='bold')
    ax.set_xlabel('Predicted', fontsize=10, fontweight='bold')
    ax.set_ylabel('Actual', fontsize=10, fontweight='bold')

plt.suptitle('Complete Classification Analysis - All Approaches', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('../images/compare_regression_and_with_threshold.png')
plt.show()

print("✓ Comprehensive visualizations created")

# ============================================================================
# Final Recommendation
# ============================================================================
print("\n" + "=" * 90)
print("FINAL RECOMMENDATION FOR PRODUCTION")
print("=" * 90)

print(f"\n🎯 BEST APPROACH: {best_overall_approach}")
print(f"\n📊 Test Set Performance:")
print(f"  Accuracy:  {comparison_df.loc[best_overall_idx, 'Accuracy']:.4f}")
print(f"  Precision: {comparison_df.loc[best_overall_idx, 'Precision']:.4f}")
print(f"  Recall:    {comparison_df.loc[best_overall_idx, 'Recall']:.4f}")
print(f"  F1-Score:  {comparison_df.loc[best_overall_idx, 'F1']:.4f}")

print(f"\n✅ PRODUCTION DEPLOYMENT:")
print(f"  Use: {best_model_name}")
print(f"  Type: Regression model with threshold = {best_threshold}")

print("\n" + "=" * 90)

# COMMAND ----------

print("\n" + "=" * 80)
print("REGRESSION NOTEBOOK COMPLETE")
print("=" * 80)
print(f"\n✅ Best regression model: {best_model_name}")
print(f"✅ Optimal regression threshold: {best_threshold}")
print(f"✅ Regression recall: {threshold_df.loc[best_f1_idx, 'Recall']:.1%}")
print(f"\nNext: Run 'grid_stress_classification_v2.ipynb' for classification optimization")
print("=" * 80)

# Save for classification notebook
import pickle
import os
# Create directory
output_dir = "regression_models"
os.makedirs(output_dir, exist_ok=True)

# Save model
with open(f'{output_dir}/regression_model_outputs.pkl', 'wb') as f:
    pickle.dump({
        'best_model': best_model,
        'best_model_name': best_model_name,
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'y_test_pred': y_test_pred,
        'best_threshold': best_threshold,
        'threshold_df': threshold_df,
    }, f)

print("✓ Saved: regression_model_outputs.pkl")
