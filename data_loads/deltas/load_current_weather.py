# Databricks notebook source
import openmeteo_requests

import pandas as pd
import requests_cache
from retry_requests import retry

import time
import re
import warnings
import json
from datetime import date, timedelta
import traceback

from pyspark.sql import SparkSession

# Suppress annoying warnings from external libraries
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', message='.*Downcasting object dtype arrays.*')
warnings.filterwarnings('ignore', message='.*distutils Version classes are deprecated.*')
warnings.filterwarnings('ignore', message='.*is_categorical_dtype is deprecated.*')
warnings.filterwarnings('ignore', message='.*is_datetime64tz_dtype is deprecated.*')
warnings.filterwarnings('ignore', message='.*Converting to PeriodArray.*will drop timezone information.*')

# ============================================================================
# GLOBAL CONFIG
# ============================================================================

DATABASE = "workspace.european_weather_raw"
TABLE_NAME = "current_weather"

# Coordinates (same order as in your original snippet)
LATITUDES = [47.52, 50.5, 51.17, 40.46, 46.23, 45.1, 47.16, 41.87, 55.17, 52.13, 51.92, 39.4, 48.67]
LONGITUDES = [14.55, 4.47, 10.45, -3.75, 2.21, 15.2, 19.5, 12.56, 23.88, 5.29, 19.14, -8.22, 19.69]

# Default number of forecast days if widget is not provided
DEFAULT_FORECAST_DAYS = 2

# ============================================================================
# DATABRICKS JOB PARAMETERS
# ============================================================================

def get_widget_value(*widget_names: str) -> str | None:
    """Get widget value, trying multiple possible widget names."""
    for name in widget_names:
        try:
            value = dbutils.widgets.get(name)  # type: ignore[name-defined]
            if value and value.strip():
                return value.strip()
        except Exception:
            continue
    return None

try:
    forecast_days_raw = get_widget_value("forecast_days", "days_ahead", "forecast")
except Exception:
    forecast_days_raw = None

if forecast_days_raw:
    try:
        FORECAST_DAYS = int(forecast_days_raw)
        if FORECAST_DAYS < 1:
            FORECAST_DAYS = DEFAULT_FORECAST_DAYS
    except ValueError:
        FORECAST_DAYS = DEFAULT_FORECAST_DAYS
else:
    FORECAST_DAYS = DEFAULT_FORECAST_DAYS

print(f"✓ Using forecast_days = {FORECAST_DAYS}")

# ============================================================================
# SPARK INIT
# ============================================================================

def init_spark_session() -> SparkSession:
    spark = SparkSession.builder.getOrCreate()
    spark.sql(f"USE {DATABASE}")
    return spark

spark = init_spark_session()

# ============================================================================
# COLUMN SANITIZER (REUSED PATTERN)
# ============================================================================

INVALID_CHARS_PATTERN = re.compile(r"[^0-9A-Za-z_]+")

def sanitize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flatten MultiIndex columns and remove characters invalid for Delta.
    Replace any non [0-9A-Za-z_] chars with '_'
    If name starts with a digit, prefix with '_'
    """
    new_cols = []
    for col in df.columns:
        if isinstance(col, tuple):
            parts = [str(p) for p in col if p is not None and str(p) != ""]
            name = "__".join(parts) if parts else "col"
        else:
            name = str(col)

        name = name.strip().replace(" ", "_")
        name = INVALID_CHARS_PATTERN.sub("_", name)

        if name and name[0].isdigit():
            name = "_" + name

        new_cols.append(name)

    df.columns = new_cols
    return df

# ============================================================================
# TABLE MANAGEMENT
# ============================================================================

def table_exists(full_name: str) -> bool:
    return spark.catalog.tableExists(full_name)

# ============================================================================
# OPEN-METEO CLIENT SETUP
# ============================================================================

# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

# ============================================================================
# DATA DOWNLOAD
# ============================================================================

def download_weather(forecast_days: int) -> pd.DataFrame:
    """
    Download hourly weather for all configured locations for the next
    `forecast_days` days.
    """
    params = {
        "latitude": LATITUDES,
        "longitude": LONGITUDES,
        "hourly": ["temperature_2m", "wind_speed_10m", "shortwave_radiation"],
        "forecast_days": forecast_days,
        "wind_speed_unit": "ms",
        "timezone": "UTC",
    }

    print(f"→ Requesting Open-Meteo data for {len(LATITUDES)} locations "
          f"and {forecast_days} forecast days...")
    responses = openmeteo.weather_api(OPEN_METEO_URL, params=params)

    all_rows: list[pd.DataFrame] = []

    for response in responses:
        lat = response.Latitude()
        lon = response.Longitude()

        hourly = response.Hourly()
        hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
        hourly_wind_speed_10m = hourly.Variables(1).ValuesAsNumpy()
        hourly_shortwave_radiation = hourly.Variables(2).ValuesAsNumpy()

        timestamps = pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left",
        )

        hourly_data = {
            "lat": lat,
            "lon": lon,
            "timestamp": timestamps,
            "ssrd": hourly_shortwave_radiation,
            "wind_speed": hourly_wind_speed_10m,
            "temperature_c": hourly_temperature_2m,
        }

        df = pd.DataFrame(hourly_data)
        all_rows.append(df)

    if not all_rows:
        print("⚠ No data returned from Open-Meteo")
        return pd.DataFrame()

    full_table = pd.concat(all_rows, ignore_index=True)

    # Derive date column (day-level) for partitioning
    full_table["date"] = full_table["timestamp"].dt.date.astype(str)

    # Optional: stable sorting
    full_table = full_table.sort_values(["date", "timestamp", "lat", "lon"]).reset_index(drop=True)

    print(f"✓ Downloaded {len(full_table)} rows from Open-Meteo.")
    return full_table

# ============================================================================
# DELTA WRITE
# ============================================================================

def write_current_weather(df: pd.DataFrame):
    """
    Write the combined weather dataframe to Delta:
    - Table: workspace.european_weather_raw.current_weather
    - Partitioned by date (day)
    - Overwrites only the partitions present in df['date']
    """
    if df is None or df.empty:
        print("→ No weather data to write, skipping Delta write.")
        return

    df = sanitize_columns(df)

    full_name = f"{DATABASE}.{TABLE_NAME}"
    exists = table_exists(full_name)

    # Collect distinct dates present in the data
    unique_dates = sorted(df["date"].unique())
    print(f"→ Writing {len(df)} rows into {full_name} for dates: {unique_dates}")

    for d in unique_dates:
        df_day = df[df["date"] == d].copy()
        if df_day.empty:
            continue

        sdf = spark.createDataFrame(df_day)
        rows = sdf.count()

        if not exists:
            print(
                f"  💾 Creating {full_name} partitioned by date, "
                f"{rows} rows for date={d}"
            )
            (
                sdf.write
                .format("delta")
                .mode("overwrite")
                .option("mergeSchema", "true")
                .partitionBy("date")
                .saveAsTable(full_name)
            )
            exists = True
        else:
            condition = f"date = '{d}'"
            print(
                f"  💾 Overwriting {full_name} for date={d} "
                f"(rows={rows})"
            )
            (
                sdf.write
                .format("delta")
                .mode("overwrite")
                .option("mergeSchema", "true")
                .option("replaceWhere", condition)
                .saveAsTable(full_name)
            )

        print(f"  ✅ Saved {rows} rows for date={d} into {full_name}")

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=== Starting Open-Meteo weather collection ===")
    print(f"Forecast days: {FORECAST_DAYS}")

    try:
        df = download_weather(FORECAST_DAYS)
        write_current_weather(df)
        print("\n✅ Weather pipeline completed successfully.")
    except Exception as e:
        print(f"\n✗ Weather pipeline failed: {e}")
        traceback.print_exc()
        raise

# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    main()
