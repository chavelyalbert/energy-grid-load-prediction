# European Energy Grid & Weather Data Pipeline

Data collection pipelines for European electricity grid data (ENTSO-E) and weather data (ERA5) running on Databricks.

## Setup

### Prerequisites

```bash
# Grid data pipeline
pip install entsoe-py pandas pyspark

# Weather data pipeline
pip install cdsapi pygrib numpy pandas
```

### API Keys Required

1. **ENTSO-E API Key**: Register at [transparency.entsoe.eu](https://transparency.entsoe.eu/)
2. **Copernicus CDS API Key**: Register at [cds.climate.copernicus.eu](https://cds.climate.copernicus.eu/)

## 📁 Pipeline Files

This folder contains 5 pipeline scripts:

| File | Purpose | Schedule |
|------|---------|----------|
| `delta__load_grid_data.py` | Collect grid data from ENTSO-E (load, generation, forecasts, cross-border flows) | Hourly |
| `delta__load_weather_data.py` | Download ERA5 reanalysis weather data from Copernicus | Daily |
| `load_current_weather.py` | Fetch current & forecasted weather from Open-Meteo (lightweight) | Hourly |
| `partition_migration.py` | Utility: Remove partitioning from Delta tables (one-time) | Manual |
| `table_migration.py` | Utility: Add/backfill date column in Delta tables (one-time) | Manual |

## Configuration

### Grid Data Pipeline (`load_grid_data.py`)

Update these variables:

```python
API_KEY = 'your-entsoe-api-key'
START_DATE = '2023-01-01'
END_DATE = '2025-10-31'
DATABASE = "european_grid_raw"
```

### Weather Data Pipeline (`load_weather__lon_lat.py`)

Update the `WeatherConfig` class:

```python
class WeatherConfig:
    CDS_API_KEY = "your-cds-api-key"
    MIN_DATE = "2023-06-12"
    MAX_DATE = "2025-11-08"
    TARGET_TABLE = "workspace.european_weather_raw.weather_hourly"
    FULL_RELOAD = False  # Set True to drop and recreate table
```

### Weather Data Pipeline (`delta__load_weather_data.py`)

Same idempotent approach as grid data. Overwrites per (year, month, day) partition.

**Requirements:** High-memory cluster (24+ GB RAM)

### Current Weather Pipeline (`load_current_weather.py`)

Lightweight real-time weather from Open-Meteo. No API key required. Simple per-date partition overwrite.

## Running the Pipelines

### Grid Data

1. Upload `load_grid_data.py` to Databricks
2. Configure API key and date range
3. Run the notebook - it will automatically collect data for all 27 European countries

### Weather Data

1. Upload `load_weather__lon_lat.py` to Databricks
2. Configure CDS API key and date range
3. Ensure cluster has HIGH memory setting
4. Run the notebook - it processes one day at a time sequentially

### Weather Data Pipelines

1. ERA5: Upload `delta__load_weather_data.py` to Databricks, configure CDS API key, use high-memory cluster
2. Current Weather: Upload `load_current_weather.py` to Databricks (no API key needed for Open-Meteo)

Both follow the same idempotent partition overwrite pattern as grid data.

## Output

- **Grid data**: Delta tables in `european_grid_raw` database
- **Weather data**: Delta table at `workspace.european_weather_raw.weather_hourly`

## Notes

- Both pipelines are idempotent - safe to re-run
- Weather pipeline uses partition overwrite by day (no duplicates)
- Grid pipeline truncates tables before full reload
- All temporary files are automatically cleaned up