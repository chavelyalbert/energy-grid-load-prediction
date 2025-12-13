# Databricks notebook source
import os
import uuid
from datetime import date, datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import cdsapi
import numpy as np
import pandas as pd
import pygrib
from pyspark.sql import SparkSession


# ============================================================================
# CONFIGURATION
# ============================================================================

class WeatherConfig:
    # Copernicus CDS API (in real life: use secrets / env vars!)
    CDS_API_URL = "https://cds.climate.copernicus.eu/api"
    CDS_API_KEY = "your API key here"  

    # Europe bounding box [North, West, South, East]
    EUROPE_BBOX = [72, -25, 35, 45]

    # REQUIRED date range (inclusive)
    MIN_DATE = "2023-12-01"      # restart from failed date
    MAX_DATE = "2025-11-08"      # adjust if you want to limit to a shorter window

    # ERA5 dataset & variables
    DATASET = "reanalysis-era5-single-levels"
    VARIABLES = [
        "10m_u_component_of_wind",            # -> shortName 10u
        "10m_v_component_of_wind",            # -> shortName 10v
        "2m_temperature",                     # -> shortName 2t
        "surface_solar_radiation_downwards",  # -> shortName ssrd
    ]

    # Where to store temporary GRIB files (local/DBFS path)
    OUTPUT_DIR = "data/weather_grib_daily"

    # Delta target (Unity Catalog)
    CATALOG = "workspace"
    TARGET_DB = "european_weather_raw"
    TARGET_TABLE = f"{CATALOG}.{TARGET_DB}.weather_hourly"

    # Full reload toggle: if True, drop the table at the beginning
    FULL_RELOAD: bool = False

    # Parallelism (number of days processed concurrently)
    MAX_WORKERS: int = int(os.getenv("WEATHER_MAX_WORKERS", "2"))


# ============================================================================
# DATABRICKS JOB PARAMETERS (DATE RANGE & PARALLELISM)
# ============================================================================

def get_widget_value(*widget_names: str) -> str | None:
    """
    Get widget value, trying multiple possible widget names.
    Returns the first non-empty value or None if none found / dbutils not available.
    """
    try:
        _ = dbutils.widgets  # type: ignore[name-defined]
    except Exception:
        return None

    for name in widget_names:
        try:
            value = dbutils.widgets.get(name)  # type: ignore[name-defined]
            if value and value.strip():
                return value.strip()
        except Exception:
            continue
    return None


# Defaults: relative to "today" if nothing explicit is passed
today = date.today()
today_str = today.isoformat()

# Try to read job parameters
try:
    load_mode   = get_widget_value("load_mode", "mode")        # optional, e.g. "today"
    from_date   = get_widget_value("from_date", "--from", "from")
    to_date     = get_widget_value("to_date", "--to", "to")
    days_back   = get_widget_value("days_back", "lookback_days", "days")
    max_workers = get_widget_value("max_workers", "workers")
    full_reload = get_widget_value("full_reload", "FULL_RELOAD")
except Exception:
    load_mode = from_date = to_date = days_back = max_workers = full_reload = None


# -------------------------------
# Date logic
# -------------------------------
if from_date or to_date:
    # 1) Explicit range via widgets wins
    if from_date:
        WeatherConfig.MIN_DATE = from_date
    else:
        WeatherConfig.MIN_DATE = today_str  # fallback

    if to_date:
        WeatherConfig.MAX_DATE = to_date
    else:
        WeatherConfig.MAX_DATE = WeatherConfig.MIN_DATE  # single-day

    print(f"✓ Using job parameters date range: {WeatherConfig.MIN_DATE} → {WeatherConfig.MAX_DATE}")

elif days_back:
    # 2) Relative days_back window
    try:
        n = int(days_back)
        if n < 1:
            n = 1
    except ValueError:
        n = 1

    end_dt = today
    start_dt = end_dt - timedelta(days=n - 1)

    WeatherConfig.MIN_DATE = start_dt.isoformat()
    WeatherConfig.MAX_DATE = end_dt.isoformat()

    print(f"✓ Using job parameter days_back={n}: {WeatherConfig.MIN_DATE} → {WeatherConfig.MAX_DATE}")

elif (load_mode or "").lower() == "today":
    # 3) Explicit "today" mode
    WeatherConfig.MIN_DATE = today_str
    WeatherConfig.MAX_DATE = today_str
    print(f"✓ load_mode='today' → MIN_DATE = MAX_DATE = {today_str}")

else:
    # 4) No date parameters → default to today only
    WeatherConfig.MIN_DATE = today_str
    WeatherConfig.MAX_DATE = today_str
    print(f"✓ No date parameters → defaulting to today: {today_str}")


# -------------------------------
# Parallelism & full reload
# -------------------------------
if max_workers:
    try:
        WeatherConfig.MAX_WORKERS = int(max_workers)
        print(f"✓ Using job parameter max_workers: {WeatherConfig.MAX_WORKERS}")
    except ValueError:
        print(f"⚠️ Invalid max_workers='{max_workers}', keeping default {WeatherConfig.MAX_WORKERS}")

if full_reload:
    if full_reload.lower() in ("1", "true", "yes", "y"):
        WeatherConfig.FULL_RELOAD = True
    elif full_reload.lower() in ("0", "false", "no", "n"):
        WeatherConfig.FULL_RELOAD = False
    print(f"✓ Using job parameter full_reload: {WeatherConfig.FULL_RELOAD}")

print(
    f"→ Effective date window: {WeatherConfig.MIN_DATE} → {WeatherConfig.MAX_DATE}, "
    f"FULL_RELOAD={WeatherConfig.FULL_RELOAD}, MAX_WORKERS={WeatherConfig.MAX_WORKERS}"
)


# ============================================================================
# DATE HELPERS
# ============================================================================

def iter_dates():
    """Return list of dates to process, based on REQUIRED MIN_DATE/MAX_DATE."""
    if not WeatherConfig.MIN_DATE or not WeatherConfig.MAX_DATE:
        raise RuntimeError("WeatherConfig.MIN_DATE and MAX_DATE must be set.")

    start = date.fromisoformat(WeatherConfig.MIN_DATE)
    end = date.fromisoformat(WeatherConfig.MAX_DATE)
    if end < start:
        raise RuntimeError("MAX_DATE must be >= MIN_DATE.")

    current = start
    days = []
    while current <= end:
        days.append(current)
        current += timedelta(days=1)
    return days


# ============================================================================
# SPARK SESSION HELPER
# ============================================================================

def init_spark_session() -> SparkSession:
    """Create or get SparkSession and ensure catalog/schema are set."""
    spark = SparkSession.builder.getOrCreate()

    # spark.sql(f"CREATE CATALOG IF NOT EXISTS {WeatherConfig.CATALOG}")
    # spark.sql(
    #     f"CREATE SCHEMA IF NOT EXISTS {WeatherConfig.CATALOG}.{WeatherConfig.TARGET_DB}"
    # )
    spark.sql(f"USE CATALOG {WeatherConfig.CATALOG}")
    spark.sql(f"USE {WeatherConfig.TARGET_DB}")

    return spark


# ============================================================================
# DOWNLOAD
# ============================================================================

def build_target_path(day: date) -> str:
    """Local/DBFS file path for a given day."""
    os.makedirs(WeatherConfig.OUTPUT_DIR, exist_ok=True)
    return os.path.join(
        WeatherConfig.OUTPUT_DIR,
        f"era5_europe_all_{day.strftime('%Y%m%d')}.grib",
    )


def download_one_day(day: date) -> str | None:
    """Download one day of ERA5 data as GRIB. Returns local path or None."""
    target = build_target_path(day)

    # Remove any existing partial file
    if os.path.exists(target):
        try:
            os.remove(target)
        except OSError:
            pass

    abs_target = os.path.abspath(target)
    print(f"\n📥 Downloading ERA5 for {day.isoformat()} → {abs_target}")

    client = cdsapi.Client(
        url=WeatherConfig.CDS_API_URL,
        key=WeatherConfig.CDS_API_KEY,
        quiet=True,
    )

    req = {
        "product_type": "reanalysis",
        "variable": WeatherConfig.VARIABLES,
        "year": f"{day.year:04d}",
        "month": f"{day.month:02d}",
        "day": f"{day.day:02d}",
        "time": [f"{h:02d}:00" for h in range(24)],
        "area": WeatherConfig.EUROPE_BBOX,
        "format": "grib",
    }

    try:
        client.retrieve(WeatherConfig.DATASET, req, target)
        print(f"   ✓ Downloaded {day.isoformat()} → {abs_target}")
        return target
    except Exception as e:
        print(f"   ✗ Failed to download {day.isoformat()}: {e}")
        return None


# ============================================================================
# GRIB → PANDAS
# ============================================================================

def grib_day_to_dataframe(grib_path: str, load_id: str, load_ts: datetime) -> pd.DataFrame:
    """
    Parse a GRIB file for a single day into a flat DataFrame.

    Columns:
      timestamp, lat, lon, u10, v10, t2m, ssrd,
      wind_speed, temperature_c,
      year, month, day, load_id, load_ts, source_file
    """
    abs_path = os.path.abspath(grib_path)
    source_file = os.path.basename(grib_path)
    print(f"🔍 Parsing GRIB file: {abs_path}")

    # Parse date from filename (era5_europe_all_YYYYMMDD.grib)
    token = source_file.replace("era5_europe_all_", "").replace(".grib", "")
    year = int(token[0:4])
    month = int(token[4:6])
    day = int(token[6:8])
    day_date = date(year, month, day)

    with pygrib.open(grib_path) as grbs:
        messages = list(grbs)

    if not messages:
        raise ValueError(f"No GRIB messages found in {grib_path}")

    sample = messages[0]
    lats, lons = sample.latlons()
    n_lat, n_lon = lats.shape
    n_points = n_lat * n_lon

    timestamps_list = sorted({g.validDate for g in messages})
    n_times = len(timestamps_list)

    print(
        f"   🗓️ Date: {day_date.isoformat()}, "
        f"timestamps: {n_times}, grid: {n_lat}x{n_lon} → {n_points:,} points/time"
    )
    print(f"   📄 source_file = {source_file}")

    # Map timestamp → index
    ts_index = {ts: i for i, ts in enumerate(timestamps_list)}

    # Pre-allocate arrays (float32 for memory)
    arr_u10 = np.full((n_times, n_points), np.nan, dtype=np.float32)
    arr_v10 = np.full((n_times, n_points), np.nan, dtype=np.float32)
    arr_t2m = np.full((n_times, n_points), np.nan, dtype=np.float32)
    arr_ssrd = np.full((n_times, n_points), np.nan, dtype=np.float32)

    # Fill arrays according to shortName
    for msg in messages:
        ts = msg.validDate
        idx = ts_index[ts]
        vals = msg.values.astype(np.float32).reshape(-1)
        short = msg.shortName

        if short == "10u":
            arr_u10[idx, :] = vals
        elif short == "10v":
            arr_v10[idx, :] = vals
        elif short == "2t":
            arr_t2m[idx, :] = vals
        elif short == "ssrd":
            arr_ssrd[idx, :] = vals

    # Flatten time × grid
    u10_flat = arr_u10.reshape(-1)
    v10_flat = arr_v10.reshape(-1)
    t2m_flat = arr_t2m.reshape(-1)
    ssrd_flat = arr_ssrd.reshape(-1)

    # Derived variables
    wind_speed = np.sqrt(u10_flat ** 2 + v10_flat ** 2)
    temperature_c = t2m_flat - 273.15

    # Timestamps repeated per grid point
    ts_array = np.array(timestamps_list, dtype="datetime64[ns]")
    ts_repeated = np.repeat(ts_array, n_points)

    # Lat/lon tiled over time
    lats_flat = lats.astype(np.float32).ravel()
    lons_flat = lons.astype(np.float32).ravel()
    lat_tiled = np.tile(lats_flat, n_times)
    lon_tiled = np.tile(lons_flat, n_times)

    n_rows = len(ts_repeated)
    print(f"   Building DataFrame: rows={n_rows:,}")

    df = pd.DataFrame(
        {
            "timestamp": ts_repeated,
            "lat": lat_tiled,
            "lon": lon_tiled,
            "u10": u10_flat,
            "v10": v10_flat,
            "t2m": t2m_flat,
            "ssrd": ssrd_flat,
            "wind_speed": wind_speed,
            "temperature_c": temperature_c,
        }
    )

    # Partition & load metadata
    df["year"] = year
    df["month"] = month
    df["day"] = day
    df["load_id"] = load_id
    df["load_ts"] = load_ts
    df["source_file"] = source_file

    print(
        "   ✅ DataFrame ready: "
        f"shape={df.shape}, "
        f"timestamp range=[{df['timestamp'].min()} … {df['timestamp'].max()}]"
    )

    return df


# ============================================================================
# DELTA WRITE (PARTITION OVERWRITE PER DAY)
# ============================================================================

def write_df_to_delta(df: pd.DataFrame):
    """
    Overwrite the partition for (year, month, day) using Delta + replaceWhere.
    This makes the pipeline idempotent per day (no duplicates if re-run).
    """

    year = int(df["year"].iloc[0])
    month = int(df["month"].iloc[0])
    day = int(df["day"].iloc[0])

    condition = f"year = {year} AND month = {month} AND day = {day}"

    print(
        f"   💾 Writing to {WeatherConfig.TARGET_TABLE} "
        f"(overwrite partition {year}-{month:02d}-{day:02d})"
    )
    print(f"   🔎 Rows to write: {len(df):,}")
    print(f"   🔎 Columns: {list(df.columns)}")

    spark = init_spark_session()
    sdf = spark.createDataFrame(df)

    (
        sdf.write
        .format("delta")
        .mode("overwrite")
        .option("replaceWhere", condition)
        .saveAsTable(WeatherConfig.TARGET_TABLE)
    )

    print(
        f"   ✅ Partition overwrite complete for {year}-{month:02d}-{day:02d}"
    )


# ============================================================================
# PER-DAY PIPELINE (download → parse → write → clean)
# ============================================================================

def process_one_day(day: date, run_load_id: str, run_load_ts: datetime) -> bool:
    """
    Full pipeline for a single day, executed in series for that day:

      download_one_day → grib_day_to_dataframe → write_df_to_delta → delete GRIB

    Returns True if successful, False otherwise.
    """
    print(f"\n🚀 Starting job for day {day.isoformat()}")

    # 1) Download
    grib_path = download_one_day(day)
    if not grib_path:
        print(f"   ⚠️ Skipping {day.isoformat()} due to download failure.")
        return False

    # 2) Parse GRIB → DataFrame
    try:
        df_day = grib_day_to_dataframe(grib_path, run_load_id, run_load_ts)
    except Exception as e:
        print(f"   ✗ Error parsing {day.isoformat()}: {e}")
        # still try to clean up file
        try:
            os.remove(grib_path)
            print(f"   🧹 Deleted temp file (after parse error): {os.path.abspath(grib_path)}")
        except OSError:
            print(f"   ⚠️ Could not delete temp file (after parse error): {os.path.abspath(grib_path)}")
        return False

    # 3) Write to Delta (partition overwrite)
    try:
        write_df_to_delta(df_day)
    except Exception as e:
        print(f"   ✗ Error writing to Delta for {day.isoformat()}: {e}")
        # do not retry here to keep threading simple; day is marked failed
        try:
            os.remove(grib_path)
            print(f"   🧹 Deleted temp file (after write error): {os.path.abspath(grib_path)}")
        except OSError:
            print(f"   ⚠️ Could not delete temp file (after write error): {os.path.abspath(grib_path)}")
        return False

    # 4) Delete GRIB file
    try:
        abs_grib = os.path.abspath(grib_path)
        os.remove(grib_path)
        print(f"   🧹 Deleted temp file: {abs_grib}")
    except OSError:
        print(f"   ⚠️ Could not delete temp file: {os.path.abspath(grib_path)}")

    print(f"✅ Finished job for day {day.isoformat()}")
    return True


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("\n🌤️  EUROPEAN WEATHER DATA PIPELINE (GRIB → DELTA, DAILY, PARALLEL-BY-DAY) 🌤️\n")
    print("📦 Required Python packages on the cluster:")
    print("   pip install cdsapi pygrib numpy pandas\n")

    if WeatherConfig.CDS_API_KEY in (None, "", "YOUR_UID:YOUR_API_KEY"):
        raise RuntimeError("CDS_API_KEY is not configured correctly in WeatherConfig.")

    days = iter_dates()
    print(f"📅 Date window: {WeatherConfig.MIN_DATE} → {WeatherConfig.MAX_DATE}")
    print(f"   Total days to process: {len(days)}")
    print(f"   Parallel workers (days in-flight): {WeatherConfig.MAX_WORKERS}\n")

    # Prepare Spark and handle FULL_RELOAD once (single-threaded)
    spark = init_spark_session()

    if WeatherConfig.FULL_RELOAD:
        print(f"🧨 FULL_RELOAD = TRUE → dropping table {WeatherConfig.TARGET_TABLE}…")
        spark.sql(f"DROP TABLE IF EXISTS {WeatherConfig.TARGET_TABLE}")
        print("   ✅ Table dropped. A fresh one will be created on first write.\n")

    # Run-level load metadata
    run_load_id = str(uuid.uuid4())
    run_load_ts = datetime.utcnow()

    print(f"🆔 run_load_id = {run_load_id}")
    print(f"🕒 run_load_ts = {run_load_ts.isoformat()}Z\n")

    processed_days = 0
    total_days = len(days)

    # Execute per-day pipeline in parallel
    with ThreadPoolExecutor(max_workers=WeatherConfig.MAX_WORKERS) as executor:
        future_to_day = {
            executor.submit(process_one_day, d, run_load_id, run_load_ts): d for d in days
        }

        for future in as_completed(future_to_day):
            d = future_to_day[future]
            try:
                ok = future.result()
                if ok:
                    processed_days += 1
                else:
                    print(f"❌ Day {d.isoformat()} failed (see logs above).")
            except Exception as e:
                print(f"❌ Unhandled exception for day {d.isoformat()}: {e}")

    print("\n🎉 Pipeline finished!")
    print(f"   Days successfully processed: {processed_days}/{total_days}")
    print(f"   Delta table: {WeatherConfig.TARGET_TABLE}")
    print(
        f"""
🔍 Example query (SQL):

SELECT year, month, day, COUNT(*) AS rows
FROM {WeatherConfig.TARGET_TABLE}
GROUP BY year, month, day
ORDER BY year, month, day;
"""
    )
