# Databricks notebook source
from entsoe import EntsoePandasClient
import pandas as pd
import time
import re
import warnings
import json
from datetime import date, timedelta
import traceback

from pyspark.sql import SparkSession

# Suppress annoying warnings from external libraries
warnings.filterwarnings('ignore', category=FutureWarning, module='entsoe')
warnings.filterwarnings('ignore', category=DeprecationWarning, module='pyspark')
warnings.filterwarnings('ignore', category=DeprecationWarning, module='pandas')
warnings.filterwarnings('ignore', message='.*Downcasting object dtype arrays.*')
warnings.filterwarnings('ignore', message='.*distutils Version classes are deprecated.*')
warnings.filterwarnings('ignore', message='.*is_categorical_dtype is deprecated.*')
warnings.filterwarnings('ignore', message='.*is_datetime64tz_dtype is deprecated.*')
warnings.filterwarnings('ignore', message='.*Converting to PeriodArray.*will drop timezone information.*')

# ============================================================================
# GLOBAL CONFIG
# ============================================================================

API_KEY = "your API key here"

COUNTRIES = {
    "AT": "Austria",
    "BE": "Belgium",
    "DE": "Germany",
    "ES": "Spain",
    "FR": "France",
    "HR": "Croatia",
    "HU": "Hungary",
    "IT": "Italy",
    "LT": "Lithuania",
    "NL": "Netherlands",
    "PL": "Poland",
    "PT": "Portugal",
    "SK": "Slovakia",
}

VALID_BORDERS = {
    ("ES", "PT"), ("ES", "FR"),
    ("FR", "BE"), ("FR", "DE"), ("FR", "IT"),
    ("BE", "NL"), ("BE", "DE"),
    ("NL", "DE"),
    ("DE", "PL"), ("DE", "AT"),
    ("AT", "DE"), ("AT", "IT"),
    ("LT", "PL"),
    ("PL", "SK"),
    ("SK", "PL"), ("SK", "HU"),
    ("HR", "HU"),
    ("HU", "AT"), ("HU", "HR"), ("HU", "SK"),
}

START_DATE = "2023-01-01"
END_DATE   = "2025-10-31"

DATABASE = "european_grid_raw__v2"

DATASETS = [
    "load_actual",
    "load_forecast",
    "generation",
    # "generation_total",
    "wind_forecast",
    "solar_forecast",
    # "installed_capacity",
    "crossborder_flows",
]

# ============================================================================
# DATABRICKS JOB PARAMETERS (DATE RANGE & DATASETS)
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
    from_date  = get_widget_value("from_date", "--from", "from")
    to_date    = get_widget_value("to_date", "--to", "to")
    days_back  = get_widget_value("days_back", "lookback_days", "days")
    data_types = get_widget_value("data_types", "--type", "type")
except Exception:
    from_date = to_date = days_back = data_types = None

# --- date window resolution ---
if from_date or to_date:
    # explicit range wins
    if from_date:
        START_DATE = from_date
    if to_date:
        END_DATE = to_date
    print(f"✓ Using job parameter date range: {START_DATE} → {END_DATE}")
elif days_back:
    # relative window: last N days including today
    try:
        n = int(days_back)
        if n < 1:
            n = 1
    except ValueError:
        n = 1
    today = date.today()
    start_dt = today - timedelta(days=n - 1)
    START_DATE = start_dt.isoformat()
    END_DATE = today.isoformat()
    print(f"✓ Using job parameter days_back={n}: {START_DATE} → {END_DATE}")
else:
    # keep defaults (full range)
    print(f"✓ Using default START_DATE / END_DATE: {START_DATE} → {END_DATE}")

# --- datasets selection ---
if data_types:
    if data_types.lower().strip() == "all":
        SELECTED_DATASETS = DATASETS
    else:
        SELECTED_DATASETS = [ds.strip() for ds in data_types.split(",")]
    print(f"✓ Using job parameter data_types: {data_types} -> {SELECTED_DATASETS}")
else:
    SELECTED_DATASETS = DATASETS

print("✓ FULL_LOAD_MODE = True (script can still do full loads by giving a wide date range)")

# ============================================================================
# SPARK INIT
# ============================================================================

def init_spark_session() -> SparkSession:
    spark = SparkSession.builder.getOrCreate()
    spark.sql(f"USE {DATABASE}")
    return spark

spark = init_spark_session()

# ============================================================================
# COLUMN SANITIZER
# ============================================================================

INVALID_CHARS_PATTERN = re.compile(r"[^0-9A-Za-z_]+")

def sanitize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flatten MultiIndex columns and remove characters invalid for Delta.
    - MultiIndex ('Biomass', 'Actual Aggregated') -> 'Biomass__Actual_Aggregated'
    - Replace any non [0-9A-Za-z_] chars with '_'
    - If name starts with a digit, prefix with '_'
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

def drop_table_if_exists(dataset_name: str):
    """Drop a Delta table entirely (for full reload)."""
    full_name = f"{DATABASE}.{dataset_name}"
    print(f"  → Dropping table IF EXISTS {full_name}")
    spark.sql(f"DROP TABLE IF EXISTS {full_name}")

def get_partition_columns(full_table_name: str) -> list[str]:
    """
    Return the partition columns for an existing Delta table.
    If the table does not exist, return [].
    """
    if not spark.catalog.tableExists(full_table_name):
        return []
    detail = spark.sql(f"DESCRIBE DETAIL {full_table_name}").collect()[0]
    return list(detail["partitionColumns"] or [])

# ============================================================================
# HELPER: ROBUST TIMESTAMP EXTRACTION
# ============================================================================

def _extract_timestamp(df: pd.DataFrame, context: str) -> pd.Series:
    """
    Try to find a timestamp column in df and return it as a pandas Series (UTC).

    Priority:
      1. Columns named "index", "time", "timestamp", "datetime"
      2. First column that can be parsed as datetime with at least one non-NaT

    Raises RuntimeError if no suitable column is found.
    """
    candidates = ["index", "time", "timestamp", "datetime"]

    # 1) Named candidates
    for col in candidates:
        if col in df.columns:
            ts = pd.to_datetime(df[col], errors="coerce", utc=True)
            if ts.notna().any():
                return ts

    # 2) Any column that parses as datetime
    for col in df.columns:
        ts = pd.to_datetime(df[col], errors="coerce", utc=True)
        if ts.notna().any():
            return ts

    raise RuntimeError(
        f"Could not find a timestamp column for context '{context}'. "
        f"Available columns: {list(df.columns)}"
    )

# ============================================================================
# DELTA WRITERS – PER-DATE PARTITION
# ============================================================================

def write_dataset(dataset_name: str, country_code: str, df: pd.DataFrame):
    """
    Writer for a single dataset and country over the chosen date range.

    - Derives date/month from the actual data timestamp (no job-date usage).
    - Always computes:
        * date (YYYY-MM-DD)
        * month (YYYY-MM)  [normal column]
    - Writes *per date*:
        * For each date d, overwrite rows where (country = X AND date = d)
    - If table does not exist:
        * Create partitioned by (country, date).
    """
    if df is None or len(df) == 0:
        print(f"    → No data returned for {dataset_name} {country_code}, skipping.")
        return

    print(f"    [DEBUG] Source columns (before reset_index): {list(df.columns)}")

    df = df.reset_index(drop=False)
    print(f"    [DEBUG] Columns after reset_index: {list(df.columns)}")

    # Robustly get the timestamp series
    ts = _extract_timestamp(df, context=f"{dataset_name}/{country_code}")

    # Derive date/month from the real timestamps
    df["date"] = ts.dt.date.astype(str)
    df["month"] = ts.dt.to_period("M").astype(str)
    df["country"] = country_code

    df = sanitize_columns(df)
    print(f"    [DEBUG] Columns after sanitization: {list(df.columns)}")

    full_name = f"{DATABASE}.{dataset_name}"
    table_exists = spark.catalog.tableExists(full_name)

    # We *always* group by date now
    group_col = "date"
    values = sorted(df[group_col].unique())

    for val in values:
        df_part = df[df[group_col] == val].copy()
        if df_part.empty:
            continue

        sdf = spark.createDataFrame(df_part)
        rows = sdf.count()

        if not table_exists:
            # First write: create table partitioned by (country, date)
            print(
                f"  💾 Creating {full_name} partitioned by country,date, {rows} rows "
                f"for date={val}, country={country_code}"
            )
            (
                sdf.write
                .format("delta")
                .mode("overwrite")
                .option("mergeSchema", "true")
                .partitionBy("country", "date")
                .saveAsTable(full_name)
            )
            table_exists = True
        else:
            # Overwrite only the specific date for this country
            condition = f"country = '{country_code}' AND date = '{val}'"
            print(
                f"  💾 Writing to {full_name} "
                f"(overwrite country='{country_code}', date='{val}', {rows} rows)"
            )
            (
                sdf.write
                .format("delta")
                .mode("overwrite")
                .option("mergeSchema", "true")
                .option("replaceWhere", condition)
                .saveAsTable(full_name)
            )

        print(f"  ✅ Saved {rows} rows for country={country_code}, date={val} into {full_name}")


def _write_crossborder_pair(df: pd.DataFrame, from_c: str, to_c: str):
    """
    Writer for a cross-border pair over the chosen date range.

    FIXED VERSION:
    - Avoids double reset_index → removes 'level_0'
    - Ensures EXACTLY ONE value column named 'value'
    - Prevents Spark case-insensitive duplicate-column errors
    - Keeps partitioning by (from_country, to_country, date)
    """

    if df is None or len(df) == 0:
        print(f"    → No data for {from_c}->{to_c}, skipping.")
        return

    print(f"      [DEBUG] Crossborder source columns (before fixing): {list(df.columns)}")

    # IMPORTANT: df was *already* reset_index() in collect_crossborder_full_range
    # → NO SECOND reset_index HERE
    if "index" not in df.columns:
        raise RuntimeError(
            f"'index' column missing for crossborder {from_c}->{to_c}. Columns={df.columns}"
        )

    # Extract timestamp from the 'index' column
    ts = _extract_timestamp(df, context=f"crossborder {from_c}->{to_c}")

    df = df.copy()
    df["date"] = ts.dt.date.astype(str)
    df["month"] = ts.dt.to_period("M").astype(str)
    df["from_country"] = from_c
    df["to_country"] = to_c

    # Identify data columns (numeric series)
    exclude = {"index", "date", "month", "from_country", "to_country"}
    candidates = [c for c in df.columns if c not in exclude]

    if not candidates:
        raise RuntimeError(
            f"No numeric value column found for {from_c}->{to_c}. Columns={df.columns}"
        )

    # Select the first candidate column as the value column
    value_col = candidates[0]
    extra_cols = candidates[1:]

    # Drop all extra numeric columns (to avoid Spark duplicate case-insensitive column issues)
    if extra_cols:
        df = df.drop(columns=extra_cols)

    # Standardize name to 'value'
    if value_col.lower() != "value":
        df = df.rename(columns={value_col: "value"})

    df = sanitize_columns(df)
    print(f"      [DEBUG] Crossborder columns after fix: {list(df.columns)}")

    full_name = f"{DATABASE}.crossborder_flows"
    table_exists = spark.catalog.tableExists(full_name)

    # Write each date separately
    for date_val in sorted(df["date"].unique()):
        df_day = df[df["date"] == date_val].copy()
        if df_day.empty:
            continue

        sdf = spark.createDataFrame(df_day)
        rows = sdf.count()

        if not table_exists:
            print(
                f"    💾 Creating {full_name} partitioned by (from_country,to_country,date) "
                f"with {rows} rows for date {date_val} {from_c}->{to_c}"
            )
            (
                sdf.write
                .format("delta")
                .mode("overwrite")
                .option("mergeSchema", "true")
                .partitionBy("from_country", "to_country", "date")
                .saveAsTable(full_name)
            )
            table_exists = True
        else:
            condition = (
                f"from_country = '{from_c}' AND "
                f"to_country = '{to_c}' AND "
                f"date = '{date_val}'"
            )
            print(
                f"    💾 Overwriting {full_name} for {from_c}->{to_c} date={date_val} ({rows} rows)"
            )
            (
                sdf.write
                .format("delta")
                .mode("overwrite")
                .option("mergeSchema", "true")
                .option("replaceWhere", condition)
                .saveAsTable(full_name)
            )

        print(
            f"    ✅ Saved {rows} rows for {from_c}->{to_c}, date={date_val} into {full_name}"
        )

# ============================================================================
# DATA COLLECTOR (FULL RANGE = [START_DATE, END_DATE])
# ============================================================================

class EuropeanGridDataCollector:

    def __init__(self, api_key, start_date: str = None, end_date: str = None, selected_datasets: list = None):
        if not api_key or api_key.strip() == "":
            raise ValueError("API_KEY must be set")

        self.client = EntsoePandasClient(api_key=api_key)
        self.countries = COUNTRIES
        self.selected_datasets = selected_datasets or DATASETS

        start = start_date or START_DATE
        end   = end_date   or END_DATE

        self.start = pd.Timestamp(start, tz="UTC")

        # Inclusive / exclusive handling:
        # - end_inclusive: what the user means (last full day)
        # - end_exclusive: what the ENTSO-E API expects (exclusive upper bound)
        self.end_inclusive = pd.Timestamp(end, tz="UTC")
        self.end_exclusive = self.end_inclusive + pd.Timedelta(days=1)

    # -------------------------------
    # SINGLE COUNTRY DATA - GIVEN DATE RANGE
    # -------------------------------
    def collect_country_full_range(self, country_code: str):
        c = country_code
        print(f"\n==== Collecting RANGE for {c} ({self.countries[c]}) ====")
        print(f"    Date range: {self.start.date()} to {self.end_inclusive.date()}")

        if "load_actual" in self.selected_datasets:
            try:
                print(f"    → load_actual...")
                df = self.client.query_load(c, start=self.start, end=self.end_exclusive)
                write_dataset("load_actual", c, df)
            except Exception as e:
                print(f"    ✗ load_actual: {e}")
                traceback.print_exc()
            time.sleep(1)

        if "load_forecast" in self.selected_datasets:
            try:
                print(f"    → load_forecast...")
                df = self.client.query_load_forecast(c, start=self.start, end=self.end_exclusive)
                write_dataset("load_forecast", c, df)
            except Exception as e:
                print(f"    ✗ load_forecast: {e}")
                traceback.print_exc()
            time.sleep(1)

        if "generation" in self.selected_datasets:
            try:
                print(f"    → generation...")
                df = self.client.query_generation(c, start=self.start, end=self.end_exclusive)
                write_dataset("generation", c, df)
            except Exception as e:
                print(f"    ✗ generation: {e}")
                traceback.print_exc()
            time.sleep(1)

        if "generation_total" in self.selected_datasets:
            try:
                print(f"    → generation_total...")
                df = self.client.query_generation(c, start=self.start, end=self.end_exclusive, psr_type=None)
                write_dataset("generation_total", c, df)
            except Exception as e:
                print(f"    ✗ generation_total: {e}")
                traceback.print_exc()
            time.sleep(1)

        if "wind_forecast" in self.selected_datasets:
            try:
                print(f"    → wind_forecast...")
                df = self.client.query_wind_and_solar_forecast(
                    c, start=self.start, end=self.end_exclusive, psr_type="B19"
                )
                write_dataset("wind_forecast", c, df)
            except Exception as e:
                print(f"    ✗ wind_forecast: {e}")
                traceback.print_exc()
            time.sleep(1)

        if "solar_forecast" in self.selected_datasets:
            try:
                print(f"    → solar_forecast...")
                df = self.client.query_wind_and_solar_forecast(
                    c, start=self.start, end=self.end_exclusive, psr_type="B16"
                )
                write_dataset("solar_forecast", c, df)
            except Exception as e:
                print(f"    ✗ solar_forecast: {e}")
                traceback.print_exc()
            time.sleep(1)

        if "installed_capacity" in self.selected_datasets:
            try:
                print(f"    → installed_capacity...")
                df = self.client.query_installed_generation_capacity(
                    c, start=self.start, end=self.end_exclusive
                )
                write_dataset("installed_capacity", c, df)
            except Exception as e:
                print(f"    ✗ installed_capacity: {e}")
                traceback.print_exc()

    # -------------------------------
    # CROSS-BORDER FLOWS - GIVEN DATE RANGE
    # -------------------------------
    def collect_crossborder_full_range(self):
        if "crossborder_flows" not in self.selected_datasets:
            return

        print(f"\n=== Collecting Cross-Border Flows (RANGE) ===")
        print(f"    Date range: {self.start.date()} to {self.end_inclusive.date()}")

        # Build a set of unique *undirected* borders from VALID_BORDERS
        unique_borders = set()
        for a, b in VALID_BORDERS:
            if a == b:
                continue  # ignore self-links just in case
            border_key = tuple(sorted((a, b)))
            unique_borders.add(border_key)

        # For each border, pull *both* directions: A->B and B->A
        for a, b in sorted(unique_borders):
            for from_c, to_c in ((a, b), (b, a)):
                print(f"  → {from_c} ↔ {to_c}...", end="")
                try:
                    flow = self.client.query_crossborder_flows(
                        from_c, to_c, start=self.start, end=self.end_exclusive
                    )

                    if flow is None or len(flow) == 0:
                        print(" ✗ No data")
                        time.sleep(0.5)
                        continue

                    # Normalize to DataFrame and ensure we have a timestamp
                    if isinstance(flow, pd.Series):
                        df = flow.to_frame(name="value").reset_index()
                    elif isinstance(flow, pd.DataFrame):
                        df = flow.reset_index()
                    else:
                        df = pd.DataFrame(flow).reset_index()

                    _write_crossborder_pair(df, from_c, to_c)
                    print(" ✓")
                except Exception as e:
                    print(f" ✗ Failed: {e}")
                    traceback.print_exc()

                time.sleep(0.5)

    # -------------------------------
    # MAIN
    # -------------------------------
    def collect_all(self):
        print(f"=== Starting ENTSO-E data collection ===")
        print(f"Date range: {self.start.date()} to {self.end_inclusive.date()}")
        print(f"Selected datasets: {', '.join(self.selected_datasets)}")
        print(f"Mode: RANGE LOAD (idempotent per partition)")

        # Per-country range collection
        for c in self.countries.keys():
            self.collect_country_full_range(c)

        # Cross-border range collection
        self.collect_crossborder_full_range()

        print("\n✅ ENTSO-E range-load pipeline completed successfully.")

# ============================================================================
# RUN PIPELINE
# ============================================================================

collector = EuropeanGridDataCollector(
    api_key=API_KEY,
    start_date=START_DATE,
    end_date=END_DATE,
    selected_datasets=SELECTED_DATASETS,
)
collector.collect_all()
print("\nCOMPLETE.")
