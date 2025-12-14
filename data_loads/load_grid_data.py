# Databricks notebook source
from entsoe import EntsoePandasClient
import pandas as pd
import time
import re
import warnings
import json

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

API_KEY = "YOUR API KEY HERE"

COUNTRIES = {
    "ES": "Spain", "PT": "Portugal", "FR": "France", "DE": "Germany",
    "IT": "Italy", "GB": "Great Britain", "NL": "Netherlands",
    "BE": "Belgium", "AT": "Austria", "CH": "Switzerland", "PL": "Poland",
    "CZ": "Czechia", "DK": "Denmark", "SE": "Sweden", "NO": "Norway",
    "FI": "Finland", "GR": "Greece", "IE": "Ireland", "RO": "Romania",
    "BG": "Bulgaria", "HU": "Hungary", "SK": "Slovakia", "SI": "Slovenia",
    "HR": "Croatia", "EE": "Estonia", "LT": "Lithuania", "LV": "Latvia",
}

# YOU REQUESTED THIS EXACT BLOCK KEPT UNCHANGED
VALID_BORDERS = {
    ("ES", "PT"), ("ES", "FR"),
    ("FR", "BE"), ("FR", "CH"), ("FR", "DE"), ("FR", "IT"),
    ("BE", "NL"), ("BE", "DE"),
    ("NL", "DE"), ("NL", "GB"),
    ("GB", "NL"), ("GB", "FR"), ("GB", "IE"),
    ("DE", "CZ"), ("DE", "PL"), ("DE", "CH"), ("DE", "DK"), ("DE", "AT"),
    ("DK", "DE"), ("DK", "NO"), ("DK", "SE"),
    ("SE", "NO"), ("SE", "FI"), ("SE", "DK"),
    ("NO", "NL"), ("NO", "GB"), ("NO", "SE"), ("NO", "DK"),
    ("FI", "EE"), ("FI", "SE"),
    ("EE", "LV"),
    ("LV", "LT"),
    ("LT", "PL"),
    ("PL", "SK"), ("PL", "CZ"),
    ("CZ", "AT"), ("CZ", "SK"),
    ("AT", "SI"), ("AT", "IT"), ("AT", "CH"), ("AT", "CZ"), ("AT", "DE"),
    ("SI", "HR"), ("SI", "IT"), ("SI", "AT"),
    ("HR", "HU"), ("HR", "SI"),
    ("HU", "SK"), ("HU", "RO"), ("HU", "HR"), ("HU", "AT"),
    ("SK", "HU"), ("SK", "CZ"), ("SK", "PL"),
    ("RO", "BG"), ("RO", "HU"),
    ("BG", "GR"), ("BG", "RO"),
    ("GR", "BG"),
}

START_DATE = "2023-01-01"
END_DATE   = "2025-10-31"

DATABASE = "european_grid_raw__v2"

DATASETS = [
    "load_actual",
    "load_forecast",
    "generation",
    "generation_total",
    "wind_forecast",
    "solar_forecast",
    "installed_capacity",
    "crossborder_flows",
]

# ============================================================================
# DATABRICKS JOB PARAMETERS (FULL LOAD ONLY, BUT DATE RANGE & DATASETS ARE PARAMETRIC)
# ============================================================================

def get_widget_value(*widget_names: str) -> str:
    """Get widget value, trying multiple possible widget names."""
    for name in widget_names:
        try:
            value = dbutils.widgets.get(name)
            if value and value.strip():
                return value
        except:
            continue
    return None

try:
    from_date = get_widget_value("from_date", "--from", "from")
    to_date = get_widget_value("to_date", "--to", "to")
    data_types = get_widget_value("data_types", "--type", "type")
except:
    from_date = None
    to_date = None
    data_types = None

if from_date:
    START_DATE = from_date
    print(f"✓ Using job parameter from_date: {from_date}")
if to_date:
    END_DATE = to_date
    print(f"✓ Using job parameter to_date: {to_date}")

if data_types:
    if data_types.lower().strip() == "all":
        SELECTED_DATASETS = DATASETS
    else:
        SELECTED_DATASETS = [ds.strip() for ds in data_types.split(",")]
    print(f"✓ Using job parameter data_types: {data_types} -> {SELECTED_DATASETS}")
else:
    SELECTED_DATASETS = DATASETS

print("✓ FULL_LOAD_MODE = True (script only implements full load)")

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

# ============================================================================
# DELTA WRITERS (FULL LOAD, PER-MONTH PARTITIONS)
# ============================================================================

def write_dataset(dataset_name: str, country_code: str, df: pd.DataFrame):
    """
    Full-load writer for a single dataset and country over the whole date range.

    - Input df:
        * index = timestamp (ENTSO-E index)
        * various value columns
    - We:
        * keep 'index' as timestamp column
        * add 'month' (YYYY-MM) computed from 'index'
        * add 'country'
        * sanitize column names
        * partition by (country, month)
        * overwrite partitions via replaceWhere
        * use mergeSchema = true so new columns are added and missing ones become NULL
    """
    if df is None or len(df) == 0:
        print(f"    → No data returned for {dataset_name} {country_code}, skipping.")
        return

    # Debug original columns (index is in df.index here)
    print(f"    [DEBUG] Source columns (before reset_index): {list(df.columns)}")

    # Bring index into a column named 'index'
    df = df.reset_index()
    print(f"    [DEBUG] Columns after reset_index: {list(df.columns)}")

    if "index" not in df.columns:
        raise RuntimeError(f"'index' column missing after reset_index for {dataset_name}/{country_code}")

    # Compute month from timestamp
    df["month"] = pd.to_datetime(df["index"]).dt.to_period("M").astype(str)
    df["country"] = country_code

    # Sanitize column names
    df = sanitize_columns(df)
    print(f"    [DEBUG] Columns after sanitization: {list(df.columns)}")

    full_name = f"{DATABASE}.{dataset_name}"
    month_values = sorted(df["month"].unique())

    for month_val in month_values:
        df_month = df[df["month"] == month_val].copy()
        if df_month.empty:
            continue

        sdf = spark.createDataFrame(df_month)
        rows = sdf.count()

        if not spark.catalog.tableExists(full_name):
            # First write: create table partitioned by (country, month)
            print(
                f"  💾 Creating {full_name} with "
                f"(country='{country_code}', month='{month_val}'), {rows} rows"
            )
            (
                sdf.write
                .format("delta")
                .mode("overwrite")
                .option("mergeSchema", "true")
                .partitionBy("country", "month")
                .saveAsTable(full_name)
            )
        else:
            # Overwrite just this (country, month) partition
            condition = f"country = '{country_code}' AND month = '{month_val}'"
            print(
                f"  💾 Writing to {full_name} "
                f"(overwrite country='{country_code}', month='{month_val}', {rows} rows)"
            )
            (
                sdf.write
                .format("delta")
                .mode("overwrite")
                .option("mergeSchema", "true")
                .option("replaceWhere", condition)
                .saveAsTable(full_name)
            )

        print(f"  ✅ Saved {rows} rows for country={country_code}, month={month_val} into {full_name}")


def _write_crossborder_pair(df: pd.DataFrame, from_c: str, to_c: str):
    """
    Full-load writer for a cross-border pair over the whole date range.

    - Input df:
        * MUST already have 'index' column with timestamp (caller does reset_index())
    - We:
        * keep 'index' as timestamp column
        * add 'month' (YYYY-MM) from index
        * add from_country, to_country
        * sanitize columns
        * partition by (from_country, to_country, month)
        * overwrite partitions via replaceWhere
        * use mergeSchema = true for schema evolution
    """
    if df is None or len(df) == 0:
        print(f"    → No data for {from_c}->{to_c}, skipping.")
        return

    print(f"      [DEBUG] Crossborder source columns (before month/from/to): {list(df.columns)}")

    if "index" not in df.columns:
        raise RuntimeError(f"'index' column missing in crossborder df for {from_c}->{to_c}")

    df["month"] = pd.to_datetime(df["index"]).dt.to_period("M").astype(str)
    df["from_country"] = from_c
    df["to_country"] = to_c

    df = sanitize_columns(df)
    print(f"      [DEBUG] Crossborder columns after sanitization: {list(df.columns)}")

    table_name = "crossborder_flows"
    full_name = f"{DATABASE}.{table_name}"
    month_values = sorted(df["month"].unique())

    for month_val in month_values:
        df_month = df[df["month"] == month_val].copy()
        if df_month.empty:
            continue

        # Try to create a stable "Value" column name (optional)
        exclude_cols = {"index", "from_country", "to_country", "month"}
        value_cols = [c for c in df_month.columns if c not in exclude_cols]
        if len(value_cols) == 1:
            df_month = df_month.rename(columns={value_cols[0]: "Value"})
        elif len(value_cols) > 1:
            for c in value_cols:
                if df_month[c].dtype.kind in ("i", "u", "f"):
                    df_month = df_month.rename(columns={c: "Value"})
                    break

        sdf = spark.createDataFrame(df_month)
        rows = sdf.count()

        if not spark.catalog.tableExists(full_name):
            print(
                f"    💾 Creating {full_name} with "
                f"(from='{from_c}', to='{to_c}', month='{month_val}'), {rows} rows"
            )
            (
                sdf.write
                .format("delta")
                .mode("overwrite")
                .option("mergeSchema", "true")
                .partitionBy("from_country", "to_country", "month")
                .saveAsTable(full_name)
            )
        else:
            condition = (
                f"from_country = '{from_c}' AND "
                f"to_country = '{to_c}' AND "
                f"month = '{month_val}'"
            )
            print(
                f"    💾 Writing to {full_name} "
                f"(overwrite from='{from_c}', to='{to_c}', month='{month_val}', {rows} rows)"
            )
            (
                sdf.write
                .format("delta")
                .mode("overwrite")
                .option("mergeSchema", "true")
                .option("replaceWhere", condition)
                .saveAsTable(full_name)
            )

        print(f"    ✅ Saved {rows} rows for {from_c}->{to_c}, month={month_val} into {full_name}")

# ============================================================================
# DATA COLLECTOR (FULL LOAD ONLY)
# ============================================================================

class EuropeanGridDataCollector:

    def __init__(self, api_key, start_date: str = None, end_date: str = None, selected_datasets: list = None):
        if not api_key or api_key.strip() == "":
            raise ValueError("API_KEY must be set")

        self.client = EntsoePandasClient(api_key=api_key)
        self.countries = COUNTRIES
        self.selected_datasets = selected_datasets or DATASETS

        start = start_date or START_DATE
        end = end_date or END_DATE
        self.start = pd.Timestamp(start, tz="UTC")
        self.end   = pd.Timestamp(end, tz="UTC")

    # -------------------------------
    # SINGLE COUNTRY DATA - FULL DATE RANGE
    # -------------------------------
    def collect_country_full_range(self, country_code: str):
        c = country_code
        print(f"\n==== Collecting FULL RANGE for {c} ({self.countries[c]}) ====")
        print(f"    Date range: {self.start.date()} to {self.end.date()}")

        if "load_actual" in self.selected_datasets:
            try:
                print(f"    → load_actual...")
                df = self.client.query_load(c, start=self.start, end=self.end)
                write_dataset("load_actual", c, df)
            except Exception as e:
                print(f"    ✗ load_actual: {e}")
            time.sleep(1)

        if "load_forecast" in self.selected_datasets:
            try:
                print(f"    → load_forecast...")
                df = self.client.query_load_forecast(c, start=self.start, end=self.end)
                write_dataset("load_forecast", c, df)
            except Exception as e:
                print(f"    ✗ load_forecast: {e}")
            time.sleep(1)

        if "generation" in self.selected_datasets:
            try:
                print(f"    → generation...")
                df = self.client.query_generation(c, start=self.start, end=self.end)
                write_dataset("generation", c, df)
            except Exception as e:
                print(f"    ✗ generation: {e}")
            time.sleep(1)

        if "generation_total" in self.selected_datasets:
            try:
                print(f"    → generation_total...")
                df = self.client.query_generation(c, start=self.start, end=self.end, psr_type=None)
                write_dataset("generation_total", c, df)
            except Exception as e:
                print(f"    ✗ generation_total: {e}")
            time.sleep(1)

        if "wind_forecast" in self.selected_datasets:
            try:
                print(f"    → wind_forecast...")
                df = self.client.query_wind_and_solar_forecast(
                    c, start=self.start, end=self.end, psr_type="B19"
                )
                write_dataset("wind_forecast", c, df)
            except Exception as e:
                print(f"    ✗ wind_forecast: {e}")
            time.sleep(1)

        if "solar_forecast" in self.selected_datasets:
            try:
                print(f"    → solar_forecast...")
                df = self.client.query_wind_and_solar_forecast(
                    c, start=self.start, end=self.end, psr_type="B16"
                )
                write_dataset("solar_forecast", c, df)
            except Exception as e:
                print(f"    ✗ solar_forecast: {e}")
            time.sleep(1)

        if "installed_capacity" in self.selected_datasets:
            try:
                print(f"    → installed_capacity...")
                df = self.client.query_installed_generation_capacity(
                    c, start=self.start, end=self.end
                )
                write_dataset("installed_capacity", c, df)
            except Exception as e:
                print(f"    ✗ installed_capacity: {e}")

    # -------------------------------
    # CROSS-BORDER FLOWS - FULL DATE RANGE
    # -------------------------------
        # -------------------------------
    # CROSS-BORDER FLOWS - FULL DATE RANGE
    # -------------------------------
    def collect_crossborder_full_range(self):
        if "crossborder_flows" not in self.selected_datasets:
            return

        print(f"\n=== Collecting Cross-Border Flows (FULL RANGE) ===")
        print(f"    Date range: {self.start.date()} to {self.end.date()}")

        # Build a set of unique *undirected* borders from VALID_BORDERS
        # e.g. ("DE", "DK") and ("DK", "DE") -> only one entry ("DE", "DK")
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
                        from_c, to_c, start=self.start, end=self.end
                    )

                    if flow is None or len(flow) == 0:
                        print(" ✗ No data")
                        time.sleep(0.5)
                        continue

                    # Normalize to DataFrame and ensure we have 'index'
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

                time.sleep(0.5)

    # -------------------------------
    # MAIN - FULL LOAD ONLY
    # -------------------------------
    def collect_all(self):
        print(f"=== Starting ENTSO-E data collection ===")
        print(f"Date range: {self.start.date()} to {self.end.date()}")
        print(f"Selected datasets: {', '.join(self.selected_datasets)}")
        print(f"Mode: FULL LOAD (drop & reload)")

        # Drop selected tables once at the beginning
        print("\n🧨 FULL LOAD MODE → dropping selected tables before reload")
        # for ds in self.selected_datasets:
        #     drop_table_if_exists(ds)

        # Per-country full-range collection
        for c in self.countries.keys():
            self.collect_country_full_range(c)

        # Cross-border full-range collection
        self.collect_crossborder_full_range()

        print("\n✅ ENTSO-E full-load pipeline completed successfully.")

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
