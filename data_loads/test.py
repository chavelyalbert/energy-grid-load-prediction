"""
build_master_generation_dataset.py

Purpose:
  Build master generation-by-type + total generation dataset for 2023-01-01 -> 2025-11-07
  Merges ENTSO-E (A75/A76) with national TSO APIs and optional ERA5 modeled renewables.

Dependencies:
  pip install entsoe-py pandas requests pyarrow parquetlib cdsapi retrying beautifulsoup4 lxml

Notes:
  - Put your ENTSO-E API key in ENTSEE_API_KEY.
  - For ERA5 via Copernicus, set up CDS API key if you plan to use ERA5 gap-filling.
  - Some TSO endpoints require scraping or token retrieval (Terna, TenneT); script includes examples.
"""

import os
import time
import json
import logging
from datetime import datetime, timedelta
from retrying import retry
import pandas as pd
import requests
from entsoe import EntsoePandasClient

# ---------- CONFIG ----------
ENTSOE_API_KEY = os.getenv("ENTSOE_API_KEY", "7b785108-53d7-42f8-931e-3d28c4323c68")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "/mnt/data")
START = pd.Timestamp("2023-01-01T00:00Z")
END   = pd.Timestamp("2025-11-07T23:00Z")
COUNTRIES = [
    "AT","BE","BG","CH","CY","CZ","DE","DK","EE","ES",
    "FI","FR","GB","GR","HR","HU","IE","IT","LT","LU",
    "LV","NL","NO","PL","PT","RO","SE","SI","SK"
]
# prefer TSO data for these countries (TSO fetchers implemented below)
TSO_PRIORITY = {
    "ES":"ree", "FR":"rte", "DE":"smard", "DK":"energinet",
    "FI":"fingrid", "PT":"ren", "BE":"elia", "NO":"statnett",
    "SE":"svk", "NL":"tennet", "IT":"terna", "GB":"ngeso"
}

# Where to write outputs
MASTER_PARQUET = os.path.join(OUTPUT_DIR, "master_generation_20230101_20251107.parquet")
LOG_FILE = os.path.join(OUTPUT_DIR, "build_master_generation.log")

# ---------- Logging ----------
os.makedirs(OUTPUT_DIR, exist_ok=True)
logging.basicConfig(filename=LOG_FILE, level=logging.INFO,
                    format="%(asctime)s %(levelname)s: %(message)s")
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger("").addHandler(console)

# ---------- Helpers ----------
client = EntsoePandasClient(api_key=ENTSOE_API_KEY)

def _safe_ts_to_utc(ts):
    if isinstance(ts, str):
        return pd.Timestamp(ts).tz_convert("UTC") if pd.Timestamp(ts).tz is not None else pd.Timestamp(ts).tz_localize("UTC")
    return ts.tz_localize("UTC") if ts.tz is None else ts.tz_convert("UTC")

def _hour_index():
    return pd.date_range(start=START, end=END, freq="H", tz="UTC")

# Retry on exceptions
@retry(stop_max_attempt_number=5, wait_exponential_multiplier=1000, wait_exponential_max=10000)
def fetch_entsoe_total(country):
    """
    Fetch A75 total actual generation for one country as a pandas.Series indexed by timestamp.
    Use psr_type=None to request aggregated total.
    """
    logging.info(f"ENTSO-E: fetching total generation for {country}")
    s = client.query_generation(country_code=country, start=START, end=END, psr_type=None)
    # entsoe-py returns pandas.Series; ensure timezone-aware UTC
    s.index = s.index.tz_convert("UTC")
    s.name = "Actual_Generation_Total"
    return s

@retry(stop_max_attempt_number=5, wait_exponential_multiplier=1000, wait_exponential_max=10000)
def fetch_entsoe_by_psr(country, psr_type=None):
    """
    Fetch A76 generation for specific psr code (or None returns all types separately).
    If psr_type is None, entsoe returns a Series/DF of aggregated by PSR depending on client version.
    """
    logging.info(f"ENTSO-E: fetching generation by PSR for {country} (psr_type={psr_type})")
    df = client.query_generation(country_code=country, start=START, end=END, psr_type=psr_type)
    # entsoe-py may return Series or DataFrame
    if isinstance(df, pd.Series):
        df = df.to_frame()
    df.index = df.index.tz_convert("UTC")
    return df

# -------------------- TSO fetchers --------------------
# These TSO fetchers are examples and may need small adjustments for tokens or pagination.
# They return DataFrame indexed UTC hourly with columns for generation types (standard names).

def fetch_tso_ree():
    """Spain REE API: good coverage, returns structure by technology."""
    logging.info("Fetching REE (Spain) generation")
    url = "https://apidatos.ree.es/en/datos/generacion/estructura"
    params = {
        "start_date": START.strftime("%Y-%m-%dT%H:%M"),
        "end_date":   END.strftime("%Y-%m-%dT%H:%M"),
        "time_trunc": "hour"
    }
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    js = r.json()
    # REE returns nested structure: process to DataFrame
    records = []
    for serie in js.get("included", []) if "included" in js else js.get("data", []):
        # older API shapes vary - search for 'attributes' / 'values'
        attrs = serie.get("attributes") if isinstance(serie, dict) and "attributes" in serie else serie
        if not attrs:
            continue
        # many blocks: each 'values' contain {"value","datetime"}
        for v in attrs.get("values", []):
            dt = pd.Timestamp(v["datetime"]).tz_convert("UTC") if pd.Timestamp(v["datetime"]).tz is not None else pd.Timestamp(v["datetime"]).tz_localize("UTC")
            tech = attrs.get("indicator") or attrs.get("group") or attrs.get("name") or "unknown"
            records.append({"timestamp": dt, "technology": tech, "value": float(v.get("value") or 0.0)})
    if not records:
        logging.warning("REE response parsing produced no records; returning empty DF")
        return pd.DataFrame(index=_hour_index())
    df = pd.DataFrame(records)
    df = df.pivot_table(index="timestamp", columns="technology", values="value", aggfunc="sum")
    df.index = pd.DatetimeIndex(df.index).tz_convert("UTC")
    return df

def fetch_tso_rte():
    """RTE Eco2mix CSV direct download (France)."""
    logging.info("Fetching RTE (France) eco2mix")
    csv_url = "https://opendata.rte-france.com/explore/dataset/eco2mix-production-consommation/download?format=csv&timezone=UTC"
    r = requests.get(csv_url, timeout=60)
    r.raise_for_status()
    df = pd.read_csv(pd.compat.StringIO(r.text))
    # RTE columns usually include 'date' and many technology columns - basic parsing:
    if "heure" in df.columns:
        df.rename(columns={"heure":"timestamp"}, inplace=True)
    # try to guess time column
    time_col = [c for c in df.columns if "date" in c.lower() or "time" in c.lower() or "heure" in c.lower()]
    if not time_col:
        raise RuntimeError("RTE CSV time column not found")
    tcol = time_col[0]
    df[tcol] = pd.to_datetime(df[tcol], utc=True)
    df = df.set_index(tcol)
    # drop irrelevant columns, keep numeric
    df = df.select_dtypes(include="number")
    df.index = df.index.tz_convert("UTC")
    return df

def fetch_tso_smard():
    """SMARD Germany - example market id usage - returns JSON"""
    logging.info("Fetching SMARD (Germany)")
    url = "https://www.smard.de/app/chart_data/410/DE/index.json"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    js = r.json()
    # json structure includes 'data' lists per technology -> build DF
    rows = []
    for serie in js.get("series", []):
        tech = serie.get("label", "unknown")
        for point in serie.get("data", []):
            # data: [timestamp_ms, value]
            ts = pd.to_datetime(point[0], unit="ms", utc=True)
            val = point[1]
            rows.append({"timestamp": ts, "technology": tech, "value": val})
    if not rows:
        return pd.DataFrame(index=_hour_index())
    df = pd.DataFrame(rows)
    df = df.pivot_table(index="timestamp", columns="technology", values="value", aggfunc="sum")
    df.index = df.index.tz_convert("UTC")
    return df

def fetch_tso_fingrid():
    """Fingrid dataset example - uses API dataset 124 or 137 depending on data shape"""
    logging.info("Fetching Fingrid (Finland)")
    # dataset 124 provides production per type as zip csv; here we illustrate a JSON API if available
    url = f"https://data.fingrid.fi/api/3/action/datastore_search?resource_id=0573f2f6-9b1c-4a98-8e4f-0c4f3a6ba3b2&limit=500000"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    js = r.json()
    records = js.get("result", {}).get("records", [])
    if not records:
        return pd.DataFrame(index=_hour_index())
    df = pd.DataFrame(records)
    # assume 'Date' or 'Datetime' column exists
    time_col = [c for c in df.columns if "time" in c.lower() or "date" in c.lower()]
    if not time_col:
        return pd.DataFrame(index=_hour_index())
    df[time_col[0]] = pd.to_datetime(df[time_col[0]], utc=True)
    df = df.set_index(time_col[0])
    df = df.select_dtypes(include="number")
    df.index = df.index.tz_convert("UTC")
    return df

def fetch_tso_energinet():
    logging.info("Fetching Energinet (Denmark)")
    url = "https://api.energidataservice.dk/dataset/ProductionConsumptionSettlement?limit=500000"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    js = r.json()
    records = js.get("records", [])
    if not records:
        return pd.DataFrame(index=_hour_index())
    df = pd.DataFrame(records)
    # mapping fields and cleaning to get hourly generation per type
    # typical columns: 'HourUTC', 'Coal', 'Gas', 'WindOnshore' etc -- adapt to returned fields
    time_col = [c for c in df.columns if 'hourutc' in c.lower() or 'time' in c.lower()]
    if time_col:
        df[time_col[0]] = pd.to_datetime(df[time_col[0]], utc=True)
        df = df.set_index(time_col[0])
    df = df.select_dtypes(include="number")
    df.index = df.index.tz_convert("UTC")
    return df

def fetch_tso_ren():
    """Portugal REN example (unofficial endpoint used earlier)"""
    logging.info("Fetching REN (Portugal)")
    r = requests.get("https://app.ren.pt/data/production?type=hourly", timeout=30)
    r.raise_for_status()
    js = r.json()
    # expects keys per timestamp
    rows = []
    for rec in js.get("data", []):
        ts = pd.to_datetime(rec["ts"], utc=True)
        for k,v in rec.get("production", {}).items():
            rows.append({"timestamp": ts, "technology": k, "value": v})
    df = pd.DataFrame(rows).pivot_table(index="timestamp", columns="technology", values="value", aggfunc="sum")
    df.index = df.index.tz_convert("UTC")
    return df

def fetch_tso_elia():
    logging.info("Fetching Elia (Belgium)")
    # example API; actual endpoints can differ
    url = "https://griddata.elia.be/eliadata/api/production?type=raw"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    js = r.json()
    # parse as needed
    # fallback: return empty DF to avoid halting pipeline
    return pd.DataFrame(index=_hour_index())

def fetch_tso_statnett():
    logging.info("Fetching Statnett (Norway)")
    url = "https://driftsdata.statnett.no/rest/productionhourly"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    df = pd.read_json(r.text)
    # normalize
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], utc=True)
        df = df.set_index("time")
    df = df.select_dtypes(include="number")
    df.index = df.index.tz_convert("UTC")
    return df

def fetch_tso_svk():
    logging.info("Fetching Svenska Kraftnät (Sweden)")
    # example placeholder - real endpoint must be adapted:
    url = "https://api.svk.se/production/v1/hourly"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    df = pd.read_json(r.text)
    return df

# Add more fetchers as needed...

# Mapping fetcher names to functions
TSO_FETCHERS = {
    "ree": fetch_tso_ree,
    "rte": fetch_tso_rte,
    "smard": fetch_tso_smard,
    "fingrid": fetch_tso_fingrid,
    "energinet": fetch_tso_energinet,
    "ren": fetch_tso_ren,
    "elia": fetch_tso_elia,
    "statnett": fetch_tso_statnett,
    "svk": fetch_tso_svk,
    # "tennet": fetch_tso_tennet, "terna": fetch_tso_terna, etc.
}

# ---------- Merge logic ----------
def unify_column_names(df):
    # Map many vendor-specific names to standardized PSR names.
    # This is a minimal example; expand mapping as you encounter names.
    mapping = {
        "Wind Onshore": "Wind_Onshore",
        "Wind offshore": "Wind_Offshore",
        "Solar": "Solar",
        "Photovoltaic": "Solar",
        "Nuclear": "Nuclear",
        "Hydro": "Hydro_Total",
        "Hydro Pumped storage": "Hydro_Pumped_Storage",
        "Coal": "Fossil_Coal",
        "Gas": "Fossil_Gas",
        # add more...
    }
    df = df.rename(columns={c: mapping.get(c, c) for c in df.columns})
    return df

def merge_country_data(country, entsoe_total, entsoe_by_psr, tso_df=None):
    """
    Priority:
     1) Use TSO per-tech if available (tso_df)
     2) Use ENTSO-E per-tech (entsoe_by_psr)
     3) As fallback, use entsoe_total as Actual_Generation_Total and NaN per-type
    """
    # create base hourly index
    idx = _hour_index()
    base = pd.DataFrame(index=idx)
    # attach total from ENTSO-E
    if entsoe_total is not None:
        total = entsoe_total.reindex(idx)
        base["Actual_Generation_Total_entsoe"] = total
    # per-psr from ENTSO-E
    if entsoe_by_psr is not None and not entsoe_by_psr.empty:
        df_psr = entsoe_by_psr.reindex(idx).astype(float)
        df_psr = unify_column_names(df_psr)
        # rename columns with prefix
        df_psr.columns = [f"ENTSOE_{c}" for c in df_psr.columns]
        base = base.join(df_psr)
    # TSO
    if tso_df is not None and not tso_df.empty:
        df_tso = tso_df.reindex(idx).astype(float)
        df_tso = unify_column_names(df_tso)
        df_tso.columns = [f"TSO_{c}" for c in df_tso.columns]
        base = base.join(df_tso)
    # Compute final total: prefer TSO total if available else ENTSOE total
    if "TSO_Actual_Generation_Total" in base.columns:
        base["Actual_Generation_Total"] = base["TSO_Actual_Generation_Total"]
    elif "Actual_Generation_Total_entsoe" in base.columns:
        base["Actual_Generation_Total"] = base["Actual_Generation_Total_entsoe"]
    else:
        base["Actual_Generation_Total"] = None
    # Now assemble per-technology columns: prefer TSO_ then ENTSOE_
    tech_cols = set([c.split("_",1)[1] for c in base.columns if "_" in c and c.split("_",1)[1] != "Actual_Generation_Total"])
    for tech in tech_cols:
        tso_col = f"TSO_{tech}"
        ent_col = f"ENTSOE_{tech}"
        out_col = tech
        if tso_col in base.columns:
            base[out_col] = base[tso_col]
        elif ent_col in base.columns:
            base[out_col] = base[ent_col]
        # else NaN
    # Keep country column
    base["country"] = country
    # Keep timestamp as column too
    base = base.reset_index().rename(columns={"index":"timestamp"})
    return base

# ---------- Main pipeline ----------
def build_master():
    all_countries = []
    for c in COUNTRIES:
        try:
            logging.info(f"Processing country {c}")
            # fetch ENTSO-E total
            try:
                entsoe_total = fetch_entsoe_total(c)
            except Exception as e:
                logging.warning(f"Failed ENTSO-E total for {c}: {e}")
                entsoe_total = None
            # fetch ENTSOE breakdown (may be empty)
            try:
                entsoe_psr = fetch_entsoe_by_psr(c, psr_type=None)
            except Exception as e:
                logging.warning(f"Failed ENTSO-E psr for {c}: {e}")
                entsoe_psr = None
            # fetch TSO if available
            tso_df = None
            if c in TSO_PRIORITY:
                fetcher_name = TSO_PRIORITY[c]
                fetcher = TSO_FETCHERS.get(fetcher_name)
                if fetcher:
                    try:
                        tso_df = fetcher()
                    except Exception as e:
                        logging.warning(f"TSO fetcher {fetcher_name} failed for {c}: {e}")
                        tso_df = None
                else:
                    logging.info(f"No TSO fetcher implemented for {fetcher_name} ({c})")
            # merge and produce country DF
            country_df = merge_country_data(c, entsoe_total, entsoe_psr, tso_df)
            all_countries.append(country_df)
            # save intermediate per-country file for caching
            country_df.to_parquet(os.path.join(OUTPUT_DIR, f"country_{c}_gen.parquet"), index=False)
            logging.info(f"Saved country {c} parquet")
        except Exception as e:
            logging.exception(f"Unhandled error for country {c}: {e}")
    # concat all
    if all_countries:
        master = pd.concat(all_countries, ignore_index=True)
        # unify columns / cast types
        # ensure timestamp is datetime UTC
        master["timestamp"] = pd.to_datetime(master["timestamp"], utc=True)
        # filter to requested period (safety)
        master = master[(master["timestamp"] >= START) & (master["timestamp"] <= END)]
        # write master
        master.to_parquet(MASTER_PARQUET, index=False)
        csv_out = MASTER_PARQUET.replace(".parquet", ".csv")
        master.to_csv(csv_out, index=False)
        logging.info(f"Wrote master dataset to {MASTER_PARQUET} and {csv_out}")
        return master
    else:
        logging.error("No country data fetched; master dataset empty")
        return None

if __name__ == "__main__":
    logging.info("START building master generation dataset")
    master = build_master()
    if master is None:
        logging.error("Pipeline failed")
    else:
        logging.info("Pipeline finished successfully")
