"""
================================================================================
EUROPEAN POWER GRID STRESS PREDICTOR - STREAMLIT DASHBOARD
================================================================================
Author: Team 6 - GridWatch
Chavely Albert Fernandez, Pedro Miguel da Câmara Leme, Ya-Chi Hsiao and Maria Sokotushchenko
Project: Capstone - European Power Grid Stress Prediction
Date: December 2025

Description:
Real-time stress prediction dashboard for European power grids using
XGBoost classifier model with class imbalance handling trained on ENTSOE transparency platform data and Copernicus
Reanalysis weather data.

Model Performance:
- Recall=0.807
- FI=0.765
- Countries: 13 European nations
================================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import math
from databricks import sql
import os
from dotenv import load_dotenv
import pandas as pd
import pytz
load_dotenv()

# Additional runtime configuration for local vs remote asset loading
import requests
import io
import logging
from pathlib import Path

# Page configuration

# Determine LOCAL_MODE: prefer explicit `LOCAL_MODE` env var; otherwise fall back
# to the older `LOCAL_DEV_MODE` env var if present.
_local_env_val = os.getenv("LOCAL_MODE")
if _local_env_val is None:
    LOCAL_MODE = os.getenv("LOCAL_DEV_MODE", "true").lower() in ("1", "true", "yes")
else:
    LOCAL_MODE = str(_local_env_val).lower() in ("1", "true", "yes")

# Public GCP base URL used when LOCAL_MODE is False. Example:
# https://storage.googleapis.com/my-public-bucket/path/to/streamlit_dashboard
GCP_PUBLIC_BASE_URL = os.getenv("GCP_PUBLIC_BASE_URL", "").rstrip('/')

logger = logging.getLogger(__name__)


def _fetch_bytes_from_gcp(path: str) -> bytes:
    """Fetch raw bytes from a public GCP URL constructed from base + path.

    `path` is the path inside the bucket, e.g. "xgboost_model.pkl" or
    "arima_models/ARIMA_DE.pkl". Raises if `GCP_PUBLIC_BASE_URL` is not set.
    """
    if not GCP_PUBLIC_BASE_URL:
        raise ValueError("GCP_PUBLIC_BASE_URL is not set. Set it to the public bucket base URL.")

    url = f"{GCP_PUBLIC_BASE_URL}/{path.lstrip('/')}"
    logger.info("Fetching remote asset from %s", url)

    # Use a session with simple retry behavior to handle transient failures
    session = requests.Session()
    try:
        # Stream the response to avoid loading the whole body at once implicitly
        # Set a generous read timeout for large objects
        resp = session.get(url, timeout=(5, 300), stream=True)
        try:
            resp.raise_for_status()
        except Exception:
            logger.error("Failed fetching %s: status=%s; content=%s", url, getattr(resp, 'status_code', None), resp.text[:500] if hasattr(resp, 'text') else None)
            raise

        total = resp.headers.get('Content-Length')
        try:
            total = int(total) if total is not None else None
        except Exception:
            total = None

        chunk_size = 8192
        data_buf = bytearray()
        downloaded = 0
        for chunk in resp.iter_content(chunk_size=chunk_size):
            if not chunk:
                continue
            data_buf.extend(chunk)
            downloaded += len(chunk)
            # Log progress every ~8MB
            if downloaded % (8 * 1024 * 1024) < chunk_size:
                logger.info("Downloading %s: %d bytes%s", path, downloaded, f" / {total}" if total else "")

        logger.info("Finished downloading %s: %d bytes%s", path, downloaded, f" / {total}" if total else "")
        return bytes(data_buf)
    finally:
        try:
            session.close()
        except Exception:
            pass


def _check_remote_asset(path: str) -> dict:
    """Check remote asset with HEAD and return diagnostics.

    Returns a dict with keys: url, ok (bool), status (int), content_length (int|None)
    """
    if not GCP_PUBLIC_BASE_URL:
        return {"url": None, "ok": False, "status": None, "content_length": None}

    url = f"{GCP_PUBLIC_BASE_URL}/{path.lstrip('/')}"
    try:
        resp = requests.head(url, timeout=15)
        content_length = resp.headers.get('Content-Length')
        try:
            content_length = int(content_length) if content_length is not None else None
        except Exception:
            content_length = None
        return {"url": url, "ok": resp.status_code == 200, "status": resp.status_code, "content_length": content_length}
    except Exception as e:
        logger.warning("HEAD check failed for %s: %s", url, e)
        return {"url": url, "ok": False, "status": None, "content_length": None}

st.set_page_config(
    page_title="EU Grid Stress Predictor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Save the time when the user enters the page
if "entry_time" not in st.session_state:
    st.session_state.entry_time = datetime.utcnow().replace(tzinfo=pytz.UTC)

entry_time = st.session_state.entry_time

# Custom CSS for dark industrial theme
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #0a1628;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #0d1f3c;
        border-right: 1px solid #1e3a5f;
    }
    
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] h3 {
        color: #8ba3c7 !important;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #4fc3f7 !important;
    }
    
    /* Metric cards */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: bold;
    }
    
    /* Custom metric styling */
    .metric-container {
        background: linear-gradient(135deg, #1a2942 0%, #0d1f3c 100%);
        border: 1px solid #2a4a6e;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    
    .metric-label {
        color: #8ba3c7;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 5px;
    }
    
    .metric-value {
        color: #ffffff;
        font-size: 2.2rem;
        font-weight: bold;
    }
    
    .metric-delta {
        font-size: 0.85rem;
        margin-top: 5px;
    }
    
    .delta-positive { color: #4caf50; }
    .delta-negative { color: #f44336; }
    .delta-warning { color: #ff9800; }
    
    /* Stress gauge colors */
    .stress-normal { color: #4caf50; }
    .stress-moderate { color: #ff9800; }
    .stress-high { color: #f44336; }
    
    /* Card styling */
    .card {
        background: linear-gradient(135deg, #1a2942 0%, #0d1f3c 100%);
        border: 1px solid #2a4a6e;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    
    .card-title {
        color: #4fc3f7;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 15px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    /* Scenario preset buttons */
    .stButton > button {
        background-color: transparent;
        border: 1px solid #2a4a6e;
        color: #4fc3f7;
        border-radius: 5px;
        padding: 8px 16px;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background-color: #2a4a6e;
        border-color: #4fc3f7;
    }
    
    /* Warning banner */
    .warning-banner {
        background: linear-gradient(90deg, #f44336 0%, #ff5722 100%);
        color: white;
        padding: 10px 20px;
        border-radius: 5px;
        font-weight: bold;
        text-align: center;
        animation: pulse 2s infinite;
    }
            
            /* Custom banner for Live Data timestamp */
    .live-data-banner {
        background: linear-gradient(135deg, #1a2942 0%, #0d1f3c 100%);
        border: 1px solid #4fc3f7;
        color: #4fc3f7;
        padding: 10px 16px;
        border-radius: 6px;
        font-size: 0.95rem;
        margin: 10px 0 18px 0;
    }

    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    
    /* Target breakdown bars */
    .target-bar {
        height: 25px;
        border-radius: 4px;
        margin: 5px 0;
    }
    
    /* Footer */
    .footer {
        color: #5a7a9a;
        font-size: 0.8rem;
        text-align: center;
        margin-top: 30px;
        padding-top: 20px;
        border-top: 1px solid #2a4a6e;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Keep Streamlit menu visible */
    #footer {visibility: hidden;}

    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #1a2942;
        border-radius: 5px;
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div {
        background-color: #1a2942;
        border-color: #2a4a6e;
    }
                       
    /* SIDEBAR STYLES */
    [data-testid="stSidebar"] {
        color: white !important;
    }
    
    [data-testid="stSidebar"] .element-container {
        color: white !important;
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: white !important;
    }
    
    /* "Select Mode" text - match "Net Imports" color */
    [data-testid="stSidebar"] label {
        color: #8ba3c7 !important;
    }
    
    /* Sidebar section titles (Mode, Country, Scenario Presets) - smaller size */
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: white !important;
        font-size: 1rem !important;
    }
    
    /* "Scenarios Note:" title - smaller size */
    [data-testid="stSidebar"] h4,
    [data-testid="stSidebar"] .stMarkdown h4 {
        font-size: 0.9rem !important;
    }
    
    /* Text under "Scenarios Note:" - smaller size */
    [data-testid="stSidebar"] p {
        font-size: 0.85rem !important;
    }

    /* Radio button labels */
    [data-testid="stSidebar"] [data-baseweb="radio"] label span {
        color: white !important;
    }
    
    [data-testid="stSidebar"] label[data-baseweb="radio"] p {
        color: white !important;
    }
    
    /* Selectbox options */
    [data-testid="stSidebar"] [data-baseweb="select"] span,
    [data-testid="stSidebar"] [data-baseweb="select"] > div > div {
        color: white !important;
    }
    
    /* Expander content in sidebar */
    [data-testid="stSidebar"] [data-testid="stExpander"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] .streamlit-expanderContent p,
    [data-testid="stSidebar"] details > div p {
        color: white !important;
    }
    
    /* MAIN CONTENT AREA STYLES */
    /* Expander titles */
    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] .streamlit-expanderHeader,
    details summary {
        color: white !important;
    }
    
    /* All text inside expanders (but exclude code blocks) */
    [data-testid="stExpander"] p:not(:has(code)),
    [data-testid="stExpander"] span:not(code):not(code *),
    [data-testid="stExpander"] div:not(:has(code)):not(pre) {
        color: #a8b9cc !important;
        font-weight: normal !important;
    }
    
    /* Regular paragraphs inside expanders - match "TOP 10 FEATURE IMPORTANCE" color */
    [data-testid="stExpander"] p {
        font-size: 1rem !important;
        font-weight: normal !important;
        color: #a8b9cc !important;
    }
    
    /* Code blocks should have dark text on light background */
    [data-testid="stExpander"] code,
    [data-testid="stExpander"] pre,
    [data-testid="stExpander"] pre code,
    [data-testid="stExpander"] pre span {
        color: #1a1a1a !important;
        background-color: #f5f5f5 !important;
    }
    
    /* Bullet point text inside expanders - match "TOP 10 FEATURE IMPORTANCE" color */
    [data-testid="stExpander"] ul li,
    [data-testid="stExpander"] ol li,
    [data-testid="stExpander"] li {
        color: #a8b9cc !important;
        font-size: 1rem !important;
        font-weight: normal !important;
    }
    
    /* Tables inside expanders */
    [data-testid="stExpander"] table,
    [data-testid="stExpander"] thead,
    [data-testid="stExpander"] tbody,
    [data-testid="stExpander"] tr,
    [data-testid="stExpander"] th,
    [data-testid="stExpander"] td {
        color: #a8b9cc !important;
        font-size: 1rem !important;
    }
    
    /* Dataframe styling */
    [data-testid="stExpander"] [data-testid="stDataFrame"],
    [data-testid="stExpander"] .dataframe {
        color: #a8b9cc !important;
    }
    
    [data-testid="stExpander"] .dataframe td,
    [data-testid="stExpander"] .dataframe th {
        color: #a8b9cc !important;
    }
    
    /* Section titles in main area headers */
    .main h1,
    .main h2,
    .main h3,
    .main h4,
    .main h5,
    .main h6 {
        color: #5DADE2 !important;
    }
    
    /* Target all paragraph text in main area */
    .main p {
        color: #5DADE2 !important;
    }
    
    /* More specific targeting for markdown containers */
    .main [data-testid="stMarkdownContainer"] {
        color: #5DADE2 !important;
    }
    
    .main [data-testid="stMarkdownContainer"] * {
        color: #5DADE2 !important;
    }
    
    /* All list items in main content */
    .main ul li,
    .main ol li {
        color: white !important;
    }
    
    /* H1 Headers inside expanders - Match "Documentation & Methodology" size */
    [data-testid="stExpander"] h1 {
        color: #5DADE2 !important;
        font-size: 1.5rem !important;
        font-weight: 600 !important;
    }
    
    /* H2 Headers inside expanders - Slightly smaller than H1 */
    [data-testid="stExpander"] h2 {
        color: #5DADE2 !important;
        font-size: 1.3rem !important;
        font-weight: 600 !important;
    }
    
    /* H3 Headers inside expanders */
    [data-testid="stExpander"] h3 {
        color: #5DADE2 !important;
        font-size: 1.2rem !important;
        font-weight: 600 !important;
    }
    
    /* H4-H6 Headers inside expanders */
    [data-testid="stExpander"] h4,
    [data-testid="stExpander"] h5,
    [data-testid="stExpander"] h6 {
        color: #5DADE2 !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
    }
    
    /* Strong/Bold emphasis text - only bold when explicitly marked */
    [data-testid="stExpander"] strong:not(code *),
    [data-testid="stExpander"] b:not(code *) {
        color: #a8b9cc !important;
        font-weight: 700 !important;
    }
    
    /* Category headers like "Load Features (35 features)" */
    [data-testid="stExpander"] p strong:first-child,
    [data-testid="stExpander"] p b:first-child {
        font-weight: 700 !important;
        color: #a8b9cc !important;
    }
            
    /* Selectbox, multiselect, date input arrows */
[data-baseweb="select"] svg,
[data-baseweb="date-picker"] svg,
.stDateInput svg {
    fill: #ffffff !important;   /* ← your color */
}

/* Expander arrow (the “>” icon) */
details summary svg,
.streamlit-expanderHeader svg {
    stroke: #ffffff !important; 
    fill: #ff9800 !important;
}

/* Fix for some Streamlit internal nested containers */
.css-1j77add svg,
.css-1v0mbdj svg {
    fill: #ffffff !important;
}
            
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* Change color of the help "?" icon */
.stTooltipIcon svg {
    fill: #ff9800 !important;
}
</style>
""", unsafe_allow_html=True)


# ============================================================================
# DATA LOADING
# ============================================================================
@st.cache_resource
def load_model():
    """Load the trained XGBoost model (local file only)."""
    try:
        model_path = Path(__file__).parent / 'xgboost_model.pkl'
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None
    


def load_arima_model_for_country(country_code):
    """Load the ARIMA model for a specific country code only."""
    try:
        # Local path: arima_models/ARIMA_{CODE}.pkl
        local_rel_path = f'arima_models/ARIMA_{country_code}.pkl'
        if LOCAL_MODE:
            local_path = Path(__file__).parent / local_rel_path
            if local_path.exists():
                with open(local_path, 'rb') as f:
                    return pickle.load(f)
            else:
                logger.warning("Local ARIMA model not found for %s: %s", country_code, local_path)
                return None
        else:
            # Remote path: arima_{CODE}.pkl (lowercase)
            remote_name = f'arima_{country_code}.pkl'
            diag = _check_remote_asset(remote_name)
            logger.info("Checking remote ARIMA asset for %s: %s", country_code, diag)
            if diag.get('ok'):
                try:
                    data = _fetch_bytes_from_gcp(remote_name)
                    logger.info("Loaded remote ARIMA model for %s from %s (bytes=%d)", country_code, diag.get('url'), len(data))
                    return pickle.loads(data)
                except requests.HTTPError as he:
                    logger.error("HTTP error loading ARIMA model %s from %s: %s", country_code, diag.get('url'), he)
                    st.error(f"HTTP error loading ARIMA model for {country_code}. URL: {diag.get('url')} (status={diag.get('status')}).")
                    return None
                except Exception as e:
                    logger.exception(e)
                    st.error(f"Error loading ARIMA model for {country_code} from {diag.get('url')}: {e}")
                    return None
            else:
                logger.error("Remote ARIMA model not found for %s at expected path: %s (diag=%s)", country_code, remote_name, diag)
                return None
    except Exception as e:
        st.error(f"Error loading ARIMA model for {country_code}: {e}")
        logger.exception(e)
        return None

# For example: Get the ARIMA model for Germany
#model_de = all_models_arima['DE']

@st.cache_resource
def load_feature_names():
    """Load feature names used by the model"""
    try:
        features_path = Path(__file__).parent / 'feature_names.pkl'
        with open(features_path, 'rb') as f:
            features = pickle.load(f)
        return features
    except Exception as e:
        st.error(f"Error loading feature names: {e}")
        return None

@st.cache_data(ttl=3600)
def fetch_data_from_databricks():
    """Load data from Databricks"""
    try:
        ## import env vars
        DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN")
        DATABRICKS_HOSTNAME = os.getenv("DATABRICKS_HOSTNAME")
        DATABRICKS_HTTP_PATH = os.getenv("DATABRICKS_HTTP_PATH")

        connection = sql.connect(
                                server_hostname = DATABRICKS_HOSTNAME,
                                http_path = DATABRICKS_HTTP_PATH,
                                access_token = DATABRICKS_TOKEN)

        cursor = connection.cursor()

        cursor.execute("SELECT * from workspace.default.x_test_imputed_with_features_countries")
        column_names = [desc[0] for desc in cursor.description]
        data = cursor.fetchall()
        df = pd.DataFrame(data, columns=column_names)

        # Load table for live prediction
        cursor.execute("SELECT * FROM workspace.live_data.electricity_and_weather_europe_imputed_with_features")
        column_names_live = [desc[0] for desc in cursor.description]
        data_live = cursor.fetchall()
        df_live = pd.DataFrame(data_live, columns=column_names_live)

        # Load stress table for ARIMA
        cursor.execute("SELECT * FROM workspace.live_data.grid_stress_scores_real")
        column_names_live = [desc[0] for desc in cursor.description]
        data_scores_real = cursor.fetchall()
        data_scores_real = pd.DataFrame(data_scores_real, columns=column_names_live)

        cursor.close()
        connection.close()

        return df, df_live, data_scores_real
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

@st.cache_data
def load_country_stats():
    """Load country statistics"""
    try:
        stats_path = Path(__file__).parent / 'country_stats.csv'
        df = pd.read_csv(stats_path)
        return df
    except Exception as e:
        st.error(f"Error loading country stats: {e}")
        return None


# ============================================================================
# COUNTRY CONFIGURATION
# ============================================================================
COUNTRY_INFO = {
    'AT': {'name': 'Austria', 'flag': '🇦🇹', 'avg_load': 6669, 'avg_stress': 29.5, 'iso_3': 'AUT'},
    'BE': {'name': 'Belgium', 'flag': '🇧🇪', 'avg_load': 9113, 'avg_stress': 28.6, 'iso_3': 'BEL'},
    'DE': {'name': 'Germany', 'flag': '🇩🇪', 'avg_load': 52664, 'avg_stress': 29.2, 'iso_3': 'DEU'},
    'ES': {'name': 'Spain', 'flag': '🇪🇸', 'avg_load': 26307, 'avg_stress': 25.2, 'iso_3': 'ESP'},
    'FR': {'name': 'France', 'flag': '🇫🇷', 'avg_load': 48737, 'avg_stress': 29.4, 'iso_3': 'FRA'},
    'HR': {'name': 'Croatia', 'flag': '🇭🇷', 'avg_load': 2059, 'avg_stress': 26.8, 'iso_3': 'HRV'},
    'HU': {'name': 'Hungary', 'flag': '🇭🇺', 'avg_load': 4871, 'avg_stress': 31.8, 'iso_3': 'HUN'},
    'IT': {'name': 'Italy', 'flag': '🇮🇹', 'avg_load': 31638, 'avg_stress': 19.9, 'iso_3': 'ITA'},
    'LT': {'name': 'Lithuania', 'flag': '🇱🇹', 'avg_load': 1360, 'avg_stress': 28.9, 'iso_3': 'LTU'},
    'NL': {'name': 'Netherlands', 'flag': '🇳🇱', 'avg_load': 12789, 'avg_stress': 41.2, 'iso_3': 'NLD'},
    'PL': {'name': 'Poland', 'flag': '🇵🇱', 'avg_load': 18794, 'avg_stress': 26.0, 'iso_3': 'POL'},
    'PT': {'name': 'Portugal', 'flag': '🇵🇹', 'avg_load': 5819, 'avg_stress': 24.2, 'iso_3': 'PRT'},
    'SK': {'name': 'Slovakia', 'flag': '🇸🇰', 'avg_load': 2913, 'avg_stress': 28.5, 'iso_3': 'SVK'},
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def get_stress_category(score):
    """Categorize stress score (0-100 scale)"""
    if score < 33:
        return "NORMAL", "#4caf50"
    elif score < 66:
        return "MODERATE", "#ff9800"
    else:
        return "HIGH RISK", "#f44336"


def create_stress_gauge(score, title="Stress Score"):
    """Create a half-circle stress gauge (0-100 scale)"""
    category, color = get_stress_category(score)
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        number={
            'font': {'size': 60, 'color': color},
            'suffix': ' pts'
        },
        gauge={
            'axis': {
                'range': [0, 100],
                'tickwidth': 2,
                'tickcolor': "#4fc3f7",
                'tickfont': {'color': '#8ba3c7', 'size': 12},
                'tickvals': [0, 25, 50, 75, 100],
            },
            'bar': {'color': color, 'thickness': 0.3},
            'bgcolor': "#1a2942",
            'borderwidth': 2,
            'bordercolor': "#2a4a6e",
            'steps': [
                {'range': [0, 33], 'color': 'rgba(76, 175, 80, 0.2)'},
                {'range': [33, 66], 'color': 'rgba(255, 152, 0, 0.2)'},
                {'range': [66, 100], 'color': 'rgba(244, 67, 54, 0.2)'}
            ],
            'threshold': {
                'line': {'color': "#f44336", 'width': 3},
                'thickness': 0.8,
                'value': 66
            }
        }
    ))
    
    fig.add_annotation(
        x=0.5, y=-0.15,
        text=f"HIGH RISK THRESHOLD: 66 pts",
        font={'size': 11, 'color': '#8ba3c7'},
        showarrow=False
    )
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#8ba3c7'},
        height=280,
        margin=dict(l=30, r=30, t=30, b=50)
    )
    
    return fig


def create_target_breakdown(components):
    """Create horizontal bar chart for target breakdown"""
    fig = go.Figure()
    
    labels = list(components.keys())
    values = list(components.values())
    colors = ['#4caf50' if (v >= 0 and v<12.5) else '#ff9800' if (v >= 12.5 and v<25) else '#f44336' for v in values]
    
    fig.add_trace(go.Bar(
        y=labels,
        x=values,
        orientation='h',
        marker_color=colors,
        text=[f"{v:.0f} pts" if v > 0 else "—" for v in values],
        textposition='outside',
        textfont={'color': '#ffffff', 'size': 11}
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#8ba3c7'},
        height=250,
        margin=dict(l=10, r=60, t=10, b=10),
        xaxis={
            'range': [0, 30],
            'gridcolor': '#2a4a6e',
            'zerolinecolor': '#2a4a6e',
            'title': 'Points',
            'titlefont': {'size': 11}
        },
        yaxis={
            'tickfont': {'size': 10}
        },
        showlegend=False
    )
    
    return fig


def plot_24h_with_6h_forecast(last_24h_data, forecast_6h, current_hour):
    """
    last_24h_data: list/array of 24 historical stress values
    forecast_6h: list/array of 6 forecast values
    current_hour: int, 0-23 (the last point of historical data)
    """
    import plotly.graph_objects as go

    # Create a continuous timeline for ordering
    hist_hours = list(range(current_hour - 23, current_hour + 1))   # 24h
    forecast_hours = list(range(current_hour + 1, current_hour + 7)) # 6h

    # Clock labels (00–23) but DO NOT affect ordering
    hist_labels = [f"{h % 24:02d}:00" for h in hist_hours]
    forecast_labels = [f"{h % 24:02d}:00" for h in forecast_hours]

    # Build figure
    fig = go.Figure()

    # Historical data
    fig.add_trace(go.Scatter(
        x=hist_hours,                     # chronological ordering
        y=last_24h_data,
        mode='lines+markers',
        line=dict(color='#4fc3f7', width=2),
        marker=dict(size=6, color='#4fc3f7'),
        name='Last 24h',
        text=hist_labels,
        hovertemplate="%{text}: %{y}<extra></extra>"
    ))

    # Forecast
    fig.add_trace(go.Scatter(
        x=forecast_hours,                 # chronological ordering
        y=forecast_6h,
        mode='lines+markers',
        line=dict(color='#ff9800', width=2, dash='dash'),
        marker=dict(size=6, color='#ff9800'),
        name='Next 6h Forecast',
        text=forecast_labels,
        hovertemplate="%{text}: %{y}<extra></extra>"
    ))

    # Mark "Now"
    fig.add_trace(go.Scatter(
        x=[current_hour],
        y=[last_24h_data[-1]],
        mode='markers+text',
        marker=dict(size=12, color='#ff5722', symbol='diamond'),
        text=['Now'],
        textposition='top center',
        name='Current'
    ))

    # Threshold line
    fig.add_hline(
        y=66, line_dash="dash", line_color="#f44336",
        annotation_text="High Risk"
    )

    all_hours = hist_hours + forecast_hours
    all_labels = hist_labels + forecast_labels

    tickvals = all_hours[::3]
    ticktext = all_labels[::3]

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#8ba3c7'),
        height=300,
        margin=dict(l=40, r=20, t=20, b=40),

        # X-axis is continuous (hist + forecast)
        xaxis=dict(
        title='Hour of Day',
        tickmode='array',
        tickvals=tickvals,
        ticktext=ticktext,
        gridcolor='#2a4a6e',
        tickangle=-45
        ),
        yaxis=dict(
            title='Stress Score',
            range=[0, 105],
            gridcolor='#2a4a6e'
        ),

        showlegend=True
    )

    return fig



def generate_6h_forecast(model):
    """
    Use the given ARIMA model to forecast the next 6 hours.
    
    model: a trained pmdarima ARIMA model
    current_stress_value: the current stress score from live data
    returns: numpy array of 6 forecasted values
    """
    if model is None:
        logger.warning("generate_6h_forecast called with None model")
        # Return NaNs so downstream plotting can handle missing forecast
        return np.full(6, np.nan)

    # Some models may not have `predict` method; guard against that.
    if not hasattr(model, 'predict'):
        logger.warning("ARIMA model does not have predict(): %s", type(model))
        return np.full(6, np.nan)

    try:
        forecast = model.predict(n_periods=6)  # predict next 6 steps
        forecast = np.clip(forecast, 0, 100)  # clip to 0-100 scale if needed
        return forecast
    except Exception as e:
        logger.exception(e)
        st.warning(f"ARIMA forecast failed: {e}")
        return np.full(6, np.nan)


def get_last_24h_stress_data(data_scores_real, country_code):
    """
    Extract the last 24 consecutive hourly stress scores from live data.
    
    data_scores_real: DataFrame with grid stress computed with real data. Updated every 1 hour.
    country_code: Country to filter
    
    Returns: numpy array of last 24 hourly stress scores
    """
    try:
        # Filter for the country
        country_data = data_scores_real[data_scores_real["country"] == country_code].copy()
        
        # Sort by timestamp (assuming it exists)
        if "index" in country_data.columns:
            country_data = country_data.sort_values("index")
        
        # Get last 24 rows
        last_24 = country_data.tail(24)
        
        # Extract grid_stress_score or return placeholder
        if "grid_stress_score" in last_24.columns:
            return last_24["grid_stress_score"].values
        else:
            # Fallback: generate synthetic data if column doesn't exist
            return np.linspace(30, 45, 24)
    
    except Exception as e:
        st.warning(f"Could not load 24h data: {e}")
        return np.linspace(30, 45, 24)  # placeholder


def create_feature_importance():
    """Create feature importance chart with real model values"""
    # Real feature importance from LightGBM (boosted) model
    features = {
    'Imports (1-hour lag)': 3828,
    '24-hour load change': 1364,
    '24-hour rolling mean of load': 1277,
    '1-hour load change': 1205,
    '24-hour load variability (std)': 1065,
    'Load deviation from daily average': 1025,
    'Time-of-day (cyclic feature 1)': 998,
    'Actual load': 807,
    '24-hour rolling mean of imports': 727,
    'Time-of-day (cyclic feature 2)': 681
    }
    
    # Convert to percentage of total
    total = sum(features.values())
    features_pct = {k: (v / total * 100) for k, v in features.items()}
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=list(features_pct.keys())[::-1],
        x=list(features_pct.values())[::-1],
        orientation='h',
        marker_color='#4fc3f7',
        text=[f"{v:.1f}%" for v in list(features_pct.values())[::-1]],
        textposition='outside',
        textfont={'color': '#ffffff', 'size': 10}
    ))
    
    fig.update_layout(
        title={'text': 'TOP 10 FEATURE IMPORTANCE', 'font': {'size': 12, 'color': '#8ba3c7'}},
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#8ba3c7'},
        height=300,
        margin=dict(l=10, r=80, t=40, b=10),
        xaxis={
            'title': 'Importance (%)',
            'gridcolor': '#2a4a6e',
            'range': [0, 35]
        },
        yaxis={
            'tickfont': {'size': 9}
        },
        showlegend=False
    )
    
    return fig

def create_eu_map(stress_scores):
    """Create choropleth map of EU grid stress"""
    locations = []
    stress_values = []
    hover_texts = []
    
    for country_code, info in COUNTRY_INFO.items():
        score = stress_scores.get(country_code, info['avg_stress'])
        locations.append(info['iso_3'])
        stress_values.append(score)
        hover_texts.append(f"{info['flag']} {info['name']}<br>{score:.1f} pts")
    
    fig = go.Figure(data=go.Choropleth(
        locations=locations,
        z=stress_values,
        text=hover_texts,
        hovertemplate='<b>%{text}</b><extra></extra>',
        colorscale=[
            [0.0, '#4caf50'],
            [0.25, '#8bc34a'],
            [0.50, "#ddeb18"],
            [0.75, '#ff9800'],
            [1.0, '#f44336']
        ],
        zmin=0,
        zmax=100,
        marker_line_width=2,
        marker_line_color='#0a1628',
        colorbar=dict(
            title=dict(text='Stress Score', font=dict(color='#8ba3c7')),
            thickness=20,
            len=0.7,
            tickfont={'size': 10, 'color': '#8ba3c7'},
            tickcolor='#8ba3c7'
        )
    ))
    
    fig.update_geos(
        scope='europe',
        fitbounds="locations",
        projection_type='natural earth',
        showland=True,
        landcolor='#0d1f3c',
        coastlinecolor='#1e3a5f',
        showocean=False,
        oceancolor='#0a0e1a',
        showcountries=True,
        countrycolor='#1e3a5f',
        countrywidth=1
    )
    
    fig.update_layout(
        title='🗺️ EU Grid Stress Heatmap',
        paper_bgcolor='#0a1628',
        font={'color': '#5DADE2', 'size': 12},
        title_font={'color': '#5DADE2', 'size': 16},
        height=550,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    return fig


import pandas as pd

def prepare_features(country, params, feature_names, country_stats, df):
    """
    Build the feature vector for the given country using:
    - Databricks base row
    - Defaults from country_stats
    - Slider overrides mapping to model feature names
    """

    # -------------------------------
    # 1. Start with an empty feature dict
    # -------------------------------
    features = {}

    # -------------------------------
    # 2. Get Databricks row for that country
    # -------------------------------
    base_row = df[df["country"] == country]

    if base_row.empty:
        # fallback: fill with NaN
        row_data = {col: np.nan for col in df.columns}
    else:
        row_data = base_row.iloc[0].to_dict()

    # Add all Databricks values
    for col in feature_names:
        features[col] = row_data.get(col, np.nan)

    # -------------------------------
    # 3. Get country defaults
    # -------------------------------
    if country in country_stats:
        defaults = country_stats[country]
    else:
        defaults = {}

    for col in feature_names:
        if pd.isna(features[col]):
            if col in defaults:
                features[col] = defaults[col]

    # -------------------------------
    # 4. Override using slider parameters
    # -------------------------------
    param_to_feature = {
        "actual_load": "Actual_Load",
        "forecasted_load": "Forecasted_Load",
        "net_imports": ["imports_lag_1h", "imports_rolling_mean_24h"],
        "temperature": "mean_temperature_c"
    }

    for param_key, model_cols in param_to_feature.items():
        if param_key in params:
            value = params[param_key]

            if isinstance(model_cols, list):
                for mc in model_cols:
                    if mc in features:
                        features[mc] = value
            else:
                if model_cols in features:
                    features[model_cols] = value

    # -------------------------------
    # 5. Set one-hot country flags
    # -------------------------------
    for col in feature_names:
        if col.startswith("country_"):
            features[col] = 1 if col == f"country_{country}" else 0

    # -------------------------------
    # 6. Return as one-row DataFrame
    # -------------------------------
    return pd.DataFrame([features], columns=feature_names), row_data


def compute_net_imports(row_data, params):
    """
    Compute a single net_imports value for a row.
    - Uses user-provided value if available.
    - Otherwise combines 'imports_lag_1h' and 'imports_rolling_mean_24h'.
    """
    user_value = params.get("net_imports", None)
    if user_value is not None:
        return user_value
    
    imports_1h = row_data.get("imports_lag_1h", 0)
    imports_24h = row_data.get("imports_rolling_mean_24h", 0)
    
    # Example: simple average
    return (imports_1h + imports_24h) / 2


def calculate_percentiles(row_data, user_params):
    """
    Compute P10/P90 thresholds.
    - Uses user-provided thresholds if available.
    - Otherwise computes percentiles from series.
    """
    P10 = user_params.get("P10_net", None)
    P90 = user_params.get("P90_net", None)

    net_imports_series = (row_data.get("imports_lag_1h", 0) + row_data.get("imports_rolling_mean_24h", 0))/2
    
    if P10 is None:
        P10 = np.percentile(net_imports_series, 10)
    if P90 is None:
        P90 = np.percentile(net_imports_series, 90)
    
    return P10, P90



def calculate_target_components(row_data, params):
    """
    Compute target components using:
    - Databricks base row (row_data)
    - Slider overrides (params)
    """

    # ----------------------------------
    # 1. Resolve inputs from row + sliders
    # ----------------------------------
    actual_load = params.get("actual_load", row_data.get("Actual_Load", np.nan))
    forecasted_load = params.get("forecasted_load", row_data.get("Forecasted_Load", actual_load))

    # Net imports
    net_imports = compute_net_imports(row_data, params)
    
    # P10/P90 thresholds
    P10_threshold, P90_threshold = calculate_percentiles(row_data, params)
    

    # ----------------------------------
    # 2. Compute score metrics
    # ----------------------------------
    reserve_margin = (forecasted_load - actual_load) / max(actual_load, 1)
    load_error_pct = abs(forecasted_load - actual_load) / max(forecasted_load, 1) * 100

    components = {}

    # ----------------------------------
    # 3. Apply original scoring rules
    # ----------------------------------

    # Reserve Margin
    if reserve_margin < -0.15:
        components["Reserve Margin"] = 25
    elif reserve_margin < -0.05:
        components["Reserve Margin"] = 12.5
    else:
        components["Reserve Margin"] = 0

    # Load Error
    if load_error_pct > 10:
        components["Load Forecast Error"] = 25
    elif load_error_pct > 5:
        components["Load Forecast Error"] = 12.5
    else:
        components["Load Forecast Error"] = 0

    # T7: High Exports
    components["T7: High Exports"] = 25 if net_imports < P10_threshold else 0

    # T8: High Imports
    components["T8: High Imports"] = 25 if net_imports > P90_threshold else 0

    return components


def predict_stress(country, params, model, feature_names, country_stats, df):
    """
    Predict stress score AND return row_data for target breakdown.
    """
    # prepare_features must return (features_df, row_data)
    features_df, row_data = prepare_features(country, params, feature_names, country_stats, df)

    try:
        prediction = model.predict(features_df)[0]

        # Probability if classifier
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(features_df)[0]

            # Probability of class 1
            class1_prob = proba[1] if len(proba) > 1 else prediction

            stress_score = class1_prob * 100
            return stress_score, class1_prob, row_data

        else:
            # Regression fallback
            stress_score = min(100, max(0, prediction))
            return stress_score, None, row_data

    except Exception:
        # SIMPLE fallback scoring (your existing logic)
        default_load = COUNTRY_INFO.get(country, {}).get('Actual_Load', 10000)
        base_stress = COUNTRY_INFO.get(country, {}).get('grid_stress_score', 25)

        actual_load = params.get('actual_load', default_load)
        forecasted_load = params.get('forecasted_load', actual_load * 1.05)
        forecast_error = abs(forecasted_load - actual_load) / max(forecasted_load, 1) * 100
        net_imports = params.get('net_imports', 0)
        temperature = params.get('temperature', 15)

        load_factor = (actual_load / default_load - 1) * 20
        forecast_factor = abs(forecast_error) * 0.5
        import_factor = abs(net_imports) / 1000 * 2
        temp_factor = max(0, abs(temperature - 15) - 10) * 0.5

        stress_score = base_stress + load_factor + forecast_factor + import_factor + temp_factor
        stress_score = max(0, min(100, stress_score))

        # No probability or row_data during fallback
        return stress_score, None, row_data



# ============================================================================
# MAIN APPLICATION
# ============================================================================
def main():
    # Load resources
    model = load_model()
    feature_names = load_feature_names()
    data_for_simulations, data_for_live, data_scores_real = fetch_data_from_databricks()
    country_stats = load_country_stats()

    # Only load the ARIMA model for the selected country
    # selected_country is set in the sidebar logic below
    
    # Header
    st.markdown("""
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 20px;">
            <div>
                <h1 style="margin: 0; font-size: 1.5rem;">⚡ GridWatch</h1>
                <p style="color: #f44336; font-size: 0.8rem; margin: 0;">STRESS PREDICTOR</p>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # ========================================================================
    # SIDEBAR
    # ========================================================================

    with st.sidebar:
        st.markdown("### 🔄 Mode")
        mode = st.radio(
            "Select Mode",
            options=["Simulated", "Live"],
            index=0,
            help="Simulated: Tweak parameters. Live: Real-time data"
        )

        st.markdown("### 🌍 Country")
        country_options = {f"{v['flag']} {v['name']} ({k})": k for k, v in COUNTRY_INFO.items()}
        selected_display = st.selectbox(
            "Select Country",
            options=list(country_options.keys()),
            index=4,  # Default to Germany
            label_visibility="collapsed"
        )
        selected_country = country_options[selected_display]

        # ===================================================================
        # DEFINE DEFAULTS FOR BOTH MODES
        # ===================================================================
        # Get base values from country_stats (used by both modes)
        sim_row = country_stats[country_stats["country"] == selected_country]
        if sim_row.empty:
            st.error(f"No data for {selected_country}")
            st.stop()
        
        sim_row = sim_row.iloc[0]
        default_load_country = float(sim_row.get("Actual_Load", 10000))
        default_net_imports_country = float(sim_row.get("net_imports", 0))
        default_temperature_country = float(sim_row.get("mean_temperature_c", 15))

        # ===================================================================
        # LOAD DATA BASED ON MODE
        # ===================================================================
        if mode == "Live":
            # Use live data
            live_row = data_for_live[data_for_live["country"] == selected_country]
            if live_row.empty:
                st.error(f"No live data for {selected_country}")
                st.stop()

            # Case 1 — "index" column exists and is a timestamp
            if "index" in live_row.columns:

                # Convert to datetime
                live_row["index"] = pd.to_datetime(live_row["index"])

                # Find row closest to entry_time
                live_row["time_diff"] = abs(live_row["index"] - entry_time)

                # Select best row
                live_row = live_row.loc[live_row["time_diff"].idxmin()]

                closest_timestamp = live_row["index"]

                # Convert to Berlin time
                berlin_tz = pytz.timezone("Europe/Berlin")
                berlin_time = closest_timestamp.tz_convert(berlin_tz) if closest_timestamp.tzinfo else closest_timestamp.tz_localize('UTC').tz_convert(berlin_tz)

                st.markdown(
                    f"""
                    <div class="live-data-banner">
                        📍 Live data from: <strong>{berlin_time.strftime('%Y-%m-%d %H:%M %Z')}</strong>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


            else:
                # Case 2 — No index/timestamp column → fallback
                st.warning("⚠️ No timestamp in live data, using most recent data")
                live_row = live_row.iloc[-1]

                data_time_str = "Unknown time"

                # If the Series still contains 'index'
                if "index" in live_row.index:
                    try:
                        ts = pd.to_datetime(live_row["index"])
                        data_time_str = ts.strftime('%H:%M')
                    except:
                        pass

                st.markdown(
                    f"""
                    <div class="live-data-banner">
                        📍 Live data from: <strong>{data_time_str}</strong>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


            #EXTRACT ALL VALUES FROM LIVE DATA
            slider_actual_load = int(live_row.get("Actual_Load", 10000))
            slider_forecasted_load = int(live_row.get("Forecasted_Load", slider_actual_load))
            if "imports_lag_1h" in live_row.index and "imports_rolling_mean_24h" in live_row.index:
                slider_net_imports = int((live_row.get("imports_lag_1h", 0) + live_row.get("imports_rolling_mean_24h", 0)) / 2)
            else:
                slider_net_imports = 0
            slider_temperature = float(live_row.get("mean_temperature_c", 15))
            slider_wind = float(live_row.get("mean_wind_speed", 5.0))
            slider_solar = int(live_row.get("mean_ssrd", 500))

            # Extract timestamp for hour_of_day and day_of_week
            if "index" in live_row.index or isinstance(live_row.get("index"), (pd.Timestamp, datetime)):
                try:
                    ts = pd.to_datetime(live_row.get("index", entry_time))
                    hour_of_day = ts.hour
                    day_of_week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][ts.weekday()]
                except:
                    hour_of_day = datetime.now().hour
                    day_of_week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][datetime.now().weekday()]
            else:
                hour_of_day = datetime.now().hour
                day_of_week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][datetime.now().weekday()]
            
            # No scenario multiplier in LIVE
            load_multiplier = 1.0
            scenario = None
            
        else:  # Simulated mode
            # Use simulated data
            sim_row = country_stats[country_stats["country"] == selected_country]
            if sim_row.empty:
                st.error(f"No data for {selected_country}")
                st.stop()
            
            sim_row = sim_row.iloc[0]

        # select arima model for the country the user selected
        model_arima_for_country = load_arima_model_for_country(selected_country)
        
        st.markdown("---")
        
        # get country info
        country_info = COUNTRY_INFO.get(selected_country, {})
        # Get default values based on country
        default_load = country_info.get('avg_load', 10000)
    

        # ===================================================================
        # SCENARIO BUTTONS (ONLY IN SIMULATED MODE)
        # ===================================================================
        if mode == "Simulated":
            st.markdown("### 🎯 Scenario Presets")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("NORMAL\nOPERATIONS", use_container_width=True):
                    st.session_state['scenario'] = 'normal'
                if st.button("COLD SNAP", use_container_width=True):
                    st.session_state['scenario'] = 'cold_snap'
                if st.button("IMPORT CRISIS", use_container_width=True):
                    st.session_state['scenario'] = 'import_crisis'
            with col2:
                if st.button("HEAT WAVE", use_container_width=True):
                    st.session_state['scenario'] = 'heat_wave'
                if st.button("WIND DROUGHT", use_container_width=True):
                    st.session_state['scenario'] = 'wind_drought'
                if st.button("FORECAST\nERROR", use_container_width=True):
                    st.session_state['scenario'] = 'forecast_error'
            
            col3, col4 = st.columns(2)
            with col3:
                if st.button("PEAK HOUR\nSTRESS", use_container_width=True):
                    st.session_state['scenario'] = 'peak_hour'
            with col4:
                pass
            
            st.markdown("---")
            
            col_sim, col_reset = st.columns(2)
            with col_sim:
                simulate_blackout = st.button("⚠️ SIMULATE\nBLACKOUT", use_container_width=True)
            with col_reset:
                if st.button("🔄 RESET", use_container_width=True):
                    st.session_state['scenario'] = 'normal'
                    st.rerun()
            
            st.markdown("""
            **ℹ️ Scenarios Note:**  
            Preset scenarios modify grid conditions, but the final stress score always comes from the model's learned patterns.
            """)
            st.markdown("---")
            
            # Apply scenario presets
            scenario = st.session_state.get('scenario', 'normal')
            
            if scenario == 'blackout':
                default_temp = 38
                default_wind = 1
                default_solar = 50
                load_multiplier = 2.0
            elif scenario == 'heat_wave':
                default_temp = 38
                default_wind = 2
                default_solar = 900
                load_multiplier = 1.3
            elif scenario == 'cold_snap':
                default_temp = -15
                default_wind = 3
                default_solar = 100
                load_multiplier = 1.4
            elif scenario == 'wind_drought':
                default_temp = 15
                default_wind = 0.5
                default_solar = 300
                load_multiplier = 1.0
            elif scenario == 'import_crisis':
                default_temp = 10
                default_wind = 5
                default_solar = 400
                load_multiplier = 1.1
            elif scenario == 'forecast_error':
                default_temp = 15
                default_wind = 5
                default_solar = 500
                load_multiplier = 1.2
            elif scenario == 'peak_hour':
                default_temp = 20
                default_wind = 4
                default_solar = 600
                load_multiplier = 1.25
            else:  # normal
                default_temp = 15
                default_wind = 5
                default_solar = 500
                load_multiplier = 1.0
            
            if simulate_blackout:
                st.session_state['scenario'] = 'blackout'
                st.rerun()
            
            # In Simulated mode, use slider values with defaults
            country_info = COUNTRY_INFO.get(selected_country, {})
            default_load = country_info.get('avg_load', 10000)
            
            slider_actual_load = int(default_load_country * load_multiplier)
            slider_forecasted_load = int(default_load_country * 1.0)
            slider_net_imports = int(default_net_imports_country)
            slider_temperature = float(default_temperature_country)
            slider_wind = float(default_wind)
            slider_solar = int(default_solar)
            
        else:  # LIVE MODE
            # No sliders, just display current values
            st.markdown("### 📊 Current Grid Status")
            st.markdown(
                    f"""
                    <div class="live-data-banner">
                        Live data enabled - Using real-time measurements from {selected_country}</strong>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            
            # Display the current values (non-interactive)
            slider_actual_load = int(live_row.get("Actual_Load", 10000))
            slider_forecasted_load = int(live_row.get("Forecasted_Load", slider_actual_load))
            if "imports_lag_1h" in live_row.index and "imports_rolling_mean_24h" in live_row.index:
                slider_net_imports = int((live_row.get("imports_lag_1h", 0) + live_row.get("imports_rolling_mean_24h", 0)) / 2)
            else:
                slider_net_imports = 0
            slider_temperature = float(live_row.get("mean_temperature_c", 15))
            slider_wind = float(live_row.get("mean_wind_speed", 5.0))
            slider_solar = int(live_row.get("mean_ssrd", 500))
            

        
        # ===================================================================
        # SLIDERS (ONLY IN SIMULATED MODE)
        # ===================================================================
        
        st.markdown("### ⚡ Load Parameters")
        
        actual_load = st.slider(
            "Actual Load (MW)",
            min_value=int(default_load_country * 0.5),
            max_value=int(default_load_country * 2),
            value=slider_actual_load,
            step=100,
            disabled=(mode == "Live"),
        )
        
        forecasted_load = st.slider(
            "Forecasted Load (MW)",
            min_value=int(default_load_country * 0.5),
            max_value=int(default_load_country * 2),
            value=slider_forecasted_load,
            step=100,
            disabled=(mode == "Live")
        )
        
        forecast_error = ((actual_load - forecasted_load) / max(forecasted_load, 1)) * 100
        error_color = "#4caf50" if abs(forecast_error) < 5 else "#ff9800" if abs(forecast_error) < 10 else "#f44336"
        st.markdown(f"<p style='color: {error_color}; font-size: 0.9rem;'>FORECAST ERROR: {forecast_error:+.1f}%</p>", unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 🔗 Cross-Border Flow")
        
        net_imports = st.slider(
            "Net Imports (MW)",
            min_value=-10000,
            max_value=10000,
            value=slider_net_imports,
            step=100,
            disabled=(mode == "Live")
        )
        
        st.markdown("---")
        st.markdown("### 🌤️ Weather")
        
        temperature = st.slider(
            "Temperature (°C)",
            min_value=float(-20),
            max_value=float(45),
            value=slider_temperature,
            step=0.5,
            disabled=(mode == "Live")
        )

        wind_speed = st.slider(
            "Wind Speed (m/s)",
            0.0,
            25.0,
            slider_wind,
            disabled=(mode == "Live")
        )
        
        solar_radiation = st.slider(
            "Solar Radiation (W/m²)",
            0,
            1000,
            slider_solar,
            disabled=(mode == "Live")
        )
        
        
        # ===================================================================
        # TIME SELECTION (BOTH MODES)
        # ===================================================================
        st.markdown("---")
        st.markdown("### 🕐 Time")
        hour_of_day = st.slider(
            "Hour of Day",
            0,
            23,
            datetime.now().hour,
            disabled=(mode == "Live")
        )
        day_of_week = st.selectbox(
            "Day of Week",
            options=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
            index=datetime.now().weekday(),
            disabled=(mode == "Live")
        )
    
    # Prepare parameters
    params = {
        'actual_load': actual_load,
        'forecasted_load': forecasted_load,
        'net_imports': net_imports,
        'temperature': temperature,
        'wind_speed': wind_speed,
        'solar_radiation': solar_radiation,
        'hour': hour_of_day,
        'day_of_week': ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"].index(day_of_week),
        'month': datetime.now().month
    }

    # Calculate stress for all countries
    all_stress_scores = {}
    all_row_data = {}

    if model is not None and feature_names is not None:
        dataset_to_use = data_for_live if mode == "Live" else data_for_simulations
        
        for country_code in COUNTRY_INFO.keys():
            if mode == "Live":
                # In Live mode: Get actual live data for THIS country
                live_country_row = data_for_live[data_for_live["country"] == country_code]
                
                if live_country_row.empty:
                    # No live data for this country, use baseline
                    all_stress_scores[country_code] = COUNTRY_INFO[country_code]['avg_stress']
                    all_row_data[country_code] = {}
                    continue
                
                # Get the most recent row for this country
                live_country_row = live_country_row.iloc[-1]
                
                # Compute net_imports from components
                imports_1h = float(live_country_row.get("imports_lag_1h", 0))
                imports_24h = float(live_country_row.get("imports_rolling_mean_24h", 0))
                net_imports_value = (imports_1h + imports_24h) / 2
                
                # Create params from actual live data
                params_for_country = {
                    'actual_load': float(live_country_row.get("Actual_Load", 10000)),
                    'forecasted_load': float(live_country_row.get("Forecasted_Load", 10000)),
                    'net_imports': float(net_imports_value) if net_imports_value else 0,
                    'temperature': float(live_country_row.get("mean_temperature_c", 15)),
                    'wind_speed': float(live_country_row.get("mean_wind_speed", 5.0)),
                    'solar_radiation': float(live_country_row.get("mean_ssrd", 500)),
                    'hour': int(live_country_row.get("hour", datetime.now().hour)),
                    'day_of_week': int(live_country_row.get("day_of_week", datetime.now().weekday())),
                    'month': datetime.now().month
                }
                
            else:
                # In Simulated mode: Scale selected country's params for other countries
                country_load_ratio = COUNTRY_INFO[country_code]['avg_load'] / COUNTRY_INFO[selected_country]['avg_load']
                params_for_country = params.copy()
                params_for_country['actual_load'] = actual_load * country_load_ratio
                params_for_country['forecasted_load'] = forecasted_load * country_load_ratio

            stress_score, stress_prob, row_data = predict_stress(
                country_code,
                params_for_country,
                model,
                feature_names,
                country_stats,
                dataset_to_use
            )

            all_stress_scores[country_code] = stress_score
            all_row_data[country_code] = row_data

    else:
        for country_code, info in COUNTRY_INFO.items():
            all_stress_scores[country_code] = info['avg_stress']

    # GET SELECTED COUNTRY'S DATA (AFTER LOOP)

    # Make prediction for selected country
    stress_score = all_stress_scores[selected_country]
    selected_row_data = all_row_data.get(selected_country, {})  # row_data for breakdown

    # Calculate components
    target_components = calculate_target_components(selected_row_data, params)
    active_targets = sum(1 for v in target_components.values() if v > 0)


    # ========================================================================
    # MAIN CONTENT
    # ========================================================================
    

    st.markdown("<br>", unsafe_allow_html=True)

    # Country header
    country_name = COUNTRY_INFO[selected_country]['name']
    country_flag = COUNTRY_INFO[selected_country]['flag']
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
    
    st.markdown(f"""
        <h2 style="margin-bottom: 5px;">{country_flag} {selected_country} {country_name} Grid Status</h2>
        <p style="color: #8ba3c7; font-size: 0.9rem;">Stress prediction • {current_time}</p>
    """, unsafe_allow_html=True)
    
    # Warning banner if high stress (66+ is high risk on 0-100 scale)
    if stress_score >= 66:
        st.markdown("""
            <div class="warning-banner">
                ⚠️ WARNING: HIGH BLACKOUT RISK DETECTED
            </div>
        """, unsafe_allow_html=True)
    
    # Top metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        category, color = get_stress_category(stress_score)
        delta = stress_score - country_info.get('avg_stress', 25)
        delta_color = "delta-negative" if delta > 0 else "delta-positive"
        st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">STRESS SCORE</div>
                <div class="metric-value" style="color: {color};">{stress_score:.1f}</div>
                <div class="metric-delta {delta_color}">{'▲' if delta > 0 else '▼'} {abs(delta):.1f} vs Baseline</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        load_delta = ((actual_load / default_load) - 1) * 100
        st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">CURRENT LOAD</div>
                <div class="metric-value">{actual_load:,.0f} MW</div>
                <div class="metric-delta {'delta-warning' if abs(load_delta) > 10 else 'delta-positive'}">
                    {'▲' if load_delta > 0 else '▼'} {abs(load_delta):.1f}% vs forecast
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        import_color = "delta-warning" if abs(net_imports) > 3000 else "delta-positive"
        st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">NET IMPORTS</div>
                <div class="metric-value">{abs(net_imports):,.0f} MW</div>
                <div class="metric-delta {import_color}">
                    {'🔌 Importing' if net_imports > 0 else '📤 Exporting' if net_imports < 0 else '⚖️ Balanced'}
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">TARGETS TRIGGERED</div>
                <div class="metric-value" style="color: {'#f44336' if active_targets >= 3 else '#ff9800' if active_targets >= 1 else '#4caf50'};">{active_targets}/4</div>
                <div class="metric-delta {'delta-negative' if active_targets >= 3 else 'delta-warning' if active_targets >= 1 else 'delta-positive'}">
                    + {sum(target_components.values()):.0f} pts
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Main dashboard grid
    col_left, col_right = st.columns([1.2, 1])
    
    with col_left:
        # Stress Gauge
        st.markdown('<div class="card-title">📊 Stress Gauge</div>', unsafe_allow_html=True, help="Gauge represents the model-predicted likelihood of high stress (0–100), not a direct sum of components.")
        gauge_fig = create_stress_gauge(stress_score)
        st.plotly_chart(gauge_fig, use_container_width=True, config={'displayModeBar': False})
        
        # Large score display
        category, color = get_stress_category(stress_score)
        st.markdown(f"""
            <div style="text-align: center; margin-top: -20px;">
                <span style="font-size: 4rem; font-weight: bold; color: {color};">{stress_score:.1f} pts</span>
            </div>
        """, unsafe_allow_html=True)
    
    with col_right:
        # Target Breakdown
        st.markdown('<div class="card-title">🔍 Underlying cause (component contribution)</div>', unsafe_allow_html=True, help="Bars represent the contribution of each indicator to grid stress.")
        breakdown_fig = create_target_breakdown(target_components)
        st.plotly_chart(breakdown_fig, use_container_width=True, config={'displayModeBar': False})
    
    # Target Analysis section
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="card-title">📋 Target Analysis</div>', unsafe_allow_html=True)
    
    cols = st.columns(4)
    target_items = list(target_components.items())
    
    for i, (target, value) in enumerate(target_items):
        with cols[i % 4]:
            bg_color = "#1a2942" if value == 0 else "#2d4a2d" if value <= 12.5 else "#4a2d2d"
            border_color = "#2a4a6e" if value == 0 else "#4caf50" if value <= 12.5 else "#f44336"
            text_color = "#8ba3c7" if value == 0 else "#4caf50" if value <= 12.5 else "#f44336"
            
            st.markdown(f"""
                <div style="background: {bg_color}; border: 1px solid {border_color}; 
                            border-radius: 8px; padding: 12px; margin: 5px 0;">
                    <div style="color: {text_color}; font-weight: bold; font-size: 0.9rem;">
                        {target}
                    </div>
                    <div style="color: {text_color}; font-size: 0.8rem; margin-top: 5px;">
                        {'✓ Normal' if value == 0 else f'+{value:.0f} pts'}
                    </div>
                </div>
            """, unsafe_allow_html=True)
    
    # Bottom row: 24h Projection and Feature Importance
    st.markdown("<br>", unsafe_allow_html=True)
    col_proj, col_feat = st.columns(2)
    
    with col_proj:
        st.markdown('<div class="card-title">📈 6-Hour Projection</div>', unsafe_allow_html=True)

        # Get last 24h data for ARIMA visualization
        if mode == "Live":
            last_24h_data = get_last_24h_stress_data(data_scores_real, selected_country)
        else:
            # In simulated mode, generate or use placeholder
            last_24h_data = [np.random.randint(30, 70) for _ in range(24)]

        # Generate 6-hour forecast
        forecast_6h = generate_6h_forecast(model_arima_for_country)

        # Plot combined
        fig = plot_24h_with_6h_forecast(last_24h_data, forecast_6h, hour_of_day)
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    
    with col_feat:
        st.markdown('<div class="card-title">🔬 Feature Importance</div>', unsafe_allow_html=True)
        importance_fig = create_feature_importance()
        st.plotly_chart(importance_fig, use_container_width=True, config={'displayModeBar': False})

    # EU Map
    map_fig = create_eu_map(all_stress_scores)
    st.plotly_chart(map_fig, use_container_width=True, config={'displayModeBar': False})
    
    
    # ========================================================================
    # DOCUMENTATION EXPANDERS
    # ========================================================================
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="card-title">📚 Documentation & Methodology</div>', unsafe_allow_html=True)
    
    # Expander 1: Stress Score & Targets
    with st.expander("🎯 Stress Score Description & Underlying Target Components"):
        st.markdown("""
        ### Grid Stress Score Calculation
        
        The Grid Stress Score is a 0–100 point metric derived from a classification model predicting the likelihood of high grid stress. It reflects the predicted probability, reflecting that the grid may experience stress when the score is higher than 50.
        
        ---
        
        ### The model uses four underlying targets:
        
        #### 1. Reserve Margin Score (0, 12.5, or 25 points)
        Measures the capacity buffer between forecasted and actual load.
        
        | Condition | Points | Interpretation |
        |-----------|--------|----------------|
        | Reserve margin >= -5% | 0 | Adequate capacity buffer |
        | Reserve margin between -5% and -15% | 12.5 | Moderate capacity concern |
        | Reserve margin < -15% | 25 | Critical capacity shortage |
        
        **Formula:** `reserve_margin = (forecasted_load - actual_load) / actual_load`
        
        ---
        
        #### 2. Load Forecast Error Score (0, 12.5, or 25 points)
        Measures the accuracy of day-ahead demand forecasting.
        
        | Condition | Points | Interpretation |
        |-----------|--------|----------------|
        | Forecast error <= 5% | 0 | Accurate forecast |
        | Forecast error between 5% and 10% | 12.5 | Moderate forecast deviation |
        | Forecast error > 10% | 25 | Significant forecast miss |
        
        **Formula:** `forecast_error_pct = |forecasted_load - actual_load| / forecasted_load × 100`
        
        ---
        
        #### 3. T7 - High Exports Score (0 or 25 points)
        Triggered when a country is exporting large amounts of electricity.
        
        | Condition | Points | Interpretation |
        |-----------|--------|----------------|
        | Net imports >= P10 threshold | 0 | Normal export levels |
        | Net imports < P10 threshold | 25 | Excessive exports strain |
        
        **Threshold:** P10 = -1,500 MW (10th percentile of historical net imports)
        
        ---
        
        #### 4. T8 - High Imports Score (0 or 25 points)
        Triggered when a country is importing large amounts of electricity.
        
        | Condition | Points | Interpretation |
        |-----------|--------|----------------|
        | Net imports <= P90 threshold | 0 | Normal import levels |
        | Net imports > P90 threshold | 25 | High import dependency |
        
        **Threshold:** P90 = 2,000 MW (90th percentile of historical net imports)
        
        ---
        
        ### Score Interpretation
        
        | Score Range | Status | Recommended Action |
        |-------------|--------|-------------------|
        | 0-32 | 🟢 **Normal** | Standard operations |
        | 33-65 | 🟡 **Moderate** | Increased monitoring |
        | 66-99 | 🔴 **High Risk** | Activate reserves, prepare interventions |
        | 100 | ⚫ **Critical** | Emergency protocols, load shedding |
        """)
    
    # Expander 2: Model Information
    with st.expander("🤖 Machine Learning Model Details"):
        st.markdown("""
        ### XGBoost Regression Model
        
        The prediction engine uses an XGBoost (Extreme Gradient Boosting) regressor optimized for time-series grid stress prediction.
        
        ---
        
        ### Model Performance Metrics
        
        | Metric | Value | Description |
        |--------|-------|-------------|
        | **R² Score** | 0.999878 | Explains 99.99% of variance |
        | **AUC-ROC** | 0.826 | Binary classification performance |
        | **Training Records** | 550,000+ | Hourly observations |
        | **Time Period** | 2023-2025 | ~2.5 years of data |
        | **Countries** | 13 | European nations |
        | **Features** | 100 | Engineered predictors |
        
        ---
        
        ### Feature Categories
        
        **Load Features (35 features)**
        - Actual and forecasted load values
        - Lag features: 1h, 2h, 3h, 24h, 168h (1 week)
        - Rolling statistics: mean, std, max over 24h window
        - Load changes and deviations
        
        **Import/Export Features (15 features)**
        - Net imports current and lagged
        - Rolling import statistics
        - Import dependency indicators
        - Cross-border flow ratios
        
        **Weather Features (12 features)**
        - Temperature (current and lagged)
        - Wind speed and power index
        - Solar radiation and forecast
        - Extreme weather indicators
        
        **Temporal Features (14 features)**
        - Cyclical encoding (sin/cos) for hour, day, month
        - Weekend indicators
        - Peak hour flags (morning: 7-9, evening: 17-20)
        
        **Country Indicators (13 features)**
        - One-hot encoded country identifiers
        
        ---
        
        ### Top 10 Most Important Features
        
        | Rank | Feature | Importance |
        |------|---------|------------|
        | 1 | stress_lag_1h | 43.1% |
        | 2 | load_forecast_ratio | 17.1% |
        | 3 | import_magnitude | 10.3% |
        | 4 | stress_change_24h | 8.5% |
        | 5 | load_rel_error | 7.5% |
        | 6 | reserve_margin_ml | 4.2% |
        | 7 | load_rolling_mean_24h | 3.1% |
        | 8 | import_dependency_ratio | 2.5% |
        | 9 | stress_lag_24h | 1.9% |
        | 10 | temp_load_interaction | 1.7% |
        
        ---
        
        ### Model Validation
        
        - **Temporal Split**: Training data before test data (no data leakage)
        - **Cross-Validation**: 5-fold time-series CV
        - **Real Event Validation**: Tested against April 28, 2025 Spain/Portugal blackout
        """)
    
    # Expander 3: Data Sources
    with st.expander("📊 Data Sources & Coverage"):
        st.markdown("""
        ### Primary Data Source
        
        **ENTSO-E Transparency Platform**
        - Official European electricity transmission data
        - Hourly resolution across all metrics
        - URL: [transparency.entsoe.eu](https://transparency.entsoe.eu)
                    
         **Copernicus Reanalysis**
        - Global climate reanalysis dataset providing meteorological variables at high temporal and spatial resolution
        - URL: [https://climate.copernicus.eu](https://climate.copernicus.eu) 
        
        ---
        
        ### Data Categories
        
        **Generation Data**
        - Actual generation by fuel type (nuclear, gas, coal, wind, solar, hydro, etc.)
        - Generation forecasts
        - Installed capacity
        
        **Load Data**
        - Actual electricity demand
        - Day-ahead load forecasts
        - Week-ahead forecasts
        
        **Cross-Border Flows**
        - Physical electricity flows between countries
        - Net import/export positions
        - Interconnector capacities
        
        **Weather Data**
        - Temperature (hourly, country-averaged)
        - Wind speed at turbine height
        - Solar radiation (surface downwelling)
        
        ---
        
        ### Country Coverage (13 Nations)
        
        | Region | Countries |
        |--------|-----------|
        | **Central Europe** | Germany (DE), France (FR), Netherlands (NL), Belgium (BE), Austria (AT) |
        | **Southern Europe** | Spain (ES), Portugal (PT), Italy (IT), Croatia (HR) |
        | **Eastern Europe** | Poland (PL), Hungary (HU), Slovakia (SK) |
        | **Baltic States** | Lithuania (LT) |
        
        ---
        
        ### Data Quality
        
        | Metric | Value |
        |--------|-------|
        | Total Records | 550,000+ |
        | Missing Data Rate | < 2% |
        | Temporal Coverage | Jan 2023 - Nov 2025 |
        | Update Frequency | Hourly |
        """)
    
    # Expander 4: How to Use
    with st.expander("💡 How to Use This Dashboard"):
        st.markdown("""
        ### Quick Start Guide
        
        ---
        
        #### 1. Select a Country
        Use the dropdown in the sidebar to choose from 13 European countries. The dashboard will automatically load country-specific baseline values.
        
        ---
        
        #### 2. Adjust Parameters
        
        **Load Parameters**
        - **Actual Load**: Current electricity demand in MW
        - **Forecasted Load**: Day-ahead predicted demand
        - The forecast error percentage is calculated automatically
        
        **Cross-Border Flow**
        - **Net Imports**: Positive = importing, Negative = exporting
        - Values above 2,000 MW trigger the T8 (High Imports) target
        - Values below -1,500 MW trigger the T7 (High Exports) target
        
        **Weather**
        - **Temperature**: Affects heating/cooling demand
        - **Wind Speed**: Impacts wind generation availability
        - **Solar Radiation**: Affects solar generation output
        
        **Time**
        - **Hour**: Peak hours (7-9 AM, 5-8 PM) typically show higher stress
        - **Day**: Weekdays generally have higher demand than weekends
        
        ---
        
        #### 3. Use Scenario Presets
        
        | Preset | Description | Key Changes |
        |--------|-------------|-------------|
        | **Normal Operations** | Baseline conditions | Default values |
        | **Heat Wave** | Summer extreme | +38°C, +30% load, low wind |
        | **Cold Snap** | Winter extreme | -15°C, +40% load, low solar |
        | **Wind Drought** | Low renewable output | 0.5 m/s wind |
        | **Import Crisis** | High dependency | +6,000 MW imports |
        | **Forecast Error** | Planning failure | +20% actual vs forecast |
        | **Peak Hour Stress** | High demand period | +25% load |
        | **Simulate Blackout** | Critical scenario | All factors extreme |
        
        ---
        
        #### 4. Interpret Results
        
        **Stress **: Shows current predicted stress level (0-100)
        
        **Target Breakdown**: Which components are contributing to stress
        
        **24-Hour Projection**: Expected stress pattern throughout the day
        
        **Feature Importance**: Which factors most influence predictions
        
        ---
        
        #### 5. Take Action
        
        | Stress Level | Recommended Response |
        |--------------|---------------------|
        | 🟢 Normal (0-32) | Continue standard monitoring |
        | 🟡 Moderate (33-65) | Review reserve availability, alert operators |
        | 🔴 High Risk (66-99) | Activate demand response, prepare load shedding |
        | ⚫ Critical (100) | Execute emergency protocols |
        """)
    
    # Footer
    st.markdown("""
        <div class="footer">
            <p>European Power Grid Stress Prediction System</p>
            <p>Team 6 - GridWatch | Capstone Project | December 2025</p>
            <p>Authors: Chavely Albert Fernandez, Pedro Miguel da Câmara Leme, Ya-Chi Hsiao and Maria Sokotushchenko</p>
            <p>XGBoost Classifier with class imbalance handling (F1 = 0.765, Recall = 0.807) | 13 Countries | 550K+ hourly records</p>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()