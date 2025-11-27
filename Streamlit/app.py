# ============================================================
# STREAMLIT GRID STRESS FORECAST DASHBOARD (DE)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

# ------------------------------------------------------------
# 1. Page setup
# ------------------------------------------------------------
st.set_page_config(
    page_title="Grid Stress Forecast Dashboard",
    layout="wide",
)

st.title("⚡ Grid Stress Forecast Dashboard – Germany (DE)")
st.markdown("Forecast grid stress for the next 1–6 hours using trained XGBoost models.")


# ------------------------------------------------------------
# 2. Load DE models (correct relative path)
# ------------------------------------------------------------
@st.cache_resource
def load_models():
    models = {}

    # ✅ Correct path when running from Streamlit/ folder
    model_path = "models/de"

    for h in [1, 2, 3, 4, 5, 6]:
        fname = os.path.join(model_path, f"stress_plus_{h}h.pkl")

        if os.path.exists(fname):
            with open(fname, "rb") as f:
                models[f"stress_plus_{h}h"] = pickle.load(f)
        else:
            st.warning(f"⚠️ Missing model file: {fname}")

    return models


models = load_models()

if len(models) == 0:
    st.error("❌ No models found! Please upload your pickle files into models/de/")
    st.stop()

st.success("✅ XGBoost models loaded!")


# ------------------------------------------------------------
# 3. Load DE training dataset (correct relative path)
# ------------------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv(
        "data_streamlit/train_XGBoost_DE.csv", 
        parse_dates=["timestamp"]
    )


df = load_data()
df = df.sort_values("timestamp")

feature_cols = [
    c for c in df.columns
    if c not in ["timestamp", "country", "grid_stress_score"]
    and not c.startswith("stress_plus_")
]


# ------------------------------------------------------------
# 4. Sidebar – Select timestamp
# ------------------------------------------------------------
st.sidebar.header("🔧 Controls")

selected_ts = st.sidebar.selectbox(
    "Select timestamp",
    df["timestamp"].astype(str).tolist()
)

row = df[df["timestamp"].astype(str) == selected_ts]

if row.empty:
    st.error("Timestamp not found!")
    st.stop()

X_input = row[feature_cols].values


# ------------------------------------------------------------
# 5. Make predictions for all horizons
# ------------------------------------------------------------
st.subheader("📈 Predicted Grid Stress (Next 1–6 Hours)")

predictions = {}
for h in [1, 2, 3, 4, 5, 6]:
    key = f"stress_plus_{h}h"
    model = models.get(key)
    if model:
        pred = model.predict(X_input)[0]
        predictions[h] = pred
    else:
        predictions[h] = np.nan

pred_df = pd.DataFrame({
    "Hours Ahead": list(predictions.keys()),
    "Predicted Stress": list(predictions.values())
})

st.dataframe(pred_df, hide_index=True)


# ------------------------------------------------------------
# 6. Plot prediction curve
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(pred_df["Hours Ahead"], pred_df["Predicted Stress"], marker="o")
ax.set_title("Predicted Grid Stress (1–6 Hours Ahead)")
ax.set_xlabel("Hours Ahead")
ax.set_ylabel("Stress Score")
st.pyplot(fig)


# ------------------------------------------------------------
# 7. Feature importance (select horizon)
# ------------------------------------------------------------
st.subheader("🔍 Feature Importance")

h_selected = st.selectbox("Select horizon:", [1, 2, 3, 4, 5, 6])
model_key = f"stress_plus_{h_selected}h"

model = models.get(model_key)
if model:
    importances = model.feature_importances_

    imp_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": importances
    }).sort_values("importance", ascending=False).head(20)

    fig, ax = plt.subplots(figsize=(8, 10))
    ax.barh(imp_df["feature"], imp_df["importance"])
    ax.set_title(f"Top 20 Feature Importances (+{h_selected}h)")
    ax.invert_yaxis()
    st.pyplot(fig)
else:
    st.warning("Feature importance data not available for this horizon.")


# ------------------------------------------------------------
# 8. Plot historical stress
# ------------------------------------------------------------
st.subheader("📉 Last 300 Hours of Stress (DE)")

N = 300
df_last = df.tail(N)

fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(df_last["timestamp"], df_last["grid_stress_score"])
ax.set_title("Recent Grid Stress History")
ax.set_ylabel("Stress Score")
st.pyplot(fig)

st.info("Dashboard ready — more countries will be added soon!")
