# Databricks notebook source
# MAGIC %md
# MAGIC # European Energy Grid Stress Prediction
# MAGIC
# MAGIC A comprehensive machine learning project for predicting electricity grid stress across Europe by integrating weather data, electricity generation, load forecasts, and cross-border flows.
# MAGIC
# MAGIC ## 🎯 Project Objective
# MAGIC
# MAGIC Develop a predictive model to classify grid stress levels (LOW, MEDIUM, HIGH) based on real-time and forecasted data from 27 European countries. This enables grid operators to anticipate stability issues and take preventive measures.
# MAGIC
# MAGIC ## 📊 Project Workflow
# MAGIC ```
# MAGIC 🌤️  RAW DATA SOURCES
# MAGIC     │
# MAGIC     ├─ Weather Data (hourly)
# MAGIC     ├─ Generation (by fuel type)
# MAGIC     ├─ Electricity Load
# MAGIC     ├─ Cross-border Flows
# MAGIC     └─ Load/Solar/Wind Forecasts
# MAGIC     
# MAGIC     ⬇️
# MAGIC     
# MAGIC 📁 DATA PROCESSING (data_processing/)
# MAGIC     │
# MAGIC     ├─ 01: Weather aggregation & geocoding
# MAGIC     ├─ 02: Generation normalization
# MAGIC     ├─ 03: Data integration & merging
# MAGIC     ├─ 04: Target variable (grid stress score)
# MAGIC     ├─ 05: Train/Validation/Test split
# MAGIC     └─ 06: Missing value imputation
# MAGIC     
# MAGIC     ✓ Output: train_set_imputed, validation_set_imputed, test_set_imputed
# MAGIC     
# MAGIC     ⬇️
# MAGIC     
# MAGIC 📁 MODELS (models/)
# MAGIC     │
# MAGIC     ├─ Model training & experiments
# MAGIC     ├─ Hyperparameter tuning
# MAGIC     ├─ Performance evaluation
# MAGIC     └─ Best model selection
# MAGIC     
# MAGIC     ✓ Output: Trained models & predictions
# MAGIC     
# MAGIC     ⬇️
# MAGIC     
# MAGIC 📁 ANALYSIS (analysis/)
# MAGIC     │
# MAGIC     ├─ Results visualization
# MAGIC     ├─ Feature importance analysis
# MAGIC     ├─ Error analysis
# MAGIC     └─ Insights & recommendations
# MAGIC     
# MAGIC     ✓ Output: Reports & dashboards
# MAGIC ```
# MAGIC
# MAGIC ## 📁 Repository Structure
# MAGIC
# MAGIC ```
# MAGIC european-energy-grid/
# MAGIC ├── README.md                          # This file
# MAGIC ├── data_processing/                   # Data pipeline stage
# MAGIC │   ├── README.md                      # Detailed data pipeline docs
# MAGIC │   ├── 01_weather_data_processing.py
# MAGIC │   ├── 02_generation_data_processing.py
# MAGIC │   ├── 03_all_tables_processing.py
# MAGIC │   ├── 04_define_target_variable.py
# MAGIC │   ├── 05_train_val_test_split.py
# MAGIC │   ├── 06a_filling_nans_train.py
# MAGIC │   ├── 06b_filling_nans_validation.py
# MAGIC │   └── 06c_filling_nans_test.py
# MAGIC │
# MAGIC ├── models/                            # Model training & evaluation
# MAGIC │   ├── README.md                      # Model documentation
# MAGIC │   ├── model_1_baseline.py
# MAGIC │   ├── model_2_advanced.py
# MAGIC │   ├── model_evaluation.py
# MAGIC │   └── hyperparameter_tuning.py
# MAGIC │
# MAGIC ├── analysis/                          # Results & insights
# MAGIC │   ├── README.md                      # Analysis documentation
# MAGIC │   ├── results_visualization.py
# MAGIC │   └── feature_importance.py
# MAGIC │
# MAGIC └── utils/                             # Shared utilities (if applicable)
# MAGIC     ├── helpers.py
# MAGIC     └── config.py
# MAGIC ```
# MAGIC
# MAGIC ## 🚀 Quick Start
# MAGIC
# MAGIC ### Prerequisites
# MAGIC
# MAGIC - **Databricks** workspace with Apache Spark
# MAGIC - **Python 3.8+** with PySpark
# MAGIC - **Libraries**: pyspark, pandas, scikit-learn, matplotlib, seaborn
# MAGIC - Access to the raw data: `curlybyte_solutions_rawdata_europe_grid_load` database
# MAGIC
# MAGIC ### Installation
# MAGIC
# MAGIC 1. Clone this repository
# MAGIC 2. Import notebooks into your Databricks workspace
# MAGIC 3. Install required libraries (if not already in cluster):
# MAGIC    ```
# MAGIC    %pip install reverse_geocode scikit-learn matplotlib seaborn
# MAGIC    ```
# MAGIC
# MAGIC ### Running the Pipeline
# MAGIC
# MAGIC **Step 1: Data Processing**
# MAGIC ```
# MAGIC Navigate to data_processing/ and run notebooks in order (01 → 06c)
# MAGIC See data_processing/README.md for detailed instructions
# MAGIC ```
# MAGIC
# MAGIC **Step 2: Model Development**
# MAGIC ```
# MAGIC Navigate to models/ and run training notebooks
# MAGIC See models/README.md for model architecture and tuning details
# MAGIC ```
# MAGIC
# MAGIC **Step 3: Analysis** (Optional)
# MAGIC ```
# MAGIC Navigate to analysis/ for results visualization and insights
# MAGIC See analysis/README.md for available analyses
# MAGIC ```
# MAGIC
# MAGIC ## 📚 Documentation
# MAGIC
# MAGIC Each folder has its own detailed README:
# MAGIC
# MAGIC - **[data_processing/README.md](./data_processing/README.md)** - Complete pipeline documentation
# MAGIC   - Data sources and transformations
# MAGIC   - Feature engineering details
# MAGIC   - Target variable definition (grid stress score)
# MAGIC   - Imputation methodology
# MAGIC   
# MAGIC - **[models/README.md](./models/README.md)** - Model development documentation
# MAGIC   - Model architectures
# MAGIC   - Hyperparameter tuning results
# MAGIC   - Performance metrics & comparisons
# MAGIC   - Best model selection criteria
# MAGIC
# MAGIC - **[analysis/README.md](./analysis/README.md)** - Analysis and results
# MAGIC   - Visualization outputs
# MAGIC   - Feature importance rankings
# MAGIC   - Error analysis and interpretations
# MAGIC
# MAGIC ## 🎯 Key Metrics & Target Variable
# MAGIC
# MAGIC **Grid Stress Score** (0-100 points):
# MAGIC
# MAGIC The target combines three indicators to measure grid stability:
# MAGIC
# MAGIC 1. **Reserve Margin** (0-25 pts): Current load vs. 24h historical average
# MAGIC 2. **Load Forecast Error** (0-25 pts): Prediction accuracy of demand
# MAGIC 3. **Cross-Border Flows** (0-50 pts): Unusual import/export levels
# MAGIC
# MAGIC **Stress Levels**:
# MAGIC - 🟢 **LOW** (< 33): Grid is stable
# MAGIC - 🟡 **MEDIUM** (33-66): Grid under moderate stress
# MAGIC - 🔴 **HIGH** (> 66): Grid under significant stress
# MAGIC
# MAGIC ## 🌍 Coverage
# MAGIC
# MAGIC **27 European Countries:**
# MAGIC Spain, Portugal, France, Germany, Italy, Great Britain, Netherlands, Belgium, Austria, Switzerland, Poland, Czech Republic, Denmark, Sweden, Norway, Finland, Greece, Ireland, Romania, Bulgaria, Hungary, Slovakia, Slovenia, Croatia, Estonia, Lithuania, Latvia
# MAGIC
# MAGIC ## 📊 Data Sources
# MAGIC
# MAGIC | Source | Frequency | Coverage |
# MAGIC |--------|-----------|----------|
# MAGIC | Weather Data | Hourly | Coordinates (lat/lon) → countries |
# MAGIC | Electricity Generation | 15-min → Hourly | By fuel type & country |
# MAGIC | Actual Load | 15-min → Hourly | By country |
# MAGIC | Cross-border Flows | 15-min → Hourly | Country pairs |
# MAGIC | Load Forecast | Hourly | By country |
# MAGIC | Solar Forecast | Hourly | By country |
# MAGIC | Wind Forecast | Hourly | By country |
# MAGIC
# MAGIC ## 🛠 Technologies & Stack
# MAGIC
# MAGIC - **Platform**: Databricks
# MAGIC - **Processing**: Apache Spark (PySpark)
# MAGIC - **ML Frameworks**: Scikit-learn, MLflow (if used)
# MAGIC - **Data Processing**: Pandas, PySpark SQL
# MAGIC - **Visualization**: Matplotlib, Seaborn
# MAGIC - **Language**: Python
# MAGIC
# MAGIC ## 📈 Expected Outputs
# MAGIC
# MAGIC ### From Data Processing
# MAGIC - `train_set_imputed` - Training data (features + target)
# MAGIC - `validation_set_imputed` - Validation data
# MAGIC - `test_set_imputed` - Test data
# MAGIC
# MAGIC ### From Models
# MAGIC - Trained classification models (LOW/MEDIUM/HIGH stress)
# MAGIC - Performance metrics (Accuracy, F1, Precision, Recall)
# MAGIC - Feature importance rankings
# MAGIC - Predictions on test set
# MAGIC
# MAGIC ### From Analysis
# MAGIC - Confusion matrices & ROC curves
# MAGIC - Feature importance visualizations
# MAGIC - Error case analysis
# MAGIC - Insights & recommendations
# MAGIC
# MAGIC ## 🔄 Data Flow Summary
# MAGIC
# MAGIC ```
# MAGIC Raw Data → Processing → Feature Engineering → Model Training → Evaluation → Results
# MAGIC   ↓           ↓              ↓                   ↓               ↓           ↓
# MAGIC 7 sources  Normalize     Weather +           Classification  Metrics &    Dashboards
# MAGIC            Aggregate    Generation         Random Forest     Insights     & Reports
# MAGIC                         Load + Flows       XGBoost
# MAGIC                         Forecasts          Neural Networks
# MAGIC ```
# MAGIC
# MAGIC ## 📝 Notes
# MAGIC
# MAGIC - All notebooks are designed to run in Databricks environment
# MAGIC - Data is aggregated to hourly intervals for consistency
# MAGIC - Countries with incomplete generation data are excluded (11 countries)
# MAGIC - Missing values are imputed using method-specific optimizations per column
# MAGIC - Temporal train/val/test split preserves time-series nature of data
# MAGIC - Reserve margin uses 24-hour rolling windows per country
# MAGIC
# MAGIC **Last Updated**: December 2025  
