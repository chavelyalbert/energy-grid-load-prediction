# Databricks notebook source
# This notebook is used to generate the target variable for the forecasting model.
# It uses the table 'electricity_and_weather_europe'.
# 1. Drop columns representin same info
# 2. Compute relative forecast errors
# 3. Compute reserve margin
# 4. Define target variable based on the relative forecast error, the reserve margin and the high import or exports:
# Total score: 100 points
# Each stress indicator contributes part of the total.
# Each indicator has 3 levels → low / medium / high → assign points accordingly.

# COMMAND ----------

# DBTITLE 1,Load table
df = spark.table("workspace.default.electricity_and_weather_europe")

# COMMAND ----------

df.columns

# COMMAND ----------

# DBTITLE 1,There are columns that represent the same kind of generation but correspond to the old database
columns_to_drop = [
    'Hydro_Pumped_Storage',
    'Hydro_Run_of_river_and_poundage',
    'Hydro_Water_Reservoir',
    'Nuclear',
    'Solar',
    'Wind_Onshore',
    'Biomass',
    'Fossil_Brown_coal_Lignite',
    'Fossil_Coal_derived_gas',
    'Fossil_Gas',
    'Fossil_Hard_coal',
    'Fossil_Oil',
    'Waste',
    'Wind_Offshore',
    'Other',
    'Other_renewable',
    'Fossil_Peat',
    'Energy_storage',
    'Fossil_Oil_shale'
]
df = df.drop(*columns_to_drop)

# COMMAND ----------

display(df)

# COMMAND ----------

# DBTITLE 1,Reserve margin
# Compare current load to the average load over the past 24 hours
# If current load is high relative to recent history → stress increases
# If current load is low → system is relaxed

from pyspark.sql.functions import col, try_divide
from pyspark.sql import Window
import pyspark.sql.functions as F

window_24h = Window.partitionBy("country").orderBy("index").rowsBetween(-24, 0)   # past 24 rows INCLUDING current

df = df.withColumn(
    "reserve_margin_ml",
    (F.avg("Actual_Load").over(window_24h) - F.col("Actual_Load")) 
    / F.avg("Actual_Load").over(window_24h)
)

# COMMAND ----------

display(df)

# COMMAND ----------

from pyspark.sql import functions as F
df.agg(F.max("reserve_margin_ml")).show()

# COMMAND ----------

# DBTITLE 1,Forecast error that can produce unexpected cases of high demand
df = df.withColumn("forecast_load_error", col("Forecasted_Load") - col("Actual_Load"))

# COMMAND ----------

# Compute relative forecast errors
df = df.withColumn("load_rel_error",
                       F.abs(F.col("forecast_load_error")) / (F.col("Actual_Load") + F.lit(1e-6)))

# COMMAND ----------

display(df)

# COMMAND ----------

# DBTITLE 1,90th percentile of Electricity imports and exports
# net_imports > 0 : import
# net_imports < 0 : export
# Since exports are negative, the 90th percentile of exports corresponds to the 10th percentile of net_imports.

w = Window.partitionBy("country")

df = df.withColumn("P10_net", F.expr("percentile_approx(net_imports, 0.10)").over(w))
df = df.withColumn("P90_net", F.expr("percentile_approx(net_imports, 0.90)").over(w))

# COMMAND ----------

# DBTITLE 1,Defining grid instability criteria
# Concept: Grid Stress Score
# Total score: 100 points
# Each stress indicator contributes part of the total.
# Each indicator has 3 levels → low / medium / high → assign points accordingly.

def calculate_grid_stress_score(df):
    """
    Compute a grid stress score based on multiple indicators with 3-level scoring.
    Returns the dataframe with total score and stress level.
    """

    # --- Reserve margin ---
    df = df.withColumn(
        "score_reserve_margin",
        F.when(F.abs(F.col("reserve_margin_ml")) >= 0.2, 0)
         .when((F.abs(F.col("reserve_margin_ml")) >= 0.1) & (F.abs(F.col("reserve_margin_ml")) < 0.2), 12.5)
         .otherwise(25)
    )

    # --- Load forecast error ---
    df = df.withColumn(
        "score_load_error",
        F.when(F.col("load_rel_error") <= 0.03, 0)
         .when((F.col("load_rel_error") > 0.03) & (F.col("load_rel_error") <= 0.1), 12.5)
         .otherwise(25)
    )

    # --- Cross-border flows (binary conditions T7–T9) ---
    df = df.withColumn("T7_high_exports", (F.col("net_imports") < F.col("P10_net")).cast("int"))
    df = df.withColumn("T8_high_imports", (F.col("net_imports") > F.col("P90_net")).cast("int"))

    # Assign points for cross-border flags
    df = df.withColumn("score_T7", F.col("T7_high_exports") * 25)
    df = df.withColumn("score_T8", F.col("T8_high_imports") * 25)

    # --- Total grid stress score ---
    df = df.withColumn(
        "grid_stress_score",
        F.col("score_reserve_margin") +
        F.col("score_load_error") +
        F.col("score_T7") +
        F.col("score_T8")
    )

    # LOW / MEDIUM / HIGH stress level: 
    # < 33, "LOW"
    # >33 and < 66, "MEDIUM"
    # > 66, "HIGH"

    return df


# COMMAND ----------

df = calculate_grid_stress_score(df)

# COMMAND ----------

display(df)

# COMMAND ----------

df.write \
  .mode("overwrite") \
  .option("overwriteSchema", "true") \
  .saveAsTable("electricity_and_weather_europe_with_target")
