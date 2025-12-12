# Databricks notebook source
# This notebook is used to fill NULLS values in the train set.
# It creates a mask of known values and substitutes them with the mean, median and fbfill of that type of generation for the country. If all values per type are NULL the values will keep as NULL
# Computes the MAE of the filled values and the known, and selectsselect per column the best filling method for each column (smallest MAE)

# COMMAND ----------

# DBTITLE 1,Load dataset
all_data = spark.table("workspace.live_data.electricity_and_weather_europe")

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
all_data = all_data.drop(*columns_to_drop)

# COMMAND ----------

# DBTITLE 1,Remove all rows corresponding o countries that don't have information about generation
missing_countries = ["DK", "FI", "LV", "SE", "EE", "GR", "RO", "SI", "NO", "CH", "BG"]
all_data = all_data.filter(~all_data["country"].isin(missing_countries))

# COMMAND ----------

# DBTITLE 1,Select only columns that should be imputed (we know from EDA process)
numeric_cols = [
    c for c, t in all_data.dtypes
    if t in ("double", "float", "int", "bigint")
    and (
        c.endswith("__Actual_Aggregated")
        or c.endswith("__Actual_Consumption")
    )
]

# COMMAND ----------

# DBTITLE 1,Remove columns that are fully empty
import pyspark.sql.functions as F

# Count nulls per column
null_counts = all_data.select([F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in numeric_cols]).collect()[0].asDict()

# Identify columns where all values are null
all_null_cols = [c for c, cnt in null_counts.items() if cnt == all_data.count()]

# Drop these columns
if all_null_cols:
    all_data = all_data.drop(*all_null_cols)

# Update numeric_cols list
numeric_cols = [c for c in numeric_cols if c not in all_null_cols]

# COMMAND ----------

# DBTITLE 1,Best method to fill (determined in exploratory notebook)
best_method_map = {
    "Biomass__Actual_Aggregated": "ffill_bfill",
    "Biomass__Actual_Consumption": "mean",
    "Energy_storage__Actual_Aggregated": "median",
    "Energy_storage__Actual_Consumption": "median",
    "Fossil_Brown_coal_Lignite__Actual_Aggregated": "mean",
    "Fossil_Coal_derived_gas__Actual_Aggregated": "mean",
    "Fossil_Gas__Actual_Aggregated": "ffill_bfill",
    "Fossil_Gas__Actual_Consumption": "mean",
    "Fossil_Hard_coal__Actual_Aggregated": "mean",
    "Fossil_Hard_coal__Actual_Consumption": "mean",
    "Fossil_Oil__Actual_Aggregated": "mean",
    "Fossil_Oil__Actual_Consumption": "mean",
    "Fossil_Oil_shale__Actual_Aggregated": "mean",
    "Fossil_Peat__Actual_Aggregated": "mean",
    "Geothermal__Actual_Aggregated": "mean",
    "Geothermal__Actual_Consumption": "mean",
    "Hydro_Pumped_Storage__Actual_Aggregated": "ffill_bfill",
    "Hydro_Pumped_Storage__Actual_Consumption": "ffill_bfill",
    "Hydro_Run_of_river_and_poundage__Actual_Aggregated": "mean",
    "Hydro_Run_of_river_and_poundage__Actual_Consumption": "mean",
    "Hydro_Water_Reservoir__Actual_Aggregated": "ffill_bfill",
    "Hydro_Water_Reservoir__Actual_Consumption": "mean",
    "Marine__Actual_Aggregated": "mean",
    "Nuclear__Actual_Aggregated": "ffill_bfill",
    "Nuclear__Actual_Consumption": "ffill_bfill",
    "Other__Actual_Aggregated": "mean",
    "Other__Actual_Consumption": "mean",
    "Other_renewable__Actual_Aggregated": "mean",
    "Other_renewable__Actual_Consumption": "mean",
    "Solar__Actual_Aggregated": "ffill_bfill",
    "Solar__Actual_Consumption": "mean",
    "Waste__Actual_Aggregated": "mean",
    "Waste__Actual_Consumption": "mean",
    "Wind_Offshore__Actual_Aggregated": "mean",
    "Wind_Offshore__Actual_Consumption": "ffill_bfill",
    "Wind_Onshore__Actual_Aggregated": "ffill_bfill",
    "Wind_Onshore__Actual_Consumption": "mean"
}


# COMMAND ----------

from pyspark.sql.window import Window# Window definitions for forward/backward fill

w_ff = Window.partitionBy("country").orderBy("index").rowsBetween(Window.unboundedPreceding, 0)
w_bf = Window.partitionBy("country").orderBy(F.col("index").desc()).rowsBetween(Window.unboundedPreceding, 0)

# COMMAND ----------

# Impute columns efficiently by batching transformations
mean_cols = [col for col, method in best_method_map.items() if method == "mean"]
median_cols = [col for col, method in best_method_map.items() if method == "median"]
ffill_bfill_cols = [col for col, method in best_method_map.items() if method == "ffill_bfill"]

# Mean imputation
if mean_cols:
    mean_exprs = [F.mean(col).alias(f"mean_{col}") for col in mean_cols]
    mean_df = all_data.groupBy("country").agg(*mean_exprs)
    all_data = all_data.join(mean_df, on="country", how="left")
    for col in mean_cols:
        all_data = all_data.withColumn(col, F.when(F.col(col).isNull(), F.col(f"mean_{col}")).otherwise(F.col(col)))
        all_data = all_data.drop(f"mean_{col}")

# Median imputation
if median_cols:
    median_exprs = [F.expr(f'percentile_approx({col}, 0.5)').alias(f"median_{col}") for col in median_cols]
    median_df = all_data.groupBy("country").agg(*median_exprs)
    all_data = all_data.join(median_df, on="country", how="left")
    for col in median_cols:
        all_data = all_data.withColumn(col, F.when(F.col(col).isNull(), F.col(f"median_{col}")).otherwise(F.col(col)))
        all_data = all_data.drop(f"median_{col}")

# Forward/backward fill imputation
for col in ffill_bfill_cols:
    all_data = all_data.withColumn(f"{col}_ffill", F.last(F.col(col), ignorenulls=True).over(w_ff))
    all_data = all_data.withColumn(f"{col}_fbfill", F.coalesce(F.col(f"{col}_ffill"), F.last(F.col(col), ignorenulls=True).over(w_bf)))
    all_data = all_data.drop(col, f"{col}_ffill").withColumnRenamed(f"{col}_fbfill", col)

# COMMAND ----------

# DBTITLE 1,Fill with zero the remaining NULLS
# some countries don't provide information about certain kind of energy sources, so we can fill them with zeros, like nuclear energy

all_data = all_data.fillna(0, subset=numeric_cols)

# COMMAND ----------

schema_name = "live_data"

all_data.write \
  .mode("overwrite") \
  .option("overwriteSchema", "true") \
  .saveAsTable(f"{schema_name}.electricity_and_weather_europe_imputed")
