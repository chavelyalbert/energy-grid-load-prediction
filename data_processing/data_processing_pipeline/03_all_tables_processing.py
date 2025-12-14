# Databricks notebook source
# This notebook:
# 1. load the data from the database into Spark DataFrames (weather, generation, load, crossborder flows)
# 2. take all the data into hourly means (like weather data originally)
# 3. Create another column with net flow for each country and timestamp, considering imports and exports in crossborder_hourly
# 4. Merge the tables on countries and timestamp (only rows common for all tables)
# 5. Save into Spark Table called "electricity_and_weather_europe"

# COMMAND ----------

# DBTITLE 1,Load weather and electricity tables
weather = spark.table("workspace.live_data.weather_europe")
generation = spark.table("workspace.live_data.generation_clean")
load = spark.table("curlybyte_solutions_rawdata_europe_grid_load.european_grid_raw__v2.load_actual")
crossborder = spark.table("curlybyte_solutions_rawdata_europe_grid_load.european_grid_raw__v2.crossborder_flows")
load_forecast = spark.table("curlybyte_solutions_rawdata_europe_grid_load.european_grid_raw__v2.load_forecast")
solar_forecast = spark.table("curlybyte_solutions_rawdata_europe_grid_load.european_grid_raw__v2.solar_forecast")
wind_forecast = spark.table("curlybyte_solutions_rawdata_europe_grid_load.european_grid_raw__v2.wind_forecast")


# COMMAND ----------

# need to rename timestamp column to index
weather = weather.withColumnRenamed("timestamp", "index")
crossborder = crossborder.withColumnRenamed("timestamp", "index")

# COMMAND ----------

# DBTITLE 1,Compute hourly means in case some country give information for more timestamps within the hour
from pyspark.sql import functions as F

# for load dataframe

# Identify numeric columns in load
numeric_cols = [
    c for c, t in load.dtypes
    if t in ("double", "float", "int", "bigint")
]

# Truncate timestamp to hour
load = load.withColumn("hour", F.date_trunc("hour", F.col("index")))

# Aggregate by country and hour
agg_exprs = [F.mean(F.col(c)).alias(c) for c in numeric_cols]

load_hourly = load.groupBy("country", "hour").agg(*agg_exprs).orderBy("hour")

load_hourly = load_hourly.withColumnRenamed("hour", "index")

# COMMAND ----------

# DBTITLE 1,Compute hourly means for crossborder dataframe
# for crossborder dataframe

# Identify numeric columns in crossborder
dtypes = crossborder.dtypes
numeric_cols = [c for c, t in dtypes if t in ("double", "float", "int", "bigint")]

# Truncate timestamp to hour
crossborder = crossborder.withColumn("hour", F.date_trunc("hour", F.col("index")))

# Aggregate by from_country, to_country, and hour
agg_exprs = [F.mean(F.col(c)).alias(c) for c in numeric_cols]
crossborder_hourly = crossborder.groupBy("from_country", "to_country", "hour").agg(*agg_exprs).orderBy("hour")

# Rename hour column to index
crossborder_hourly = crossborder_hourly.withColumnRenamed("hour", "index")

# COMMAND ----------

# for load forecast dataframe

# Identify numeric columns in load
numeric_cols = [
    c for c, t in load_forecast.dtypes
    if t in ("double", "float", "int", "bigint")
]

# Truncate timestamp to hour
load_forecast = load_forecast.withColumn("hour", F.date_trunc("hour", F.col("index")))

# Aggregate by country and hour
agg_exprs = [F.mean(F.col(c)).alias(c) for c in numeric_cols]

load_forecast_hourly = load_forecast.groupBy("country", "hour").agg(*agg_exprs).orderBy("hour")

load_forecast_hourly = load_forecast_hourly.withColumnRenamed("hour", "index")

# COMMAND ----------

# for solar generation forecast dataframe

# Identify numeric columns in load
numeric_cols = [
    c for c, t in solar_forecast.dtypes
    if t in ("double", "float", "int", "bigint")
]

# Truncate timestamp to hour
solar_forecast = solar_forecast.withColumn("hour", F.date_trunc("hour", F.col("index")))

# Aggregate by country and hour
agg_exprs = [F.mean(F.col(c)).alias(c) for c in numeric_cols]

solar_forecast_hourly = solar_forecast.groupBy("country", "hour").agg(*agg_exprs).orderBy("hour")

solar_forecast_hourly = solar_forecast_hourly.withColumnRenamed("hour", "index")

# COMMAND ----------

# for wind generation forecast dataframe

# Identify numeric columns in load
numeric_cols = [
    c for c, t in wind_forecast.dtypes
    if t in ("double", "float", "int", "bigint")
]

# Truncate timestamp to hour
wind_forecast = wind_forecast.withColumn("hour", F.date_trunc("hour", F.col("index")))

# Aggregate by country and hour
agg_exprs = [F.mean(F.col(c)).alias(c) for c in numeric_cols]

wind_forecast_hourly = wind_forecast.groupBy("country", "hour").agg(*agg_exprs).orderBy("hour")

wind_forecast_hourly = wind_forecast_hourly.withColumnRenamed("hour", "index")

# COMMAND ----------

# DBTITLE 1,Compute net flow for each country and timestamp  considering imports and exports in crossborder_hourly
# Net flow per country
imports = crossborder_hourly.groupBy("to_country", "index").agg(F.sum("Value").alias("import_mw"))
exports = crossborder_hourly.groupBy("from_country", "index").agg(F.sum("Value").alias("export_mw"))

# COMMAND ----------

# Rename index columns before join
imports = imports.withColumnRenamed("index", "import_index")
exports = exports.withColumnRenamed("index", "export_index")

# Join imports and exports for each country and timestamp
net_flow = imports.join(
    exports,
    (imports["to_country"] == exports["from_country"]) & (imports["import_index"] == exports["export_index"]),
    how="full_outer"
).fillna(0)

# COMMAND ----------

# Compute net imports
net_flow = net_flow.withColumn(
    "index",
    F.coalesce("import_index", "export_index")
).withColumn(
    "country",
    F.coalesce("to_country", "from_country")
).withColumn(
    "net_imports",
    F.col("import_mw") - F.col("export_mw")
).select(
    "country", "index", "net_imports"
)

# COMMAND ----------

# DBTITLE 1,Renaming columns
solar_forecast_hourly = solar_forecast_hourly.withColumnRenamed("Solar", "solar_forecast")
wind_forecast_hourly = wind_forecast_hourly.withColumnRenamed("Wind_Onshore", "wind_forecast")

# COMMAND ----------

# DBTITLE 1,Merge the tables on countries and timestamp (only rows common for all tables)
# since we have less timestamps for load_hourly, we will join to this table
df = load_hourly
df = df.join(generation, on=["index", "country"], how="inner")
df = df.join(net_flow, on=["index", "country"], how="inner")
df = df.join(weather, on=["index", "country"], how="inner")
df = df.join(solar_forecast_hourly, on=["index", "country"], how="inner")
df = df.join(wind_forecast_hourly, on=["index", "country"], how="inner")
df = df.join(load_forecast_hourly, on=["index", "country"], how="inner")
df = df.drop("month")


# COMMAND ----------

from collections import Counter
col_counts = Counter(df.columns)
duplicates = [col for col, count in col_counts.items() if count > 1]
print(duplicates)

# COMMAND ----------

# DBTITLE 1,Save Spark Table
schema_name = "live_data"

df.write \
  .mode("overwrite") \
  .option("overwriteSchema", "true") \
  .saveAsTable(f"{schema_name}.electricity_and_weather_europe")
