# Databricks notebook source
# This notebook:
# 1. load the data from the database into Spark DataFrames (weather, generation, load, crossborder flows)
# 2. take all the data into hourly means (like weather data originally)
# 3. Create another column with net flow for each country and timestamp, considering imports and exports in crossborder_hourly
# 4. Merge the tables on countries and timestamp (only rows common for all tables)
# 5. Save into Spark Table called "electricity_and_weather_europe"

# COMMAND ----------

# DBTITLE 1,Load weather and electricity tables
weather = spark.table("workspace.default.weather_europe")
generation = spark.table("workspace.schema_capstone.generation_clean")
load = spark.table("curlybyte_solutions_rawdata_europe_grid_load.european_grid_raw__v2.load_actual")
crossborder = spark.table("curlybyte_solutions_rawdata_europe_grid_load.european_grid_raw__v2.crossborder_flows")
load_forecast = spark.table("curlybyte_solutions_rawdata_europe_grid_load.european_grid_raw__v2.load_forecast")
solar_forecast = spark.table("curlybyte_solutions_rawdata_europe_grid_load.european_grid_raw__v2.solar_forecast")
wind_forecast = spark.table("curlybyte_solutions_rawdata_europe_grid_load.european_grid_raw__v2.wind_forecast")


# COMMAND ----------

# DBTITLE 1,See columns
print("load:", load.columns)
print("generation:", generation.columns)
print("weather:", weather.columns)
print("crossborder flows:", weather.columns)
print("load forecast:", load_forecast.columns)
print("solar forecast:", solar_forecast.columns)
print("wind forecast:", wind_forecast.columns)

# COMMAND ----------

# need to rename timestamp column to index
weather = weather.withColumnRenamed("timestamp", "index")
crossborder = crossborder.withColumnRenamed("timestamp", "index")

# COMMAND ----------

# DBTITLE 1,Get size of tables
num_rows, num_columns = load.count(), len(load.columns)
print(f"Rows: {num_rows}, Columns: {num_columns}")

# COMMAND ----------

num_rows, num_columns = generation.count(), len(generation.columns)
print(f"Rows: {num_rows}, Columns: {num_columns}")

# COMMAND ----------

num_rows, num_columns = crossborder.count(), len(crossborder.columns)
print(f"Rows: {num_rows}, Columns: {num_columns}")

# COMMAND ----------

num_rows, num_columns = load_forecast.count(), len(load_forecast.columns)
print(f"Rows: {num_rows}, Columns: {num_columns}")

# COMMAND ----------

num_rows, num_columns = solar_forecast.count(), len(solar_forecast.columns)
print(f"Rows: {num_rows}, Columns: {num_columns}")

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
display(load_hourly)


# COMMAND ----------

# DBTITLE 1,Get size of tables
num_rows, num_columns = load_hourly.count(), len(load_hourly.columns)
print(f"Rows: {num_rows}, Columns: {num_columns}")

# COMMAND ----------

display(crossborder)

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
display(crossborder_hourly)

# COMMAND ----------

# DBTITLE 1,Get size of table
num_rows, num_columns = crossborder_hourly.count(), len(crossborder_hourly.columns)
print(f"Rows: {num_rows}, Columns: {num_columns}")

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
display(load_forecast_hourly)

# COMMAND ----------

num_rows, num_columns = load_forecast_hourly.count(), len(load_forecast_hourly.columns)
print(f"Rows: {num_rows}, Columns: {num_columns}")

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
display(solar_forecast_hourly)

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
display(wind_forecast_hourly)

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

display(net_flow)

# COMMAND ----------

# DBTITLE 1,Get size of table
num_rows, num_columns = net_flow.count(), len(net_flow.columns)
print(f"Rows: {num_rows}, Columns: {num_columns}")

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

# DBTITLE 1,Get size of table
num_rows, num_columns = df.count(), len(df.columns)
print(f"Rows: {num_rows}, Columns: {num_columns}")

# COMMAND ----------

display(df.orderBy("index"))

# COMMAND ----------

# DBTITLE 1,Save Spark Table
df.write \
  .mode("overwrite") \
  .option("overwriteSchema", "true") \
  .saveAsTable("electricity_and_weather_europe")

# COMMAND ----------

# DBTITLE 1,Check for missing values
df_nulls = df.select([
    F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in df.columns
])

# Convert to table: column_name | null_count
expr = ", ".join([f"'{c}', {c}" for c in df.columns])
df_nulls_long = df_nulls.selectExpr(f"stack({len(df.columns)}, {expr}) as (column_name, null_count)")

df_nulls_long.show(truncate=False)
