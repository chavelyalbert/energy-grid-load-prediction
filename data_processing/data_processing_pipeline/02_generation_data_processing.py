# Databricks notebook source
# This notebook:
# 1. Loads the raw generation data from the curlybyte_solutions_rawdata_europe_grid_load database
# 2. Compute hourly averages for each country and time
# 3. Save the cleaned generation table under "workspace.schema_capstone.generation_clean"


# COMMAND ----------

# DBTITLE 1,Load electricity generation table
generation = spark.table("curlybyte_solutions_rawdata_europe_grid_load.european_grid_raw__v2.generation")

# COMMAND ----------

# DBTITLE 1,Get numeric columns of the table
numeric_cols = [
    c for c, t in generation.dtypes 
    if t in ("double", "float", "int", "bigint")
]

# COMMAND ----------

from pyspark.sql import functions as F
generation = generation.withColumn("hour",F.date_trunc("hour", F.col("index")))

# COMMAND ----------

agg_exprs = [F.mean(c).alias(c) for c in numeric_cols]  # keep same names

generation_hourly = generation.groupBy("country","hour").agg(*agg_exprs).orderBy("hour")
generation_hourly = generation_hourly.withColumnRenamed("hour", "index")


# COMMAND ----------

# Create schema under workspace
spark.sql("CREATE SCHEMA IF NOT EXISTS workspace.schema_capstone")

# COMMAND ----------

# Save the cleaned generation table under workspace
schema_name = "live_data"
generation_hourly.write.format("delta").mode("overwrite").saveAsTable(f"{schema_name}.generation_clean")
