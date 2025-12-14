# Databricks notebook source
# DBTITLE 1,Set schema name variable
schema_name = "live_data"
# Use schema_name when saving tables, e.g. f"{schema_name}.weather_europe"

# COMMAND ----------

# This notebook:
#   1. Reads the hourly weather data from the curlybyte_solutions_rawdata_europe_grid_load database. This database gets live weather data from Open-Meteo.com every 1 hour.
#   2. Create a column with the country code for each coordinate.
#   4. Save into spark table "weather_europe" only data from Europe.

# COMMAND ----------

# DBTITLE 1,Load weather raw data
weather_raw = spark.table("curlybyte_solutions_rawdata_europe_grid_load.european_weather_raw.current_weather")

# COMMAND ----------

# DBTITLE 1,Creating "country" column based on latitude and longitude
import reverse_geocode
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, DoubleType, StringType

# Getting distinct coordinates
coordinates = weather_raw.select("lat", "lon").distinct().collect()
coord_list = [(row['lat'], row['lon']) for row in coordinates]

# Getting country codes for each coordinate
country = [loc['country_code'] for loc in reverse_geocode.search(coord_list)]

# Creating a DataFrame mapping coordinates to country codes
coord_country_df = spark.createDataFrame(
    [(lat, lon, c) for (lat, lon), c in zip(coord_list, country)],
    schema=StructType([
        StructField("lat", DoubleType(), True),
        StructField("lon", DoubleType(), True),
        StructField("country", StringType(), True)
    ])
)

# Join back to the original weather_raw table
weather_with_country = weather_raw.join(coord_country_df, on=["lat", "lon"], how="left")

# COMMAND ----------

# DBTITLE 1,Rename columns
weather_with_country = weather_with_country.withColumnRenamed("temperature_c", "mean_temperature_c")
weather_with_country = weather_with_country.withColumnRenamed("wind_speed", "mean_wind_speed")
weather_with_country = weather_with_country.withColumnRenamed("ssrd", "mean_ssrd")

# COMMAND ----------

# DBTITLE 1,Create schema if not exists
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {schema_name}")

# COMMAND ----------

# DBTITLE 1,Save to table
weather_with_country.write.mode("overwrite").saveAsTable(f"{schema_name}.weather_europe")
