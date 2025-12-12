# Databricks notebook source
# This notebook:
#   1. Reads the hourly weather data from the curlybyte_solutions_rawdata_europe_grid_load database
#   2. Create a column with the country code for each coordinate
#   3. Compute mean for each country and hour
#   4. Save into spark table "weather_europe" only data from Europe

# COMMAND ----------

# DBTITLE 1,Load weather raw data
weather_raw = spark.table("curlybyte_solutions_rawdata_europe_grid_load.european_weather_raw.weather_hourly")

# COMMAND ----------

# DBTITLE 1,See first 5 rows
weather_raw.show(5)

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

display(weather_with_country.orderBy("timestamp"))


# COMMAND ----------

# DBTITLE 1,Getting mean value for each country and for each time stamp
columns_to_average = ["ssrd", "wind_speed", "temperature_c"]
agg_exprs = [F.mean(c).alias(f"mean_{c}") for c in columns_to_average]
country_time_mean = weather_with_country.groupBy("country", "timestamp").agg(*agg_exprs)

country_time_mean = country_time_mean.orderBy("timestamp")

display(country_time_mean)

# COMMAND ----------

# DBTITLE 1,Reducing to only European countries
countries_europe = ['ES', 'PT', 'FR', 'DE', 'IT', 'GB', 'NL', 'BE', 'AT', 'CH', 'PL', 'CZ', 'DK', 'SE', 'NO', 'FI', 'GR', 'IE', 'RO', 'BG', 'HU', 'SK', 'SI', 'HR', 'EE', 'LT', 'LV']

#print(len(countries_europe))  
country_time_mean = country_time_mean.filter(country_time_mean.country.isin(countries_europe))

display(country_time_mean)        

# COMMAND ----------

# DBTITLE 1,Save to table
country_time_mean.write.mode("overwrite").saveAsTable("weather_europe")

# COMMAND ----------

# DBTITLE 1,Show first 5 rows of table
spark.table('weather_europe').show(5)
