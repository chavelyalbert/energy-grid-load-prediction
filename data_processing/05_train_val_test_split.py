# Databricks notebook source
# This notebook splits the data into train, validation, and test sets based on time.

# COMMAND ----------

from pyspark.sql import functions as F

def split_by_date(df, time_col="index", train_end="2024-12-31", val_end="2025-07-31"):
    """
    Split a DataFrame into train, validation, and test sets based on time.

    Parameters:
    - df: PySpark DataFrame
    - time_col: name of the column with datetime info
    - train_end: last date for training set (inclusive)
    - val_end: last date for validation set (inclusive); test is after this

    Returns:
    - df_train, df_val, df_test
    """

    # Ensure time_col is timestamp
    df = df.withColumn(time_col, F.to_timestamp(F.col(time_col)))

    # Train: time <= train_end
    df_train = df.filter(F.col(time_col) <= F.lit(train_end))

    # Validation: train_end < time <= val_end
    df_val = df.filter((F.col(time_col) > F.lit(train_end)) & (F.col(time_col) <= F.lit(val_end)))

    # Test: time > val_end
    df_test = df.filter(F.col(time_col) > F.lit(val_end))

    return df_train, df_val, df_test


# COMMAND ----------

df = spark.table("workspace.default.electricity_and_weather_europe_with_target")
df_train, df_val, df_test = split_by_date(df)

# COMMAND ----------

df_train.write \
  .mode("overwrite") \
  .option("overwriteSchema", "true") \
  .saveAsTable("train_set")

# COMMAND ----------

df_val.write \
  .mode("overwrite") \
  .option("overwriteSchema", "true") \
  .saveAsTable("validation_set")

# COMMAND ----------

df_test.write \
  .mode("overwrite") \
  .option("overwriteSchema", "true") \
  .saveAsTable("test_set")
