# Databricks notebook source
# This notebook is used to fill NULLS values in the validation set.
# It creates a mask of known values and substitutes them with the mean, median and fbfill of that type of generation for the country. If all values per type are NULL the values will keep as NULL
# Computes the MAE of the filled values and the known, and selectsselect per column the best filling method for each column (smallest MAE)

# COMMAND ----------

# DBTITLE 1,Load train dataset
train = spark.table("workspace.default.validation_set")

# COMMAND ----------

# DBTITLE 1,See first 5 rows
display(train.limit(5))

# COMMAND ----------

# DBTITLE 1,Remove all rows corresponding o countries that don't have information about generation
missing_countries = ["DK", "FI", "LV", "SE", "EE", "GR", "RO", "SI", "NO", "CH", "BG"]
train = train.filter(~train["country"].isin(missing_countries))

# COMMAND ----------

train.select("country").distinct().show()

# COMMAND ----------

# DBTITLE 1,Select only columns that should be imputed (we know from EDA process)
numeric_cols = [
    c for c, t in train.dtypes
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
null_counts = train.select([F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in numeric_cols]).collect()[0].asDict()

# Identify columns where all values are null
all_null_cols = [c for c, cnt in null_counts.items() if cnt == train.count()]

print("Columns with all null values:", all_null_cols)

# Drop these columns
if all_null_cols:
    train = train.drop(*all_null_cols)

# Update numeric_cols list
numeric_cols = [c for c in numeric_cols if c not in all_null_cols]

# COMMAND ----------

# DBTITLE 1,Create artificial mask (10% NULLS per country per column)
import pyspark.sql.functions as F
df_all = train

# Preserve original values
for c in numeric_cols:
    df_all = df_all.withColumn(f"{c}_original", F.col(c))

# Create deliberate 10% mask per country
for c in numeric_cols:
    df_all = df_all.withColumn(
        f"{c}_was_masked",
        ((F.rand(seed=42) < 0.1) & F.col(c).isNotNull()).cast("int")
    )
    df_all = df_all.withColumn(
        f"{c}_mask",
        F.when(F.col(f"{c}_was_masked") == 1, None).otherwise(F.col(c))
    )

# COMMAND ----------

# DBTITLE 1,Trying different methods to fill NULLS
# Aggregation-based per group
# Substitute nulls with the mean of that type of generation for the country. If all values per type are NULL the values will keep as NULL

from pyspark.sql import Window
import pyspark.sql.functions as F

# Use df_all from previous cell, which contains mask columns
w = Window.partitionBy("country")

for c in numeric_cols:
    df_all = df_all.withColumn(
        f"{c}_mean",
        F.when(F.col(f"{c}_mask").isNull(), F.avg(f"{c}_mask").over(w))
         .otherwise(F.col(f"{c}_mask"))
    )


# COMMAND ----------

display(df_all)

# COMMAND ----------

# Forward + Backward Fill: fills nulls with the last known value. Then, fill with next known value (reverse order) for filling the remaining nulls.

w_ff = Window.partitionBy("country").orderBy("index") \
          .rowsBetween(Window.unboundedPreceding, 0)

w_bf = Window.partitionBy("country").orderBy(F.col("index").desc()).rowsBetween(Window.unboundedPreceding, 0)

for c in numeric_cols:
    df_all = df_all.withColumn(f"{c}_ffill", F.last(f"{c}_mask", ignorenulls=True).over(w_ff))
    df_all = df_all.withColumn(f"{c}_fbfill",
                               F.coalesce(F.col(f"{c}_ffill"), F.last(f"{c}_mask", ignorenulls=True).over(w_bf)))

# COMMAND ----------

# Median

for c in numeric_cols:
    df_all = df_all.withColumn(
        f"{c}_median",
        F.when(F.col(f"{c}_mask").isNull(), F.expr(f"percentile_approx({c}_mask, 0.5)").over(w))
         .otherwise(F.col(f"{c}_mask"))
    )

# COMMAND ----------

# DBTITLE 1,Evaluate all methods (per method per country)
methods = {
    "mean": "_mean",
    "ffill_bfill": "_fbfill",
    "median": "_median"
}

mae_results = []

for c in numeric_cols:
    mask_condition = F.col(f"{c}_was_masked") == 1  # only rows deliberately masked

    for method_name, suffix in methods.items():
        mae_df = (
            df_all.filter(mask_condition)
                  .groupBy("country")
                  .agg(F.mean(F.abs(F.col(f"{c}{suffix}") - F.col(f"{c}_original"))).alias("mae"))
                  .withColumn("column", F.lit(c))
                  .withColumn("method", F.lit(method_name))
        )
        mae_results.append(mae_df)

# Combine all results
mae_all_spark = mae_results[0]
for m in mae_results[1:]:
    mae_all_spark = mae_all_spark.unionByName(m)

# Convert to pandas for plotting
mae_pdf = mae_all_spark.toPandas()


# COMMAND ----------

mae_pdf.head(50)

# COMMAND ----------

# DBTITLE 1,Select best imputation method per column
best_method_per_col = (
    mae_pdf.groupby("column")
           .apply(lambda x: x.loc[x["mae"].idxmin()])
           .reset_index(drop=True)
)

# COMMAND ----------

# DBTITLE 1,Turn it into dictionary
best_method_map = {
    (row.country, row.column): row.method
    for _, row in best_method_per_col.iterrows()
}

# COMMAND ----------

# DBTITLE 1,Display best method per column
best_method_per_col[["column", "method", "mae"]]

# COMMAND ----------

# DBTITLE 1,Apply best method to fill NULLS in every column
print("Best methods to impute nulls are: n" + best_method_per_col["method"].unique()) 
# Compute the values for mean and median columns 
mean_values = {}

for (country, column), method in best_method_map.items():
    if method == "mean":
        val = (
            df_all.filter(F.col("country") == country)
                  .agg(F.mean(column).alias(column))
                  .collect()[0][column]
        )
        mean_values[(country, column)] = val

median_values = {}

for (country, column), method in best_method_map.items():
    if method == "median":
        val = (
            df_all.filter(F.col("country") == country)
                  .approxQuantile(column, [0.5], 0.001)[0]
        )
        median_values[(country, column)] = val

df_imputed = df_all

for (country, column), value in mean_values.items():
    df_imputed = df_imputed.withColumn(
        column,
        F.when(
            (F.col("country") == country) & F.col(column).isNull(),
            F.lit(value)
        ).otherwise(F.col(column))
    )

for (country, column), value in median_values.items():
    df_imputed = df_imputed.withColumn(
        column,
        F.when(
            (F.col("country") == country) & F.col(column).isNull(),
            F.lit(value)
        ).otherwise(F.col(column))
    )

# COMMAND ----------

# Identify columns that need forward+backward fill
# Define windows for forward and backward fills 
w_ff = Window.partitionBy("country").orderBy("index").rowsBetween(Window.unboundedPreceding, 0) 
w_bf = Window.partitionBy("country").orderBy(F.col("index").desc()).rowsBetween(Window.unboundedPreceding, 0) 

# Apply forward + backward fill per column
for col in numeric_cols:
    df_imputed = df_imputed.withColumn(
        f"{col}_ffill", F.last(F.col(col), ignorenulls=True).over(w_ff)
    )
    df_imputed = df_imputed.withColumn(
        f"{col}_fbfill", F.coalesce(F.col(f"{col}_ffill"), F.last(F.col(col), ignorenulls=True).over(w_bf))
    )
    # Replace the original column with the filled one
    df_imputed = df_imputed.drop(col).withColumnRenamed(f"{col}_fbfill", col)

# COMMAND ----------

# check for remaining nulls
remaining_nans = df_imputed.select([F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in numeric_cols])
display(remaining_nans)

# COMMAND ----------

# DBTITLE 1,Fill with zero the remaining NULLS
# some countries don't provide information about certain kind of energy sources, so we can fill them with zeros, like nuclear energy

df_imputed = df_imputed.fillna(0, subset=numeric_cols)

# COMMAND ----------

# check for remaining nulls
remaining_nans = df_imputed.select([F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in numeric_cols])
display(remaining_nans)

# COMMAND ----------

original_cols =display(df_imputed)

# COMMAND ----------

# DBTITLE 1,Eliminate not necessary columns with intermediate steps
original_cols = train.columns

helper_cols = [c for c in df_imputed.columns if c not in original_cols]

# 3. Remove them
df_final = df_imputed.drop(*helper_cols)


# COMMAND ----------

df_final.write \
  .mode("overwrite") \
  .option("overwriteSchema", "true") \
  .saveAsTable("validation_set_imputed")
