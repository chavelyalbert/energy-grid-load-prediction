# Databricks notebook source
# Remove SparkSession creation; use the existing spark object

# Make sure we are in the right catalog & schema
spark.sql("USE CATALOG workspace")
spark.sql("USE european_grid_raw__v2")

tables = [
    "load_actual",
    "load_forecast",
    "generation",
    "generation_total",
    "wind_forecast",
    "solar_forecast",
    "installed_capacity",
    "crossborder_flows",
]

for tbl in tables:
    print(f"\n=== Migrating table: {tbl} ===")

    # Check if 'date' column exists
    columns = [f.name for f in spark.table(tbl).schema.fields]
    if "date" not in columns:
        spark.sql(f"ALTER TABLE {tbl} ADD COLUMN date STRING")

    # Backfill date from index for existing rows
    spark.sql(
        f"""
        UPDATE {tbl}
        SET date = date_format(CAST(index AS DATE), 'yyyy-MM-dd')
        WHERE date IS NULL
        """
    )

    print(f"   ✓ date column added & backfilled in {tbl}")
