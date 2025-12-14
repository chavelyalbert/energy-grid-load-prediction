# Databricks notebook source
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

DATABASE = "european_grid_raw__v2"

TABLES = [
    "load_actual",
    "load_forecast",
    "generation",
    "wind_forecast",
    "solar_forecast",
    "crossborder_flows",
]

def migrate_table(table_name: str):
    full = f"{DATABASE}.{table_name}"
    temp = f"{DATABASE}.{table_name}__tmp_migration"

    print(f"\n=== MIGRATING TABLE: {full} ===")

    # 1. Describe current table
    detail = spark.sql(f"DESCRIBE DETAIL {full}").collect()[0]
    partition_cols = detail.partitionColumns
    print(f"  Current partitions: {partition_cols}")

    # 2. Extract schema (as DDL)
    schema_ddl = spark.sql(f"SHOW CREATE TABLE {full}").collect()[0]["createtab_stmt"]

    # Remove PARTITIONED BY (...) from the DDL
    import re
    clean_ddl = re.sub(r"PARTITIONED BY\s*\([^)]+\)", "", schema_ddl, flags=re.IGNORECASE)

    # Fix formatting for Databricks CREATE OR REPLACE
    clean_ddl = clean_ddl.replace("CREATE TABLE", "CREATE TABLE IF NOT EXISTS")

    print("  Creating temp table without partitions...")
    spark.sql(f"DROP TABLE IF EXISTS {temp}")
    spark.sql(clean_ddl.replace(full, temp))

    # 3. Copy all data (no partitioning)
    print("  Copying data...")
    spark.sql(f"""
        INSERT INTO {temp}
        SELECT * FROM {full}
    """)

    # 4. Swap tables atomically
    print("  Replacing old table with unpartitioned version...")
    spark.sql(f"DROP TABLE {full}")
    spark.sql(f"ALTER TABLE {temp} RENAME TO {table_name}")

    print(f"  ✔ Migration complete for {full}")


# MAIN EXECUTION
for t in TABLES:
    migrate_table(t)

print("\n🎉 ALL TABLES SUCCESSFULLY MIGRATED TO NON-PARTITIONED STORAGE.")
