# utils/data_processing_bronze_telco.py
import os
from datetime import datetime
from pyspark.sql.functions import col, to_date # Keep to_date for the filter comparison

def _process_bronze_generic(snapshot_date_str, bronze_base_dir, data_category, raw_csv_path, spark):
    """
    Generic function to process a raw CSV to its bronze location, filtered by snapshot_date.
    The Snapshot_Date column from the CSV is used directly for filtering after an on-the-fly cast.
    No new 'Snapshot_Date_parsed' column is created or persisted in the bronze output.
    """
    snapshot_date_dt_object = datetime.strptime(snapshot_date_str, "%Y-%m-%d").date() # Python date object for comparison

    bronze_category_directory = os.path.join(bronze_base_dir, data_category)
    if not os.path.exists(bronze_category_directory):
        os.makedirs(bronze_category_directory)

    print(f"Processing Bronze for {data_category} on {snapshot_date_str} from {raw_csv_path}")

    # Load entire CSV with inferSchema=False, so Snapshot_Date will be string
    df = spark.read.csv(raw_csv_path, header=True, inferSchema=False)

    if 'Snapshot_Date' not in df.columns:
        print(f"  Critical Error: Snapshot_Date column not found in {raw_csv_path}. Cannot filter for {data_category}. Skipping.")
        return None

    # Filter by Snapshot_Date.
    # Cast the string 'Snapshot_Date' column to a DateType ON-THE-FLY for the comparison.
    # This cast is only for the filter operation and does not add a new column to df_filtered.
    df_filtered = df.filter(to_date(col('Snapshot_Date'), "yyyy-MM-dd") == snapshot_date_dt_object)

    print(f"  {data_category} - {snapshot_date_str} - Raw row count (before filter for this date): {df.count()}, Filtered row count: {df_filtered.count()}")

    if df_filtered.count() == 0:
        print(f"  No data found for {data_category} on snapshot_date {snapshot_date_str}. Skipping save.")
        return None

    partition_name = f"bronze_{data_category}_{snapshot_date_str}.csv"
    filepath = os.path.join(bronze_category_directory, partition_name)

    df_filtered.toPandas().to_csv(filepath, index=False) # df_filtered contains original columns
    print(f'  Saved {data_category} bronze to: {filepath}')
    return df_filtered



def process_bronze_features(snapshot_date_str, bronze_base_dir, spark):
    raw_csv_path = "data/telco_features.csv"
    return _process_bronze_generic(snapshot_date_str, bronze_base_dir, "features", raw_csv_path, spark)


def process_bronze_meta(snapshot_date_str, bronze_base_dir, spark):
    raw_csv_path = "data/telco_meta.csv"
    return _process_bronze_generic(snapshot_date_str, bronze_base_dir, "meta", raw_csv_path, spark)


def process_bronze_labels(snapshot_date_str, bronze_base_dir, spark):
    raw_csv_path = "data/telco_labels.csv"
    return _process_bronze_generic(snapshot_date_str, bronze_base_dir, "labels", raw_csv_path, spark)