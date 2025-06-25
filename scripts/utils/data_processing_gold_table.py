# utils/data_processing_gold_telco.py
import os
from pyspark.sql.functions import col

def process_gold_feature_store(snapshot_date_str, silver_base_dir, gold_base_dir, spark):
    silver_features_path = os.path.join(silver_base_dir, "features", f"silver_features_{snapshot_date_str}.parquet")
    silver_meta_path = os.path.join(silver_base_dir, "meta", f"silver_meta_{snapshot_date_str}.parquet")
    
    gold_feature_store_directory = os.path.join(gold_base_dir, "feature_store")
    if not os.path.exists(gold_feature_store_directory):
        os.makedirs(gold_feature_store_directory)

    print(f"Processing Gold Feature Store for {snapshot_date_str}")

    if not os.path.exists(silver_features_path) or not os.path.exists(silver_meta_path):
        print(f"  Missing silver files for {snapshot_date_str} (features: {os.path.exists(silver_features_path)}, meta: {os.path.exists(silver_meta_path)}). Skipping Gold Feature Store.")
        return None

    df_features = spark.read.parquet(silver_features_path)
    df_meta = spark.read.parquet(silver_meta_path)

    print(f"  Loaded Silver Features: {df_features.count()} rows, Silver Meta: {df_meta.count()} rows.")

    # Join features and meta data
    # Both should have customerID and Snapshot_Date. Let's ensure Snapshot_Date from features is used.
    # df_meta might have Snapshot_Date aliased if there's a conflict, but Spark handles it well with array of join cols
    join_cols = ["customerID", "Snapshot_Date"]
    df_gold_fs = df_features.join(df_meta, join_cols, "inner") # Inner join assumes a customer must exist in both

    print(f"  Joined features and meta. Resulting count: {df_gold_fs.count()}")
    
    # Drop one of the Snapshot_Date columns if Spark duplicated it (usually doesn't with join_cols array)
    # Or select explicitly:
    # Example: df_gold_fs = df_features.alias("feat").join(df_meta.alias("meta"), 
    #                                                  (col("feat.customerID") == col("meta.customerID")) & \
    #                                                  (col("feat.Snapshot_Date") == col("meta.Snapshot_Date")), 
    #                                                  "inner") \
    #                                     .select("feat.*", "meta.gender", ...) # to be explicit

    gold_filepath = os.path.join(gold_feature_store_directory, f"gold_feature_store_{snapshot_date_str}.parquet")
    df_gold_fs.write.mode("overwrite").parquet(gold_filepath)
    print(f"  Saved Gold Feature Store to: {gold_filepath}, row count: {df_gold_fs.count()}")
    # df_gold_fs.printSchema()
    return df_gold_fs



def process_gold_label_store(snapshot_date_str, silver_base_dir, gold_base_dir, spark):
    silver_labels_path = os.path.join(silver_base_dir, "labels", f"silver_labels_{snapshot_date_str}.parquet")
    
    gold_label_store_directory = os.path.join(gold_base_dir, "label_store") # Corrected directory
    if not os.path.exists(gold_label_store_directory):
        os.makedirs(gold_label_store_directory)

    print(f"Processing Gold Label Store for {snapshot_date_str}")

    if not os.path.exists(silver_labels_path):
        print(f"  Missing silver labels file for {snapshot_date_str}. Skipping Gold Label Store.")
        return None
        
    df_labels = spark.read.parquet(silver_labels_path)
    print(f"  Loaded Silver Labels: {df_labels.count()} rows.")

    # For labels, gold might just be a direct copy of silver if no further transformations are needed at this stage
    gold_filepath = os.path.join(gold_label_store_directory, f"gold_label_store_{snapshot_date_str}.parquet")
    df_labels.write.mode("overwrite").parquet(gold_filepath)
    print(f"  Saved Gold Label Store to: {gold_filepath}, row count: {df_labels.count()}")
    # df_labels.printSchema()
    return df_labels