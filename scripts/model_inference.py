import argparse
import os
import glob
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint
import pyspark
import pyspark.sql.functions as F

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, f1_score, roc_auc_score
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split



def main(snapshotdate, modelname):
    print('\n\n---starting job---\n\n')
    
    # Initialize SparkSession
    spark = pyspark.sql.SparkSession.builder \
        .appName("gold_model_prediction") \
        .master("local[*]") \
        .getOrCreate()
    
    # Set log level to ERROR to hide warnings
    spark.sparkContext.setLogLevel("ERROR")

    
    # --- set up config ---

    config = {}
    config["snapshot_date_str"] = snapshotdate
    config["snapshot_date"] = datetime.strptime(config["snapshot_date_str"], "%Y-%m-%d")
    config["model_name"] = modelname
    config["model_directory"] = "model_artifacts/"
    config["model_artefact_filepath"] = config["model_directory"] + config["model_name"]
    
    pprint.pprint(config)


    # --- Load Model Artifact ---
    
    with open(config["model_artefact_filepath"], 'rb') as f:
        obj = pickle.load(f)

    preprocessor = obj['preprocessor']
    model = obj['model']

    
    # --- load gold feature store ---
    parquet_path = f"datamart/gold/feature_store/gold_feature_store_{config['snapshot_date_str']}.parquet"
    features_store_sdf = spark.read.parquet(parquet_path)


    # extract feature store
    features_sdf = features_store_sdf.filter((col("snapshot_date") == config["snapshot_date"]))
    print("extracted features_sdf", features_sdf.count(), config["snapshot_date"])
    
    # Extract IDs before dropping them
    id_cols_pdf = features_sdf.select("customerID", "snapshot_date").toPandas()
    
    features_pdf = features_sdf.toPandas()
    features_pdf.drop(columns=[c for c in ["customerID", "snapshot_date"] if c in features_pdf.columns], inplace=True)


    # --- preprocess data for modeling ---

    # Drop identifiers 
    features_pdf.drop(columns=[c for c in ["customerID", "snapshot_date"] if c in features_pdf.columns], inplace=True)
    
    # Recreate tenure groups
    def create_tenure_groups(df):
        df = df.copy()
        df["tenure_group"] = pd.cut(
            df["tenure"],
            bins=[0, 12, 24, 36, 48, 60, 72, np.inf],
            labels=["0-1yr", "1-2yr", "2-3yr", "3-4yr", "4-5yr", "5-6yr", "6+yr"]
        )
        return df
    
    features_pdf = create_tenure_groups(features_pdf)


    # Handle missing values
    numerical_cols = features_pdf.select_dtypes(include=["float64", "int64"]).columns
    categorical_cols = features_pdf.select_dtypes(include=["object", "category"]).columns
    
    for num_col in numerical_cols:
        median_val = features_pdf[num_col].median()
        features_pdf[num_col] = features_pdf[num_col].fillna(median_val)

    for cat_col in categorical_cols:
        if pd.api.types.is_categorical_dtype(features_pdf[cat_col]):
            if "Unknown" not in features_pdf[cat_col].cat.categories:
                features_pdf[cat_col] = features_pdf[cat_col].cat.add_categories(["Unknown"])
        mode_val = features_pdf[cat_col].mode()[0] if not features_pdf[cat_col].mode().empty else "Unknown"
        features_pdf[cat_col] = features_pdf[cat_col].fillna(mode_val)


    # --- model prediction inference ---

    # Transform and predict
    X_inference = preprocessor.transform(features_pdf)
    y_inference = model.predict(X_inference)

    y_inference_pdf = id_cols_pdf.copy()
    y_inference_pdf["model_name"] = config["model_name"]
    y_inference_pdf["model_predictions"] = y_inference
    
    print(y_inference_pdf.head())
    print("Number of predictions:", len(y_inference_pdf))


    # --- save model inference to datamart gold table ---

    gold_directory = f"datamart/gold/model_predictions/{config['model_name'][:-4]}/"
    os.makedirs(gold_directory, exist_ok=True)

    partition_name = f"{config['model_name'][:-4]}_predictions_{config['snapshot_date_str'].replace('-', '_')}.parquet"
    filepath = os.path.join(gold_directory, partition_name)

    # Save Parquet
    spark.createDataFrame(y_inference_pdf).write.mode("overwrite").parquet(filepath)
    print("Saved inference results to:", filepath)

    # Save CSV
    csv_filepath = os.path.join(gold_directory, partition_name.replace(".parquet", ".csv"))
    y_inference_pdf.to_csv(csv_filepath, index=False)
    print("Saved inference results to CSV:", csv_filepath)

    
    # --- end spark session --- 
    spark.stop()
    
    print('\n\n---completed job---\n\n')


if __name__ == "__main__":
    # Setup argparse to parse command-line arguments
    parser = argparse.ArgumentParser(description="run job")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--modelname", type=str, required=True, help="model_name")
    
    args = parser.parse_args()
    
    # Call main with arguments explicitly passed
    main(args.snapshotdate, args.modelname)
