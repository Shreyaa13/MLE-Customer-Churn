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
    model_name = "log_reg_churn_model.joblib"

    config = {}
    config["snapshot_date_str"] = snapshot_date_str
    config["snapshot_date"] = datetime.strptime(config["snapshot_date_str"], "%Y-%m-%d")
    config["model_name"] = model_name
    config["model_directory"] = "model_artifacts/"
    config["model_artefact_filepath"] = config["model_directory"] + config["model_name"]
    
    pprint.pprint(config)


    # --- Load Model Artifact ---
    # Load the model from the pickle file
    with open(config["model_artefact_filepath"], 'rb') as file:
        model_artefact = pickle.load(file)
    print("Model loaded successfully! " + config["model_artefact_filepath"])


    # --- load gold feature store ---
    gold_feature_store_path = "datamart/gold/feature_store/"
    
    config["train_test_start_date_str"] = "2024-04-01"
    config["oot_end_date_str"] = "2024-07-01"
    config["train_test_start_date"] = datetime.strptime(config["train_test_start_date_str"], "%Y-%m-%d")
    config["oot_end_date"] = datetime.strptime(config["oot_end_date_str"], "%Y-%m-%d")
    
    available_files = os.listdir(gold_feature_store_path)
    target_dates = ["2024-04-01", "2024-05-01", "2024-06-01", "2024-07-01"]
    
    target_files = [
        os.path.join(gold_feature_store_path, f"gold_feature_store_{d}.parquet")
        for d in target_dates if f"gold_feature_store_{d}.parquet" in available_files
    ]
    
    if not target_files:
        raise FileNotFoundError("No matching Parquet files found for the given date range.")
    
    features_sdf = spark.read.parquet(*target_files)
    features_sdf = features_sdf.filter(
        (col("snapshot_date") >= config["train_test_start_date"]) &
        (col("snapshot_date") <= config["oot_end_date"])
    )
    
    print("Extracted features_sdf:", features_sdf.count(), config["train_test_start_date"], config["oot_end_date"])
    
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
    
    for col in numerical_cols:
        median_val = features_pdf[col].median()
        features_pdf[col] = features_pdf[col].fillna(median_val)
    
    for col in categorical_cols:
        mode_val = features_pdf[col].mode()[0] if not features_pdf[col].mode().empty else "Unknown"
        features_pdf[col] = features_pdf[col].fillna(mode_val)


    # --- model prediction inference ---

    # Load model and preprocessor 
    pipeline = load("model_artifacts/model_pipeline.pkl")
    preprocessor = pipeline["preprocessor"]
    model = pipeline["model"]

    # Transform and predict
    X_inference = preprocessor.transform(features_pdf)
    y_inference = model.predict_proba(X_inference)[:, 1]

    y_inference_pdf = id_cols_pdf.copy()
    y_inference_pdf["model_name"] = config["model_name"]
    y_inference_pdf["model_predictions"] = y_inference
    
    print(y_inference_pdf.head())
    print("Number of predictions:", len(y_inference_pdf))


    # --- save model inference to datamart gold table ---

    snapshot_range_str = f"{config['train_test_start_date_str']}_to_{config['oot_end_date_str']}".replace("-", "_")
    gold_directory = f"datamart/gold/model_predictions/{config['model_name'][:-4]}/"
    
    if not os.path.exists(gold_directory):
        os.makedirs(gold_directory)
    
    partition_name = f"{config['model_name'][:-4]}_predictions_{snapshot_range_str}.parquet"
    filepath = os.path.join(gold_directory, partition_name)
    
    spark.createDataFrame(y_inference_pdf).write.mode("overwrite").parquet(filepath)
    print("Saved inference results to:", filepath)

    # Save as CSV
    csv_filepath = filepath.replace(".parquet", ".csv")  # Change the extension to .csv

    # Save the predictions as CSV using pandas
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
