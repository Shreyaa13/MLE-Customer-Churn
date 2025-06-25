import argparse
import os
import glob
import pandas as pd
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

import utils.data_processing_bronze_table
import utils.data_processing_silver_table
import utils.data_processing_gold_table

# to call this script: python gold_feature_store.py --snapshotdate "2023-01-01"

def main(snapshotdate):
    print('\n\n---starting job---\n\n')
    
    # Initialize SparkSession
    spark = pyspark.sql.SparkSession.builder \
        .appName("dev") \
        .master("local[*]") \
        .getOrCreate()
    
    # Set log level to ERROR to hide warnings
    spark.sparkContext.setLogLevel("ERROR")

    # load arguments
    date_str = snapshotdate
    
    # create silver datalake
    base_silver_dir = "datamart/silver"
    base_gold_dir = "datamart/gold"
    
    if not os.path.exists(base_gold_dir):
        os.makedirs(base_gold_dir)


    # --- BRONZE LAYER PROCESSING ---
    print("\n--- STARTING GOLD FEATURE STORE PROCESSING ---")
    print(f"\nProcessing Gold for Snapshot Date: {date_str}")
    utils.data_processing_gold_table.process_gold_feature_store(date_str, base_silver_dir, base_gold_dir, spark)
    print("--- GOLD FEATURE STORE COMPLETED ---")

    
    # end spark session
    spark.stop()
    
    print('\n\n---completed job---\n\n')

if __name__ == "__main__":
    # Setup argparse to parse command-line arguments
    parser = argparse.ArgumentParser(description="run job")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    
    args = parser.parse_args()
    
    # Call main with arguments explicitly passed
    main(args.snapshotdate)
