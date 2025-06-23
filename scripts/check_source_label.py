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


# to call this script: python bronze_label_store.py --snapshotdate "2023-01-01"

def main():
    print('\n\n---starting job---\n\n')
    

    # check raw label
    raw_data_directory = "./data/"
    filename = 'telco_labels.csv'
    
    # Full path to the file
    file_path = os.path.join(raw_data_directory, filename)

    # Check if the file exists
    if os.path.isfile(file_path):
        print(f"{filename} exists in {raw_data_directory}")
    else:
        print(f"{filename} does NOT exist in {raw_data_directory}")
    
    print('\n\n---completed job---\n\n')

if __name__ == "__main__":
    main()
