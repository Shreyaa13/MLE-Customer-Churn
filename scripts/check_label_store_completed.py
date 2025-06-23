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


    # eng_directory
    label_store_directory = "./datamart/gold/label_store/"
    
    # Check if the directory exists
    if os.path.isdir(label_store_directory):
        print(f"label_store_directory exists: {label_store_directory}")
    else:
        print(f"label_store_directory does NOT exist: {label_store_directory}")


    print('\n\n---completed job---\n\n')

if __name__ == "__main__":
    main()
