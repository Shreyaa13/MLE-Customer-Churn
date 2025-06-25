# utils/data_processing_silver_telco.py
import os
from pyspark.sql.functions import col, when, trim, to_date, regexp_extract, lower
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType, DoubleType

# --- Helper for Yes/No/Unknown to Int ---
def yes_no_unknown_to_int(column_obj, yes_val=1, no_val=0, unknown_val=None):
    """Converts a column with Yes/No/Unknown to Integer."""
    return when(lower(column_obj) == "yes", yes_val) \
           .when(lower(column_obj) == "no", no_val) \
           .when(lower(column_obj) == "unknown", unknown_val) \
           .otherwise(None) # Default for other unexpected values, or could be unknown_val

def _apply_cleaning_and_schema_features(df):
    print("  Applying specific cleaning and schema for Silver Features:")

    # customerID: StringType (already handled by generic cast if needed)
    df = df.withColumn("customerID", col("customerID").cast(StringType()))

    # tenure: IntegerType, between 0 and 73
    df = df.withColumn("tenure", col("tenure").cast(IntegerType()))
    df = df.filter((col("tenure") >= 0) & (col("tenure") <= 73))
    print(f"    features - After tenure filter (0-73): {df.count()} rows")

    # PhoneService: Drop Null, 'Not Specified', 'Unknown'. Convert Yes/No to 1/0. Int.
    values_to_drop = ["not specified", "unknown"]
    df = df.filter(~(lower(col("PhoneService")).isNull() | lower(col("PhoneService")).isin(values_to_drop)))
    df = df.withColumn("PhoneService", yes_no_unknown_to_int(col("PhoneService")).cast(IntegerType()))
    df = df.filter(col("PhoneService").isNotNull()) # Ensure only 1/0 remain after conversion
    print(f"    features - After PhoneService cleaning: {df.count()} rows")

    # Columns to drop Null, 'Not Specified', 'Unknown' and keep as String
    string_cols_to_clean_simple = [
        "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "Contract"
    ]
    for col_name in string_cols_to_clean_simple:
        if col_name in df.columns:
            df = df.filter(~(lower(col(col_name)).isNull() | lower(col(col_name)).isin(values_to_drop)))
            # Map "No phone service" and "No internet service" to "No" for relevant columns
            if col_name == "MultipleLines":
                 df = df.withColumn(col_name, when(lower(col(col_name)) == "no phone service", "No").otherwise(col(col_name)))
            elif col_name in ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]:
                 df = df.withColumn(col_name, when(lower(col(col_name)) == "no internet service", "No").otherwise(col(col_name)))
            df = df.withColumn(col_name, col(col_name).cast(StringType()))
            print(f"    features - After {col_name} cleaning: {df.count()} rows")


    # PaperlessBilling: Convert null, 'Not Specified' to 'Unknown'. Then Yes/No to 1/0. Int.
    df = df.withColumn("PaperlessBilling",
                       when(lower(col("PaperlessBilling")).isNull() | (lower(col("PaperlessBilling")) == "not specified"), "Unknown")
                       .otherwise(col("PaperlessBilling")))
    df = df.withColumn("PaperlessBilling", yes_no_unknown_to_int(col("PaperlessBilling"), unknown_val=None).cast(IntegerType())) # Assuming Unknown maps to null int
    # If Unknown should be a specific int (e.g., -1 or 2), adjust unknown_val in yes_no_unknown_to_int
    print(f"    features - After PaperlessBilling cleaning: {df.count()} rows")


    # PaymentMethod: Convert null, 'Not Specified' to 'Unknown'. String.
    df = df.withColumn("PaymentMethod",
                       when(lower(col("PaymentMethod")).isNull() | (lower(col("PaymentMethod")) == "not specified"), "Unknown")
                       .otherwise(col("PaymentMethod")).cast(StringType()))
    print(f"    features - After PaymentMethod cleaning: {df.count()} rows")

    # MonthlyCharges: Extract number, >0 & <124. Double.
    # Regex to extract a valid float number, allows for decimal.
    df = df.withColumn("MonthlyCharges_extracted", regexp_extract(col("MonthlyCharges"), r"(\d+\.?\d*)", 1))
    df = df.withColumn("MonthlyCharges",
                       when(col("MonthlyCharges_extracted") == "", None)
                       .otherwise(col("MonthlyCharges_extracted")).cast(DoubleType()))
    df = df.filter((col("MonthlyCharges") > 0) & (col("MonthlyCharges") < 124))
    df = df.drop("MonthlyCharges_extracted")
    print(f"    features - After MonthlyCharges cleaning: {df.count()} rows")

    # TotalCharges: Extract number, >0. Double. Handle empty strings as null.
    df = df.withColumn("TotalCharges_extracted", regexp_extract(col("TotalCharges"), r"(\d+\.?\d*)", 1))
    df = df.withColumn("TotalCharges",
                       when(trim(col("TotalCharges_extracted")) == "", None) # handle if regex gives empty string
                       .otherwise(col("TotalCharges_extracted")).cast(DoubleType()))
    df = df.filter(col("TotalCharges") > 0)
    df = df.drop("TotalCharges_extracted")
    print(f"    features - After TotalCharges cleaning: {df.count()} rows")

    # Snapshot_Date: DateType
    if 'Snapshot_Date' in df.columns:
        df = df.withColumn("Snapshot_Date", to_date(col("Snapshot_Date"), "yyyy-MM-dd").cast(DateType()))

    # Select final feature columns in desired order (optional, but good practice)
    final_feature_columns = [
        "customerID", "tenure", "PhoneService", "MultipleLines", "InternetService",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
        "PaymentMethod", "MonthlyCharges", "TotalCharges", "Snapshot_Date"
    ]
    # Filter out any columns that might not exist if source CSV was incomplete, though ideally all should be there
    existing_final_cols = [c for c in final_feature_columns if c in df.columns]
    df = df.select(existing_final_cols)

    return df



def _apply_cleaning_and_schema_meta(df):
    print("  Applying specific cleaning and schema for Silver Meta:")
    values_to_drop_generic = ["not specified", "unknown"]

    # customerID: StringType
    df = df.withColumn("customerID", col("customerID").cast(StringType()))

    # gender: Drop Null, 'Not Specified', 'Unknown'. Male=1, Female=0. Int.
    df = df.filter(~(lower(col("gender")).isNull() | lower(col("gender")).isin(values_to_drop_generic)))
    df = df.withColumn("gender",
                       when(lower(col("gender")) == "male", 1)
                       .when(lower(col("gender")) == "female", 0)
                       .otherwise(None).cast(IntegerType()))
    df = df.filter(col("gender").isNotNull()) # Keep only 0 and 1
    print(f"    meta - After gender cleaning: {df.count()} rows")

    # SeniorCitizen: Keep 0, 1. Int.
    df = df.withColumn("SeniorCitizen", col("SeniorCitizen").cast(IntegerType()))
    df = df.filter(col("SeniorCitizen").isin([0, 1]))
    print(f"    meta - After SeniorCitizen cleaning: {df.count()} rows")

    # Partner: Drop Null, 'Not Specified', 'Unknown'. Yes/No to 1/0. Int.
    df = df.filter(~(lower(col("Partner")).isNull() | lower(col("Partner")).isin(values_to_drop_generic)))
    df = df.withColumn("Partner", yes_no_unknown_to_int(col("Partner")).cast(IntegerType()))
    df = df.filter(col("Partner").isNotNull())
    print(f"    meta - After Partner cleaning: {df.count()} rows")

    # Dependents: Drop Null, 'Not Specified', 'Unknown'. Yes/No to 1/0. Int.
    df = df.filter(~(lower(col("Dependents")).isNull() | lower(col("Dependents")).isin(values_to_drop_generic)))
    df = df.withColumn("Dependents", yes_no_unknown_to_int(col("Dependents")).cast(IntegerType()))
    df = df.filter(col("Dependents").isNotNull())
    print(f"    meta - After Dependents cleaning: {df.count()} rows")

    # Snapshot_Date: DateType
    if 'Snapshot_Date' in df.columns:
        df = df.withColumn("Snapshot_Date", to_date(col("Snapshot_Date"), "yyyy-MM-dd").cast(DateType()))

    final_meta_columns = ["customerID", "gender", "SeniorCitizen", "Partner", "Dependents", "Snapshot_Date"]
    existing_final_cols = [c for c in final_meta_columns if c in df.columns]
    df = df.select(existing_final_cols)
    return df


def _apply_cleaning_and_schema_labels(df):
    print("  Applying specific cleaning and schema for Silver Labels:")

    # customerID: StringType
    df = df.withColumn("customerID", col("customerID").cast(StringType()))

    # Churn: Yes/No to 1/0. Int.
    df = df.withColumn("Churn", yes_no_unknown_to_int(col("Churn")).cast(IntegerType()))
    # df = df.filter(col("Churn").isNotNull()) # Decide if rows with Churn not Yes/No should be dropped or kept as null
    print(f"    labels - After Churn cleaning: {df.count()} rows")


    # Snapshot_Date: DateType
    if 'Snapshot_Date' in df.columns:
        df = df.withColumn("Snapshot_Date", to_date(col("Snapshot_Date"), "yyyy-MM-dd").cast(DateType()))

    final_labels_columns = ["customerID", "Churn", "Snapshot_Date"]
    existing_final_cols = [c for c in final_labels_columns if c in df.columns]
    df = df.select(existing_final_cols)
    return df


def _process_silver_generic(snapshot_date_str, bronze_base_dir, silver_base_dir, data_category, cleaning_schema_func, spark):
    bronze_file_path = os.path.join(bronze_base_dir, data_category, f"bronze_{data_category}_{snapshot_date_str}.csv")
    silver_category_directory = os.path.join(silver_base_dir, data_category)

    if not os.path.exists(bronze_file_path):
        print(f"  Bronze file not found: {bronze_file_path}. Skipping Silver for {data_category} on {snapshot_date_str}.")
        return None

    if not os.path.exists(silver_category_directory):
        os.makedirs(silver_category_directory)

    print(f"Processing Silver for {data_category} on {snapshot_date_str} from {bronze_file_path}")
    df = spark.read.csv(bronze_file_path, header=True, inferSchema=False) # Read as string initially
    initial_count = df.count()
    print(f"  Loaded {data_category} from Bronze: {bronze_file_path}, initial row count: {initial_count}")

    if initial_count == 0:
        print(f"  No data in bronze file for {data_category} on {snapshot_date_str}. Skipping Silver processing.")
        # Optionally, still write an empty parquet for consistency if downstream expects a file
        # df.write.mode("overwrite").parquet(os.path.join(silver_category_directory, f"silver_{data_category}_{snapshot_date_str}.parquet"))
        return None

    # Apply specific cleaning and schema function
    df = cleaning_schema_func(df)

    final_count = df.count()
    print(f"  Silver {data_category} - Rows dropped/filtered: {initial_count - final_count}")

    if final_count == 0:
        print(f"  Silver {data_category} - All rows were filtered out. Not saving empty parquet.")
        return None

    silver_filepath = os.path.join(silver_category_directory, f"silver_{data_category}_{snapshot_date_str}.parquet")
    df.write.mode("overwrite").parquet(silver_filepath)
    print(f"  Saved Silver {data_category} to: {silver_filepath}, final row count: {final_count}")
    # df.printSchema() # For debugging
    return df




def process_silver_features(snapshot_date_str, bronze_base_dir, silver_base_dir, spark):
    return _process_silver_generic(snapshot_date_str, bronze_base_dir, silver_base_dir, "features", _apply_cleaning_and_schema_features, spark)

def process_silver_meta(snapshot_date_str, bronze_base_dir, silver_base_dir, spark):
    return _process_silver_generic(snapshot_date_str, bronze_base_dir, silver_base_dir, "meta", _apply_cleaning_and_schema_meta, spark)

def process_silver_labels(snapshot_date_str, bronze_base_dir, silver_base_dir, spark):
    return _process_silver_generic(snapshot_date_str, bronze_base_dir, silver_base_dir, "labels", _apply_cleaning_and_schema_labels, spark)