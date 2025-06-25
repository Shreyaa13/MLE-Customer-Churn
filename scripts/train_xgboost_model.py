import os
import glob
import pandas as pd
import pickle
import numpy as np
import argparse
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pyspark
import pyspark.sql.functions as F
from pyspark.sql.functions import col

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier

def train_model(model_train_date_str):
    try:
        """
        Main function to train and save the XGBoost churn model.
        """
        print(f"Starting XGBoost model training for date: {model_train_date_str}")

        # --- 1. Setup Spark and Config ---
        spark = pyspark.sql.SparkSession.builder \
            .appName("xgboost_training") \
            .master("local[*]") \
            .getOrCreate()
        spark.sparkContext.setLogLevel("ERROR")

        # Build config based on the provided training date
        config = {}
        config["model_train_date_str"] = model_train_date_str
        config["model_train_date"] = datetime.strptime(model_train_date_str, "%Y-%m-%d")
        config["oot_end_date"] = config['model_train_date'] - timedelta(days=1)
        config["oot_start_date"] = config['model_train_date'] - relativedelta(months=1)
        config["train_test_end_date"] = config["oot_start_date"] - timedelta(days=1)
        config["train_test_start_date"] = config["oot_start_date"] - relativedelta(months=3)
        config["train_test_ratio"] = 0.8
        print("Configuration set:")
        print(config)

        # --- 2. Load Data ---
        # Load Label Store
        label_store_path = "datamart/gold/label_store/"
        label_files = glob.glob(os.path.join(label_store_path, '*.parquet'))
        label_store_sdf = spark.read.option("header", "true").parquet(*label_files)
        labels_sdf = label_store_sdf.filter(
            (col("Snapshot_Date") >= config["train_test_start_date"]) &
            (col("Snapshot_Date") <= config["oot_end_date"])
        )

        # Load Feature Store
        feature_store_path = "datamart/gold/feature_store/"
        feature_files = glob.glob(os.path.join(feature_store_path, '*.parquet'))
        valid_feature_files = [f for f in feature_files if os.path.getsize(f) > 0]
        features_store_sdf = spark.read.option("header", "true").parquet(*valid_feature_files)
        features_sdf = features_store_sdf.filter(
            (col("snapshot_date") >= config["train_test_start_date"]) &
            (col("snapshot_date") <= config["oot_end_date"])
        )
        
        # --- 3. Prepare Data for Modeling ---
        data_pdf = labels_sdf.join(features_sdf, on=["customerID", "snapshot_date"], how="left").toPandas()
        data_pdf['Snapshot_Date'] = pd.to_datetime(data_pdf['Snapshot_Date']).dt.date
        print(f"Joined data loaded with {data_pdf.shape[0]} rows.")

        # Split data into train-test-oot
        oot_pdf = data_pdf[(data_pdf['Snapshot_Date'] >= config["oot_start_date"].date()) & (data_pdf['Snapshot_Date'] <= config["oot_end_date"].date())]
        train_test_pdf = data_pdf[(data_pdf['Snapshot_Date'] >= config["train_test_start_date"].date()) & (data_pdf['Snapshot_Date'] <= config["train_test_end_date"].date())]

        feature_cols = [fcol for fcol in oot_pdf.columns if fcol not in ["customerID", "Snapshot_Date"]]

        X_train_full, X_test, y_train_full, y_test = train_test_split(
            train_test_pdf[feature_cols], train_test_pdf["Churn"],
            test_size=1 - config["train_test_ratio"],
            random_state=88,
            shuffle=True,
            stratify=train_test_pdf["Churn"]
        )
        X_train = X_train_full.drop('Churn', axis=1)
        y_train = y_train_full
        X_oot = oot_pdf.drop('Churn', axis=1)
        y_oot = oot_pdf['Churn']


        # --- 4. Preprocess Data ---
        print("Starting preprocessing...")
        # Impute missing values
        numerical_cols = X_train.select_dtypes(include=['float64', 'int64']).columns
        categorical_cols = X_train.select_dtypes(include=['object']).columns

        for num_col in numerical_cols:
            median_val = X_train[num_col].median()
            X_train.loc[:, num_col] = X_train[num_col].fillna(median_val)
            X_test.loc[:, num_col] = X_test.loc[:, num_col].fillna(median_val)
            X_oot.loc[:, num_col] = X_oot.loc[:, num_col].fillna(median_val)

        for cat_col in categorical_cols:
            mode_val = X_train[cat_col].mode()[0] if not X_train[cat_col].mode().empty else 'Unknown'
            X_train.loc[:, cat_col] = X_train[cat_col].fillna(mode_val)
            X_test.loc[:, cat_col] = X_test.loc[:, cat_col].fillna(mode_val)
            X_oot.loc[:, cat_col] = X_oot.loc[:, cat_col].fillna(mode_val)

        # Feature Engineering
        def create_tenure_groups(df):
            df = df.copy()
            df['tenure_group'] = pd.cut(df['tenure'],
                                        bins=[0, 12, 24, 36, 48, 60, 72, np.inf],
                                        labels=['0-1yr', '1-2yr', '2-3yr', '3-4yr', '4-5yr', '5-6yr', '6+yr'])
            return df

        X_train = create_tenure_groups(X_train)
        X_test = create_tenure_groups(X_test)
        X_oot = create_tenure_groups(X_oot)
        
        # Define preprocessing pipeline
        categorical_features = X_train.select_dtypes(include=['object', 'category']).columns
        numerical_features = X_train.select_dtypes(include=['float64', 'int64']).columns
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ])
        
        # Fit and transform data
        X_train_preprocessed = preprocessor.fit_transform(X_train)
        X_test_preprocessed = preprocessor.transform(X_test)

        # --- 5. Train XGBoost Model ---
        print("Starting XGBoost model training with RandomizedSearchCV...")
        xgb_model = XGBClassifier(objective='binary:logistic', eval_metric='auc', random_state=42, use_label_encoder=False)
        
        param_grid = {
            'max_depth': [3, 5, 7, 9],
            'learning_rate': [0.01, 0.1, 0.2],
            'n_estimators': [100, 200, 300],
            'gamma': [0, 0.1, 0.2],
            'min_child_weight': [1, 3, 5],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0]
        }
        
        random_search = RandomizedSearchCV(
            estimator=xgb_model,
            param_distributions=param_grid,
            n_iter=50,
            scoring='roc_auc',
            cv=5,
            verbose=1,
            random_state=42,
            n_jobs=-1
        )
        
        random_search.fit(X_train_preprocessed, y_train)
        best_xgb = random_search.best_estimator_

        # Evaluate best model on test set
        y_pred_proba_best = best_xgb.predict_proba(X_test_preprocessed)[:, 1]
        auc_score = roc_auc_score(y_test, y_pred_proba_best)
        print(f"\nBest XGBoost Model Test ROC AUC Score: {auc_score:.4f}")
        print(f"Best Parameters: {random_search.best_params_}")

        # --- 6. Save Model Artifact ---

        # import joblib
        artifact_path = 'model_artifacts'
        os.makedirs(artifact_path, exist_ok=True)

        pipeline_to_save = {'preprocessor': preprocessor, 'model': best_xgb}

        joblib.dump(pipeline_to_save, f'{artifact_path}/xgb_model_pipeline.joblib')

        print(f"XGBoost model pipeline saved to {artifact_path}/xgb_model_pipeline.joblib")



        # artifact_path = 'model_artifacts'
        # os.makedirs(artifact_path, exist_ok=True)
        
        # pipeline_to_save = {'preprocessor': preprocessor, 'model': best_xgb}
        
        # with open(f'{artifact_path}/xgb_model_pipeline.pkl', 'wb') as f:
        #     pickle.dump(pipeline_to_save, f)
            
        # print(f"XGBoost model pipeline saved to {artifact_path}/xgb_model_pipeline.pkl")


    except Exception as e:
        print(e)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train XGBoost Churn Model")
    parser.add_argument("--model_train_date", required=True, help="The date for model training in YYYY-MM-DD format.")
    args = parser.parse_args()
    
    train_model(args.model_train_date)