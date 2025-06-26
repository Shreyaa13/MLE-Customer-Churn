from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'dag',
    default_args=default_args,
    description='data pipeline run once a month',
    schedule_interval='0 0 1 * *',  # At 00:00 on day-of-month 1
    start_date=datetime(2024, 4, 1),
    end_date=datetime(2024, 7, 1),
    catchup=True,
) as dag:

    # data pipeline

    # --- label store ---
    dep_check_source_label_data = BashOperator(
        task_id='dep_check_source_label_data',
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 check_source_label.py'
        ),
    )

    bronze_label_store = BashOperator(
        task_id='run_bronze_label',
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 bronze_label.py '
            '--snapshotdate "{{ ds }}"'
        ),
    )

    silver_label_store = BashOperator(
        task_id='run_silver_label',
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 silver_label.py '
            '--snapshotdate "{{ ds }}"'
        ),
    )

    gold_label_store =  BashOperator(
        task_id='run_gold_label_store',
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 gold_label_store.py '
            '--snapshotdate "{{ ds }}"'
        ),
    )

    label_store_completed = BashOperator(
        task_id='label_store_completed',
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 check_label_store_completed.py'
        ),
    )
    

    # Define task dependencies to run scripts sequentially
    dep_check_source_label_data >> bronze_label_store >> silver_label_store >> gold_label_store >> label_store_completed
 
 
    # --- feature store ---
    dep_check_source_data_bronze_feature = BashOperator(
        task_id='check_source_data_bronze_feature',
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 check_bronze_feature.py'
        ),
    )

    dep_check_source_data_bronze_meta = BashOperator(
        task_id='check_source_data_bronze_meta',
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 check_bronze_meta.py'
        ),
    )


    bronze_feature = BashOperator(
        task_id='run_bronze_feature',
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 bronze_feature.py '
            '--snapshotdate "{{ ds }}"'
        ),
    )
    
    bronze_meta = BashOperator(
        task_id='run_bronze_meta',
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 bronze_meta.py '
            '--snapshotdate "{{ ds }}"'
        ),
    )

    silver_feature = BashOperator(
        task_id='run_silver_feature',
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 silver_feature.py '
            '--snapshotdate "{{ ds }}"'
        ),
    )
    
    silver_meta = BashOperator(
        task_id='run_silver_meta',
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 silver_meta.py '
            '--snapshotdate "{{ ds }}"'
        ),
    )


    gold_feature_store = BashOperator(
        task_id='run_gold_feature_store',
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 gold_feature_store.py '
            '--snapshotdate "{{ ds }}"'
        ),
    )

    feature_store_completed = BashOperator(
        task_id='feature_store_completed',
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 check_gold_feature.py'
        ),
    )
    
    
    # Define task dependencies to run scripts sequentially
    dep_check_source_data_bronze_feature >> bronze_feature >> silver_feature >> gold_feature_store
    dep_check_source_data_bronze_meta >> bronze_meta >> silver_meta >> gold_feature_store
    gold_feature_store >> feature_store_completed


    # --- model inference ---
    model_inference_start = EmptyOperator(task_id="model_inference_start")

    model_1_inference = BashOperator(
        task_id='xgb_inference',
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 model_inference_xgb.py '
            '--snapshotdate "{{ ds }}" '
            '--modelname "xgb_model_pipeline.pkl"'
        ),
    )

    
    
    model_inference_completed = EmptyOperator(task_id="model_inference_completed")
    
    # Define task dependencies to run scripts sequentially
    feature_store_completed >> model_inference_start
    model_inference_start >> model_1_inference >> model_inference_completed


    # --- model monitoring ---
    model_monitor_start = EmptyOperator(task_id="model_monitor_start")

    model_1_monitor = BashOperator(
        task_id='xgb_monitor',
        bash_command=(
            'cd /opt/airflow && '
            'python3 /opt/airflow/monitor.py'
        ),
    )


    model_monitor_completed = EmptyOperator(task_id="model_monitor_completed")
    
    # Define task dependencies to run scripts sequentially
    model_inference_completed >> model_monitor_start
    model_monitor_start >> model_1_monitor >> model_monitor_completed


    # --- model auto training ---

    model_automl_start = EmptyOperator(task_id="model_automl_start")
    

    model_1_automl = BashOperator(
        task_id='automl_xgboost',
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 train_xgboost_model.py '
            '--model_train_date "{{ ds }}"'
        ),
    )

    model_automl_completed = EmptyOperator(task_id="model_automl_completed")
    
    # Define task dependencies to run scripts sequentially
    feature_store_completed >> model_automl_start
    label_store_completed >> model_automl_start
    model_automl_start >> model_1_automl >> model_automl_completed