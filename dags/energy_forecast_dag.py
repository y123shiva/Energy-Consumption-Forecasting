from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

with DAG(
    dag_id="energy_forecasting",
    start_date=datetime(2024,1,1),
    schedule_interval="@daily",
    catchup=False
) as dag:

    train = BashOperator(
        task_id="train_model",
        bash_command="python train.py"
    )
