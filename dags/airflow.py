from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
from src.lab import (
    load_data, 
    data_preprocessing, 
    build_save_model, 
    load_model_elbow,
    save_elbow_plot,           # NEW FEATURE 1
    save_cluster_results,      # NEW FEATURE 2
    print_cluster_statistics,  # NEW FEATURE 3
    calculate_silhouette_score # NEW FEATURE 4
)
from airflow import configuration as conf

# Enable pickle support for XCom
conf.set('core', 'enable_xcom_pickling', 'True')

# Default arguments
default_args = {
    'owner': 'sanjana',
    'start_date': datetime(2023, 9, 17),
    'retries': 0,
    'retry_delay': timedelta(minutes=5),
}

# Create DAG
dag = DAG(
    'Airflow_Lab1',
    default_args=default_args,
    description='K-Means Clustering Pipeline with Advanced Analytics',
    schedule_interval=None,
    catchup=False,
)

# Task 1: Load data
load_data_task = PythonOperator(
    task_id='load_data_task',
    python_callable=load_data,
    dag=dag,
)

# Task 2: Preprocess data
data_preprocessing_task = PythonOperator(
    task_id='data_preprocessing_task',
    python_callable=data_preprocessing,
    op_args=[load_data_task.output],
    dag=dag,
)

# Task 3: Build and save model
build_save_model_task = PythonOperator(
    task_id='build_save_model_task',
    python_callable=build_save_model,
    op_args=[data_preprocessing_task.output, "model.sav"],
    provide_context=True,
    dag=dag,
)

# Task 4: Load model and find optimal clusters
load_model_task = PythonOperator(
    task_id='load_model_task',
    python_callable=load_model_elbow,
    op_args=["model.sav", build_save_model_task.output],
    dag=dag,
)

# ============================================================
# NEW FEATURE 1: Save Elbow Plot
# ============================================================
save_plot_task = PythonOperator(
    task_id='save_elbow_plot_task',
    python_callable=save_elbow_plot,
    op_args=[build_save_model_task.output, load_model_task.output],
    dag=dag,
)

# ============================================================
# NEW FEATURE 2: Save Cluster Results to CSV
# ============================================================
save_results_task = PythonOperator(
    task_id='save_cluster_results_task',
    python_callable=save_cluster_results,
    op_args=[data_preprocessing_task.output, "model.sav", load_model_task.output],
    dag=dag,
)

# ============================================================
# NEW FEATURE 3: Print Cluster Statistics
# ============================================================
print_stats_task = PythonOperator(
    task_id='print_cluster_statistics_task',
    python_callable=print_cluster_statistics,
    op_args=[data_preprocessing_task.output, "model.sav", load_model_task.output],
    dag=dag,
)

# ============================================================
# NEW FEATURE 4: Calculate Silhouette Score
# ============================================================
silhouette_task = PythonOperator(
    task_id='calculate_silhouette_score_task',
    python_callable=calculate_silhouette_score,
    op_args=[data_preprocessing_task.output, "model.sav", load_model_task.output],
    dag=dag,
)

# Set dependencies
# Original flow
load_data_task >> data_preprocessing_task >> build_save_model_task >> load_model_task

# New features run in parallel after load_model_task
load_model_task >> [save_plot_task, save_results_task, print_stats_task, silhouette_task]

if __name__ == "__main__":
    dag.cli()