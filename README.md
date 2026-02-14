# Airflow Lab 1: K-Means Clustering Pipeline

**Course:** MLOps - Northeastern University  
**Author:** Sanjana Menon

## Project Overview

This project builds an automated machine learning workflow using Apache Airflow. The pipeline performs K-Means clustering on customer data and finds the optimal number of clusters using the Elbow Method.

Everything runs inside Docker containers, so you don't need to install Airflow directly on your computer. This makes it easy to run on Windows, Mac, or Linux.

---

## What Does This Pipeline Do?

The workflow has 4 tasks that run one after another:

1. **load_data_task** - Loads customer data from a CSV file (100 rows, 5 columns)
2. **data_preprocessing_task** - Cleans the data and scales the features using StandardScaler
3. **build_save_model_task** - Trains K-Means models for k=1 to k=10 and saves them
4. **load_model_task** - Uses the Elbow Method to find the best number of clusters

**Result:** The pipeline found that **4 clusters** is the optimal number for this dataset.


## Project Structure

airflow-lab1/
├── dags/
│   ├── data/
│   │   ├── file.csv          # Main dataset (100 customers)
│   │   └── test.csv          # Test dataset
│   ├── model/
│   │   └── model.sav         # Saved K-Means models (created after running)
│   ├── src/
│   │   ├── __init__.py       # Makes src a Python package
│   │   └── lab.py            # Contains all ML functions
│   └── airflow.py            # DAG definition file
├── config/                    # Airflow config (auto-generated)
├── logs/                      # Airflow logs (auto-generated)
├── plugins/                   # Airflow plugins (auto-generated)
├── .env                       # Environment variables
├── .gitignore                 # Files to ignore in Git
├── docker-compose.yaml        # Docker configuration for Airflow
└── README.md                  # This file


## Screenshots

### DAG Graph View - All Tasks Successful

![DAG Graph](images/dag_graph.png)


This screenshot shows the Airflow Graph view after the pipeline finished running. You can see all 4 tasks connected in sequence, and each one shows a green "success" status. The tasks flow from left to right: load_data_task → data_preprocessing_task → build_save_model_task → load_model_task.

### Task Logs - Optimal Clusters Found

![Task Logs](images/task_logs.png)

This screenshot shows the logs from the final task (load_model_task). You can see:
- The SSE (Sum of Squared Errors) values for each k from 1 to 10
- The model was loaded successfully from the saved file
- The final result: **OPTIMAL NUMBER OF CLUSTERS: 4**


## How the Elbow Method Works

The Elbow Method helps us find the best number of clusters:

1. We train K-Means with different values of k (1 to 10)
2. For each k, we calculate the SSE (how spread out the points are from their cluster centers)
3. We plot SSE vs k and look for the "elbow" - where adding more clusters stops helping much
4. The KneeLocator library automatically finds this elbow point

In our case, the elbow was at k=4, meaning 4 clusters best represents our customer data.


## Prerequisites

Before running this project, you need:

1. **Docker Desktop** installed and running
   - [Download for Windows](https://docs.docker.com/desktop/install/windows-install/)
   - [Download for Mac](https://docs.docker.com/desktop/install/mac-install/)
   - [Download for Linux](https://docs.docker.com/desktop/install/linux-install/)

2. At least **4GB of memory** allocated to Docker (I used 5GB)

### For Windows Users (WSL2)

If you're using Windows with WSL2, you need to create a `.wslconfig` file to allocate enough memory:

1. Create a file at `C:\Users\YourUsername\.wslconfig`
2. Add this content:
```
   [wsl2]
   memory=5GB
   processors=2
```
3. Run `wsl --shutdown` in Command Prompt
4. Restart Docker Desktop

---

## Setup Instructions

### Step 1: Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/airflow-lab1.git
cd airflow-lab1
```

### Step 2: Download docker-compose.yaml
```bash
# Mac/Linux
curl -LfO 'https://airflow.apache.org/docs/apache-airflow/2.9.2/docker-compose.yaml'

# Windows
curl -o docker-compose.yaml https://airflow.apache.org/docs/apache-airflow/2.9.2/docker-compose.yaml
```

### Step 3: Create the .env File
```bash
# Mac/Linux
echo -e "AIRFLOW_UID=$(id -u)" > .env

# Windows - create .env file with this content:
AIRFLOW_UID=50000
```

### Step 4: Edit docker-compose.yaml

Open the file and make these changes:
```yaml
# Don't load example DAGs
AIRFLOW__CORE__LOAD_EXAMPLES: 'false'

# Add required Python packages
_PIP_ADDITIONAL_REQUIREMENTS: ${_PIP_ADDITIONAL_REQUIREMENTS:- pandas scikit-learn kneed}

# Change login credentials
_AIRFLOW_WWW_USER_USERNAME: ${_AIRFLOW_WWW_USER_USERNAME:-airflow2}
_AIRFLOW_WWW_USER_PASSWORD: ${_AIRFLOW_WWW_USER_PASSWORD:-airflow2}
```

### Step 5: Initialize the Database
```bash
docker compose up airflow-init
```

Wait until you see "airflow-init exited with code 0"

### Step 6: Start Airflow
```bash
docker compose up
```

Wait until you see the health check message in the terminal.

### Step 7: Open Airflow UI

1. Go to http://localhost:8080 in your browser
2. Login with:
   - Username: `airflow2`
   - Password: `airflow2`

### Step 8: Run the DAG

1. Find "Airflow_Lab1" in the DAGs list
2. Toggle the switch to turn it ON
3. Click the play button to trigger the DAG
4. Watch the tasks turn green in the Graph view

### Step 9: Check the Results

1. Click on "load_model_task" in the Graph
2. Click on "Logs" tab
3. Look for "OPTIMAL NUMBER OF CLUSTERS: 4"

### Step 10: Stop Airflow
```bash
docker compose down
```

---

## Python Functions Explained

### load_data()
Reads the CSV file and converts it to a format that can be passed between Airflow tasks using pickle serialization.

### data_preprocessing(data)
Takes the raw data, selects only numeric columns, removes any missing values, and scales all features to have mean=0 and standard deviation=1.

### build_save_model(data, filename)
Trains 10 different K-Means models (k=1 to k=10), calculates the SSE for each, and saves all models to a file.

### load_model_elbow(filename, sse)
Loads the saved models and uses the KneeLocator library to automatically find the elbow point in the SSE curve.

---

## Technologies Used

- **Apache Airflow** - Workflow orchestration
- **Docker** - Containerization
- **Python** - Programming language
- **pandas** - Data manipulation
- **scikit-learn** - K-Means clustering and StandardScaler
- **kneed** - Elbow point detection

---

## Results

| Metric | Value |
|--------|-------|
| Dataset Size | 100 rows, 5 columns |
| K Values Tested | 1 to 10 |
| Optimal Clusters | 4 |
| Pipeline Status | Success ✅ |

---

## References

- [Apache Airflow Documentation](https://airflow.apache.org/docs/)
- [Running Airflow in Docker](https://airflow.apache.org/docs/apache-airflow/stable/howto/docker-compose/index.html)
- [Prof. Ramin Mohammadi's MLOps Course](https://www.mlwithramin.com/blog/airflow-lab1)
- [K-Means Clustering](https://scikit-learn.org/stable/modules/clustering.html#k-means)

---

## License

This project is for educational purposes as part of the MLOps course at Northeastern University.

## License

This project is for educational purposes as part of the MLOps course at Northeastern University.
