import pandas as pd
import pickle
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from kneed import KneeLocator

def load_data():
    """
    Loads data from a CSV file, serializes it, and returns the serialized data.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, '..', 'data', 'file.csv')
    
    df = pd.read_csv(data_path)
    print(f"Data loaded. Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    serialized_data = pickle.dumps(df)
    return serialized_data


def data_preprocessing(data):
    """
    Deserializes data, performs data preprocessing, and returns serialized clustered data.
    """
    df = pickle.loads(data)
    print(f"Data deserialized. Shape: {df.shape}")
    
    # Select numeric columns only
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    print(f"Numeric columns: {numeric_df.columns.tolist()}")
    
    # Drop missing values
    numeric_df = numeric_df.dropna()
    
    # Standardize features
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_df)
    print(f"Data scaled. Shape: {scaled_data.shape}")
    
    serialized_data = pickle.dumps(scaled_data)
    return serialized_data


def build_save_model(data, filename):
    """
    Builds a K-Means clustering model, saves it to a file, and returns SSE values.
    """
    scaled_data = pickle.loads(data)
    print(f"Building model. Data shape: {scaled_data.shape}")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(script_dir, '..', 'model')
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, filename)
    
    # Calculate SSE for k=1 to 10
    sse_values = []
    k_range = range(1, 11)
    models = {}
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(scaled_data)
        sse_values.append(kmeans.inertia_)
        models[k] = kmeans
        print(f"K={k}, SSE={kmeans.inertia_:.2f}")
    
    # Save models
    with open(model_path, 'wb') as f:
        pickle.dump(models, f)
    print(f"Models saved to {model_path}")
    
    serialized_sse = pickle.dumps(sse_values)
    return serialized_sse


def load_model_elbow(filename, sse):
    """
    Loads a saved K-Means clustering model and determines the number of clusters using the elbow method.
    """
    sse_values = pickle.loads(sse)
    print(f"SSE values: {sse_values}")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, '..', 'model', filename)
    
    with open(model_path, 'rb') as f:
        models = pickle.load(f)
    print(f"Models loaded from {model_path}")
    
    # Find elbow point
    k_range = range(1, 11)
    kneedle = KneeLocator(
        list(k_range), 
        sse_values, 
        curve='convex', 
        direction='decreasing'
    )
    
    optimal_k = kneedle.elbow
    print(f"\n{'='*50}")
    print(f"OPTIMAL NUMBER OF CLUSTERS: {optimal_k}")
    print(f"{'='*50}\n")
    
    return optimal_k


# ============================================================
# NEW FEATURE 1: Save Elbow Plot Visualization
# ============================================================
def save_elbow_plot(sse, optimal_k):
    """
    Creates and saves an elbow plot showing SSE vs K with the optimal point marked.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    sse_values = pickle.loads(sse)
    k_range = list(range(1, 11))
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, '..', 'output')
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, 'elbow_plot.png')
    
    # Create simple plot
    plt.figure(figsize=(8, 5))
    plt.plot(k_range, sse_values, 'b-o')
    plt.axvline(x=optimal_k, color='r', linestyle='--', label=f'Optimal K={optimal_k}')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('SSE')
    plt.title('Elbow Method')
    plt.legend()
    plt.savefig(plot_path)
    plt.close()
    
    print(f"\n{'='*50}")
    print(f"ELBOW PLOT SAVED TO: {plot_path}")
    print(f"{'='*50}\n")
    
    return plot_path

# ============================================================
# NEW FEATURE 2: Save Cluster Results to CSV
# ============================================================
def save_cluster_results(data, filename, optimal_k):
    """
    Assigns each data point to a cluster and saves results to CSV.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load original data
    data_path = os.path.join(script_dir, '..', 'data', 'file.csv')
    original_df = pd.read_csv(data_path)
    
    # Load scaled data
    scaled_data = pickle.loads(data)
    
    # Load the optimal model
    model_path = os.path.join(script_dir, '..', 'model', filename)
    with open(model_path, 'rb') as f:
        models = pickle.load(f)
    
    optimal_model = models[optimal_k]
    
    # Predict clusters
    clusters = optimal_model.predict(scaled_data)
    
    # Add cluster column to original data
    original_df['cluster'] = clusters
    
    # Save to CSV
    output_dir = os.path.join(script_dir, '..', 'output')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'clustered_customers.csv')
    original_df.to_csv(output_path, index=False)
    
    print(f"\n{'='*50}")
    print(f"CLUSTER RESULTS SAVED TO: {output_path}")
    print(f"Total customers: {len(original_df)}")
    print(f"Clusters assigned: {original_df['cluster'].nunique()}")
    print(f"{'='*50}\n")
    
    return output_path


# ============================================================
# NEW FEATURE 3: Print Cluster Statistics
# ============================================================
def print_cluster_statistics(data, filename, optimal_k):
    """
    Prints detailed statistics about each cluster.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load original data
    data_path = os.path.join(script_dir, '..', 'data', 'file.csv')
    original_df = pd.read_csv(data_path)
    
    # Load scaled data
    scaled_data = pickle.loads(data)
    
    # Load the optimal model
    model_path = os.path.join(script_dir, '..', 'model', filename)
    with open(model_path, 'rb') as f:
        models = pickle.load(f)
    
    optimal_model = models[optimal_k]
    
    # Predict clusters
    clusters = optimal_model.predict(scaled_data)
    original_df['cluster'] = clusters
    
    print(f"\n{'='*60}")
    print(f"CLUSTER STATISTICS (K={optimal_k})")
    print(f"{'='*60}\n")
    
    # Get numeric columns automatically (exclude cluster column)
    numeric_cols = original_df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    if 'cluster' in numeric_cols:
        numeric_cols.remove('cluster')
    
    for cluster_id in range(optimal_k):
        cluster_data = original_df[original_df['cluster'] == cluster_id]
        print(f"--- CLUSTER {cluster_id} ---")
        print(f"    Size: {len(cluster_data)} customers ({len(cluster_data)/len(original_df)*100:.1f}%)")
        
        for col in numeric_cols:
            print(f"    Avg {col}: {cluster_data[col].mean():.2f}")
        print()
    
    print(f"{'='*60}\n")
    
    return "Statistics printed successfully"


# ============================================================
# NEW FEATURE 4: Calculate Silhouette Score
# ============================================================
def calculate_silhouette_score(data, filename, optimal_k):
    """
    Calculates the silhouette score for the optimal clustering.
    """
    from sklearn.metrics import silhouette_score as sklearn_silhouette_score
    
    scaled_data = pickle.loads(data)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, '..', 'model', filename)
    
    with open(model_path, 'rb') as f:
        models = pickle.load(f)
    
    optimal_model = models[optimal_k]
    clusters = optimal_model.predict(scaled_data)
    
    # Calculate silhouette score
    score = sklearn_silhouette_score(scaled_data, clusters)
    
    print(f"\n{'='*60}")
    print(f"SILHOUETTE SCORE ANALYSIS")
    print(f"{'='*60}")
    print(f"Optimal K: {optimal_k}")
    print(f"Silhouette Score: {score:.4f}")
    print()
    
    # Interpret the score
    if score >= 0.7:
        interpretation = "EXCELLENT - Clusters are well defined"
    elif score >= 0.5:
        interpretation = "GOOD - Clusters have reasonable structure"
    elif score >= 0.25:
        interpretation = "FAIR - Clusters are overlapping but meaningful"
    else:
        interpretation = "WEAK - Clusters may not be well separated"
    
    print(f"Interpretation: {interpretation}")
    print(f"{'='*60}\n")
    
    return score