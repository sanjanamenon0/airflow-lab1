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