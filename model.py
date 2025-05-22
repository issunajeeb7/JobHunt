import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

MODEL_PATH = 'data/model.pkl'
VECTORIZER_PATH = 'data/vectorizer.pkl'
CLUSTERED_JOBS_PATH = 'data/jobs_clustered.csv'
ELBOW_PLOT_PATH = 'data/elbow_plot.png'


def preprocess_skills(skills_series):
    # Lowercase, remove punctuation, etc. (simple version)
    return skills_series.fillna('').str.lower()


def find_optimal_clusters(X, max_clusters=15):
    """
    Find the optimal number of clusters using the elbow method.
    Returns the optimal number of clusters and saves an elbow plot.
    """
    distortions = []
    silhouette_scores = []
    K = range(2, max_clusters + 1)
    
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        distortions.append(kmeans.inertia_)
        
        # Calculate silhouette score
        if k > 1:  # Silhouette score requires at least 2 clusters
            labels = kmeans.labels_
            silhouette_scores.append(silhouette_score(X, labels))
    
    # Plot elbow curve
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('Elbow Method')
    
    # Plot silhouette scores
    plt.subplot(1, 2, 2)
    plt.plot(K[1:], silhouette_scores, 'rx-')
    plt.xlabel('k')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Analysis')
    
    plt.tight_layout()
    os.makedirs('data', exist_ok=True)
    plt.savefig(ELBOW_PLOT_PATH)
    plt.close()
    
    # Find the elbow point using the second derivative
    distortions = np.array(distortions)
    second_derivative = np.gradient(np.gradient(distortions))
    optimal_k = K[np.argmax(second_derivative)]
    
    return optimal_k


def cluster_jobs(jobs_csv_path, n_clusters=None):
    df = pd.read_csv(jobs_csv_path)
    skills = preprocess_skills(df['Skills'])
    vectorizer = TfidfVectorizer(token_pattern=r'[^,;]+')
    X = vectorizer.fit_transform(skills)
    
    # Find optimal number of clusters if not specified
    if n_clusters is None:
        n_clusters = find_optimal_clusters(X)
        print(f"Optimal number of clusters determined: {n_clusters}")
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X)
    df['Cluster'] = clusters

    # Save model and vectorizer
    os.makedirs('data', exist_ok=True)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(kmeans, f)
    with open(VECTORIZER_PATH, 'wb') as f:
        pickle.dump(vectorizer, f)
    df.to_csv(CLUSTERED_JOBS_PATH, index=False)
    return df


def load_model():
    with open(MODEL_PATH, 'rb') as f:
        kmeans = pickle.load(f)
    with open(VECTORIZER_PATH, 'rb') as f:
        vectorizer = pickle.load(f)
    return kmeans, vectorizer


def assign_cluster_to_new_jobs(jobs_df):
    kmeans, vectorizer = load_model()
    skills = preprocess_skills(jobs_df['Skills'])
    X = vectorizer.transform(skills)
    clusters = kmeans.predict(X)
    jobs_df['Cluster'] = clusters
    return jobs_df 