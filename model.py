import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import pickle
import os

MODEL_PATH = 'data/model.pkl'
VECTORIZER_PATH = 'data/vectorizer.pkl'
CLUSTERED_JOBS_PATH = 'data/jobs_clustered.csv'


def preprocess_skills(skills_series):
    # Lowercase, remove punctuation, etc. (simple version)
    return skills_series.fillna('').str.lower()


def cluster_jobs(jobs_csv_path, n_clusters=8):
    df = pd.read_csv(jobs_csv_path)
    skills = preprocess_skills(df['Skills'])
    vectorizer = TfidfVectorizer(token_pattern=r'[^,;]+')
    X = vectorizer.fit_transform(skills)
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