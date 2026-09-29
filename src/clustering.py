import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

def train_kmeans(X_scaled, n_clusters, random_state=42):
    """
    Train a KMeans model on standardized feature matrix X_scaled.
    
    Returns:
    - model: fitted KMeans object
    - labels: cluster assignments array
    - inertia: within-cluster sum of squares
    - silhouette: silhouette score metric
    """
    model = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init='auto'
    )
    labels = model.fit_predict(X_scaled)
    inertia = model.inertia_
    
    # Calculate Silhouette score if n_clusters >= 2 and sample size allows
    if n_clusters >= 2 and X_scaled.shape[0] > n_clusters:
        # Sample for speed if dataset is extremely large, but 4580 is small enough for full calculation
        sil_score = float(silhouette_score(X_scaled, labels))
    else:
        sil_score = 0.0
        
    return model, labels, inertia, sil_score

def evaluate_k_range(X_scaled, k_range=range(2, 11), random_state=42):
    """
    Evaluate K-Means across a range of K values to perform Elbow Method & Silhouette analysis.
    
    Returns:
    - k_results: pandas DataFrame containing [k, inertia, silhouette_score]
    - optimal_k: integer K with highest silhouette score
    """
    results = []
    
    for k in k_range:
        model = KMeans(
            n_clusters=k,
            random_state=random_state,
            n_init='auto'
        )
        labels = model.fit_predict(X_scaled)
        inertia = float(model.inertia_)
        sil = float(silhouette_score(X_scaled, labels))
        results.append({
            'k': k,
            'inertia': inertia,
            'silhouette_score': sil
        })
        
    k_df = pd.DataFrame(results)
    
    # Optimal K based on maximum Silhouette Score
    optimal_idx = k_df['silhouette_score'].idxmax()
    optimal_k = int(k_df.loc[optimal_idx, 'k'])
    
    return k_df, optimal_k

def compute_pca_2d(X_scaled, random_state=42):
    """
    Compute 2D PCA projection ONLY for visual representation of clusters.
    
    Returns:
    - pca_df: DataFrame with ['PCA1', 'PCA2']
    - pca_obj: fitted PCA object
    """
    pca = PCA(n_components=2, random_state=random_state)
    components = pca.fit_transform(X_scaled)
    
    pca_df = pd.DataFrame(components, columns=['PCA1', 'PCA2'])
    pca_df['explained_variance_ratio'] = pca.explained_variance_ratio_.sum()
    return pca_df, pca
