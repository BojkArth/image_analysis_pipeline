"""
Similarity Analysis Module

Performs comprehensive similarity analysis on image features including
dimensionality reduction, clustering, and similarity metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import logging
from pathlib import Path

# Machine learning and analysis
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.neighbors import NearestNeighbors
import umap

# Statistical analysis
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.stats import pearsonr, spearmanr

logger = logging.getLogger(__name__)


class SimilarityAnalyzer:
    """
    Comprehensive similarity analyzer for image features.
    
    Performs:
    - Dimensionality reduction (PCA, t-SNE, UMAP)
    - Clustering analysis (K-means, hierarchical, DBSCAN)
    - Similarity matrix calculation
    - Distance-based analysis
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the similarity analyzer.
        
        Parameters:
        -----------
        **kwargs : dict
            Configuration parameters for similarity analysis
        """
        self.config = {
            'distance_metrics': ['cosine', 'euclidean', 'correlation'],
            'clustering_methods': ['kmeans', 'hierarchical'],
            'n_clusters': 5,
            'pca_n_components': 50,
            'tsne_n_components': 2,
            'tsne_perplexity': 30,
            'umap_n_components': 2,
            'umap_n_neighbors': 15,
            'umap_min_dist': 0.1,
            'random_state': 42,
            **kwargs
        }
        
        # Initialize scaler
        self.scaler = StandardScaler()
        self.feature_scaler = None
    
    def analyze(self, features_df: pd.DataFrame) -> Dict:
        """
        Perform comprehensive similarity analysis on feature matrix.
        
        Parameters:
        -----------
        features_df : pd.DataFrame
            Feature matrix with images as rows and features as columns
            
        Returns:
        --------
        dict
            Dictionary containing similarity matrices, embeddings, and clustering results
        """
        logger.info("Starting similarity analysis...")
        
        # Remove non-numeric columns
        numeric_features = features_df.select_dtypes(include=[np.number])
        non_numeric_cols = features_df.select_dtypes(exclude=[np.number])
        
        # Scale features
        scaled_features = self.scaler.fit_transform(numeric_features)
        self.feature_scaler = self.scaler
        
        # Calculate similarity matrices
        similarity_matrices = self._calculate_similarity_matrices(scaled_features, features_df.index)
        
        # Perform dimensionality reduction
        embeddings = self._perform_dimensionality_reduction(scaled_features)
        
        # Perform clustering
        clustering_results = self._perform_clustering(scaled_features, embeddings)
        
        # Calculate additional statistics
        statistics = self._calculate_statistics(scaled_features, similarity_matrices)
        
        results = {
            'similarity_matrix': similarity_matrices['cosine'],  # Default to cosine
            'similarity_matrices': similarity_matrices,
            'embeddings': embeddings,
            'clustering': clustering_results,
            'statistics': statistics,
            'feature_names': numeric_features.columns.tolist(),
            'non_numeric_features': non_numeric_cols.to_dict('index')
        }
        
        logger.info("Similarity analysis complete")
        return results
    
    def _calculate_similarity_matrices(self, features: np.ndarray, index: pd.Index) -> Dict:
        """Calculate similarity matrices using different distance metrics."""
        similarity_matrices = {}
        
        for metric in self.config['distance_metrics']:
            try:
                if metric == 'cosine':
                    # Cosine similarity (higher = more similar)
                    sim_matrix = cosine_similarity(features)
                elif metric == 'euclidean':
                    # Euclidean distance (lower = more similar), convert to similarity
                    dist_matrix = euclidean_distances(features)
                    sim_matrix = 1 / (1 + dist_matrix)  # Convert to similarity
                elif metric == 'correlation':
                    # Correlation distance (lower = more similar), convert to similarity
                    # Calculate correlation distance using scipy
                    dist_matrix = squareform(pdist(features, metric='correlation'))
                    sim_matrix = 1 - dist_matrix  # Convert to similarity
                else:
                    logger.warning(f"Unknown distance metric: {metric}")
                    continue
                
                # Create DataFrame
                sim_df = pd.DataFrame(sim_matrix, index=index, columns=index)
                similarity_matrices[metric] = sim_df
                
                logger.info(f"Calculated {metric} similarity matrix")
                
            except Exception as e:
                logger.error(f"Error calculating {metric} similarity matrix: {e}")
        
        return similarity_matrices
    
    def _perform_dimensionality_reduction(self, features: np.ndarray) -> Dict:
        """Perform dimensionality reduction using multiple methods."""
        embeddings = {}
        
        # PCA
        try:
            # Ensure n_components doesn't exceed min(n_samples, n_features)
            max_components = min(self.config['pca_n_components'], features.shape[0], features.shape[1])
            pca = PCA(n_components=max_components)
            pca_embeddings = pca.fit_transform(features)
            embeddings['pca'] = {
                'embeddings': pca_embeddings,
                'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
                'cumulative_variance_ratio': np.cumsum(pca.explained_variance_ratio_).tolist(),
                'n_components': pca.n_components_
            }
            logger.info(f"PCA completed with {pca.n_components_} components")
        except Exception as e:
            logger.error(f"Error in PCA: {e}")
        
        # t-SNE
        try:
            tsne = TSNE(
                n_components=self.config['tsne_n_components'],
                perplexity=self.config['tsne_perplexity'],
                random_state=self.config['random_state']
            )
            tsne_embeddings = tsne.fit_transform(features)
            embeddings['tsne'] = {
                'embeddings': tsne_embeddings,
                'perplexity': self.config['tsne_perplexity']
            }
            logger.info("t-SNE completed")
        except Exception as e:
            logger.error(f"Error in t-SNE: {e}")
        
        # UMAP
        try:
            umap_reducer = umap.UMAP(
                n_components=self.config['umap_n_components'],
                n_neighbors=self.config['umap_n_neighbors'],
                min_dist=self.config['umap_min_dist'],
                random_state=self.config['random_state']
            )
            umap_embeddings = umap_reducer.fit_transform(features)
            embeddings['umap'] = {
                'embeddings': umap_embeddings,
                'n_neighbors': self.config['umap_n_neighbors'],
                'min_dist': self.config['umap_min_dist']
            }
            logger.info("UMAP completed")
        except Exception as e:
            logger.error(f"Error in UMAP: {e}")
        
        return embeddings
    
    def _perform_clustering(self, features: np.ndarray, embeddings: Dict) -> Dict:
        """Perform clustering analysis using multiple methods."""
        clustering_results = {}
        
        for method in self.config['clustering_methods']:
            try:
                if method == 'kmeans':
                    kmeans_results = self._kmeans_clustering(features, embeddings)
                    clustering_results['kmeans'] = kmeans_results
                    
                elif method == 'hierarchical':
                    hierarchical_results = self._hierarchical_clustering(features, embeddings)
                    clustering_results['hierarchical'] = hierarchical_results
                    
                elif method == 'dbscan':
                    dbscan_results = self._dbscan_clustering(features, embeddings)
                    clustering_results['dbscan'] = dbscan_results
                    
                logger.info(f"{method.capitalize()} clustering completed")
                
            except Exception as e:
                logger.error(f"Error in {method} clustering: {e}")
        
        return clustering_results
    
    def _kmeans_clustering(self, features: np.ndarray, embeddings: Dict) -> Dict:
        """Perform K-means clustering."""
        results = {}
        
        # Try different numbers of clusters
        n_clusters_range = range(2, min(11, features.shape[0] // 2))
        
        for n_clusters in n_clusters_range:
            try:
                kmeans = KMeans(
                    n_clusters=n_clusters,
                    random_state=self.config['random_state'],
                    n_init=10
                )
                labels = kmeans.fit_predict(features)
                
                # Calculate clustering metrics
                if len(np.unique(labels)) > 1:
                    silhouette = silhouette_score(features, labels)
                    calinski = calinski_harabasz_score(features, labels)
                    davies = davies_bouldin_score(features, labels)
                else:
                    silhouette = calinski = davies = 0
                
                results[f'n_clusters_{n_clusters}'] = {
                    'labels': labels,
                    'centroids': kmeans.cluster_centers_,
                    'inertia': kmeans.inertia_,
                    'silhouette_score': silhouette,
                    'calinski_harabasz_score': calinski,
                    'davies_bouldin_score': davies
                }
                
            except Exception as e:
                logger.warning(f"Error in K-means with {n_clusters} clusters: {e}")
        
        # Store optimal clustering (based on silhouette score)
        if results:
            optimal_n = max(results.keys(), 
                          key=lambda x: results[x]['silhouette_score'])
            results['optimal'] = results[optimal_n]
            results['optimal_n_clusters'] = int(optimal_n.split('_')[-1])
        
        return results
    
    def _hierarchical_clustering(self, features: np.ndarray, embeddings: Dict) -> Dict:
        """Perform hierarchical clustering."""
        results = {}
        
        # Calculate linkage matrix
        try:
            linkage_matrix = linkage(features, method='ward')
            results['linkage_matrix'] = linkage_matrix
            
            # Try different numbers of clusters
            n_clusters_range = range(2, min(11, features.shape[0] // 2))
            
            for n_clusters in n_clusters_range:
                try:
                    labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
                    
                    # Calculate clustering metrics
                    if len(np.unique(labels)) > 1:
                        silhouette = silhouette_score(features, labels)
                        calinski = calinski_harabasz_score(features, labels)
                        davies = davies_bouldin_score(features, labels)
                    else:
                        silhouette = calinski = davies = 0
                    
                    results[f'n_clusters_{n_clusters}'] = {
                        'labels': labels,
                        'silhouette_score': silhouette,
                        'calinski_harabasz_score': calinski,
                        'davies_bouldin_score': davies
                    }
                    
                except Exception as e:
                    logger.warning(f"Error in hierarchical clustering with {n_clusters} clusters: {e}")
            
            # Store optimal clustering
            if results:
                optimal_n = max([k for k in results.keys() if k.startswith('n_clusters_')], 
                              key=lambda x: results[x]['silhouette_score'])
                results['optimal'] = results[optimal_n]
                results['optimal_n_clusters'] = int(optimal_n.split('_')[-1])
        
        except Exception as e:
            logger.error(f"Error in hierarchical clustering: {e}")
        
        return results
    
    def _dbscan_clustering(self, features: np.ndarray, embeddings: Dict) -> Dict:
        """Perform DBSCAN clustering."""
        results = {}
        
        # Try different epsilon values
        eps_range = [0.1, 0.5, 1.0, 2.0, 5.0]
        
        for eps in eps_range:
            try:
                dbscan = DBSCAN(eps=eps, min_samples=5)
                labels = dbscan.fit_predict(features)
                
                # Calculate clustering metrics (only if more than one cluster)
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                
                if n_clusters > 1:
                    # Remove noise points for metric calculation
                    non_noise_mask = labels != -1
                    if np.sum(non_noise_mask) > 1:
                        silhouette = silhouette_score(features[non_noise_mask], labels[non_noise_mask])
                        calinski = calinski_harabasz_score(features[non_noise_mask], labels[non_noise_mask])
                        davies = davies_bouldin_score(features[non_noise_mask], labels[non_noise_mask])
                    else:
                        silhouette = calinski = davies = 0
                else:
                    silhouette = calinski = davies = 0
                
                results[f'eps_{eps}'] = {
                    'labels': labels,
                    'n_clusters': n_clusters,
                    'n_noise': np.sum(labels == -1),
                    'silhouette_score': silhouette,
                    'calinski_harabasz_score': calinski,
                    'davies_bouldin_score': davies
                }
                
            except Exception as e:
                logger.warning(f"Error in DBSCAN with eps={eps}: {e}")
        
        return results
    
    def _calculate_statistics(self, features: np.ndarray, similarity_matrices: Dict) -> Dict:
        """Calculate additional statistics for the analysis."""
        statistics = {}
        
        # Feature statistics
        statistics['feature_stats'] = {
            'n_features': features.shape[1],
            'n_samples': features.shape[0],
            'feature_means': np.mean(features, axis=0).tolist(),
            'feature_stds': np.std(features, axis=0).tolist(),
            'feature_ranges': (np.max(features, axis=0) - np.min(features, axis=0)).tolist()
        }
        
        # Similarity statistics
        for metric, sim_matrix in similarity_matrices.items():
            sim_values = sim_matrix.values[np.triu_indices_from(sim_matrix.values, k=1)]
            statistics[f'{metric}_similarity_stats'] = {
                'mean': np.mean(sim_values),
                'std': np.std(sim_values),
                'min': np.min(sim_values),
                'max': np.max(sim_values),
                'median': np.median(sim_values)
            }
        
        # Distance distribution
        try:
            distances = pdist(features)
            statistics['distance_stats'] = {
                'mean_distance': np.mean(distances),
                'std_distance': np.std(distances),
                'min_distance': np.min(distances),
                'max_distance': np.max(distances),
                'median_distance': np.median(distances)
            }
        except Exception as e:
            logger.warning(f"Could not calculate distance statistics: {e}")
        
        return statistics
    
    def get_nearest_neighbors(self, features: np.ndarray, n_neighbors: int = 5) -> Dict:
        """Find nearest neighbors for each sample."""
        try:
            nn = NearestNeighbors(n_neighbors=n_neighbors + 1)  # +1 to exclude self
            nn.fit(features)
            distances, indices = nn.kneighbors(features)
            
            # Remove self from results
            distances = distances[:, 1:]
            indices = indices[:, 1:]
            
            return {
                'distances': distances,
                'indices': indices,
                'n_neighbors': n_neighbors
            }
        except Exception as e:
            logger.error(f"Error finding nearest neighbors: {e}")
            return {}
    
    def calculate_feature_importance(self, features: np.ndarray, labels: np.ndarray) -> Dict:
        """Calculate feature importance based on clustering."""
        try:
            from sklearn.ensemble import RandomForestClassifier
            
            # Use random forest to assess feature importance
            rf = RandomForestClassifier(n_estimators=100, random_state=self.config['random_state'])
            rf.fit(features, labels)
            
            return {
                'feature_importance': rf.feature_importances_.tolist(),
                'feature_importance_ranked': sorted(
                    enumerate(rf.feature_importances_), 
                    key=lambda x: x[1], 
                    reverse=True
                )
            }
        except Exception as e:
            logger.error(f"Error calculating feature importance: {e}")
            return {} 