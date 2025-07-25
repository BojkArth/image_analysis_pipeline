"""
Visualization Module

Generates comprehensive visualizations for image analysis results including
clustermaps, UMAP plots, feature distributions, and similarity matrices.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
import logging
from pathlib import Path

# Visualization libraries
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

# Statistical plotting
from scipy.cluster.hierarchy import dendrogram
from sklearn.manifold import TSNE
import umap

logger = logging.getLogger(__name__)


class ImageVisualizer:
    """
    Comprehensive visualizer for image analysis results.
    
    Generates:
    - Clustermaps and heatmaps
    - UMAP and t-SNE plots
    - Feature distribution plots
    - Similarity matrix visualizations
    - Clustering results plots
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the visualizer.
        
        Parameters:
        -----------
        **kwargs : dict
            Configuration parameters for visualization
        """
        self.config = {
            'plot_style': 'seaborn',
            'figure_size': (12, 8),
            'dpi': 300,
            'color_palette': 'viridis',
            'max_images_per_plot': 20,
            'save_format': 'png',
            **kwargs
        }
        
        # Set up plotting style
        self._setup_plotting_style()
    
    def _setup_plotting_style(self):
        """Set up matplotlib and seaborn plotting styles."""
        plt.style.use('default')
        sns.set_theme(style="whitegrid")
        sns.set_palette(self.config['color_palette'])
        
        # Set default figure size
        plt.rcParams['figure.figsize'] = self.config['figure_size']
        plt.rcParams['figure.dpi'] = self.config['dpi']
        plt.rcParams['savefig.dpi'] = self.config['dpi']
    
    def create_all_plots(self, features_df: pd.DataFrame, similarity_matrix: pd.DataFrame,
                        embeddings: Dict, clustering_results: Dict, output_dir: Path) -> Dict:
        """
        Create all visualization plots.
        
        Parameters:
        -----------
        features_df : pd.DataFrame
            Feature matrix
        similarity_matrix : pd.DataFrame
            Similarity matrix
        embeddings : dict
            Dimensionality reduction results
        clustering_results : dict
            Clustering results
        output_dir : Path
            Output directory for plots
            
        Returns:
        --------
        dict
            Dictionary containing paths to generated plots
        """
        plots = {}
        
        # Create output directory for plots
        plots_dir = output_dir / 'plots'
        plots_dir.mkdir(exist_ok=True)
        
        # 1. Similarity matrix clustermap
        try:
            clustermap_path = self._create_clustermap(similarity_matrix, plots_dir)
            plots['clustermap'] = clustermap_path
        except Exception as e:
            logger.error(f"Error creating clustermap: {e}")
        
        # 2. UMAP visualization
        try:
            umap_path = self._create_umap_plot(embeddings, clustering_results, plots_dir)
            plots['umap'] = umap_path
        except Exception as e:
            logger.error(f"Error creating UMAP plot: {e}")
        
        # 3. t-SNE visualization
        try:
            tsne_path = self._create_tsne_plot(embeddings, clustering_results, plots_dir)
            plots['tsne'] = tsne_path
        except Exception as e:
            logger.error(f"Error creating t-SNE plot: {e}")
        
        # 4. Feature distribution plots
        try:
            feature_dist_path = self._create_feature_distribution_plots(features_df, plots_dir)
            plots['feature_distributions'] = feature_dist_path
        except Exception as e:
            logger.error(f"Error creating feature distribution plots: {e}")
        
        # 5. Clustering evaluation plots
        try:
            clustering_eval_path = self._create_clustering_evaluation_plots(clustering_results, plots_dir)
            plots['clustering_evaluation'] = clustering_eval_path
        except Exception as e:
            logger.error(f"Error creating clustering evaluation plots: {e}")
        
        # 6. Similarity matrix heatmap
        try:
            similarity_heatmap_path = self._create_similarity_heatmap(similarity_matrix, plots_dir)
            plots['similarity_heatmap'] = similarity_heatmap_path
        except Exception as e:
            logger.error(f"Error creating similarity heatmap: {e}")
        
        # 7. Feature correlation matrix
        try:
            correlation_matrix_path = self._create_feature_correlation_matrix(features_df, plots_dir)
            plots['feature_correlation'] = correlation_matrix_path
        except Exception as e:
            logger.error(f"Error creating feature correlation matrix: {e}")
        
        # 8. PCA explained variance plot
        try:
            pca_variance_path = self._create_pca_variance_plot(embeddings, plots_dir)
            plots['pca_variance'] = pca_variance_path
        except Exception as e:
            logger.error(f"Error creating PCA variance plot: {e}")
        
        logger.info(f"Generated {len(plots)} visualization plots")
        return plots
    
    def _create_clustermap(self, similarity_matrix: pd.DataFrame, output_dir: Path) -> str:
        """Create clustermap of similarity matrix."""
        fig, ax = plt.subplots(figsize=(15, 12))
        
        # Create clustermap
        g = sns.clustermap(
            similarity_matrix,
            cmap='viridis',
            center=0.5,
            square=True,
            xticklabels=True,
            yticklabels=True,
            figsize=(15, 12),
            cbar_kws={'label': 'Similarity'},
            dendrogram_ratio=(0.1, 0.1),
            colors_ratio=0.03
        )
        
        # Customize appearance
        g.ax_heatmap.set_title('Image Similarity Clustermap', fontsize=16, pad=20)
        g.ax_heatmap.set_xlabel('Images', fontsize=12)
        g.ax_heatmap.set_ylabel('Images', fontsize=12)
        
        # Rotate x-axis labels
        plt.setp(g.ax_heatmap.get_xticklabels(), rotation=45, ha='right')
        plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0)
        
        # Save plot
        output_path = output_dir / f'clustermap.{self.config["save_format"]}'
        plt.savefig(output_path, bbox_inches='tight', dpi=self.config['dpi'])
        plt.close()
        
        return str(output_path)
    
    def _create_umap_plot(self, embeddings: Dict, clustering_results: Dict, output_dir: Path) -> str:
        """Create UMAP visualization."""
        if 'umap' not in embeddings:
            raise ValueError("UMAP embeddings not found")
        
        umap_data = embeddings['umap']['embeddings']
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        
        # Plot 1: UMAP without clustering
        scatter1 = axes[0].scatter(umap_data[:, 0], umap_data[:, 1], 
                                 c=range(len(umap_data)), cmap='viridis', alpha=0.7)
        axes[0].set_title('UMAP Embedding', fontsize=14)
        axes[0].set_xlabel('UMAP 1', fontsize=12)
        axes[0].set_ylabel('UMAP 2', fontsize=12)
        plt.colorbar(scatter1, ax=axes[0], label='Sample Index')
        
        # Plot 2: UMAP with clustering (if available)
        if 'kmeans' in clustering_results and 'optimal' in clustering_results['kmeans']:
            labels = clustering_results['kmeans']['optimal']['labels']
            scatter2 = axes[1].scatter(umap_data[:, 0], umap_data[:, 1], 
                                     c=labels, cmap='tab10', alpha=0.7)
            axes[1].set_title('UMAP with K-means Clustering', fontsize=14)
            axes[1].set_xlabel('UMAP 1', fontsize=12)
            axes[1].set_ylabel('UMAP 2', fontsize=12)
            plt.colorbar(scatter2, ax=axes[1], label='Cluster')
        else:
            axes[1].text(0.5, 0.5, 'No clustering results available', 
                        ha='center', va='center', transform=axes[1].transAxes)
            axes[1].set_title('UMAP with Clustering', fontsize=14)
        
        plt.tight_layout()
        
        # Save plot
        output_path = output_dir / f'umap_visualization.{self.config["save_format"]}'
        plt.savefig(output_path, bbox_inches='tight', dpi=self.config['dpi'])
        plt.close()
        
        return str(output_path)
    
    def _create_tsne_plot(self, embeddings: Dict, clustering_results: Dict, output_dir: Path) -> str:
        """Create t-SNE visualization."""
        if 'tsne' not in embeddings:
            raise ValueError("t-SNE embeddings not found")
        
        tsne_data = embeddings['tsne']['embeddings']
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        
        # Plot 1: t-SNE without clustering
        scatter1 = axes[0].scatter(tsne_data[:, 0], tsne_data[:, 1], 
                                 c=range(len(tsne_data)), cmap='viridis', alpha=0.7)
        axes[0].set_title('t-SNE Embedding', fontsize=14)
        axes[0].set_xlabel('t-SNE 1', fontsize=12)
        axes[0].set_ylabel('t-SNE 2', fontsize=12)
        plt.colorbar(scatter1, ax=axes[0], label='Sample Index')
        
        # Plot 2: t-SNE with clustering (if available)
        if 'kmeans' in clustering_results and 'optimal' in clustering_results['kmeans']:
            labels = clustering_results['kmeans']['optimal']['labels']
            scatter2 = axes[1].scatter(tsne_data[:, 0], tsne_data[:, 1], 
                                     c=labels, cmap='tab10', alpha=0.7)
            axes[1].set_title('t-SNE with K-means Clustering', fontsize=14)
            axes[1].set_xlabel('t-SNE 1', fontsize=12)
            axes[1].set_ylabel('t-SNE 2', fontsize=12)
            plt.colorbar(scatter2, ax=axes[1], label='Cluster')
        else:
            axes[1].text(0.5, 0.5, 'No clustering results available', 
                        ha='center', va='center', transform=axes[1].transAxes)
            axes[1].set_title('t-SNE with Clustering', fontsize=14)
        
        plt.tight_layout()
        
        # Save plot
        output_path = output_dir / f'tsne_visualization.{self.config["save_format"]}'
        plt.savefig(output_path, bbox_inches='tight', dpi=self.config['dpi'])
        plt.close()
        
        return str(output_path)
    
    def _create_feature_distribution_plots(self, features_df: pd.DataFrame, output_dir: Path) -> str:
        """Create feature distribution plots."""
        # Select numeric features only
        numeric_features = features_df.select_dtypes(include=[np.number])
        
        # Create subplots for feature distributions
        n_features = min(12, len(numeric_features.columns))  # Limit to 12 features
        selected_features = numeric_features.columns[:n_features]
        
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        axes = axes.flatten()
        
        for i, feature in enumerate(selected_features):
            if i < len(axes):
                # Histogram
                axes[i].hist(numeric_features[feature], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
                axes[i].set_title(f'{feature}', fontsize=10)
                axes[i].set_xlabel('Value')
                axes[i].set_ylabel('Frequency')
                axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(selected_features), len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle('Feature Distributions', fontsize=16)
        plt.tight_layout()
        
        # Save plot
        output_path = output_dir / f'feature_distributions.{self.config["save_format"]}'
        plt.savefig(output_path, bbox_inches='tight', dpi=self.config['dpi'])
        plt.close()
        
        return str(output_path)
    
    def _create_clustering_evaluation_plots(self, clustering_results: Dict, output_dir: Path) -> str:
        """Create clustering evaluation plots."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Silhouette scores for different numbers of clusters (K-means)
        if 'kmeans' in clustering_results:
            kmeans_results = clustering_results['kmeans']
            n_clusters_list = []
            silhouette_scores = []
            
            for key, result in kmeans_results.items():
                if key.startswith('n_clusters_'):
                    n_clusters = int(key.split('_')[-1])
                    n_clusters_list.append(n_clusters)
                    silhouette_scores.append(result['silhouette_score'])
            
            if n_clusters_list:
                axes[0, 0].plot(n_clusters_list, silhouette_scores, 'bo-', linewidth=2, markersize=8)
                axes[0, 0].set_title('K-means: Silhouette Score vs Number of Clusters')
                axes[0, 0].set_xlabel('Number of Clusters')
                axes[0, 0].set_ylabel('Silhouette Score')
                axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Calinski-Harabasz scores for different numbers of clusters (K-means)
        if 'kmeans' in clustering_results:
            kmeans_results = clustering_results['kmeans']
            n_clusters_list = []
            calinski_scores = []
            
            for key, result in kmeans_results.items():
                if key.startswith('n_clusters_'):
                    n_clusters = int(key.split('_')[-1])
                    n_clusters_list.append(n_clusters)
                    calinski_scores.append(result['calinski_harabasz_score'])
            
            if n_clusters_list:
                axes[0, 1].plot(n_clusters_list, calinski_scores, 'ro-', linewidth=2, markersize=8)
                axes[0, 1].set_title('K-means: Calinski-Harabasz Score vs Number of Clusters')
                axes[0, 1].set_xlabel('Number of Clusters')
                axes[0, 1].set_ylabel('Calinski-Harabasz Score')
                axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Hierarchical clustering dendrogram
        if 'hierarchical' in clustering_results and 'linkage_matrix' in clustering_results['hierarchical']:
            linkage_matrix = clustering_results['hierarchical']['linkage_matrix']
            dendrogram(linkage_matrix, ax=axes[1, 0], orientation='top')
            axes[1, 0].set_title('Hierarchical Clustering Dendrogram')
            axes[1, 0].set_xlabel('Sample Index')
            axes[1, 0].set_ylabel('Distance')
        
        # Plot 4: DBSCAN clustering results
        if 'dbscan' in clustering_results:
            dbscan_results = clustering_results['dbscan']
            eps_values = []
            n_clusters_list = []
            
            for key, result in dbscan_results.items():
                if key.startswith('eps_'):
                    eps = float(key.split('_')[-1])
                    eps_values.append(eps)
                    n_clusters_list.append(result['n_clusters'])
            
            if eps_values:
                axes[1, 1].plot(eps_values, n_clusters_list, 'go-', linewidth=2, markersize=8)
                axes[1, 1].set_title('DBSCAN: Number of Clusters vs Epsilon')
                axes[1, 1].set_xlabel('Epsilon')
                axes[1, 1].set_ylabel('Number of Clusters')
                axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        output_path = output_dir / f'clustering_evaluation.{self.config["save_format"]}'
        plt.savefig(output_path, bbox_inches='tight', dpi=self.config['dpi'])
        plt.close()
        
        return str(output_path)
    
    def _create_similarity_heatmap(self, similarity_matrix: pd.DataFrame, output_dir: Path) -> str:
        """Create similarity matrix heatmap."""
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Create heatmap
        sns.heatmap(similarity_matrix, 
                   cmap='viridis', 
                   center=0.5,
                   square=True,
                   cbar_kws={'label': 'Similarity'},
                   ax=ax)
        
        ax.set_title('Image Similarity Matrix', fontsize=16, pad=20)
        ax.set_xlabel('Images', fontsize=12)
        ax.set_ylabel('Images', fontsize=12)
        
        # Rotate x-axis labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        # Save plot
        output_path = output_dir / f'similarity_heatmap.{self.config["save_format"]}'
        plt.savefig(output_path, bbox_inches='tight', dpi=self.config['dpi'])
        plt.close()
        
        return str(output_path)
    
    def _create_feature_correlation_matrix(self, features_df: pd.DataFrame, output_dir: Path) -> str:
        """Create feature correlation matrix."""
        # Select numeric features only
        numeric_features = features_df.select_dtypes(include=[np.number])
        
        # Calculate correlation matrix
        correlation_matrix = numeric_features.corr()
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(15, 12))
        
        # Use a diverging colormap for correlations
        sns.heatmap(correlation_matrix, 
                   cmap='RdBu_r', 
                   center=0,
                   square=True,
                   cbar_kws={'label': 'Correlation'},
                   ax=ax)
        
        ax.set_title('Feature Correlation Matrix', fontsize=16, pad=20)
        ax.set_xlabel('Features', fontsize=12)
        ax.set_ylabel('Features', fontsize=12)
        
        # Rotate x-axis labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        # Save plot
        output_path = output_dir / f'feature_correlation_matrix.{self.config["save_format"]}'
        plt.savefig(output_path, bbox_inches='tight', dpi=self.config['dpi'])
        plt.close()
        
        return str(output_path)
    
    def _create_pca_variance_plot(self, embeddings: Dict, output_dir: Path) -> str:
        """Create PCA explained variance plot."""
        if 'pca' not in embeddings:
            raise ValueError("PCA embeddings not found")
        
        pca_data = embeddings['pca']
        explained_variance_ratio = pca_data['explained_variance_ratio']
        cumulative_variance_ratio = pca_data['cumulative_variance_ratio']
        
        # Create subplots
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Individual explained variance
        axes[0].plot(range(1, len(explained_variance_ratio) + 1), 
                    explained_variance_ratio, 'bo-', linewidth=2, markersize=6)
        axes[0].set_title('PCA: Individual Explained Variance')
        axes[0].set_xlabel('Principal Component')
        axes[0].set_ylabel('Explained Variance Ratio')
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Cumulative explained variance
        axes[1].plot(range(1, len(cumulative_variance_ratio) + 1), 
                    cumulative_variance_ratio, 'ro-', linewidth=2, markersize=6)
        axes[1].axhline(y=0.95, color='green', linestyle='--', alpha=0.7, label='95% Variance')
        axes[1].axhline(y=0.90, color='orange', linestyle='--', alpha=0.7, label='90% Variance')
        axes[1].set_title('PCA: Cumulative Explained Variance')
        axes[1].set_xlabel('Principal Component')
        axes[1].set_ylabel('Cumulative Explained Variance Ratio')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        output_path = output_dir / f'pca_variance_plot.{self.config["save_format"]}'
        plt.savefig(output_path, bbox_inches='tight', dpi=self.config['dpi'])
        plt.close()
        
        return str(output_path) 