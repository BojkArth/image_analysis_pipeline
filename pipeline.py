"""
Main Image Analysis Pipeline

Orchestrates the complete workflow for analyzing multi-channel fluorescent microscopy images.
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
import logging
import yaml
from datetime import datetime
import pickle
import json

# Image processing
import cv2
from PIL import Image
import tifffile
from skimage import feature, measure, filters, morphology, segmentation
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu, gaussian
from skimage.measure import regionprops_table

# Machine learning and analysis
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import umap

# Visualization
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import local modules
from .feature_extraction import FeatureExtractor
from .similarity_analysis import SimilarityAnalyzer
from .visualization import ImageVisualizer
from .report_generator import ReportGenerator

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageAnalysisPipeline:
    """
    Main pipeline for analyzing multi-channel fluorescent microscopy images.
    
    This pipeline processes TIFF files with multiple channels, extracts comprehensive
    features, performs similarity analysis, and generates visualizations and reports.
    """
    
    def __init__(self, config_path: Optional[str] = None, **kwargs):
        """
        Initialize the image analysis pipeline.
        
        Parameters:
        -----------
        config_path : str, optional
            Path to YAML configuration file
        **kwargs : dict
            Configuration parameters to override config file
        """
        self.config = self._load_config(config_path, **kwargs)
        self.output_dir = Path(self.config.get('output_dir', 'image_analysis_results'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.feature_extractor = FeatureExtractor(**self.config.get('feature_extraction', {}))
        self.similarity_analyzer = SimilarityAnalyzer(**self.config.get('similarity_analysis', {}))
        self.visualizer = ImageVisualizer(**self.config.get('visualization', {}))
        self.report_generator = ReportGenerator(**self.config.get('report', {}))
        
        # Data storage
        self.images = {}
        self.features_df = None
        self.similarity_matrix = None
        self.embeddings = {}
        self.clustering_results = {}
        self.metadata = None
        
        logger.info(f"Image Analysis Pipeline initialized. Output directory: {self.output_dir}")
    
    def _load_config(self, config_path: Optional[str], **kwargs) -> Dict:
        """Load configuration from file and override with kwargs."""
        config = {
            'input_dir': 'images',
            'output_dir': 'image_analysis_results',
            'file_pattern': '*.tif',
            'channels': ['channel_1', 'channel_2', 'channel_3', 'channel_4'],
            'feature_extraction': {
                'statistical_features': True,
                'texture_features': True,
                'morphological_features': True,
                'haralick_features': True,
                'hog_features': True,
                'lbp_features': True,
                'color_features': True,
                'edge_features': True,
                'shape_features': True
            },
            'similarity_analysis': {
                'distance_metrics': ['cosine', 'euclidean', 'correlation'],
                'clustering_methods': ['kmeans', 'hierarchical'],
                'n_clusters': 5
            },
            'visualization': {
                'plot_style': 'seaborn',
                'figure_size': (12, 8),
                'dpi': 300
            },
            'report': {
                'include_plots': True,
                'include_tables': True,
                'include_statistics': True
            }
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                file_config = yaml.safe_load(f)
                config.update(file_config)
        
        # Override with kwargs
        config.update(kwargs)
        return config
    
    def load_images(self, input_dir: Optional[str] = None) -> None:
        """
        Load multi-channel TIFF images from the input directory.
        
        Parameters:
        -----------
        input_dir : str, optional
            Directory containing TIFF images. If None, uses config input_dir.
        """
        input_dir = Path(input_dir or self.config['input_dir'])
        file_pattern = self.config['file_pattern']
        
        # Find all TIFF files
        tiff_files = list(input_dir.glob(file_pattern))
        if not tiff_files:
            raise FileNotFoundError(f"No TIFF files found in {input_dir} matching pattern {file_pattern}")
        
        logger.info(f"Found {len(tiff_files)} TIFF files")
        
        # Load images
        for file_path in tiff_files:
            try:
                # Load multi-channel TIFF
                image_data = tifffile.imread(str(file_path))
                
                # Handle different image formats
                if len(image_data.shape) == 3:
                    # Multi-channel image
                    if image_data.shape[0] <= 4:  # Channels first
                        channels = image_data
                    else:  # Channels last
                        channels = np.transpose(image_data, (2, 0, 1))
                elif len(image_data.shape) == 2:
                    # Single channel image
                    channels = np.expand_dims(image_data, axis=0)
                else:
                    logger.warning(f"Skipping {file_path}: unexpected shape {image_data.shape}")
                    continue
                
                # Store image data
                image_name = file_path.stem
                self.images[image_name] = {
                    'channels': channels,
                    'file_path': str(file_path),
                    'shape': channels.shape,
                    'dtype': channels.dtype
                }
                
                logger.info(f"Loaded {image_name}: {channels.shape}")
                
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
                continue
        
        logger.info(f"Successfully loaded {len(self.images)} images")
    
    def extract_features(self) -> pd.DataFrame:
        """
        Extract comprehensive features from all loaded images.
        
        Returns:
        --------
        pd.DataFrame
            Feature matrix with images as rows and features as columns
        """
        if not self.images:
            raise ValueError("No images loaded. Call load_images() first.")
        
        logger.info("Starting feature extraction...")
        
        # Extract features for each image
        feature_list = []
        image_names = []
        
        for image_name, image_data in self.images.items():
            try:
                features = self.feature_extractor.extract_all_features(
                    image_data['channels'],
                    image_name
                )
                feature_list.append(features)
                image_names.append(image_name)
                
            except Exception as e:
                logger.error(f"Error extracting features from {image_name}: {e}")
                continue
        
        # Create feature matrix
        self.features_df = pd.DataFrame(feature_list, index=image_names)
        
        # Save features
        features_path = self.output_dir / 'extracted_features.csv'
        self.features_df.to_csv(features_path)
        
        logger.info(f"Feature extraction complete. Shape: {self.features_df.shape}")
        logger.info(f"Features saved to: {features_path}")
        
        return self.features_df
    
    def perform_similarity_analysis(self) -> Dict:
        """
        Perform comprehensive similarity analysis on the feature matrix.
        
        Returns:
        --------
        dict
            Dictionary containing similarity matrices, embeddings, and clustering results
        """
        if self.features_df is None:
            raise ValueError("No features extracted. Call extract_features() first.")
        
        logger.info("Starting similarity analysis...")
        
        # Perform similarity analysis
        results = self.similarity_analyzer.analyze(self.features_df)
        
        # Store results
        self.similarity_matrix = results['similarity_matrix']
        self.embeddings = results['embeddings']
        self.clustering_results = results['clustering']
        
        # Save results
        self._save_analysis_results(results)
        
        logger.info("Similarity analysis complete")
        return results
    
    def generate_visualizations(self) -> Dict:
        """
        Generate comprehensive visualizations of the analysis results.
        
        Returns:
        --------
        dict
            Dictionary containing paths to generated plots
        """
        if self.similarity_matrix is None:
            raise ValueError("No similarity analysis performed. Call perform_similarity_analysis() first.")
        
        logger.info("Generating visualizations...")
        
        # Generate visualizations
        plots = self.visualizer.create_all_plots(
            features_df=self.features_df,
            similarity_matrix=self.similarity_matrix,
            embeddings=self.embeddings,
            clustering_results=self.clustering_results,
            output_dir=self.output_dir
        )
        
        logger.info(f"Generated {len(plots)} visualizations")
        return plots
    
    def generate_report(self) -> str:
        """
        Generate a comprehensive HTML report of the analysis.
        
        Returns:
        --------
        str
            Path to the generated HTML report
        """
        logger.info("Generating HTML report...")
        
        report_path = self.report_generator.generate_report(
            features_df=self.features_df,
            similarity_matrix=self.similarity_matrix,
            embeddings=self.embeddings,
            clustering_results=self.clustering_results,
            images=self.images,
            config=self.config,
            output_dir=self.output_dir
        )
        
        logger.info(f"HTML report generated: {report_path}")
        return report_path
    
    def run_complete_pipeline(self, input_dir: Optional[str] = None) -> Dict:
        """
        Run the complete image analysis pipeline.
        
        Parameters:
        -----------
        input_dir : str, optional
            Directory containing TIFF images
            
        Returns:
        --------
        dict
            Dictionary containing all pipeline results
        """
        logger.info("Starting complete image analysis pipeline...")
        
        # Step 1: Load images
        self.load_images(input_dir)
        
        # Step 2: Extract features
        features_df = self.extract_features()
        
        # Step 3: Perform similarity analysis
        analysis_results = self.perform_similarity_analysis()
        
        # Step 4: Generate visualizations
        plots = self.generate_visualizations()
        
        # Step 5: Generate report
        report_path = self.generate_report()
        
        # Compile results
        results = {
            'features_df': features_df,
            'analysis_results': analysis_results,
            'plots': plots,
            'report_path': report_path,
            'output_dir': str(self.output_dir),
            'config': self.config,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save complete results
        results_path = self.output_dir / 'pipeline_results.pkl'
        with open(results_path, 'wb') as f:
            pickle.dump(results, f)
        
        logger.info(f"Pipeline complete! Results saved to: {results_path}")
        logger.info(f"HTML report: {report_path}")
        
        return results
    
    def _convert_numpy_to_lists(self, obj):
        """Recursively convert NumPy arrays to lists for JSON serialization."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._convert_numpy_to_lists(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_to_lists(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        else:
            return obj
    
    def _save_analysis_results(self, results: Dict) -> None:
        """Save analysis results to files."""
        # Save similarity matrix
        similarity_path = self.output_dir / 'similarity_matrix.csv'
        results['similarity_matrix'].to_csv(similarity_path)
        
        # Save embeddings
        embeddings_path = self.output_dir / 'embeddings.json'
        embeddings_data = self._convert_numpy_to_lists(results['embeddings'])
        with open(embeddings_path, 'w') as f:
            json.dump(embeddings_data, f, indent=2)
        
        # Save clustering results
        clustering_path = self.output_dir / 'clustering_results.json'
        clustering_data = self._convert_numpy_to_lists(results['clustering'])
        with open(clustering_path, 'w') as f:
            json.dump(clustering_data, f, indent=2)
        
        logger.info(f"Analysis results saved to {self.output_dir}")
    
    def get_summary_statistics(self) -> Dict:
        """Get summary statistics of the analysis."""
        if self.features_df is None:
            return {}
        
        summary = {
            'n_images': len(self.images),
            'n_features': self.features_df.shape[1],
            'feature_matrix_shape': self.features_df.shape,
            'feature_stats': {
                'mean': self.features_df.mean().to_dict(),
                'std': self.features_df.std().to_dict(),
                'min': self.features_df.min().to_dict(),
                'max': self.features_df.max().to_dict()
            }
        }
        
        if self.similarity_matrix is not None:
            summary['similarity_stats'] = {
                'mean_similarity': self.similarity_matrix.values.mean(),
                'std_similarity': self.similarity_matrix.values.std(),
                'min_similarity': self.similarity_matrix.values.min(),
                'max_similarity': self.similarity_matrix.values.max()
            }
        
        return summary 