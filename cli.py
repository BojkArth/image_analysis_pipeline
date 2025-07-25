#!/usr/bin/env python3
"""
Command Line Interface for Image Analysis Pipeline

Usage:
    python cli.py --input-dir /path/to/images --output-dir /path/to/results
    python cli.py --config config.yaml
"""

import argparse
import sys
import os
from pathlib import Path
import logging
import yaml
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from image_analysis.pipeline import ImageAnalysisPipeline

def setup_logging(verbose: bool = False):
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('image_analysis.log')
        ]
    )

def create_default_config(output_path: str):
    """Create a default configuration file."""
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
            'shape_features': True,
            'glcm_distances': [1],
            'glcm_angles': [0, 0.785, 1.571, 2.356],  # 0, 45, 90, 135 degrees
            'lbp_points': 8,
            'lbp_radius': 1,
            'hog_orientations': 9,
            'hog_pixels_per_cell': [8, 8],
            'hog_cells_per_block': [2, 2]
        },
        'similarity_analysis': {
            'distance_metrics': ['cosine', 'euclidean', 'correlation'],
            'clustering_methods': ['kmeans', 'hierarchical'],
            'n_clusters': 5,
            'pca_n_components': 50,
            'tsne_n_components': 2,
            'tsne_perplexity': 30,
            'umap_n_components': 2,
            'umap_n_neighbors': 15,
            'umap_min_dist': 0.1,
            'random_state': 42
        },
        'visualization': {
            'plot_style': 'seaborn',
            'figure_size': [12, 8],
            'dpi': 300,
            'color_palette': 'viridis',
            'max_images_per_plot': 20,
            'save_format': 'png'
        },
        'report': {
            'include_plots': True,
            'include_tables': True,
            'include_statistics': True,
            'include_interactive': True,
            'max_table_rows': 100,
            'max_table_cols': 20
        }
    }
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    print(f"Default configuration saved to: {output_path}")

def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description='Image Analysis Pipeline for Multi-channel Fluorescent Microscopy Images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings
  python cli.py --input-dir /path/to/images --output-dir /path/to/results
  
  # Run with custom configuration
  python cli.py --config my_config.yaml
  
  # Create default configuration file
  python cli.py --create-config config.yaml
  
  # Verbose output
  python cli.py --input-dir /path/to/images --output-dir /path/to/results --verbose
        """
    )
    
    # Input/output arguments
    parser.add_argument('--input-dir', type=str, help='Directory containing TIFF images')
    parser.add_argument('--output-dir', type=str, help='Output directory for results')
    
    # Configuration arguments
    parser.add_argument('--config', type=str, help='Path to YAML configuration file')
    parser.add_argument('--create-config', type=str, help='Create default configuration file')
    
    # Pipeline control arguments
    parser.add_argument('--skip-features', action='store_true', help='Skip feature extraction')
    parser.add_argument('--skip-similarity', action='store_true', help='Skip similarity analysis')
    parser.add_argument('--skip-visualizations', action='store_true', help='Skip visualization generation')
    parser.add_argument('--skip-report', action='store_true', help='Skip HTML report generation')
    
    # Other arguments
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--version', action='version', version='Image Analysis Pipeline 1.0.0')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    # Handle configuration file creation
    if args.create_config:
        create_default_config(args.create_config)
        return
    
    # Validate arguments
    if not args.config and not args.input_dir:
        parser.error("Either --config or --input-dir must be specified")
    
    if not args.config and not args.output_dir:
        parser.error("Either --config or --output-dir must be specified")
    
    try:
        # Initialize pipeline
        if args.config:
            logger.info(f"Loading configuration from: {args.config}")
            pipeline = ImageAnalysisPipeline(config_path=args.config)
        else:
            logger.info("Using command line arguments")
            pipeline = ImageAnalysisPipeline(
                input_dir=args.input_dir,
                output_dir=args.output_dir
            )
        
        # Run pipeline steps
        if not args.skip_features:
            logger.info("Step 1: Loading images...")
            pipeline.load_images()
            
            logger.info("Step 2: Extracting features...")
            features_df = pipeline.extract_features()
        else:
            logger.info("Skipping feature extraction")
            # Load existing features if available
            features_path = Path(pipeline.output_dir) / 'extracted_features.csv'
            if features_path.exists():
                features_df = pd.read_csv(features_path, index_col=0)
                pipeline.features_df = features_df
                logger.info(f"Loaded existing features from: {features_path}")
            else:
                logger.error("No existing features found. Cannot skip feature extraction.")
                return 1
        
        if not args.skip_similarity:
            logger.info("Step 3: Performing similarity analysis...")
            analysis_results = pipeline.perform_similarity_analysis()
        else:
            logger.info("Skipping similarity analysis")
        
        if not args.skip_visualizations:
            logger.info("Step 4: Generating visualizations...")
            plots = pipeline.generate_visualizations()
        else:
            logger.info("Skipping visualization generation")
        
        if not args.skip_report:
            logger.info("Step 5: Generating HTML report...")
            report_path = pipeline.generate_report()
            logger.info(f"HTML report generated: {report_path}")
        else:
            logger.info("Skipping HTML report generation")
        
        # Print summary
        summary = pipeline.get_summary_statistics()
        logger.info("Pipeline completed successfully!")
        logger.info(f"Processed {summary.get('n_images', 0)} images")
        logger.info(f"Extracted {summary.get('n_features', 0)} features")
        logger.info(f"Results saved to: {pipeline.output_dir}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main()) 