#!/usr/bin/env python3
"""
Example Usage of Image Analysis Pipeline

This script demonstrates how to use the image analysis pipeline
for processing multi-channel fluorescent microscopy images.
"""

import os
import sys
from pathlib import Path
import logging

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from image_analysis.pipeline import ImageAnalysisPipeline

def setup_logging():
    """Set up logging for the example."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def example_basic_usage():
    """Example 1: Basic usage with default settings."""
    print("=== Example 1: Basic Usage ===")
    
    # Initialize pipeline with basic settings
    pipeline = ImageAnalysisPipeline(
        input_dir='images',
        output_dir='results_basic'
    )
    
    # Run complete pipeline
    results = pipeline.run_complete_pipeline()
    
    print(f"Pipeline completed!")
    print(f"Results saved to: {results['output_dir']}")
    print(f"HTML report: {results['report_path']}")
    print(f"Number of images processed: {results['features_df'].shape[0]}")
    print(f"Number of features extracted: {results['features_df'].shape[1]}")

def example_custom_config():
    """Example 2: Using custom configuration."""
    print("\n=== Example 2: Custom Configuration ===")
    
    # Custom configuration
    config = {
        'input_dir': 'images',
        'output_dir': 'results_custom',
        'feature_extraction': {
            'statistical_features': True,
            'texture_features': True,
            'morphological_features': True,
            'haralick_features': True,
            'hog_features': False,  # Disable HOG features
            'lbp_features': True,
            'color_features': True,
            'edge_features': True,
            'shape_features': True
        },
        'similarity_analysis': {
            'distance_metrics': ['cosine', 'euclidean'],
            'clustering_methods': ['kmeans'],
            'n_clusters': 3,
            'umap_n_neighbors': 10
        },
        'visualization': {
            'figure_size': (10, 6),
            'dpi': 200,
            'color_palette': 'plasma'
        }
    }
    
    # Initialize pipeline with custom config
    pipeline = ImageAnalysisPipeline(**config)
    
    # Run complete pipeline
    results = pipeline.run_complete_pipeline()
    
    print(f"Custom pipeline completed!")
    print(f"Results saved to: {results['output_dir']}")

def example_step_by_step():
    """Example 3: Step-by-step processing."""
    print("\n=== Example 3: Step-by-Step Processing ===")
    
    # Initialize pipeline
    pipeline = ImageAnalysisPipeline(
        input_dir='images',
        output_dir='results_stepwise'
    )
    
    # Step 1: Load images
    print("Step 1: Loading images...")
    pipeline.load_images()
    print(f"Loaded {len(pipeline.images)} images")
    
    # Step 2: Extract features
    print("Step 2: Extracting features...")
    features_df = pipeline.extract_features()
    print(f"Extracted {features_df.shape[1]} features from {features_df.shape[0]} images")
    
    # Step 3: Perform similarity analysis
    print("Step 3: Performing similarity analysis...")
    analysis_results = pipeline.perform_similarity_analysis()
    print("Similarity analysis completed")
    
    # Step 4: Generate visualizations
    print("Step 4: Generating visualizations...")
    plots = pipeline.generate_visualizations()
    print(f"Generated {len(plots)} visualizations")
    
    # Step 5: Generate report
    print("Step 5: Generating HTML report...")
    report_path = pipeline.generate_report()
    print(f"HTML report generated: {report_path}")
    
    # Get summary statistics
    summary = pipeline.get_summary_statistics()
    print(f"\nSummary:")
    print(f"- Images processed: {summary['n_images']}")
    print(f"- Features extracted: {summary['n_features']}")
    print(f"- Feature matrix shape: {summary['feature_matrix_shape']}")

def example_error_handling():
    """Example 4: Error handling and validation."""
    print("\n=== Example 4: Error Handling ===")
    
    try:
        # Try to initialize with non-existent directory
        pipeline = ImageAnalysisPipeline(
            input_dir='non_existent_directory',
            output_dir='results_error'
        )
        
        # This should raise an error
        pipeline.load_images()
        
    except FileNotFoundError as e:
        print(f"Expected error caught: {e}")
        print("This demonstrates proper error handling for missing input directories")
    
    except Exception as e:
        print(f"Unexpected error: {e}")

def example_analysis_results():
    """Example 5: Working with analysis results."""
    print("\n=== Example 5: Working with Analysis Results ===")
    
    # Initialize and run pipeline
    pipeline = ImageAnalysisPipeline(
        input_dir='images',
        output_dir='results_analysis'
    )
    
    results = pipeline.run_complete_pipeline()
    
    # Access different components of results
    features_df = results['features_df']
    analysis_results = results['analysis_results']
    
    print("Analysis Results Summary:")
    print(f"- Feature matrix: {features_df.shape}")
    print(f"- Similarity matrices: {list(analysis_results['similarity_matrices'].keys())}")
    print(f"- Embeddings: {list(analysis_results['embeddings'].keys())}")
    print(f"- Clustering methods: {list(analysis_results['clustering'].keys())}")
    
    # Access specific results
    if 'kmeans' in analysis_results['clustering']:
        kmeans_results = analysis_results['clustering']['kmeans']
        if 'optimal' in kmeans_results:
            optimal = kmeans_results['optimal']
            print(f"- Optimal K-means clusters: {optimal.get('silhouette_score', 'N/A')}")
    
    # Access similarity statistics
    if 'statistics' in analysis_results:
        stats = analysis_results['statistics']
        if 'cosine_similarity_stats' in stats:
            cosine_stats = stats['cosine_similarity_stats']
            print(f"- Mean cosine similarity: {cosine_stats['mean']:.3f}")

def main():
    """Main function to run all examples."""
    setup_logging()
    
    print("Image Analysis Pipeline - Example Usage")
    print("=" * 50)
    
    # Check if images directory exists
    if not os.path.exists('images'):
        print("Warning: 'images' directory not found.")
        print("Please create an 'images' directory with your TIFF files to run these examples.")
        print("You can also modify the input_dir parameter in the examples.")
        return
    
    # Run examples
    try:
        example_basic_usage()
        example_custom_config()
        example_step_by_step()
        example_error_handling()
        example_analysis_results()
        
        print("\n" + "=" * 50)
        print("All examples completed successfully!")
        print("\nNext steps:")
        print("1. Check the generated results directories")
        print("2. Open the HTML reports in your browser")
        print("3. Examine the generated plots")
        print("4. Modify the configuration for your specific needs")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main() 