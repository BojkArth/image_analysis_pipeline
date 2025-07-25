# Image Analysis Pipeline

A comprehensive pipeline for analyzing multi-channel fluorescent microscopy images, extracting features, and performing similarity analysis with visualization outputs.

## 🎯 Overview

This pipeline processes TIFF files with multiple fluorescent channels (typically 4 channels), extracts comprehensive features, performs similarity analysis, and generates visualizations including clustermaps, UMAP plots, and HTML reports.

## 🏗️ Architecture

The pipeline consists of several key components:

### Core Modules
- **Pipeline** (`pipeline.py`): Main orchestrator that coordinates all analysis steps
- **Feature Extraction** (`feature_extraction.py`): Extracts comprehensive features from images
- **Similarity Analysis** (`similarity_analysis.py`): Performs dimensionality reduction and clustering
- **Visualization** (`visualization.py`): Generates plots and visualizations
- **Report Generator** (`report_generator.py`): Creates comprehensive HTML reports
- **CLI** (`cli.py`): Command-line interface for easy usage

### Feature Types Extracted

#### Statistical Features
- Mean, standard deviation, min, max, median
- Percentiles (10th, 25th, 75th, 90th)
- Skewness, kurtosis
- Energy, entropy

#### Texture Features
- **GLCM (Gray Level Co-occurrence Matrix)**: Contrast, dissimilarity, homogeneity, energy, correlation, ASM
- **Haralick Features**: Standard texture descriptors
- **LBP (Local Binary Pattern)**: Local texture patterns
- **HOG (Histogram of Oriented Gradients)**: Gradient-based features

#### Morphological Features
- Erosion, dilation, opening, closing areas
- Morphological ratios
- Shape descriptors

#### Edge Features
- Sobel edge magnitude and direction
- Canny edge density
- Edge direction histograms

#### Shape Features
- Area, perimeter, eccentricity
- Solidity, extent
- Major/minor axis lengths
- Circularity, aspect ratio

#### Color Features
- Channel statistics
- Channel ratios and correlations
- Multi-channel balance

## 🚀 Quick Start

### 1. Installation

The pipeline uses existing dependencies from the superfluid environment. Ensure you have the following packages installed:

```bash
# Core dependencies (should already be in your environment)
pip install numpy pandas matplotlib seaborn scikit-image scikit-learn
pip install opencv-python tifffile umap-learn plotly jinja2 pyyaml
```

### 2. Basic Usage

#### Using the CLI (Recommended)

```bash
# Create a default configuration file
python image_analysis/cli.py --create-config my_config.yaml

# Run the complete pipeline
python image_analysis/cli.py --input-dir /path/to/images --output-dir /path/to/results

# Or use a configuration file
python image_analysis/cli.py --config my_config.yaml

# Verbose output
python image_analysis/cli.py --input-dir /path/to/images --output-dir /path/to/results --verbose
```

#### Using Python API

```python
from image_analysis.pipeline import ImageAnalysisPipeline

# Initialize pipeline
pipeline = ImageAnalysisPipeline(
    input_dir='images',
    output_dir='results'
)

# Run complete pipeline
results = pipeline.run_complete_pipeline()

# Or run steps individually
pipeline.load_images()
features_df = pipeline.extract_features()
analysis_results = pipeline.perform_similarity_analysis()
plots = pipeline.generate_visualizations()
report_path = pipeline.generate_report()
```

### 3. Input Data Format

The pipeline expects:
- **File Format**: Multi-channel TIFF files (`.tif` or `.tiff`)
- **Image Structure**: 
  - 4 channels (fluorescent channels)
  - ~1000x1000 pixels per image
  - 5000 images total
- **File Organization**: All TIFF files in a single directory

### 4. Output Structure

```
image_analysis_results/
├── extracted_features.csv          # Feature matrix (images × features)
├── similarity_matrix.csv           # Similarity matrix
├── embeddings.json                 # Dimensionality reduction results
├── clustering_results.json         # Clustering results
├── pipeline_results.pkl            # Complete pipeline results
├── image_analysis_report.html      # HTML report
├── plots/                          # Generated visualizations
│   ├── clustermap.png
│   ├── umap_visualization.png
│   ├── tsne_visualization.png
│   ├── feature_distributions.png
│   ├── clustering_evaluation.png
│   ├── similarity_heatmap.png
│   ├── feature_correlation_matrix.png
│   └── pca_variance_plot.png
└── image_analysis.log              # Pipeline log file
```

## 📊 Outputs

### 1. Feature Matrix
- **Format**: CSV file with images as rows and features as columns
- **Size**: 5000 images × ~1000+ features
- **Features**: Statistical, texture, morphological, edge, shape, and color features

### 2. Similarity Matrix
- **Format**: CSV file with pairwise similarities between all images
- **Metrics**: Cosine, Euclidean, and correlation similarities
- **Size**: 5000 × 5000 matrix

### 3. Visualizations

#### Clustermap
- Hierarchical clustering of image similarities
- Dendrogram showing image relationships
- Color-coded similarity values

#### UMAP Plot
- 2D embedding of high-dimensional features
- Shows image clusters and relationships
- Interactive visualization

#### t-SNE Plot
- Alternative dimensionality reduction
- Preserves local structure
- Good for cluster visualization

#### Feature Distributions
- Histograms of feature values
- Shows feature characteristics
- Helps identify important features

#### Clustering Evaluation
- Silhouette scores vs number of clusters
- Calinski-Harabasz scores
- Hierarchical clustering dendrogram
- DBSCAN parameter evaluation

### 4. HTML Report
- Executive summary
- Statistical summaries
- Feature analysis tables
- Clustering results
- Visualization gallery
- Conclusions and recommendations

## ⚙️ Configuration

### Configuration File Format

The pipeline uses YAML configuration files. See `example_config.yaml` for a complete example.

Key configuration sections:

```yaml
# Feature extraction settings
feature_extraction:
  statistical_features: true
  texture_features: true
  morphological_features: true
  # ... more settings

# Similarity analysis settings
similarity_analysis:
  distance_metrics: ["cosine", "euclidean", "correlation"]
  clustering_methods: ["kmeans", "hierarchical"]
  n_clusters: 5
  # ... more settings

# Visualization settings
visualization:
  plot_style: "seaborn"
  figure_size: [12, 8]
  dpi: 300
  # ... more settings
```

### Command Line Options

```bash
# Basic usage
python cli.py --input-dir /path/to/images --output-dir /path/to/results

# Configuration file
python cli.py --config config.yaml

# Skip specific steps
python cli.py --input-dir /path/to/images --output-dir /path/to/results --skip-visualizations

# Verbose output
python cli.py --input-dir /path/to/images --output-dir /path/to/results --verbose

# Create default config
python cli.py --create-config config.yaml
```

## 🔧 Advanced Usage

### Custom Feature Extraction

```python
from image_analysis.feature_extraction import FeatureExtractor

# Custom configuration
config = {
    'statistical_features': True,
    'texture_features': True,
    'glcm_distances': [1, 2, 3],
    'glcm_angles': [0, np.pi/4, np.pi/2, 3*np.pi/4],
    'hog_orientations': 12
}

extractor = FeatureExtractor(**config)
features = extractor.extract_all_features(channels, image_name)
```

### Custom Similarity Analysis

```python
from image_analysis.similarity_analysis import SimilarityAnalyzer

# Custom configuration
config = {
    'distance_metrics': ['cosine', 'euclidean'],
    'clustering_methods': ['kmeans'],
    'n_clusters': 10,
    'umap_n_neighbors': 20
}

analyzer = SimilarityAnalyzer(**config)
results = analyzer.analyze(features_df)
```

### Custom Visualizations

```python
from image_analysis.visualization import ImageVisualizer

visualizer = ImageVisualizer(
    figure_size=(15, 10),
    dpi=300,
    color_palette='plasma'
)

plots = visualizer.create_all_plots(
    features_df, similarity_matrix, embeddings, clustering_results, output_dir
)
```

## 📈 Performance Considerations

### Memory Usage
- **Feature Extraction**: ~2-4GB for 5000 images
- **Similarity Analysis**: ~1-2GB for similarity matrices
- **Visualization**: ~500MB-1GB for plots

### Processing Time
- **Feature Extraction**: 10-30 minutes for 5000 images
- **Similarity Analysis**: 5-15 minutes
- **Visualization**: 2-5 minutes
- **Total Pipeline**: 20-60 minutes

### Optimization Tips
1. Use SSD storage for faster I/O
2. Increase RAM if processing large datasets
3. Use fewer features if memory is limited
4. Skip optional visualizations for faster processing

## 🐛 Troubleshooting

### Common Issues

1. **Memory Error**: Reduce number of features or process in batches
2. **Import Error**: Ensure all dependencies are installed
3. **File Not Found**: Check input directory path and file patterns
4. **Empty Results**: Verify TIFF files are valid and contain data

### Debug Mode

```bash
# Enable verbose logging
python cli.py --input-dir /path/to/images --output-dir /path/to/results --verbose
```

### Log Files

Check `image_analysis.log` for detailed error messages and processing information.

## 📚 API Reference

### ImageAnalysisPipeline

Main pipeline class that orchestrates the complete analysis.

```python
class ImageAnalysisPipeline:
    def __init__(self, config_path=None, **kwargs)
    def load_images(self, input_dir=None)
    def extract_features(self)
    def perform_similarity_analysis(self)
    def generate_visualizations(self)
    def generate_report(self)
    def run_complete_pipeline(self, input_dir=None)
```

### FeatureExtractor

Extracts comprehensive features from multi-channel images.

```python
class FeatureExtractor:
    def __init__(self, **kwargs)
    def extract_all_features(self, channels, image_name)
```

### SimilarityAnalyzer

Performs similarity analysis, dimensionality reduction, and clustering.

```python
class SimilarityAnalyzer:
    def __init__(self, **kwargs)
    def analyze(self, features_df)
```

### ImageVisualizer

Generates comprehensive visualizations.

```python
class ImageVisualizer:
    def __init__(self, **kwargs)
    def create_all_plots(self, features_df, similarity_matrix, embeddings, clustering_results, output_dir)
```

### ReportGenerator

Creates HTML reports with analysis results.

```python
class ReportGenerator:
    def __init__(self, **kwargs)
    def generate_report(self, features_df, similarity_matrix, embeddings, clustering_results, images, config, output_dir)
```

## 🤝 Contributing

To contribute to the image analysis pipeline:

1. Follow the existing code style and structure
2. Add comprehensive docstrings and type hints
3. Include unit tests for new functionality
4. Update documentation for new features
5. Ensure compatibility with existing dependencies

## 📄 License

This module is part of the Superfluid project and follows the same licensing terms.

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the log files for error messages
3. Ensure all dependencies are properly installed
4. Verify input data format and structure 