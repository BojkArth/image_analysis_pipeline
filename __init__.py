"""
Image Analysis Pipeline for Fluorescent Microscopy Data

A comprehensive pipeline for analyzing multi-channel fluorescent microscopy images,
extracting features, and performing similarity analysis with visualization outputs.

Features:
- Multi-channel TIFF image processing
- Comprehensive feature extraction (statistical, texture, morphological)
- Dimensionality reduction and clustering
- Similarity analysis and visualization
- HTML report generation
"""

from .pipeline import ImageAnalysisPipeline
from .feature_extraction import FeatureExtractor
from .similarity_analysis import SimilarityAnalyzer
from .visualization import ImageVisualizer
from .report_generator import ReportGenerator

__version__ = "1.0.0"
__author__ = "Superfluid Team"

__all__ = [
    "ImageAnalysisPipeline",
    "FeatureExtractor", 
    "SimilarityAnalyzer",
    "ImageVisualizer",
    "ReportGenerator"
] 