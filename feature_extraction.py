"""
Feature Extraction Module

Extracts comprehensive features from multi-channel fluorescent microscopy images.
Includes statistical, texture, morphological, and color-based features.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import logging
from pathlib import Path

# Image processing
import cv2
from skimage import feature, measure, filters, morphology, segmentation
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu, gaussian, sobel
from skimage.measure import regionprops_table
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from skimage.morphology import disk, ball
from skimage.segmentation import slic, watershed
from skimage.util import img_as_ubyte

# Machine learning
from sklearn.feature_extraction import image as skimage_feature

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Comprehensive feature extractor for multi-channel fluorescent microscopy images.
    
    Extracts various types of features:
    - Statistical features (mean, std, percentiles, etc.)
    - Texture features (GLCM, LBP, Haralick)
    - Morphological features (area, perimeter, shape descriptors)
    - Color features (channel statistics, ratios)
    - Edge features (Sobel, Canny)
    - Shape features (circularity, eccentricity, etc.)
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the feature extractor.
        
        Parameters:
        -----------
        **kwargs : dict
            Configuration parameters for feature extraction
        """
        self.config = {
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
            'glcm_angles': [0, np.pi/4, np.pi/2, 3*np.pi/4],
            'lbp_points': 8,
            'lbp_radius': 1,
            'hog_orientations': 9,
            'hog_pixels_per_cell': (8, 8),
            'hog_cells_per_block': (2, 2),
            **kwargs
        }
    
    def extract_all_features(self, channels: np.ndarray, image_name: str) -> Dict:
        """
        Extract all features from multi-channel image.
        
        Parameters:
        -----------
        channels : np.ndarray
            Multi-channel image array (channels, height, width)
        image_name : str
            Name of the image for logging
            
        Returns:
        --------
        dict
            Dictionary containing all extracted features
        """
        features = {}
        
        # Extract features for each channel
        for i, channel in enumerate(channels):
            channel_name = f"channel_{i+1}"
            channel_features = self._extract_channel_features(channel, channel_name)
            features.update(channel_features)
        
        # Extract cross-channel features
        if len(channels) > 1:
            cross_channel_features = self._extract_cross_channel_features(channels)
            features.update(cross_channel_features)
        
        # Add image metadata
        features['image_name'] = image_name
        features['n_channels'] = len(channels)
        features['image_height'] = channels.shape[1]
        features['image_width'] = channels.shape[2]
        
        return features
    
    def _extract_channel_features(self, channel: np.ndarray, channel_name: str) -> Dict:
        """Extract features from a single channel."""
        features = {}
        
        # Ensure channel is 2D
        if len(channel.shape) > 2:
            channel = np.squeeze(channel)
        
        # Normalize channel to 0-255 for feature extraction
        channel_norm = self._normalize_channel(channel)
        
        # Statistical features
        if self.config['statistical_features']:
            stats_features = self._extract_statistical_features(channel, channel_name)
            features.update(stats_features)
        
        # Texture features
        if self.config['texture_features']:
            texture_features = self._extract_texture_features(channel_norm, channel_name)
            features.update(texture_features)
        
        # Haralick features
        if self.config['haralick_features']:
            haralick_features = self._extract_haralick_features(channel_norm, channel_name)
            features.update(haralick_features)
        
        # LBP features
        if self.config['lbp_features']:
            lbp_features = self._extract_lbp_features(channel_norm, channel_name)
            features.update(lbp_features)
        
        # HOG features
        if self.config['hog_features']:
            hog_features = self._extract_hog_features(channel_norm, channel_name)
            features.update(hog_features)
        
        # Edge features
        if self.config['edge_features']:
            edge_features = self._extract_edge_features(channel_norm, channel_name)
            features.update(edge_features)
        
        # Morphological features
        if self.config['morphological_features']:
            morph_features = self._extract_morphological_features(channel_norm, channel_name)
            features.update(morph_features)
        
        # Shape features
        if self.config['shape_features']:
            shape_features = self._extract_shape_features(channel_norm, channel_name)
            features.update(shape_features)
        
        return features
    
    def _normalize_channel(self, channel: np.ndarray) -> np.ndarray:
        """Normalize channel to 0-255 range."""
        if channel.max() > 0:
            channel_norm = ((channel - channel.min()) / (channel.max() - channel.min()) * 255).astype(np.uint8)
        else:
            channel_norm = channel.astype(np.uint8)
        return channel_norm
    
    def _extract_statistical_features(self, channel: np.ndarray, channel_name: str) -> Dict:
        """Extract statistical features from channel."""
        features = {}
        
        # Basic statistics
        features[f'{channel_name}_mean'] = np.mean(channel)
        features[f'{channel_name}_std'] = np.std(channel)
        features[f'{channel_name}_min'] = np.min(channel)
        features[f'{channel_name}_max'] = np.max(channel)
        features[f'{channel_name}_median'] = np.median(channel)
        
        # Percentiles
        percentiles = [10, 25, 75, 90]
        for p in percentiles:
            features[f'{channel_name}_p{p}'] = np.percentile(channel, p)
        
        # Higher order moments
        features[f'{channel_name}_skewness'] = self._skewness(channel)
        features[f'{channel_name}_kurtosis'] = self._kurtosis(channel)
        
        # Energy and entropy
        features[f'{channel_name}_energy'] = np.sum(channel**2)
        features[f'{channel_name}_entropy'] = self._entropy(channel)
        
        return features
    
    def _extract_texture_features(self, channel: np.ndarray, channel_name: str) -> Dict:
        """Extract texture features using GLCM."""
        features = {}
        
        try:
            # Calculate GLCM
            glcm = graycomatrix(channel, 
                              distances=self.config['glcm_distances'],
                              angles=self.config['glcm_angles'],
                              levels=256, symmetric=True, normed=True)
            
            # Extract GLCM properties
            properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
            
            for prop in properties:
                prop_values = graycoprops(glcm, prop)
                for i, distance in enumerate(self.config['glcm_distances']):
                    for j, angle in enumerate(self.config['glcm_angles']):
                        features[f'{channel_name}_{prop}_d{distance}_a{int(np.degrees(angle))}'] = prop_values[i, j]
        
        except Exception as e:
            logger.warning(f"Could not extract texture features for {channel_name}: {e}")
        
        return features
    
    def _extract_haralick_features(self, channel: np.ndarray, channel_name: str) -> Dict:
        """Extract Haralick texture features."""
        features = {}
        
        try:
            # Calculate GLCM for Haralick features
            glcm = graycomatrix(channel, 
                              distances=[1], 
                              angles=[0], 
                              levels=256, 
                              symmetric=True, 
                              normed=True)
            
            # Extract Haralick features
            haralick_props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
            
            for prop in haralick_props:
                value = graycoprops(glcm, prop)[0, 0]
                features[f'{channel_name}_haralick_{prop}'] = value
        
        except Exception as e:
            logger.warning(f"Could not extract Haralick features for {channel_name}: {e}")
        
        return features
    
    def _extract_lbp_features(self, channel: np.ndarray, channel_name: str) -> Dict:
        """Extract Local Binary Pattern features."""
        features = {}
        
        try:
            # Calculate LBP
            lbp = local_binary_pattern(channel, 
                                     self.config['lbp_points'],
                                     self.config['lbp_radius'],
                                     method='uniform')
            
            # Calculate LBP histogram
            n_bins = self.config['lbp_points'] + 2
            hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
            
            # Store histogram features
            for i, val in enumerate(hist):
                features[f'{channel_name}_lbp_hist_{i}'] = val
            
            # LBP statistics
            features[f'{channel_name}_lbp_mean'] = np.mean(lbp)
            features[f'{channel_name}_lbp_std'] = np.std(lbp)
            features[f'{channel_name}_lbp_energy'] = np.sum(lbp**2)
        
        except Exception as e:
            logger.warning(f"Could not extract LBP features for {channel_name}: {e}")
        
        return features
    
    def _extract_hog_features(self, channel: np.ndarray, channel_name: str) -> Dict:
        """Extract Histogram of Oriented Gradients features."""
        features = {}
        
        try:
            # Calculate HOG features
            hog_features = feature.hog(channel,
                                     orientations=self.config['hog_orientations'],
                                     pixels_per_cell=self.config['hog_pixels_per_cell'],
                                     cells_per_block=self.config['hog_cells_per_block'],
                                     visualize=False)
            
            # Store HOG features
            for i, val in enumerate(hog_features):
                features[f'{channel_name}_hog_{i}'] = val
        
        except Exception as e:
            logger.warning(f"Could not extract HOG features for {channel_name}: {e}")
        
        return features
    
    def _extract_edge_features(self, channel: np.ndarray, channel_name: str) -> Dict:
        """Extract edge-based features."""
        features = {}
        
        try:
            # Sobel edges
            sobel_h = sobel(channel, axis=0)
            sobel_v = sobel(channel, axis=1)
            sobel_magnitude = np.sqrt(sobel_h**2 + sobel_v**2)
            
            features[f'{channel_name}_sobel_mean'] = np.mean(sobel_magnitude)
            features[f'{channel_name}_sobel_std'] = np.std(sobel_magnitude)
            features[f'{channel_name}_sobel_max'] = np.max(sobel_magnitude)
            
            # Canny edges
            canny_edges = feature.canny(channel, sigma=1.0)
            features[f'{channel_name}_canny_edge_density'] = np.sum(canny_edges) / canny_edges.size
            
            # Edge direction histogram
            edge_direction = np.arctan2(sobel_v, sobel_h)
            hist, _ = np.histogram(edge_direction, bins=8, range=(-np.pi, np.pi), density=True)
            
            for i, val in enumerate(hist):
                features[f'{channel_name}_edge_direction_{i}'] = val
        
        except Exception as e:
            logger.warning(f"Could not extract edge features for {channel_name}: {e}")
        
        return features
    
    def _extract_morphological_features(self, channel: np.ndarray, channel_name: str) -> Dict:
        """Extract morphological features."""
        features = {}
        
        try:
            # Threshold image for morphological analysis
            threshold = threshold_otsu(channel)
            binary = channel > threshold
            
            # Morphological operations
            selem = disk(3)
            eroded = morphology.erosion(binary, selem)
            dilated = morphology.dilation(binary, selem)
            opened = morphology.opening(binary, selem)
            closed = morphology.closing(binary, selem)
            
            # Calculate morphological features
            features[f'{channel_name}_eroded_area'] = np.sum(eroded)
            features[f'{channel_name}_dilated_area'] = np.sum(dilated)
            features[f'{channel_name}_opened_area'] = np.sum(opened)
            features[f'{channel_name}_closed_area'] = np.sum(closed)
            
            # Morphological ratios
            if np.sum(binary) > 0:
                features[f'{channel_name}_erosion_ratio'] = np.sum(eroded) / np.sum(binary)
                features[f'{channel_name}_dilation_ratio'] = np.sum(dilated) / np.sum(binary)
                features[f'{channel_name}_opening_ratio'] = np.sum(opened) / np.sum(binary)
                features[f'{channel_name}_closing_ratio'] = np.sum(closed) / np.sum(binary)
        
        except Exception as e:
            logger.warning(f"Could not extract morphological features for {channel_name}: {e}")
        
        return features
    
    def _extract_shape_features(self, channel: np.ndarray, channel_name: str) -> Dict:
        """Extract shape-based features."""
        features = {}
        
        try:
            # Threshold image
            threshold = threshold_otsu(channel)
            binary = channel > threshold
            
            # Label connected components
            labeled = measure.label(binary)
            regions = measure.regionprops(labeled)
            
            if regions:
                # Calculate shape features for the largest region
                largest_region = max(regions, key=lambda x: x.area)
                
                features[f'{channel_name}_area'] = largest_region.area
                features[f'{channel_name}_perimeter'] = largest_region.perimeter
                features[f'{channel_name}_eccentricity'] = largest_region.eccentricity
                features[f'{channel_name}_solidity'] = largest_region.solidity
                features[f'{channel_name}_extent'] = largest_region.extent
                features[f'{channel_name}_major_axis_length'] = largest_region.major_axis_length
                features[f'{channel_name}_minor_axis_length'] = largest_region.minor_axis_length
                
                # Circularity
                if largest_region.perimeter > 0:
                    features[f'{channel_name}_circularity'] = (4 * np.pi * largest_region.area) / (largest_region.perimeter**2)
                else:
                    features[f'{channel_name}_circularity'] = 0
                
                # Aspect ratio
                if largest_region.minor_axis_length > 0:
                    features[f'{channel_name}_aspect_ratio'] = largest_region.major_axis_length / largest_region.minor_axis_length
                else:
                    features[f'{channel_name}_aspect_ratio'] = 0
        
        except Exception as e:
            logger.warning(f"Could not extract shape features for {channel_name}: {e}")
        
        return features
    
    def _extract_cross_channel_features(self, channels: np.ndarray) -> Dict:
        """Extract features that involve multiple channels."""
        features = {}
        
        try:
            # Channel ratios
            for i in range(len(channels)):
                for j in range(i+1, len(channels)):
                    ratio_name = f'ratio_ch{i+1}_ch{j+1}'
                    if np.sum(channels[j]) > 0:
                        ratio = np.sum(channels[i]) / np.sum(channels[j])
                        features[ratio_name] = ratio
                    else:
                        features[ratio_name] = 0
            
            # Channel correlations
            for i in range(len(channels)):
                for j in range(i+1, len(channels)):
                    corr_name = f'correlation_ch{i+1}_ch{j+1}'
                    correlation = np.corrcoef(channels[i].flatten(), channels[j].flatten())[0, 1]
                    features[corr_name] = correlation if not np.isnan(correlation) else 0
            
            # Multi-channel statistics
            features['total_intensity'] = np.sum(channels)
            features['mean_intensity_per_channel'] = np.mean(channels)
            features['std_intensity_per_channel'] = np.std(channels)
            
            # Channel balance
            channel_sums = [np.sum(ch) for ch in channels]
            total_sum = sum(channel_sums)
            if total_sum > 0:
                for i, ch_sum in enumerate(channel_sums):
                    features[f'channel_{i+1}_fraction'] = ch_sum / total_sum
        
        except Exception as e:
            logger.warning(f"Could not extract cross-channel features: {e}")
        
        return features
    
    def _skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 3)
    
    def _kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of data."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 4) - 3
    
    def _entropy(self, data: np.ndarray) -> float:
        """Calculate entropy of data."""
        hist, _ = np.histogram(data, bins=256, density=True)
        hist = hist[hist > 0]  # Remove zero bins
        return -np.sum(hist * np.log2(hist)) 