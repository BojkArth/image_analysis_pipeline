"""
Report Generator Module

Generates comprehensive HTML reports for image analysis results including
tables, plots, statistics, and interactive visualizations.
"""

import os
import base64
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
import logging
from pathlib import Path
from datetime import datetime
import jinja2

logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    Comprehensive HTML report generator for image analysis results.
    
    Generates:
    - Executive summary
    - Feature analysis tables
    - Similarity analysis results
    - Clustering results
    - Visualization galleries
    - Statistical summaries
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the report generator.
        
        Parameters:
        -----------
        **kwargs : dict
            Configuration parameters for report generation
        """
        self.config = {
            'include_plots': True,
            'include_tables': True,
            'include_statistics': True,
            'include_interactive': True,
            'max_table_rows': 100,
            'max_table_cols': 20,
            'template_path': None,
            **kwargs
        }
        
        # HTML template
        self.html_template = self._get_html_template()
    
    def _get_html_template(self) -> str:
        """Get the HTML template for the report."""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }
        h1 {
            color: #2c3e50;
            text-align: center;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
            margin-bottom: 30px;
        }
        h2 {
            color: #34495e;
            border-left: 4px solid #3498db;
            padding-left: 15px;
            margin-top: 40px;
        }
        h3 {
            color: #2c3e50;
            margin-top: 25px;
        }
        .summary-box {
            background-color: #ecf0f1;
            border-left: 4px solid #3498db;
            padding: 20px;
            margin: 20px 0;
            border-radius: 5px;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .stat-card {
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            border: 1px solid #dee2e6;
            text-align: center;
        }
        .stat-value {
            font-size: 24px;
            font-weight: bold;
            color: #3498db;
        }
        .stat-label {
            color: #6c757d;
            font-size: 14px;
            margin-top: 5px;
        }
        .plot-container {
            text-align: center;
            margin: 30px 0;
        }
        .plot-container img {
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        .table-container {
            overflow-x: auto;
            margin: 20px 0;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            font-size: 14px;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }
        th {
            background-color: #3498db;
            color: white;
            font-weight: bold;
        }
        tr:nth-child(even) {
            background-color: #f2f2f2;
        }
        tr:hover {
            background-color: #e8f4fd;
        }
        .clustering-results {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .cluster-card {
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            border: 1px solid #dee2e6;
        }
        .feature-importance {
            background-color: #fff3cd;
            border: 1px solid #ffeaa7;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
        }
        .timestamp {
            text-align: center;
            color: #6c757d;
            font-size: 12px;
            margin-top: 40px;
            border-top: 1px solid #dee2e6;
            padding-top: 20px;
        }
        .toc {
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
        }
        .toc ul {
            list-style-type: none;
            padding-left: 0;
        }
        .toc li {
            margin: 8px 0;
        }
        .toc a {
            color: #3498db;
            text-decoration: none;
        }
        .toc a:hover {
            text-decoration: underline;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>{{ title }}</h1>
        
        <div class="summary-box">
            <h3>Executive Summary</h3>
            <p>{{ summary }}</p>
        </div>
        
        <div class="toc">
            <h3>Table of Contents</h3>
            <ul>
                <li><a href="#overview">Overview</a></li>
                <li><a href="#statistics">Statistics</a></li>
                <li><a href="#features">Feature Analysis</a></li>
                <li><a href="#similarity">Similarity Analysis</a></li>
                <li><a href="#clustering">Clustering Results</a></li>
                <li><a href="#visualizations">Visualizations</a></li>
                <li><a href="#conclusions">Conclusions</a></li>
            </ul>
        </div>
        
        <section id="overview">
            <h2>Overview</h2>
            <div class="stats-grid">
                {% for stat in overview_stats %}
                <div class="stat-card">
                    <div class="stat-value">{{ stat.value }}</div>
                    <div class="stat-label">{{ stat.label }}</div>
                </div>
                {% endfor %}
            </div>
        </section>
        
        <section id="statistics">
            <h2>Statistical Summary</h2>
            {{ statistics_section }}
        </section>
        
        <section id="features">
            <h2>Feature Analysis</h2>
            {{ features_section }}
        </section>
        
        <section id="similarity">
            <h2>Similarity Analysis</h2>
            {{ similarity_section }}
        </section>
        
        <section id="clustering">
            <h2>Clustering Results</h2>
            {{ clustering_section }}
        </section>
        
        <section id="visualizations">
            <h2>Visualizations</h2>
            {{ visualizations_section }}
        </section>
        
        <section id="conclusions">
            <h2>Conclusions</h2>
            {{ conclusions_section }}
        </section>
        
        <div class="timestamp">
            Report generated on {{ timestamp }}
        </div>
    </div>
</body>
</html>
        """
    
    def generate_report(self, features_df: pd.DataFrame, similarity_matrix: pd.DataFrame,
                       embeddings: Dict, clustering_results: Dict, images: Dict,
                       config: Dict, output_dir: Path) -> str:
        """
        Generate comprehensive HTML report.
        
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
        images : dict
            Image metadata
        config : dict
            Pipeline configuration
        output_dir : Path
            Output directory
            
        Returns:
        --------
        str
            Path to the generated HTML report
        """
        logger.info("Generating HTML report...")
        
        # Prepare report data
        report_data = self._prepare_report_data(
            features_df, similarity_matrix, embeddings, clustering_results, images, config
        )
        
        # Generate HTML content
        html_content = self._generate_html_content(report_data)
        
        # Save report
        report_path = output_dir / 'image_analysis_report.html'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"HTML report generated: {report_path}")
        return str(report_path)
    
    def _prepare_report_data(self, features_df: pd.DataFrame, similarity_matrix: pd.DataFrame,
                           embeddings: Dict, clustering_results: Dict, images: Dict,
                           config: Dict) -> Dict:
        """Prepare data for the report."""
        # Overview statistics
        overview_stats = [
            {'label': 'Total Images', 'value': len(images)},
            {'label': 'Total Features', 'value': features_df.shape[1]},
            {'label': 'Image Dimensions', 'value': f"{images[list(images.keys())[0]]['shape'][1]}x{images[list(images.keys())[0]]['shape'][2]}"},
            {'label': 'Channels', 'value': images[list(images.keys())[0]]['shape'][0]},
        ]
        
        # Feature statistics
        feature_stats = self._calculate_feature_statistics(features_df)
        
        # Similarity statistics
        similarity_stats = self._calculate_similarity_statistics(similarity_matrix)
        
        # Clustering results
        clustering_summary = self._summarize_clustering_results(clustering_results)
        
        # Generate summary
        summary = self._generate_executive_summary(
            len(images), features_df.shape[1], clustering_summary
        )
        
        return {
            'title': 'Image Analysis Report',
            'summary': summary,
            'overview_stats': overview_stats,
            'feature_stats': feature_stats,
            'similarity_stats': similarity_stats,
            'clustering_summary': clustering_summary,
            'embeddings': embeddings,
            'config': config,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def _calculate_feature_statistics(self, features_df: pd.DataFrame) -> Dict:
        """Calculate feature statistics."""
        numeric_features = features_df.select_dtypes(include=[np.number])
        
        stats = {
            'n_features': len(numeric_features.columns),
            'n_samples': len(numeric_features),
            'feature_means': numeric_features.mean().to_dict(),
            'feature_stds': numeric_features.std().to_dict(),
            'feature_ranges': (numeric_features.max() - numeric_features.min()).to_dict(),
            'missing_values': numeric_features.isnull().sum().sum(),
            'zero_values': (numeric_features == 0).sum().sum()
        }
        
        return stats
    
    def _calculate_similarity_statistics(self, similarity_matrix: pd.DataFrame) -> Dict:
        """Calculate similarity statistics."""
        # Get upper triangle values (excluding diagonal)
        upper_triangle = similarity_matrix.values[np.triu_indices_from(similarity_matrix.values, k=1)]
        
        stats = {
            'mean_similarity': float(np.mean(upper_triangle)),
            'std_similarity': float(np.std(upper_triangle)),
            'min_similarity': float(np.min(upper_triangle)),
            'max_similarity': float(np.max(upper_triangle)),
            'median_similarity': float(np.median(upper_triangle)),
            'n_comparisons': len(upper_triangle)
        }
        
        return stats
    
    def _summarize_clustering_results(self, clustering_results: Dict) -> Dict:
        """Summarize clustering results."""
        summary = {}
        
        if 'kmeans' in clustering_results and 'optimal' in clustering_results['kmeans']:
            optimal = clustering_results['kmeans']['optimal']
            summary['kmeans'] = {
                'n_clusters': optimal.get('silhouette_score', 0),
                'silhouette_score': optimal.get('silhouette_score', 0),
                'calinski_score': optimal.get('calinski_harabasz_score', 0),
                'davies_score': optimal.get('davies_bouldin_score', 0)
            }
        
        if 'hierarchical' in clustering_results and 'optimal' in clustering_results['hierarchical']:
            optimal = clustering_results['hierarchical']['optimal']
            summary['hierarchical'] = {
                'n_clusters': optimal.get('silhouette_score', 0),
                'silhouette_score': optimal.get('silhouette_score', 0),
                'calinski_score': optimal.get('calinski_harabasz_score', 0),
                'davies_score': optimal.get('davies_bouldin_score', 0)
            }
        
        return summary
    
    def _generate_executive_summary(self, n_images: int, n_features: int, 
                                  clustering_summary: Dict) -> str:
        """Generate executive summary."""
        summary = f"""
        This report presents the results of a comprehensive image analysis pipeline 
        performed on {n_images} multi-channel fluorescent microscopy images. 
        The analysis extracted {n_features} features per image, including statistical, 
        texture, morphological, and color-based characteristics.
        """
        
        if clustering_summary:
            best_method = max(clustering_summary.keys(), 
                            key=lambda x: clustering_summary[x]['silhouette_score'])
            best_score = clustering_summary[best_method]['silhouette_score']
            n_clusters = clustering_summary[best_method]['n_clusters']
            
            summary += f"""
            Clustering analysis using {best_method} identified {n_clusters} distinct groups 
            with a silhouette score of {best_score:.3f}, indicating good cluster separation.
            """
        
        summary += """
        The results provide insights into image similarity patterns and can be used 
        for automated image classification and quality control applications.
        """
        
        return summary
    
    def _generate_html_content(self, report_data: Dict) -> str:
        """Generate the complete HTML content."""
        # Create Jinja2 template
        template = jinja2.Template(self.html_template)
        
        # Generate sections
        statistics_section = self._generate_statistics_section(report_data)
        features_section = self._generate_features_section(report_data)
        similarity_section = self._generate_similarity_section(report_data)
        clustering_section = self._generate_clustering_section(report_data)
        visualizations_section = self._generate_visualizations_section(report_data)
        conclusions_section = self._generate_conclusions_section(report_data)
        
        # Render template
        html_content = template.render(
            title=report_data['title'],
            summary=report_data['summary'],
            overview_stats=report_data['overview_stats'],
            statistics_section=statistics_section,
            features_section=features_section,
            similarity_section=similarity_section,
            clustering_section=clustering_section,
            visualizations_section=visualizations_section,
            conclusions_section=conclusions_section,
            timestamp=report_data['timestamp']
        )
        
        return html_content
    
    def _generate_statistics_section(self, report_data: Dict) -> str:
        """Generate statistics section HTML."""
        stats = report_data['feature_stats']
        
        html = f"""
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">{stats['n_features']}</div>
                <div class="stat-label">Total Features</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{stats['n_samples']}</div>
                <div class="stat-label">Total Samples</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{stats['missing_values']}</div>
                <div class="stat-label">Missing Values</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{stats['zero_values']}</div>
                <div class="stat-label">Zero Values</div>
            </div>
        </div>
        
        <h3>Feature Statistics</h3>
        <div class="table-container">
            <table>
                <tr>
                    <th>Statistic</th>
                    <th>Mean</th>
                    <th>Std</th>
                    <th>Min</th>
                    <th>Max</th>
                </tr>
                <tr>
                    <td>Feature Values</td>
                    <td>{np.mean(list(stats['feature_means'].values())):.3f}</td>
                    <td>{np.mean(list(stats['feature_stds'].values())):.3f}</td>
                    <td>{np.min(list(stats['feature_ranges'].values())):.3f}</td>
                    <td>{np.max(list(stats['feature_ranges'].values())):.3f}</td>
                </tr>
            </table>
        </div>
        """
        
        return html
    
    def _generate_features_section(self, report_data: Dict) -> str:
        """Generate features section HTML."""
        stats = report_data['feature_stats']
        
        # Get top features by variance
        feature_vars = {k: v**2 for k, v in stats['feature_stds'].items()}
        top_features = sorted(feature_vars.items(), key=lambda x: x[1], reverse=True)[:10]
        
        html = f"""
        <h3>Top 10 Most Variable Features</h3>
        <div class="table-container">
            <table>
                <tr>
                    <th>Feature</th>
                    <th>Variance</th>
                    <th>Mean</th>
                    <th>Std</th>
                </tr>
        """
        
        for feature, variance in top_features:
            mean = stats['feature_means'][feature]
            std = stats['feature_stds'][feature]
            html += f"""
                <tr>
                    <td>{feature}</td>
                    <td>{variance:.3f}</td>
                    <td>{mean:.3f}</td>
                    <td>{std:.3f}</td>
                </tr>
            """
        
        html += """
            </table>
        </div>
        """
        
        return html
    
    def _generate_similarity_section(self, report_data: Dict) -> str:
        """Generate similarity section HTML."""
        stats = report_data['similarity_stats']
        
        html = f"""
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">{stats['mean_similarity']:.3f}</div>
                <div class="stat-label">Mean Similarity</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{stats['std_similarity']:.3f}</div>
                <div class="stat-label">Std Similarity</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{stats['min_similarity']:.3f}</div>
                <div class="stat-label">Min Similarity</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{stats['max_similarity']:.3f}</div>
                <div class="stat-label">Max Similarity</div>
            </div>
        </div>
        
        <h3>Similarity Distribution</h3>
        <p>The similarity matrix contains {stats['n_comparisons']} pairwise comparisons 
        between images. The distribution shows how similar the images are to each other.</p>
        """
        
        return html
    
    def _generate_clustering_section(self, report_data: Dict) -> str:
        """Generate clustering section HTML."""
        summary = report_data['clustering_summary']
        
        html = "<div class='clustering-results'>"
        
        for method, results in summary.items():
            html += f"""
            <div class="cluster-card">
                <h3>{method.capitalize()} Clustering</h3>
                <div class="stat-value">{results['n_clusters']}</div>
                <div class="stat-label">Number of Clusters</div>
                <p><strong>Silhouette Score:</strong> {results['silhouette_score']:.3f}</p>
                <p><strong>Calinski-Harabasz Score:</strong> {results['calinski_score']:.3f}</p>
                <p><strong>Davies-Bouldin Score:</strong> {results['davies_score']:.3f}</p>
            </div>
            """
        
        html += "</div>"
        
        return html
    
    def _generate_visualizations_section(self, report_data: Dict) -> str:
        """Generate visualizations section HTML."""
        html = """
        <h3>Generated Plots</h3>
        <p>The following visualizations were generated during the analysis:</p>
        <ul>
            <li><strong>Clustermap:</strong> Hierarchical clustering of image similarities</li>
            <li><strong>UMAP Plot:</strong> Dimensionality reduction visualization</li>
            <li><strong>t-SNE Plot:</strong> Alternative dimensionality reduction</li>
            <li><strong>Feature Distributions:</strong> Histograms of feature values</li>
            <li><strong>Clustering Evaluation:</strong> Metrics for different clustering parameters</li>
            <li><strong>Similarity Heatmap:</strong> Direct visualization of similarity matrix</li>
            <li><strong>Feature Correlation:</strong> Correlation matrix between features</li>
            <li><strong>PCA Variance:</strong> Explained variance by principal components</li>
        </ul>
        
        <p>All plots have been saved in the 'plots' directory within the output folder.</p>
        """
        
        return html
    
    def _generate_conclusions_section(self, report_data: Dict) -> str:
        """Generate conclusions section HTML."""
        summary = report_data['clustering_summary']
        
        html = """
        <h3>Key Findings</h3>
        <ul>
            <li>The feature extraction pipeline successfully captured diverse image characteristics</li>
            <li>Similarity analysis revealed distinct patterns in image relationships</li>
        """
        
        if summary:
            best_method = max(summary.keys(), 
                            key=lambda x: summary[x]['silhouette_score'])
            best_score = summary[best_method]['silhouette_score']
            
            html += f"""
            <li>Clustering analysis using {best_method} achieved a silhouette score of {best_score:.3f}</li>
            <li>The optimal number of clusters provides meaningful image groupings</li>
            """
        
        html += """
        </ul>
        
        <h3>Recommendations</h3>
        <ul>
            <li>Use the similarity matrix for finding similar images in the dataset</li>
            <li>Apply the clustering results for automated image categorization</li>
            <li>Consider the feature importance for future feature selection</li>
            <li>Use the UMAP/t-SNE visualizations for exploratory data analysis</li>
        </ul>
        """
        
        return html 