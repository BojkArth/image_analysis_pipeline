# Test Data Generation for Image Analysis Pipeline

This directory contains scripts to generate synthetic cell painting TIFF images for testing the image analysis pipeline.

## 🎯 Overview

The test data generator creates realistic multi-channel fluorescent microscopy images that simulate cell painting experiments with:

- **Channel 1**: DAPI (nuclei) - Blue fluorescence
- **Channel 2**: FITC (actin filaments) - Green fluorescence  
- **Channel 3**: TRITC (microtubules) - Red fluorescence
- **Channel 4**: Cy5 (mitochondria) - Far red fluorescence

## 🏗️ Image Characteristics

### **Cell Types Generated:**

1. **Normal Density** (10 images)
   - Standard cell density and distribution
   - Balanced staining across all channels

2. **High Density** (8 images)
   - Increased cell density
   - Higher fluorescence intensities
   - More crowded cell patterns

3. **Low Density** (8 images)
   - Reduced cell density
   - Lower fluorescence intensities
   - Sparse cell distribution

4. **Clustered** (8 images)
   - Cells arranged in distinct clusters
   - Grouped cell patterns
   - Varying cluster sizes

5. **Sparse** (6 images)
   - Very few, large cells
   - Well-separated cell distribution
   - Prominent cellular features

### **Technical Specifications:**

- **Image Size**: 1024 × 1024 pixels
- **Bit Depth**: 16-bit TIFF
- **Channels**: 4-channel multi-channel TIFF
- **Compression**: LZW compression
- **Total Images**: 40 test images

## 🚀 Usage

### **Generate Test Images:**

```bash
# Navigate to the image_analysis directory
cd image_analysis

# Run the test generation script
python test_data/run_test_generation.py
```

### **Test the Pipeline:**

```bash
# Run the complete pipeline on test data
python cli.py --input-dir test_images --output-dir test_results

# Or with custom configuration
python cli.py --config example_config.yaml --input-dir test_images --output-dir test_results
```

## 📁 Output Structure

After running the generation script, you'll get:

```
test_images/
├── normal_cell_painting_001.tif
├── normal_cell_painting_002.tif
├── ...
├── high_density_cell_painting_001.tif
├── high_density_cell_painting_002.tif
├── ...
├── low_density_cell_painting_001.tif
├── ...
├── clustered_cell_painting_001.tif
├── ...
├── sparse_cell_painting_001.tif
├── ...
└── preview.png                    # Preview of different cell types
```

## 🔬 Cell Painting Features

### **Channel 1 - DAPI (Nuclei):**
- Circular nuclear structures
- Variable nuclear sizes (8-20 pixel radius)
- Nuclear texture and noise
- Blue fluorescence simulation

### **Channel 2 - FITC (Actin):**
- Radial actin filament patterns
- Variable filament density (8-20 per cell)
- Cell boundary-defined staining
- Green fluorescence simulation

### **Channel 3 - TRITC (Microtubules):**
- Straight microtubule structures
- Radial organization from cell center
- Variable microtubule density (10-25 per cell)
- Red fluorescence simulation

### **Channel 4 - Cy5 (Mitochondria):**
- Punctate mitochondrial structures
- Small circular organelles (2-6 pixel radius)
- Variable mitochondrial density (15-40 per cell)
- Far red fluorescence simulation

## 🎨 Visualization

The generator creates a `preview.png` file showing:
- RGB composite of first 3 channels
- Examples of different cell types
- Visual comparison of cell patterns

## ⚙️ Customization

You can modify the generation parameters in `generate_test_images.py`:

- **Cell density**: Adjust `n_cells` parameters
- **Image size**: Change `height` and `width` parameters
- **Fluorescence intensity**: Modify intensity ranges
- **Noise levels**: Adjust `var` parameters in `random_noise()`
- **Cell patterns**: Modify clustering and distribution logic

## 🔍 Expected Pipeline Results

When running the pipeline on these test images, you should see:

1. **Feature Matrix**: ~1000+ features per image
2. **Clustering**: Clear separation between cell types
3. **UMAP Visualization**: Distinct clusters for different cell types
4. **Similarity Analysis**: Higher similarity within cell types
5. **HTML Report**: Comprehensive analysis results

## 🐛 Troubleshooting

### **Import Errors:**
```bash
pip install numpy tifffile matplotlib scikit-image
```

### **Memory Issues:**
- Reduce image size in generation script
- Process fewer images at once

### **File Permission Errors:**
- Ensure write permissions in output directory
- Check available disk space

## 📊 Validation

The generated images provide a good test case because they:
- Have known cell type categories
- Show clear morphological differences
- Include realistic noise and artifacts
- Cover various cell densities and patterns
- Simulate real fluorescent microscopy data

This allows you to validate that the pipeline correctly:
- Extracts meaningful features
- Identifies cell type differences
- Groups similar images together
- Generates informative visualizations 