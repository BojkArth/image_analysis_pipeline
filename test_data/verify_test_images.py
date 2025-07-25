#!/usr/bin/env python3
"""
Verify generated test images
"""

import numpy as np
import tifffile
from pathlib import Path
import matplotlib.pyplot as plt

def verify_test_images():
    """Verify the generated test images."""
    
    test_dir = Path("test_images")
    
    if not test_dir.exists():
        print("❌ Test images directory not found!")
        return
    
    # Get all TIFF files
    tiff_files = list(test_dir.glob("*.tif"))
    
    if not tiff_files:
        print("❌ No TIFF files found!")
        return
    
    print(f"✅ Found {len(tiff_files)} TIFF files")
    
    # Check first few images
    for i, tiff_file in enumerate(tiff_files[:3]):
        print(f"\n📊 Analyzing: {tiff_file.name}")
        
        # Load image
        image = tifffile.imread(str(tiff_file))
        
        print(f"   Shape: {image.shape}")
        print(f"   Data type: {image.dtype}")
        print(f"   Min values: {[image[c].min() for c in range(image.shape[0])]}")
        print(f"   Max values: {[image[c].max() for c in range(image.shape[0])]}")
        print(f"   Mean values: {[f'{image[c].mean():.1f}' for c in range(image.shape[0])]}")
        
        # Check if it's 4-channel
        if image.shape[0] == 4:
            print("   ✅ 4-channel image (DAPI, FITC, TRITC, Cy5)")
        else:
            print(f"   ⚠️  Unexpected number of channels: {image.shape[0]}")
    
    # Show image categories
    categories = {}
    for tiff_file in tiff_files:
        category = tiff_file.name.split('_')[0]
        if category not in categories:
            categories[category] = 0
        categories[category] += 1
    
    print(f"\n📁 Image categories:")
    for category, count in categories.items():
        print(f"   {category}: {count} images")
    
    # Check preview image
    preview_file = test_dir / "preview.png"
    if preview_file.exists():
        print(f"\n🖼️  Preview image: {preview_file}")
        print(f"   Size: {preview_file.stat().st_size / 1024 / 1024:.1f} MB")
    else:
        print("\n❌ Preview image not found!")

if __name__ == "__main__":
    verify_test_images() 