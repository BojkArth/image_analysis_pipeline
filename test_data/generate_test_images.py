#!/usr/bin/env python3
"""
Generate Synthetic Cell Painting TIFF Images for Testing

Creates realistic multi-channel fluorescent microscopy images with:
- Channel 1: DAPI (nuclei) - Blue
- Channel 2: FITC (actin) - Green  
- Channel 3: TRITC (tubulin) - Red
- Channel 4: Cy5 (mitochondria) - Far Red

Each image contains different cell patterns, densities, and staining intensities.
"""

import numpy as np
import tifffile
from pathlib import Path
import matplotlib.pyplot as plt
from skimage import morphology, filters
from skimage.draw import disk, line
from skimage.util import random_noise
import random
import os

def create_nuclei_channel(height=1024, width=1024, n_cells=50):
    """Create DAPI channel with nuclei."""
    # Create empty image
    nuclei = np.zeros((height, width), dtype=np.float32)
    
    # Generate random nuclei positions
    for _ in range(n_cells):
        # Random position
        y = random.randint(50, height-50)
        x = random.randint(50, width-50)
        
        # Random nucleus size
        radius = random.uniform(8, 20)
        
        # Create circular nucleus
        rr, cc = disk((y, x), radius, shape=(height, width))
        intensity = random.uniform(0.3, 1.0)
        nuclei[rr, cc] = intensity
    
    # Add some nuclear texture
    nuclei = filters.gaussian(nuclei, sigma=1.0)
    
    # Add noise
    nuclei = random_noise(nuclei, mode='gaussian', var=0.01)
    
    return nuclei

def create_actin_channel(height=1024, width=1024, n_cells=30):
    """Create FITC channel with actin filaments."""
    actin = np.zeros((height, width), dtype=np.float32)
    
    for _ in range(n_cells):
        # Random cell center
        y = random.randint(100, height-100)
        x = random.randint(100, width-100)
        
        # Cell size
        cell_radius = random.uniform(30, 80)
        
        # Create cell boundary
        rr, cc = disk((y, x), cell_radius, shape=(height, width))
        
        # Add actin filaments (radial patterns)
        n_filaments = random.randint(8, 20)
        for i in range(n_filaments):
            angle = 2 * np.pi * i / n_filaments
            length = random.uniform(20, cell_radius)
            
            # Create filament
            end_y = int(y + length * np.sin(angle))
            end_x = int(x + length * np.cos(angle))
            
            # Draw line
            rr_line, cc_line = line(y, x, end_y, end_x)
            
            # Keep only points within cell boundary
            mask = (rr_line >= 0) & (rr_line < height) & (cc_line >= 0) & (cc_line < width)
            rr_line = rr_line[mask]
            cc_line = cc_line[mask]
            
            # Add filament intensity
            intensity = random.uniform(0.2, 0.8)
            actin[rr_line, cc_line] = np.maximum(actin[rr_line, cc_line], intensity)
    
    # Smooth and add noise
    actin = filters.gaussian(actin, sigma=1.5)
    actin = random_noise(actin, mode='gaussian', var=0.02)
    
    return actin

def create_tubulin_channel(height=1024, width=1024, n_cells=25):
    """Create TRITC channel with microtubules."""
    tubulin = np.zeros((height, width), dtype=np.float32)
    
    for _ in range(n_cells):
        # Random cell center
        y = random.randint(100, height-100)
        x = random.randint(100, width-100)
        
        # Cell size
        cell_radius = random.uniform(25, 70)
        
        # Create cell boundary
        rr, cc = disk((y, x), cell_radius, shape=(height, width))
        
        # Add microtubules (straight lines from center)
        n_microtubules = random.randint(10, 25)
        for i in range(n_microtubules):
            angle = 2 * np.pi * i / n_microtubules
            length = random.uniform(15, cell_radius * 0.8)
            
            # Create microtubule
            end_y = int(y + length * np.sin(angle))
            end_x = int(x + length * np.cos(angle))
            
            # Draw line
            rr_line, cc_line = line(y, x, end_y, end_x)
            
            # Keep only points within cell boundary
            mask = (rr_line >= 0) & (rr_line < height) & (cc_line >= 0) & (cc_line < width)
            rr_line = rr_line[mask]
            cc_line = cc_line[mask]
            
            # Add microtubule intensity
            intensity = random.uniform(0.3, 0.9)
            tubulin[rr_line, cc_line] = np.maximum(tubulin[rr_line, cc_line], intensity)
    
    # Smooth and add noise
    tubulin = filters.gaussian(tubulin, sigma=1.0)
    tubulin = random_noise(tubulin, mode='gaussian', var=0.015)
    
    return tubulin

def create_mitochondria_channel(height=1024, width=1024, n_cells=40):
    """Create Cy5 channel with mitochondria."""
    mitochondria = np.zeros((height, width), dtype=np.float32)
    
    for _ in range(n_cells):
        # Random cell center
        y = random.randint(80, height-80)
        x = random.randint(80, width-80)
        
        # Cell size
        cell_radius = random.uniform(20, 60)
        
        # Create cell boundary
        rr, cc = disk((y, x), cell_radius, shape=(height, width))
        
        # Add mitochondria (small punctate structures)
        n_mitochondria = random.randint(15, 40)
        for i in range(n_mitochondria):
            # Random position within cell
            angle = random.uniform(0, 2 * np.pi)
            distance = random.uniform(0, cell_radius * 0.7)
            
            mito_y = int(y + distance * np.sin(angle))
            mito_x = int(x + distance * np.cos(angle))
            
            # Mitochondria size
            mito_radius = random.uniform(2, 6)
            
            # Create mitochondria
            rr_mito, cc_mito = disk((mito_y, mito_x), mito_radius, shape=(height, width))
            
            # Add intensity
            intensity = random.uniform(0.4, 1.0)
            mitochondria[rr_mito, cc_mito] = np.maximum(mitochondria[rr_mito, cc_mito], intensity)
    
    # Smooth and add noise
    mitochondria = filters.gaussian(mitochondria, sigma=0.8)
    mitochondria = random_noise(mitochondria, mode='gaussian', var=0.025)
    
    return mitochondria

def create_cell_painting_image(image_id, height=1024, width=1024):
    """Create a complete cell painting image with 4 channels."""
    
    # Set random seed for reproducibility
    random.seed(image_id)
    np.random.seed(image_id)
    
    # Vary cell density and patterns based on image ID
    base_cells = 30
    variation = int(image_id % 20)  # 0-19 variation
    
    # Create channels
    nuclei = create_nuclei_channel(height, width, base_cells + variation)
    actin = create_actin_channel(height, width, base_cells + variation - 5)
    tubulin = create_tubulin_channel(height, width, base_cells + variation - 8)
    mitochondria = create_mitochondria_channel(height, width, base_cells + variation + 5)
    
    # Normalize each channel
    nuclei = (nuclei - nuclei.min()) / (nuclei.max() - nuclei.min())
    actin = (actin - actin.min()) / (actin.max() - actin.min())
    tubulin = (tubulin - tubulin.min()) / (tubulin.max() - tubulin.min())
    mitochondria = (mitochondria - mitochondria.min()) / (mitochondria.max() - mitochondria.min())
    
    # Scale to 16-bit range
    nuclei = (nuclei * 65535).astype(np.uint16)
    actin = (actin * 65535).astype(np.uint16)
    tubulin = (tubulin * 65535).astype(np.uint16)
    mitochondria = (mitochondria * 65535).astype(np.uint16)
    
    # Stack channels (channels first)
    image = np.stack([nuclei, actin, tubulin, mitochondria], axis=0)
    
    return image

def create_different_cell_types():
    """Create images with different cell types and patterns."""
    
    # Create output directory
    output_dir = Path("test_images")
    output_dir.mkdir(exist_ok=True)
    
    print(f"Generating test images in: {output_dir}")
    
    # Generate different types of images
    image_configs = [
        # Normal cells
        {"prefix": "normal", "count": 10, "description": "Normal cell density"},
        
        # High density
        {"prefix": "high_density", "count": 8, "description": "High cell density"},
        
        # Low density  
        {"prefix": "low_density", "count": 8, "description": "Low cell density"},
        
        # Clustered
        {"prefix": "clustered", "count": 8, "description": "Clustered cell patterns"},
        
        # Sparse
        {"prefix": "sparse", "count": 6, "description": "Sparse cell distribution"}
    ]
    
    image_id = 0
    
    for config in image_configs:
        prefix = config["prefix"]
        count = config["count"]
        description = config["description"]
        
        print(f"Generating {count} {description} images...")
        
        for i in range(count):
            # Create image with specific characteristics
            if prefix == "high_density":
                # More cells, higher intensity
                image = create_cell_painting_image(image_id, height=1024, width=1024)
                # Increase intensity
                image = np.clip(image * 1.5, 0, 65535).astype(np.uint16)
                
            elif prefix == "low_density":
                # Fewer cells, lower intensity
                image = create_cell_painting_image(image_id, height=1024, width=1024)
                # Decrease intensity
                image = np.clip(image * 0.7, 0, 65535).astype(np.uint16)
                
            elif prefix == "clustered":
                # Create clustered pattern
                image = create_clustered_cells(image_id)
                
            elif prefix == "sparse":
                # Create sparse pattern
                image = create_sparse_cells(image_id)
                
            else:
                # Normal image
                image = create_cell_painting_image(image_id, height=1024, width=1024)
            
            # Save image
            filename = f"{prefix}_cell_painting_{i+1:03d}.tif"
            filepath = output_dir / filename
            
            # Save as multi-channel TIFF
            tifffile.imwrite(
                str(filepath),
                image,
                photometric='minisblack',
                compression='lzw'
            )
            
            print(f"  Saved: {filename}")
            image_id += 1
    
    print(f"\nGenerated {image_id} test images in {output_dir}")
    print("Image types:")
    for config in image_configs:
        print(f"  - {config['prefix']}: {config['count']} images ({config['description']})")
    
    return output_dir

def create_clustered_cells(image_id):
    """Create image with clustered cell patterns."""
    height, width = 1024, 1024
    
    # Set random seed
    random.seed(image_id)
    np.random.seed(image_id)
    
    # Create clusters
    n_clusters = random.randint(3, 8)
    clusters = []
    
    for _ in range(n_clusters):
        # Cluster center
        center_y = random.randint(150, height-150)
        center_x = random.randint(150, width-150)
        cluster_radius = random.uniform(80, 150)
        n_cells_in_cluster = random.randint(8, 20)
        
        clusters.append({
            'center': (center_y, center_x),
            'radius': cluster_radius,
            'n_cells': n_cells_in_cluster
        })
    
    # Create channels with clustered patterns
    nuclei = np.zeros((height, width), dtype=np.float32)
    actin = np.zeros((height, width), dtype=np.float32)
    tubulin = np.zeros((height, width), dtype=np.float32)
    mitochondria = np.zeros((height, width), dtype=np.float32)
    
    for cluster in clusters:
        center_y, center_x = cluster['center']
        cluster_radius = cluster['radius']
        n_cells = cluster['n_cells']
        
        for _ in range(n_cells):
            # Random position within cluster
            angle = random.uniform(0, 2 * np.pi)
            distance = random.uniform(0, cluster_radius * 0.8)
            
            y = int(center_y + distance * np.sin(angle))
            x = int(center_x + distance * np.cos(angle))
            
            # Create cell components
            cell_radius = random.uniform(20, 50)
            
            # Nucleus
            nuc_radius = random.uniform(6, 15)
            rr, cc = disk((y, x), nuc_radius, shape=(height, width))
            nuclei[rr, cc] = random.uniform(0.4, 1.0)
            
            # Actin filaments
            n_filaments = random.randint(6, 15)
            for i in range(n_filaments):
                angle_fil = 2 * np.pi * i / n_filaments
                length = random.uniform(15, cell_radius)
                end_y = int(y + length * np.sin(angle_fil))
                end_x = int(x + length * np.cos(angle_fil))
                rr_line, cc_line = line(y, x, end_y, end_x)
                mask = (rr_line >= 0) & (rr_line < height) & (cc_line >= 0) & (cc_line < width)
                actin[rr_line[mask], cc_line[mask]] = random.uniform(0.3, 0.8)
            
            # Microtubules
            n_microtubules = random.randint(8, 18)
            for i in range(n_microtubules):
                angle_mt = 2 * np.pi * i / n_microtubules
                length = random.uniform(10, cell_radius * 0.7)
                end_y = int(y + length * np.sin(angle_mt))
                end_x = int(x + length * np.cos(angle_mt))
                rr_line, cc_line = line(y, x, end_y, end_x)
                mask = (rr_line >= 0) & (rr_line < height) & (cc_line >= 0) & (cc_line < width)
                tubulin[rr_line[mask], cc_line[mask]] = random.uniform(0.4, 0.9)
            
            # Mitochondria
            n_mitochondria = random.randint(10, 25)
            for i in range(n_mitochondria):
                angle_mito = random.uniform(0, 2 * np.pi)
                distance_mito = random.uniform(0, cell_radius * 0.6)
                mito_y = int(y + distance_mito * np.sin(angle_mito))
                mito_x = int(x + distance_mito * np.cos(angle_mito))
                mito_radius = random.uniform(2, 5)
                rr_mito, cc_mito = disk((mito_y, mito_x), mito_radius, shape=(height, width))
                mitochondria[rr_mito, cc_mito] = random.uniform(0.5, 1.0)
    
    # Process channels
    channels = [nuclei, actin, tubulin, mitochondria]
    processed_channels = []
    
    for channel in channels:
        channel = filters.gaussian(channel, sigma=1.0)
        channel = random_noise(channel, mode='gaussian', var=0.02)
        channel = (channel - channel.min()) / (channel.max() - channel.min())
        channel = (channel * 65535).astype(np.uint16)
        processed_channels.append(channel)
    
    return np.stack(processed_channels, axis=0)

def create_sparse_cells(image_id):
    """Create image with sparse cell distribution."""
    height, width = 1024, 1024
    
    # Set random seed
    random.seed(image_id)
    np.random.seed(image_id)
    
    # Fewer cells, more spread out
    n_cells = random.randint(8, 15)
    
    # Create channels
    nuclei = np.zeros((height, width), dtype=np.float32)
    actin = np.zeros((height, width), dtype=np.float32)
    tubulin = np.zeros((height, width), dtype=np.float32)
    mitochondria = np.zeros((height, width), dtype=np.float32)
    
    for _ in range(n_cells):
        # Spread out positions
        y = random.randint(100, height-100)
        x = random.randint(100, width-100)
        
        # Larger cells
        cell_radius = random.uniform(40, 100)
        
        # Nucleus
        nuc_radius = random.uniform(10, 25)
        rr, cc = disk((y, x), nuc_radius, shape=(height, width))
        nuclei[rr, cc] = random.uniform(0.5, 1.0)
        
        # Actin filaments (more prominent)
        n_filaments = random.randint(12, 25)
        for i in range(n_filaments):
            angle = 2 * np.pi * i / n_filaments
            length = random.uniform(25, cell_radius)
            end_y = int(y + length * np.sin(angle))
            end_x = int(x + length * np.cos(angle))
            rr_line, cc_line = line(y, x, end_y, end_x)
            mask = (rr_line >= 0) & (rr_line < height) & (cc_line >= 0) & (cc_line < width)
            actin[rr_line[mask], cc_line[mask]] = random.uniform(0.4, 0.9)
        
        # Microtubules
        n_microtubules = random.randint(15, 30)
        for i in range(n_microtubules):
            angle = 2 * np.pi * i / n_microtubules
            length = random.uniform(20, cell_radius * 0.8)
            end_y = int(y + length * np.sin(angle))
            end_x = int(x + length * np.cos(angle))
            rr_line, cc_line = line(y, x, end_y, end_x)
            mask = (rr_line >= 0) & (rr_line < height) & (cc_line >= 0) & (cc_line < width)
            tubulin[rr_line[mask], cc_line[mask]] = random.uniform(0.5, 1.0)
        
        # Mitochondria (more numerous)
        n_mitochondria = random.randint(20, 50)
        for i in range(n_mitochondria):
            angle = random.uniform(0, 2 * np.pi)
            distance = random.uniform(0, cell_radius * 0.7)
            mito_y = int(y + distance * np.sin(angle))
            mito_x = int(x + distance * np.cos(angle))
            mito_radius = random.uniform(3, 8)
            rr_mito, cc_mito = disk((mito_y, mito_x), mito_radius, shape=(height, width))
            mitochondria[rr_mito, cc_mito] = random.uniform(0.6, 1.0)
    
    # Process channels
    channels = [nuclei, actin, tubulin, mitochondria]
    processed_channels = []
    
    for channel in channels:
        channel = filters.gaussian(channel, sigma=1.2)
        channel = random_noise(channel, mode='gaussian', var=0.015)
        channel = (channel - channel.min()) / (channel.max() - channel.min())
        channel = (channel * 65535).astype(np.uint16)
        processed_channels.append(channel)
    
    return np.stack(processed_channels, axis=0)

def create_preview_image(output_dir):
    """Create a preview image showing different cell types."""
    # Load a few representative images
    image_files = list(Path(output_dir).glob("*.tif"))[:4]
    
    if len(image_files) < 4:
        print("Not enough images for preview")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()
    
    for i, image_file in enumerate(image_files):
        # Load image
        image = tifffile.imread(str(image_file))
        
        # Create RGB composite (normalize each channel)
        composite = np.zeros((image.shape[1], image.shape[2], 3))
        
        # Channel 1 (DAPI) -> Blue
        composite[:, :, 2] = image[0] / image[0].max()
        
        # Channel 2 (FITC) -> Green
        composite[:, :, 1] = image[1] / image[1].max()
        
        # Channel 3 (TRITC) -> Red
        composite[:, :, 0] = image[2] / image[2].max()
        
        # Display
        axes[i].imshow(composite)
        axes[i].set_title(f"{image_file.stem}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / "preview.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Preview image saved: {output_dir / 'preview.png'}")

if __name__ == "__main__":
    # Generate test images
    output_dir = create_different_cell_types()
    
    # Create preview
    create_preview_image(output_dir)
    
    print("\nTest data generation complete!")
    print(f"Use these images to test the pipeline:")
    print(f"python image_analysis/cli.py --input-dir {output_dir} --output-dir test_results") 