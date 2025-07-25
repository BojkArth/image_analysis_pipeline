#!/usr/bin/env python3
"""
Simple script to run test image generation
"""

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def main():
    try:
        # Import and run the generation script
        from test_data.generate_test_images import create_different_cell_types, create_preview_image
        
        print("Starting test image generation...")
        
        # Generate test images
        output_dir = create_different_cell_types()
        
        # Create preview
        create_preview_image(output_dir)
        
        print(f"\n✅ Test data generation complete!")
        print(f"📁 Generated images in: {output_dir}")
        print(f"🔍 Preview image: {output_dir / 'preview.png'}")
        print(f"\n🚀 To test the pipeline, run:")
        print(f"python image_analysis/cli.py --input-dir {output_dir} --output-dir test_results")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please ensure all required packages are installed:")
        print("pip install numpy tifffile matplotlib scikit-image")
        return 1
    except Exception as e:
        print(f"❌ Error generating test images: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 