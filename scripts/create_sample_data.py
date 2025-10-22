#!/usr/bin/env python3
"""
Create sample data for testing the pneumonia detection pipeline
"""

import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import sys

def create_sample_data(data_dir: Path, num_samples: int = 20):
    """
    Create sample chest X-ray images and labels CSV
    
    Args:
        data_dir: Directory to save sample data
        num_samples: Number of sample images to create
    """
    # Create directories
    images_dir = data_dir / 'images'
    images_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating {num_samples} sample images in {images_dir}")
    
    # Create sample images
    image_files = []
    for i in range(num_samples):
        # Create realistic-looking grayscale chest X-ray (random noise)
        img = np.random.randint(50, 200, (1024, 1024), dtype=np.uint8)
        
        # Add some structure to make it look more like an X-ray
        center_y, center_x = 512, 512
        y, x = np.ogrid[:1024, :1024]
        mask = (x - center_x)**2 + (y - center_y)**2 <= 400**2
        img[mask] = np.clip(img[mask] + 30, 0, 255)
        
        filename = f'sample_xray_{i:04d}.png'
        img_pil = Image.fromarray(img, mode='L')
        img_pil.save(images_dir / filename)
        image_files.append(filename)
    
    print(f"Created {len(image_files)} images")
    
    # Create labels CSV
    labels_data = {
        'Image Index': image_files,
        'Finding Labels': [
            'Pneumonia' if i % 3 == 0 else 'No Finding' 
            for i in range(num_samples)
        ],
        'Patient Age': [f'{np.random.randint(20, 80)}Y' for _ in range(num_samples)],
        'Patient Gender': [
            np.random.choice(['M', 'F']) for _ in range(num_samples)
        ],
        'View Position': [
            np.random.choice(['PA', 'AP']) for _ in range(num_samples)
        ],
    }
    
    df = pd.DataFrame(labels_data)
    csv_path = data_dir / 'sample_labels.csv'
    df.to_csv(csv_path, index=False)
    
    print(f"Created labels CSV: {csv_path}")
    print(f"\nLabel distribution:")
    print(df['Finding Labels'].value_counts())
    print(f"\nSample data created successfully!")
    print(f"Images: {images_dir}")
    print(f"Labels: {csv_path}")

if __name__ == '__main__':
    # Default to data/raw directory
    project_root = Path(__file__).parent.parent
    data_dir = project_root / 'data' / 'raw'
    
    # Parse arguments
    if len(sys.argv) > 1:
        num_samples = int(sys.argv[1])
    else:
        num_samples = 50
    
    create_sample_data(data_dir, num_samples)
