import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile

@pytest.fixture
def temp_dir():
    """Create temporary directory"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)

@pytest.fixture
def sample_image():
    """Create sample grayscale image"""
    return np.random.randint(0, 255, (512, 512), dtype=np.uint8)

@pytest.fixture
def sample_rgb_image():
    """Create sample RGB image"""
    return np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)

@pytest.fixture
def sample_labels_df():
    """Create sample labels DataFrame"""
    data = {
        'Image Index': [f'img_{i}.png' for i in range(10)],
        'Finding Labels': ['Pneumonia' if i % 2 == 0 else 'No Finding' for i in range(10)],
        'Patient Age': [f'{30+i}Y' for i in range(10)],
        'Patient Gender': ['M' if i % 2 == 0 else 'F' for i in range(10)],
    }
    return pd.DataFrame(data)
