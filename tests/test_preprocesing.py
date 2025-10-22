import pytest
import numpy as np
from data.preprocessing import ImagePreprocessor

class TestImagePreprocessor:
    
    def test_preprocess_grayscale_to_rgb(self, sample_image):
        """Test converting grayscale to RGB"""
        preprocessor = ImagePreprocessor(target_size=(224, 224), channels=3)
        result = preprocessor.preprocess(sample_image)
        
        assert result.shape == (224, 224, 3)
        assert result.dtype == np.float32
        assert 0 <= result.min() <= result.max() <= 1.0
    
    def test_preprocess_rgb(self, sample_rgb_image):
        """Test preprocessing RGB image"""
        preprocessor = ImagePreprocessor(target_size=(224, 224), channels=3)
        result = preprocessor.preprocess(sample_rgb_image)
        
        assert result.shape == (224, 224, 3)
        assert 0 <= result.min() <= result.max() <= 1.0
    
    def test_preprocess_batch(self, sample_image):
        """Test batch preprocessing"""
        preprocessor = ImagePreprocessor(target_size=(224, 224), channels=3)
        images = np.array([sample_image, sample_image])
        result = preprocessor.preprocess_batch(images)
        
        assert result.shape == (2, 224, 224, 3)
