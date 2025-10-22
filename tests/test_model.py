import pytest
from models.vgg16_model import VGG16Model

class TestVGG16Model:
    
    def test_model_build(self):
        """Test model building"""
        model = VGG16Model(input_shape=(224, 224, 3), num_classes=1)
        keras_model = model.build()
        
        assert keras_model is not None
        assert keras_model.input_shape == (None, 224, 224, 3)
        assert keras_model.output_shape == (None, 1)
    
    def test_model_compile(self):
        """Test model compilation"""
        model = VGG16Model(input_shape=(224, 224, 3), num_classes=1)
        model.build()
        model.compile(optimizer='adam', learning_rate=1e-4)
        
        assert model.model.optimizer is not None
