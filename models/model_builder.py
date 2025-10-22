from typing import Optional
from tensorflow import keras
from .vgg16_model import VGG16Model
from .base_model import BaseModel

class ModelBuilder:
    """Factory for creating models"""
    
    SUPPORTED_ARCHITECTURES = {
        'vgg16': VGG16Model,
        # 'resnet50': ResNet50Model,
        # 'efficientnet': EfficientNetModel,
    }
    
    @classmethod
    def create_model(cls, architecture: str = 'vgg16', **kwargs) -> BaseModel:
        if architecture not in cls.SUPPORTED_ARCHITECTURES:
            raise ValueError(
                f"Unknown architecture: {architecture}. "
                f"Supported: {list(cls.SUPPORTED_ARCHITECTURES.keys())}"
            )
        
        model_class = cls.SUPPORTED_ARCHITECTURES[architecture]
        return model_class(**kwargs)
    
    @classmethod
    def from_config(cls, config) -> BaseModel:
        return cls.create_model(architecture=config.model.architecture,
                                input_shape=(config.data.image_size[0], config.data.image_size[1], config.data.channels),
                                num_classes=config.model.num_classes,
                                pretrained=config.model.pretrained,
                                freeze_base=config.model.freeze_base,
                                unfreeze_layers=config.model.unfreeze_layers,
                                dropout_rate=config.model.dropout_rate)
