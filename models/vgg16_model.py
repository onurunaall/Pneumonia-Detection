from tensorflow import keras
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from .base_model import BaseModel
from typing import Tuple

class VGG16Model(BaseModel):
    def __init__(self, 
                 input_shape: Tuple[int, int, int] = (224, 224, 3),
                 num_classes: int = 1,
                 pretrained: bool = True,
                 freeze_base: bool = True,
                 unfreeze_layers: int = 0,
                 dropout_rate: float = 0.5):
        """
        Args:
            input_shape: Input image shape (height, width, channels)
            num_classes: Number of output classes
            pretrained: Use ImageNet pretrained weights
            freeze_base: Freeze base VGG16 layers
            unfreeze_layers: Number of layers to unfreeze from the end
            dropout_rate: Dropout rate for regularization
        """
        super().__init__(input_shape, num_classes)
        self.pretrained = pretrained
        self.freeze_base = freeze_base
        self.unfreeze_layers = unfreeze_layers
        self.dropout_rate = dropout_rate
    
    def build(self) -> keras.Model:
        weights = 'imagenet' if self.pretrained else None
        base_model = VGG16(include_top=False, weights=weights, input_shape=self.input_shape)
        
        # Freeze base layers if specified
        if self.freeze_base:
            for layer in base_model.layers:
                layer.trainable = False
            
            # Unfreeze last N layers if specified
            if self.unfreeze_layers > 0:
                for layer in base_model.layers[-self.unfreeze_layers:]:
                    layer.trainable = True
        
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(self.dropout_rate)(x)
        x = Dense(256, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(self.dropout_rate)(x)
        
        # Output layer
        if self.num_classes == 1:
            output = Dense(1, activation='sigmoid', name='output')(x)
        else:
            output = Dense(self.num_classes, activation='softmax', name='output')(x)
        
        # Create model
        self.model = keras.Model(inputs=base_model.input, outputs=output)
        
        return self.model
    
    def unfreeze_base_layers(self, num_layers: int):
        """Unfreeze specified number of base layers for fine-tuning"""
        if self.model is None:
            raise ValueError("Model not built yet")
        
        # Get the base model (first layer is the VGG16 model)
        base_model = self.model.layers[0]
        
        # Freeze all layers first
        for layer in base_model.layers:
            layer.trainable = False
        
        # Unfreeze last N layers
        for layer in base_model.layers[-num_layers:]:
            layer.trainable = True
        
        print(f"Unfroze last {num_layers} layers of base model")
