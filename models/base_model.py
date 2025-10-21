from abc import ABC, abstractmethod
from tensorflow import keras
from typing import Tuple

class BaseModel(ABC):
    """Abstract base class for all models"""
    
    def __init__(self, input_shape: Tuple[int, int, int], num_classes: int = 1):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
    
    @abstractmethod
    def build(self) -> keras.Model:
        pass
    
    def compile(self, 
                optimizer: str = 'adam',
                learning_rate: float = 1e-4,
                loss: str = 'binary_crossentropy',
                metrics: list = None):
        if metrics is None:
            metrics = ['accuracy', keras.metrics.AUC(name='auc')]
        
        if optimizer == 'adam':
            opt = keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer == 'sgd':
            opt = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")
        
        self.model.compile(optimizer=opt, loss=loss, metrics=metrics)
    
    def summary(self):
        if self.model:
            return self.model.summary()
        raise ValueError("Model not built yet")
    
    def save(self, path: str):
        if self.model:
            self.model.save(path)
        else:
            raise ValueError("Model not built yet")
    
    @staticmethod
    def load(path: str) -> keras.Model:
        return keras.models.load_model(path)
