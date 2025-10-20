from tensorflow.keras.preprocessing.image import ImageDataGenerator
import albumentations as A
from typing import Optional

class DataAugmentor:
    """Handles data augmentation for training"""
    
    def __init__(self, 
                 rotation_range: int = 10,
                 width_shift_range: float = 0.1,
                 height_shift_range: float = 0.1,
                 horizontal_flip: bool = True,
                 zoom_range: float = 0.1,
                 brightness_range: Optional[tuple] = (0.9, 1.1),
                 fill_mode: str = 'nearest'):
        self.rotation_range = rotation_range
        self.width_shift_range = width_shift_range
        self.height_shift_range = height_shift_range
        self.horizontal_flip = horizontal_flip
        self.zoom_range = zoom_range
        self.brightness_range = brightness_range
        self.fill_mode = fill_mode
    
    def get_keras_generator(self) -> ImageDataGenerator:
        return ImageDataGenerator(rotation_range=self.rotation_range,
                                  width_shift_range=self.width_shift_range,
                                  height_shift_range=self.height_shift_range,
                                  horizontal_flip=self.horizontal_flip,
                                  zoom_range=self.zoom_range,
                                  brightness_range=self.brightness_range,
                                  fill_mode=self.fill_mode)
    
    def get_albumentations_transform(self) -> A.Compose:
        """Get Albumentations transform (more flexible)"""
        return A.Compose([
            A.Rotate(limit=self.rotation_range, p=0.5),
            A.ShiftScaleRotate(shift_limit=self.width_shift_range, scale_limit=self.zoom_range, rotate_limit=0, p=0.5),
            A.HorizontalFlip(p=0.5 if self.horizontal_flip else 0),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        ])

class PneumoniaAugmentor(DataAugmentor):
    """Specialized augmentor for pneumonia detection"""
    
    def __init__(self):
        """Use medical imaging appropriate augmentations"""
        super().__init__(
            rotation_range=5,  # Small rotations
            width_shift_range=0.05,
            height_shift_range=0.05,
            horizontal_flip=True,  # Chest X-rays can be flipped
            zoom_range=0.05,
            brightness_range=(0.95, 1.05),  # Conservative brightness
            fill_mode='constant'
        )
