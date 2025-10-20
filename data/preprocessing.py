import numpy as np
from skimage.transform import resize
from typing import Tuple, Optional
import cv2

class ImagePreprocessor:
    """Handles image preprocessing for pneumonia detection"""
    
    def __init__(self,
                 target_size: Tuple[int, int] = (224, 224),
                 channels: int = 3,
                 normalize: bool = True,
                 clahe: bool = False):
        """
        Args:
            target_size: Target image size (height, width)
            channels: Number of output channels
            normalize: Whether to normalize to [0, 1]
            clahe: Whether to apply CLAHE for contrast enhancement
        """
        self.target_size = target_size
        self.channels = channels
        self.normalize = normalize
        self.clahe = clahe
        
        if clahe:
            self.clahe_processor = cv2.createCLAHE(
                clipLimit=2.0, 
                tileGridSize=(8, 8)
            )
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess a single image
        
        Args:
            image: Input image array
            
        Returns:
            Preprocessed image
        """
        # Convert to float
        img = image.astype(np.float32)
        
        # Normalize to [0, 1] if needed
        if self.normalize:
            if img.max() > 1.0:
                img = img / 255.0
        
        # Apply CLAHE if enabled
        if self.clahe:
            img_uint8 = (img * 255).astype(np.uint8)
            img = self.clahe_processor.apply(img_uint8).astype(np.float32) / 255.0
        
        img = resize(img, self.target_size, anti_aliasing=True, preserve_range=True)
        
        # Convert to RGB if needed
        if self.channels == 3:
            if len(img.shape) == 2:  # Grayscale
                img = np.stack([img] * 3, axis=-1)
            elif img.shape[-1] == 1:
                img = np.repeat(img, 3, axis=-1)
        elif self.channels == 1:
            if len(img.shape) == 3:
                img = img[:, :, 0:1]
            else:
                img = np.expand_dims(img, axis=-1)
        
        return img
    
    def preprocess_batch(self, images: np.ndarray) -> np.ndarray:
        """Preprocess a batch of images"""
        return np.array([self.preprocess(img) for img in images])
    
    def preprocess_for_inference(self, image: np.ndarray) -> np.ndarray:
        """Preprocess single image and add batch dimension"""
        img = self.preprocess(image)
        return np.expand_dims(img, axis=0)

class DICOMPreprocessor(ImagePreprocessor):
    """Specialized preprocessor for DICOM images"""
    
    def __init__(self, *args, window_center: Optional[int] = None, 
                 window_width: Optional[int] = None, **kwargs):
        """
        Args:
            window_center: Window center for DICOM windowing
            window_width: Window width for DICOM windowing
        """
        super().__init__(*args, **kwargs)
        self.window_center = window_center
        self.window_width = window_width
    
    def apply_window(self, image: np.ndarray) -> np.ndarray:
        """Apply windowing to DICOM image"""
        if self.window_center is None or self.window_width is None:
            return image
        
        img_min = self.window_center - self.window_width // 2
        img_max = self.window_center + self.window_width // 2
        
        image = np.clip(image, img_min, img_max)
        image = (image - img_min) / (img_max - img_min)
        
        return image
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess DICOM image with windowing"""
        img = self.apply_window(image)
        return super().preprocess(img)
