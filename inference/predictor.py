import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict
import logging
from tensorflow import keras

logger = logging.getLogger(__name__)

class PneumoniaPredictor:
    """Handles pneumonia prediction on new images"""
    def __init__(self,
                 model_path: Path,
                 weight_path: Optional[Path] = None,
                 preprocessor = None,
                 validator = None,
                 threshold: float = 0.5):
        """
        Args:
            model_path: Path to model file (.h5 for full model, .json for architecture only)
            weight_path: Path to model weights (only needed if model_path is .json)
            preprocessor: ImagePreprocessor instance
            validator: DICOMValidator instance
            threshold: Classification threshold
        """
        self.model = self._load_model(model_path, weight_path)
        self.preprocessor = preprocessor
        self.validator = validator
        self.threshold = threshold
    
    def _load_model(self, model_path: Path, weight_path: Optional[Path] = None) -> keras.Model:
        """
        Load model - supports both:
        1. Full model from .h5 file (model_path only)
        2. Architecture from .json + weights from .h5 (both paths)
        """
        if model_path.suffix == '.h5':
            # Load full model from single .h5 file
            model = keras.models.load_model(str(model_path))
            logger.info(f"Model loaded from {model_path}")
        
        elif model_path.suffix == '.json':
            # Load architecture from JSON and weights separately
            if weight_path is None:
                raise ValueError("weight_path must be provided when loading from JSON")
            
            with open(model_path, 'r') as json_file:
                model_json = json_file.read()
            
            model = keras.models.model_from_json(model_json)
            model.load_weights(str(weight_path))
            logger.info(f"Model loaded from {model_path} and {weight_path}")
        else:
            raise ValueError(f"Unsupported model file format: {model_path.suffix}")
        
        return model
    
    
    def predict_dicom(self, dicom_path: Path) -> Tuple[Optional[str], Optional[float], Optional[Dict]]:
        """
        Returns:
            Tuple of (prediction, confidence, metadata)
        """
        from data.dicom_utils import read_dicom, extract_metadata
        
        img_array, dcm = read_dicom(dicom_path, self.validator)
        
        if img_array is None:
            logger.warning(f"Invalid DICOM file: {dicom_path}")
            return None, None, None
        
        metadata = extract_metadata(dcm)

        img_processed = self.preprocessor.preprocess_for_inference(img_array)
        
        pred_proba = self.model.predict(img_processed, verbose=0)[0][0]
        prediction = 'Pneumonia' if pred_proba > self.threshold else 'No Pneumonia'
        
        logger.info(f"Prediction: {prediction} (confidence: {pred_proba:.4f})")
        
        return prediction, float(pred_proba), metadata
    
    def predict_image(self, image: np.ndarray) -> Tuple[str, float]:
        img_processed = self.preprocessor.preprocess_for_inference(image)
        pred_proba = self.model.predict(img_processed, verbose=0)[0][0]
        prediction = 'Pneumonia' if pred_proba > self.threshold else 'No Pneumonia'
        return prediction, float(pred_proba)
    
    def predict_batch(self, images: list) -> Tuple[list, list]:
        images_processed = [self.preprocessor.preprocess(img) for img in images]
        images_batch = np.array(images_processed)
        
        pred_probas = self.model.predict(images_batch, verbose=0).flatten()
        predictions = ['Pneumonia' if p > self.threshold else 'No Pneumonia' for p in pred_probas]
        
        return predictions, pred_probas.tolist()
