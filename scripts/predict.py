#!/usr/bin/env python3
import argparse
import logging
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from config.config import Config
from data.preprocessing import ImagePreprocessor
from data.dicom_utils import DICOMValidator
from inference.predictor import PneumoniaPredictor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Predict pneumonia from X-ray')
    parser.add_argument('input', type=str, help='Path to input DICOM file')
    parser.add_argument('--model', type=str, required=True, help='Path to model file (.h5 for full model, .json for architecture)')
    parser.add_argument('--weights', type=str, default=None, help='Path to model weights (only needed if --model is .json)')
    parser.add_argument('--threshold', type=float, default=0.5, help='Classification threshold')
    
    args = parser.parse_args()
    
    config = Config()
    
    preprocessor = ImagePreprocessor(target_size=config.data.image_size,
                                     channels=config.data.channels,
                                     normalize=True)
    validator = DICOMValidator(allowed_modality=config.inference.allowed_modality,
                               allowed_body_part=config.inference.allowed_body_part,
                               allowed_positions=config.inference.allowed_positions)
    
    model_path = Path(args.model)
    weight_path = Path(args.weights) if args.weights else None
    
    predictor = PneumoniaPredictor(model_path=model_path,
                                   weight_path=weight_path,
                                   preprocessor=preprocessor,
                                   validator=validator,
                                   threshold=args.threshold)
    
    # Predict on input file
    input_path = Path(args.input)
    
    if input_path.suffix.lower() == '.dcm':
        prediction, confidence, metadata = predictor.predict_dicom(input_path)
        
        if prediction:
            print(f"\n{'='*50}")
            print(f"File: {input_path.name}")
            print(f"Prediction: {prediction}")
            print(f"Confidence: {confidence:.4f}")
            print(f"\nMetadata:")
            for key, value in metadata.items():
                print(f"  {key}: {value}")
            print(f"{'='*50}\n")
        else:
            print(f"Error: Could not process {input_path}")
    else:
        print(f"Error: Only DICOM files (.dcm) are supported")
        sys.exit(1)

if __name__ == '__main__':
    main()