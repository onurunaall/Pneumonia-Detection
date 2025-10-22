#!/usr/bin/env python3
import argparse
import logging
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from config.config import Config
from data.dataset import PneumoniaDataset
from data.preprocessing import ImagePreprocessor
from evaluation.evaluator import ModelEvaluator
from tensorflow import keras

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Evaluate pneumonia detection model')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model (.h5)')
    parser.add_argument('--data-dir', type=str, default='data/raw',
                       help='Path to data directory')
    parser.add_argument('--labels-file', type=str, default='sample_labels.csv',
                       help='Labels CSV filename')
    parser.add_argument('--images-dir', type=str, default='images',
                       help='Images directory name')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Classification threshold')
    
    args = parser.parse_args()
    
    config = Config()
    
    # Load model
    logger.info(f"Loading model from {args.model}")
    model = keras.models.load_model(args.model)
    
    # Setup paths
    data_dir = Path(args.data_dir)
    labels_path = data_dir / args.labels_file
    images_dir = data_dir / args.images_dir
    
    # Initialize preprocessor
    preprocessor = ImagePreprocessor(target_size=config.data.image_size,
                                     channels=config.data.channels,
                                     normalize=True)
    
    # Load dataset
    logger.info(f"Loading dataset from {labels_path}")
    dataset = PneumoniaDataset(labels_path=labels_path,
                               images_dir=images_dir,
                               preprocessor=preprocessor,
                               augmentor=None,  # No augmentation for evaluation
                               balance_classes=False)
    
    # Get test split
    _, _, test_ds = dataset.split(test_size=0.2, val_size=0.1)
    test_gen = test_ds.get_generator(batch_size=config.data.batch_size, shuffle=False)
    
    # Evaluate
    evaluator = ModelEvaluator(model, threshold=args.threshold)
    metrics = evaluator.evaluate(test_gen, return_predictions=False)
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"AUC-ROC: {metrics['auc_roc']:.4f}")
    print(f"AUC-PR: {metrics['auc_pr']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print(f"Sensitivity: {metrics['sensitivity']:.4f}")
    print(f"Specificity: {metrics['specificity']:.4f}")
    print(f"PPV: {metrics['ppv']:.4f}")
    print(f"NPV: {metrics['npv']:.4f}")
    print("\nConfusion Matrix:")
    cm = metrics['confusion_matrix']
    print(f"  TN: {cm['tn']}, FP: {cm['fp']}")
    print(f"  FN: {cm['fn']}, TP: {cm['tp']}")
    print("="*50 + "\n")

if __name__ == '__main__':
    main()
