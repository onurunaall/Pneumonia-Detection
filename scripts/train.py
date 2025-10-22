#!/usr/bin/env python3
import argparse
import logging
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from config.config import Config
from data.dataset import PneumoniaDataset
from data.preprocessing import ImagePreprocessor
from data.augmentation import PneumoniaAugmentor
from models.model_builder import ModelBuilder
from training.trainer import Trainer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main(args):
    config = Config()
    
    # Override config with command line arguments
    if args.epochs:
        config.training.epochs = args.epochs
    if args.batch_size:
        config.data.batch_size = args.batch_size
    if args.learning_rate:
        config.training.learning_rate = args.learning_rate
    
    logger.info("Starting pneumonia detection training")
    logger.info(f"Configuration: {config}")
    
    preprocessor = ImagePreprocessor(target_size=config.data.image_size,
                                     channels=config.data.channels,
                                     normalize=True,
                                     clahe=args.use_clahe)
    augmentor = PneumoniaAugmentor()
    
    dataset = PneumoniaDataset(labels_path=config.paths.data_dir / 'sample_labels.csv',
                               images_dir=config.paths.data_dir / 'images',
                               preprocessor=preprocessor,
                               augmentor=augmentor,
                               balance_classes=True)
    
    # Split dataset
    train_ds, val_ds, test_ds = dataset.split(test_size=config.data.test_split,
                                              val_size=config.data.validation_split)
    
    # Create data generators
    train_gen = train_ds.get_generator(batch_size=config.data.batch_size, shuffle=True)
    val_gen = val_ds.get_generator(batch_size=config.data.batch_size, shuffle=False)
    
    model = ModelBuilder.from_config(config)
    model_instance = model.build()
    
    model.compile(optimizer=config.training.optimizer,
                  learning_rate=config.training.learning_rate,
                  loss=config.training.loss_function,
                  metrics=['accuracy', 'AUC'])
    
    model.summary()
    
    # Calculate class weights
    trainer = Trainer(model_instance, config)
    class_weights = None
    if config.training.class_weights:
        y_train = train_ds.df['Pneumonia'].values
        class_weights = trainer.calculate_class_weights(y_train)
    
    logger.info("Starting training...")
    history = trainer.train(train_generator=train_gen,
                            val_generator=val_gen,
                            class_weight=class_weights)
    
    trainer.save_history()
    
    # Save final model
    final_model_path = config.paths.model_dir / 'final_model.h5'
    model_instance.save(str(final_model_path))
    logger.info(f"Model saved to {final_model_path}")
    
    # Save model architecture
    model_json = model_instance.to_json()
    json_path = config.paths.model_dir / 'model_architecture.json'
    with open(json_path, 'w') as f:
        f.write(model_json)
    logger.info(f"Model architecture saved to {json_path}")
    
    logger.info("Training completed successfully!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train pneumonia detection model')
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, help='Batch size')
    parser.add_argument('--learning-rate', type=float, help='Learning rate')
    parser.add_argument('--use-clahe', action='store_true', 
                       help='Use CLAHE for preprocessing')
    
    args = parser.parse_args()
    main(args)
