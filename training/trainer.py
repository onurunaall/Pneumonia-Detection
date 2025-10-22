import numpy as np
from pathlib import Path
from tensorflow import keras
from typing import Optional, Dict
import logging
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class Trainer:
    """Handles model training orchestration"""
    
    def __init__(self, model: keras.Model, config, checkpoint_dir: Optional[Path] = None):
        """
        Args:
            model: Compiled Keras model
            config: Configuration object
            checkpoint_dir: Directory to save checkpoints
        """
        self.model = model
        self.config = config
        self.checkpoint_dir = checkpoint_dir or config.paths.checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.history = None
        self.training_time = None
    
    def train(self,
              train_generator,
              val_generator,
              callbacks: Optional[list] = None,
              class_weight: Optional[Dict[int, float]] = None):
        """
        Train the model
        
        Args:
            train_generator: Training data generator
            val_generator: Validation data generator
            callbacks: List of Keras callbacks
            class_weight: Class weights for imbalanced data
            
        Returns:
            Training history
        """
        if callbacks is None:
            callbacks = self._get_default_callbacks()
        
        logger.info("Starting training...")
        logger.info(f"Epochs: {self.config.training.epochs}")
        logger.info(f"Batch size: {self.config.data.batch_size}")
        
        start_time = datetime.now()
        
        self.history = self.model.fit(train_generator,
                                      validation_data=val_generator,
                                      epochs=self.config.training.epochs,
                                      callbacks=callbacks,
                                      class_weight=class_weight,
                                      verbose=1)
        
        self.training_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"Training completed in {self.training_time:.2f} seconds")
        
        return self.history
    
    def _get_default_callbacks(self) -> list:
        """Get default training callbacks"""
        callbacks = []
        
        # Model checkpoint
        checkpoint_path = self.checkpoint_dir / 'best_model.h5'
        checkpoint = keras.callbacks.ModelCheckpoint(str(checkpoint_path),
                                                     monitor='val_auc',
                                                     mode='max',
                                                     save_best_only=True,
                                                     verbose=1)
        callbacks.append(checkpoint)
        
        # Early stopping
        early_stop = keras.callbacks.EarlyStopping(monitor='val_auc',
                                                   mode='max',
                                                   patience=self.config.training.early_stopping_patience,
                                                   restore_best_weights=True,
                                                   verbose=1)
        callbacks.append(early_stop)
        
        # Reduce learning rate on plateau
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                      factor=0.5,
                                                      patience=self.config.training.reduce_lr_patience,
                                                      min_lr=1e-7,
                                                      verbose=1)
        callbacks.append(reduce_lr)
        
        # TensorBoard
        log_dir = self.config.paths.log_dir / datetime.now().strftime('%Y%m%d-%H%M%S')
        tensorboard = keras.callbacks.TensorBoard(log_dir=str(log_dir),
                                                  histogram_freq=1,
                                                  write_graph=True)
        callbacks.append(tensorboard)
        
        return callbacks
    
    def save_history(self, path: Optional[Path] = None):
        """Save training history to JSON"""
        if self.history is None:
            raise ValueError("No training history to save")
        
        if path is None:
            path = self.checkpoint_dir / 'training_history.json'
        
        history_dict = {
            key: [float(val) for val in values]
            for key, values in self.history.history.items()
        }
        history_dict['training_time_seconds'] = self.training_time
        
        with open(path, 'w') as f:
            json.dump(history_dict, f, indent=2)
        
        logger.info(f"Training history saved to {path}")
    
    def calculate_class_weights(self, y_train: np.ndarray) -> Dict[int, float]:
        """
        Calculate class weights for imbalanced dataset
        
        Args:
            y_train: Training labels
            
        Returns:
            Dictionary of class weights
        """
        from sklearn.utils.class_weight import compute_class_weight
        
        classes = np.unique(y_train)
        weights = compute_class_weight(class_weight='balanced',
                                       classes=classes,
                                       y=y_train)
        
        class_weight_dict = dict(zip(classes, weights))
        logger.info(f"Class weights: {class_weight_dict}")
        
        return class_weight_dict
