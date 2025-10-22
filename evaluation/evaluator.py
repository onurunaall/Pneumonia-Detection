import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score, f1_score, confusion_matrix, classification_report
from typing import Tuple, Dict
import logging

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Evaluates model performance"""
    
    def __init__(self, model, threshold: float = 0.5):
        self.model = model
        self.threshold = threshold
    
    def evaluate(self, 
                 test_generator,
                 return_predictions: bool = False) -> Dict:
        logger.info("Evaluating model...")
        
        # Get predictions
        y_pred_proba = self.model.predict(test_generator, verbose=1)
        y_pred = (y_pred_proba > self.threshold).astype(int)
        
        # Get true labels
        y_true = test_generator.labels
        
        # Calculate metrics
        metrics = {}
        
        # AUC-ROC
        metrics['auc_roc'] = roc_auc_score(y_true, y_pred_proba)
        
        # Precision-Recall
        metrics['auc_pr'] = average_precision_score(y_true, y_pred_proba)
        
        # F1 Score
        metrics['f1_score'] = f1_score(y_true, y_pred)
        
        # Confusion Matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics['confusion_matrix'] = {
            'tn': int(tn), 'fp': int(fp),
            'fn': int(fn), 'tp': int(tp)
        }
        
        # Sensitivity & Specificity
        metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # PPV & NPV
        metrics['ppv'] = tp / (tp + fp) if (tp + fp) > 0 else 0
        metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0
        
        # Classification report
        metrics['classification_report'] = classification_report(y_true,
                                                                 y_pred,
                                                                 target_names=['No Pneumonia', 'Pneumonia'])
        
        logger.info(f"Evaluation Results:")
        logger.info(f"AUC-ROC: {metrics['auc_roc']:.4f}")
        logger.info(f"AUC-PR: {metrics['auc_pr']:.4f}")
        logger.info(f"F1 Score: {metrics['f1_score']:.4f}")
        logger.info(f"Sensitivity: {metrics['sensitivity']:.4f}")
        logger.info(f"Specificity: {metrics['specificity']:.4f}")
        
        if return_predictions:
            metrics['predictions'] = {
                'y_true': y_true,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
        
        return metrics
    
    def find_optimal_threshold(self,
                              val_generator,
                              metric: str = 'f1') -> float:
        y_pred_proba = self.model.predict(val_generator, verbose=1)
        y_true = val_generator.labels
        
        if metric == 'f1':
            # Find threshold that maximizes F1
            thresholds = np.arange(0.1, 0.9, 0.01)
            f1_scores = []
            
            for threshold in thresholds:
                y_pred = (y_pred_proba > threshold).astype(int)
                f1_scores.append(f1_score(y_true, y_pred))
            
            optimal_threshold = thresholds[np.argmax(f1_scores)]
            
        elif metric == 'youden':
            # Youden's J statistic
            fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
            j_scores = tpr - fpr
            optimal_threshold = thresholds[np.argmax(j_scores)]
            
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        logger.info(f"Optimal threshold ({metric}): {optimal_threshold:.4f}")
        return optimal_threshold
