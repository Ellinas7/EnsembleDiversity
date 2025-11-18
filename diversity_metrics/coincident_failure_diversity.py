from .diversity_metric import diversity_metric
import numpy as np

class coincident_failure_diversity(diversity_metric):
    """Coincident Failure Diversity: valori più alti = migliore distribuzione fallimenti"""
    
    def __init__(self):
        super().__init__("Coincident Failure Diversity")
    
    def _compute(self, predictions: np.ndarray, y_test: np.ndarray) -> float:
        n_classifiers = predictions.shape[0]
        n_samples = predictions.shape[1]
        
        p = np.zeros(n_classifiers + 1)
        for i in range(n_samples):
            l_x = np.sum(predictions[:, i] != y_test[i])
            p[l_x] += 1
        p = p / n_samples
        
        if p[0] == 1.0:
            return 0.0
        
        CFD_sum = sum(((n_classifiers - i) / (n_classifiers - 1)) * p[i] 
                      for i in range(1, n_classifiers + 1))
        
        return CFD_sum / (1 - p[0])