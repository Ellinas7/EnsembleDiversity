from .diversity_metric import diversity_metric
import numpy as np

class entropy_measure(diversity_metric):
    """Entropy: valori più alti = maggiore diversity"""
    
    def __init__(self):
        super().__init__("Entropy")
    
    def _compute(self, predictions: np.ndarray, y_test: np.ndarray) -> float:
        n_classifiers = predictions.shape[0]
        n_samples = predictions.shape[1]
        entropy_sum = 0
        
        for i in range(n_samples):
            l_x = np.sum(predictions[:, i] != y_test[i])
            entropy_sum += min(l_x, n_classifiers - l_x)
        
        denominator = n_samples * (n_classifiers - np.ceil(n_classifiers / 2))
        return entropy_sum / denominator if denominator != 0 else 0