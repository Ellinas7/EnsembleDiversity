from .diversity_metric import diversity_metric
import numpy as np

class kohavi_wolpert_variance(diversity_metric):
    """Kohavi-Wolpert Variance: valori più alti = maggiore diversity"""
    
    def __init__(self):
        super().__init__("Kohavi-Wolpert Variance")
    
    def _compute(self, predictions: np.ndarray, y_test: np.ndarray) -> float:
        n_classifiers = predictions.shape[0]
        n_samples = predictions.shape[1]
        kw_sum = 0
        
        for i in range(n_samples):
            l_x = np.sum(predictions[:, i] != y_test[i])
            kw_sum += l_x * (n_classifiers - l_x)
        
        return kw_sum / (n_samples * n_classifiers**2)