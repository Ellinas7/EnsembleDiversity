from .diversity_metric import diversity_metric
import numpy as np

class generalized_diversity(diversity_metric):
    """Generalized Diversity: valori più alti = maggiore diversity"""
    
    def __init__(self):
        super().__init__("Generalized Diversity")
    
    def _compute(self, predictions: np.ndarray, y_test: np.ndarray) -> float:
        n_classifiers = predictions.shape[0]
        n_samples = predictions.shape[1]
        
        p = np.zeros(n_classifiers + 1)
        for i in range(n_samples):
            l_x = np.sum(predictions[:, i] != y_test[i])
            p[l_x] += 1
        p = p / n_samples
        
        p_1 = sum((i / n_classifiers) * p[i] for i in range(1, n_classifiers + 1))
        p_2 = sum((i / n_classifiers) * ((i - 1) / (n_classifiers - 1)) * p[i] 
                  for i in range(2, n_classifiers + 1))
        
        return 1 - (p_2 / p_1) if p_1 != 0 else 0