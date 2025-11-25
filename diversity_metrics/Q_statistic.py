from .diversity_metric import diversity_metric
import numpy as np

class Q_statistic(diversity_metric):
    """Q-statistic: valori più bassi = maggiore diversity"""
    
    def __init__(self):
        super().__init__("Q-statistic")
    
    def _compute(self, predictions: np.ndarray, y_test: np.ndarray) -> float:
        n_classifiers = predictions.shape[0]
        Q_values = []
        
        for i in range(n_classifiers):
            for j in range(i+1, n_classifiers):

                valid_mask = self._get_valid_mask(predictions[i, :], predictions[j, :])
                
                if np.sum(valid_mask) == 0:
                    continue
                
                correct_i = (predictions[i, :] == y_test)
                correct_j = (predictions[j, :] == y_test)
                
                N11 = np.sum(correct_i & correct_j)
                N00 = np.sum(~correct_i & ~correct_j)
                N10 = np.sum(correct_i & ~correct_j)
                N01 = np.sum(~correct_i & correct_j)
                
                denominator = N11*N00 + N01*N10
                if denominator != 0:
                    Q = (N11*N00 - N01*N10) / denominator
                    Q_values.append(Q)
        
        return np.mean(Q_values)