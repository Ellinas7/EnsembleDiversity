from .diversity_metric import diversity_metric
import numpy as np

class disagreement_measure(diversity_metric):
    """Disagreement measure: valori più alti = maggiore diversity"""
    
    def __init__(self):
        super().__init__("Disagreement")
    
    def _compute(self, predictions: np.ndarray, y_test: np.ndarray) -> float:
        n_classifiers = predictions.shape[0]
        dis_values = []
        
        for i in range(n_classifiers):
            for j in range(i+1, n_classifiers):
                correct_i = (predictions[i, :] == y_test)
                correct_j = (predictions[j, :] == y_test)
                
                N11 = np.sum(correct_i & correct_j)
                N00 = np.sum(~correct_i & ~correct_j)
                N10 = np.sum(correct_i & ~correct_j)
                N01 = np.sum(~correct_i & correct_j)
                
                total = N11 + N10 + N01 + N00
                if total != 0:
                    dis = (N01 + N10) / total
                    dis_values.append(dis)
        
        return np.mean(dis_values)