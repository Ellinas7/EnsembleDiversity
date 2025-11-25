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
                
                # maschera per ignorare i reject
                valid_mask = self._get_valid_mask(predictions[i, :], predictions[j, :])
                
                # se non ci sono campioni validi, salto la coppia
                if np.sum(valid_mask) == 0:
                    continue

                # Applica la maschera prima di calcolare correct_i e correct_j
                pred_i_valid = predictions[i, :][valid_mask]
                pred_j_valid = predictions[j, :][valid_mask]
                y_test_valid = y_test[valid_mask]

                # Calcola la correttezza sui campioni filtrati
                correct_i = (pred_i_valid == y_test_valid)
                correct_j = (pred_j_valid == y_test_valid)
                
                N11 = np.sum(correct_i & correct_j)
                N00 = np.sum(~correct_i & ~correct_j)
                N10 = np.sum(correct_i & ~correct_j)
                N01 = np.sum(~correct_i & correct_j)
                
                denominator = N11*N00 + N01*N10
                if denominator != 0:
                    Q = (N11*N00 - N01*N10) / denominator
                    Q_values.append(Q)
        
        return np.mean(Q_values)