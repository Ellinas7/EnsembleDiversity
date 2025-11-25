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

                # Crea la maschera per ignorare i reject
                valid_mask = self._get_valid_mask(predictions[i, :], predictions[j, :])

                 # Se non ci sono campioni validi, salta questa coppia
                if np.sum(valid_mask) == 0:
                    continue

                # Applica la maschera per filtrare le predizioni e le label
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
                
                total = N11 + N10 + N01 + N00
                if total != 0:
                    dis = (N01 + N10) / total
                    dis_values.append(dis)
        
        return np.mean(dis_values)