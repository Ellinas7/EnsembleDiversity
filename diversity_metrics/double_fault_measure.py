from .diversity_metric import diversity_metric
import numpy as np

class double_fault_measure(diversity_metric):
    """Double Fault: valori più bassi = maggiore diversity"""
    
    def __init__(self):
        super().__init__("Double Fault")
    
    def _compute(self, predictions: np.ndarray, y_test: np.ndarray) -> float:
        n_classifiers = predictions.shape[0]
        df_values = []
        
        for i in range(n_classifiers):
            for j in range(i+1, n_classifiers):
                correct_i = (predictions[i, :] == y_test)
                correct_j = (predictions[j, :] == y_test)
                
                N00 = np.sum(~correct_i & ~correct_j)
                total = predictions.shape[1]
                
                if total != 0:
                    df = N00 / total
                    df_values.append(df)
        
        return np.mean(df_values)
