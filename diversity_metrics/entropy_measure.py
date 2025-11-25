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
            # Trova quali classificatori NON hanno fatto reject su questo campione
            valid_classifiers_mask = (predictions[:, i] != self.rejection_label)
            
            # Numero di classificatori validi per questo campione
            n_valid_classifiers = np.sum(valid_classifiers_mask)
            
            # Se ci sono meno di 2 classificatori validi, questo campione contribuisce 0
            if n_valid_classifiers < 2:
                continue
            
            # Filtra le predizioni valide per questo campione
            valid_predictions = predictions[:, i][valid_classifiers_mask]
            
            # Calcola quanti classificatori sbagliano (tra quelli validi)
            l_x = np.sum(valid_predictions != y_test[i])
            
            # Calcola l'entropia per questo campione
            entropy_sum += min(l_x, n_valid_classifiers - l_x)
        
        # Il denominatore usa N (numero totale di campioni) come nel paper
        denominator = n_samples * (n_classifiers - np.ceil(n_classifiers / 2))
        
        return entropy_sum / denominator