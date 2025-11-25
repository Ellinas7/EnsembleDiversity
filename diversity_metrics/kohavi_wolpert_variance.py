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
            
            # Calcola la varianza per questo campione usando il numero EFFETTIVO di classificatori
            kw_sum += l_x * (n_valid_classifiers - l_x)
        
        # Denominatore: usa N e L^2 originali come nel paper
        return kw_sum / (n_samples * n_classifiers**2)