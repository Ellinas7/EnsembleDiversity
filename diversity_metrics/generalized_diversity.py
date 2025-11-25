from .diversity_metric import diversity_metric
import numpy as np

class generalized_diversity(diversity_metric):
    """Generalized Diversity: valori più alti = maggiore diversity"""
    
    def __init__(self):
        super().__init__("Generalized Diversity")
    
    def _compute(self, predictions: np.ndarray, y_test: np.ndarray) -> float:
        n_classifiers = predictions.shape[0]
        n_samples = predictions.shape[1]
        
        # Array per contare quanti campioni hanno esattamente i errori
        p = np.zeros(n_classifiers + 1)
        
        for i in range(n_samples):
            # Trova quali classificatori NON hanno fatto reject su questo campione
            valid_classifiers_mask = (predictions[:, i] != self.rejection_label)
            
            # Numero di classificatori validi per questo campione
            n_valid_classifiers = np.sum(valid_classifiers_mask)
            
            # Se ci sono meno di 2 classificatori validi, questo campione contribuisce a p[0]
            # (come se nessuno avesse sbagliato, perché non possiamo misurare diversity)
            if n_valid_classifiers < 2:
                p[0] += 1
                continue
            
            # Filtra le predizioni valide per questo campione
            valid_predictions = predictions[:, i][valid_classifiers_mask]
            
            # Conta quanti classificatori sbagliano (tra quelli validi)
            l_x = np.sum(valid_predictions != y_test[i])
            
            # Incrementa il contatore per l_x errori
            p[l_x] += 1
        
        # Normalizza p per ottenere proporzioni
        p = p / n_samples
        
        # Calcola p(1) e p(2) usando n_classifiers originale
        p_1 = sum((i / n_classifiers) * p[i] for i in range(1, n_classifiers + 1))
        p_2 = sum((i / n_classifiers) * ((i - 1) / (n_classifiers - 1)) * p[i] 
                  for i in range(2, n_classifiers + 1))
        
        return 1 - (p_2 / p_1) if p_1 != 0 else 0