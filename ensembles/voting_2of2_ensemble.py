import numpy as np
from ensembles.ensemble import Ensemble


class Voting2of2Ensemble(Ensemble):
    """
    Voting 2 su 2: accetta una predizione solo quando entrambi 
    i classificatori concordano (include reject unanime).
    """
    
    def __init__(self, classifiers: list):
        if len(classifiers) != 2:
            raise ValueError("Voting2of2Ensemble richiede esattamente 2 classificatori")
        super().__init__(classifiers)
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        predictions = self.get_predictions(X_test)
        pred_1 = predictions[0]
        pred_2 = predictions[1]
        
        n_samples = len(pred_1)
        final_predictions = np.full(n_samples, self.rejection_label, dtype=object)
        
        for i in range(n_samples):
            if pred_1[i] == pred_2[i]:
                # Concordano (può essere classe o reject)
                final_predictions[i] = pred_1[i]
            # Altrimenti rimane reject (non concordano)
        
        return final_predictions
