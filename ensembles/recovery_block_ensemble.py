import numpy as np
from ensembles.ensemble import Ensemble


class RecoveryBlockEnsemble(Ensemble):
    """
    Recovery Block: consulta D1, se fa reject passa a D2.
    L'ordine dei classificatori nella lista è importante.
    """
    
    def __init__(self, classifiers: list):
        if len(classifiers) != 2:
            raise ValueError("RecoveryBlockEnsemble richiede esattamente 2 classificatori")
        super().__init__(classifiers)
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        predictions = self.get_predictions(X_test)
        pred_primary = predictions[0]    # D1
        pred_secondary = predictions[1]  # D2
        
        n_samples = len(pred_primary)
        final_predictions = np.empty(n_samples, dtype=object)
        
        for i in range(n_samples):
            if pred_primary[i] != self.rejection_label:
                # D1 risponde, usa la sua predizione
                final_predictions[i] = pred_primary[i]
            else:
                # D1 fa reject, passa a D2
                final_predictions[i] = pred_secondary[i]
        
        return final_predictions
