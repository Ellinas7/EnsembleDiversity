import numpy as np
from ensembles.ensemble import Ensemble


class Voting1ofNEnsemble(Ensemble):
    """
    Voting 1 su N: accetta solo se esattamente 1 classificatore risponde
    (non reject) e gli altri 2 fanno reject.
    """
    
    def __init__(self, classifiers: list):
        if len(classifiers) != 3:
            raise ValueError("Voting1ofNEnsemble richiede esattamente 3 classificatori")
        super().__init__(classifiers)
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        predictions = self.get_predictions(X_test)
        n_samples = predictions.shape[1]
        final_predictions = np.empty(n_samples, dtype=object)
        
        for i in range(n_samples):
            # Conta quanti classificatori rispondono (non reject)
            responses = [predictions[j, i] for j in range(self.n_classifiers)
                         if predictions[j, i] != self.rejection_label]
            
            if len(responses) == 1:
                # Esattamente 1 risponde, accetta la sua predizione
                final_predictions[i] = responses[0]
            else:
                # 0 o 2+ rispondono → reject
                final_predictions[i] = self.rejection_label
        
        return final_predictions
