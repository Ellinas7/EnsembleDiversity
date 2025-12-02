import numpy as np
from collections import Counter
from ensembles.ensemble import Ensemble


class MajorityVotingEnsemble(Ensemble):
    """
    Majority Voting: accetta la classe/reject che ottiene almeno 2 voti su 3.
    Il reject conta come voto.
    """
    
    def __init__(self, classifiers: list):
        if len(classifiers) != 3:
            raise ValueError("MajorityVotingEnsemble richiede esattamente 3 classificatori")
        super().__init__(classifiers)
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        predictions = self.get_predictions(X_test)
        n_samples = predictions.shape[1]
        final_predictions = np.empty(n_samples, dtype=object)
        
        for i in range(n_samples):
            votes = [predictions[j, i] for j in range(3)]
            counter = Counter(votes)
            most_common_class, count = counter.most_common(1)[0]
            
            if count >= 2:
                # Maggioranza trovata (classe o reject)
                final_predictions[i] = most_common_class
            else:
                # Nessuna maggioranza (3 voti diversi)
                final_predictions[i] = self.rejection_label
        
        return final_predictions
