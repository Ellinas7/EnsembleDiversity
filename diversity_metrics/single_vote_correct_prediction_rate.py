from .diversity_metric import diversity_metric
import numpy as np

class single_vote_correct_prediction_rate(diversity_metric):
    """
    Single Vote Correct Prediction Rate: quantifica la probabilità totale che 
    l'ensemble fornisca una classificazione corretta nel contesto di voting 1 su n.
    
    Formula: (N1?? + N?1? + N??1) / Ntot
    
    Conta i casi dove esattamente UN classificatore fornisce una predizione 
    corretta mentre gli altri DUE fanno reject.
    
    Notazione:
    - "1" = predizione corretta (non reject)
    - "0" = predizione sbagliata (non reject)
    - "?" = rejection
    """
    
    def __init__(self):
        super().__init__("Single Vote Correct Prediction Rate")
    
    def _compute(self, predictions: np.ndarray, y_test: np.ndarray,
                 X_test: np.ndarray = None, model = None) -> float:
        """
        Calcola il single vote correct prediction rate per 3 classificatori.
        
        Args:
            predictions: Array con shape [3, n_samples] contenente le predizioni
                        dei tre classificatori (possono contenere "reject")
            y_test: Labels vere
        
        Returns:
            La proporzione di campioni dove esattamente un classificatore 
            risponde correttamente e gli altri due fanno reject
        """
        # Verifica che ci siano esattamente 3 classificatori
        if predictions.shape[0] != 3:
            raise ValueError(f"Single vote correct prediction rate richiede esattamente 3 classificatori, "
                           f"ma ne sono stati forniti {predictions.shape[0]}")
        
        pred_1 = predictions[0, :]
        pred_2 = predictions[1, :]
        pred_3 = predictions[2, :]
        
        n_samples = len(y_test)
        
        # Contatori per i 3 pattern
        N1qq = 0  # Primo corretto, secondo e terzo reject
        Nq1q = 0  # Secondo corretto, primo e terzo reject
        Nqq1 = 0  # Terzo corretto, primo e secondo reject
        
        # Per ogni sample, determina il pattern
        for i in range(n_samples):
            # Determina lo stato di ogni classificatore
            # "1" = corretto, "0" = sbagliato, "?" = reject
            state_1 = self._get_state(pred_1[i], y_test[i])
            state_2 = self._get_state(pred_2[i], y_test[i])
            state_3 = self._get_state(pred_3[i], y_test[i])
            
            pattern = state_1 + state_2 + state_3
            
            # Conta i pattern
            if pattern == "1??":
                N1qq += 1
            elif pattern == "?1?":
                Nq1q += 1
            elif pattern == "??1":
                Nqq1 += 1
        
        # Calcola la metrica
        Ntot = n_samples
        
        if Ntot == 0:
            return 0.0
        
        return (N1qq + Nq1q + Nqq1) / Ntot
    
    def _get_state(self, prediction, true_label):
        """
        Determina lo stato di una predizione.
        
        Returns:
            "1" se corretta (non reject)
            "0" se sbagliata (non reject)
            "?" se reject
        """
        if prediction == "reject":
            return "?"
        elif prediction == true_label:
            return "1"
        else:
            return "0"