from .diversity_metric import diversity_metric
import numpy as np

class majority_voting_correct_prediction_rate(diversity_metric):
    """
    Majority Voting Correct Prediction Rate: quantifica la probabilità totale 
    che l'ensemble fornisca una classificazione corretta nel contesto di 
    majority voting con 3 classificatori.
    
    Formula: (N111 + N110 + N11? + N101 + N1?1 + N011 + N?11) / Ntot
    
    Conta tutti i casi dove almeno 2 classificatori predicono correttamente.
    
    Notazione:
    - "1" = predizione corretta (non reject)
    - "0" = predizione sbagliata (non reject)
    - "?" = rejection
    
    I 7 pattern considerati rappresentano tutte le combinazioni dove almeno 
    due classificatori forniscono predizioni corrette.
    """
    
    def __init__(self):
        super().__init__("Majority Voting Correct Prediction Rate")
    
    def _compute(self, predictions: np.ndarray, y_test: np.ndarray,
                 X_test: np.ndarray = None, model = None) -> float:
        """
        Calcola il majority voting correct prediction rate per 3 classificatori.
        
        Args:
            predictions: Array con shape [3, n_samples] contenente le predizioni
                        dei tre classificatori (possono contenere "reject")
            y_test: Labels vere
        
        Returns:
            La proporzione di campioni dove almeno 2 classificatori predicono 
            correttamente
        """
        # Verifica che ci siano esattamente 3 classificatori
        if predictions.shape[0] != 3:
            raise ValueError(f"Majority voting correct prediction rate richiede esattamente 3 classificatori, "
                           f"ma ne sono stati forniti {predictions.shape[0]}")
        
        pred_1 = predictions[0, :]
        pred_2 = predictions[1, :]
        pred_3 = predictions[2, :]
        
        n_samples = len(y_test)
        
        # Contatori per i 7 pattern
        N111 = 0  # Tutti e tre corretti
        N110 = 0  # Primi due corretti, terzo sbagliato
        N11q = 0  # Primi due corretti, terzo reject (q = ?)
        N101 = 0  # Primo e terzo corretti, secondo sbagliato
        N1q1 = 0  # Primo e terzo corretti, secondo reject
        N011 = 0  # Secondo e terzo corretti, primo sbagliato
        Nq11 = 0  # Secondo e terzo corretti, primo reject
        
        # Per ogni sample, determina il pattern
        for i in range(n_samples):
            # Determina lo stato di ogni classificatore
            # "1" = corretto, "0" = sbagliato, "?" = reject
            state_1 = self._get_state(pred_1[i], y_test[i])
            state_2 = self._get_state(pred_2[i], y_test[i])
            state_3 = self._get_state(pred_3[i], y_test[i])
            
            pattern = state_1 + state_2 + state_3
            
            # Conta i pattern
            if pattern == "111":
                N111 += 1
            elif pattern == "110":
                N110 += 1
            elif pattern == "11?":
                N11q += 1
            elif pattern == "101":
                N101 += 1
            elif pattern == "1?1":
                N1q1 += 1
            elif pattern == "011":
                N011 += 1
            elif pattern == "?11":
                Nq11 += 1
        
        # Calcola la metrica
        Ntot = n_samples
        
        if Ntot == 0:
            return 0.0
        
        return (N111 + N110 + N11q + N101 + N1q1 + N011 + Nq11) / Ntot
    
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