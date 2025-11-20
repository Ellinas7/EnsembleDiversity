from .diversity_metric import diversity_metric
import numpy as np

class majority_voting_rejection_prediction_rate(diversity_metric):
    """
    Majority Voting Rejection Prediction Rate: quantifica la probabilità 
    complessiva che il sistema non fornisca una classificazione nel contesto 
    di majority voting con 3 classificatori.
    
    Formula: (N10? + N1?0 + N1?? + N01? + N0?1 + N0?? + N?10 + N?1? + 
              N?01 + N?0? + N??1 + N??0 + N???) / Ntot
    
    Conta tutti i casi dove NON c'è una maggioranza (almeno 2) né di corretti 
    né di sbagliati. Include situazioni con:
    - Predizioni discordanti (un corretto, un sbagliato, uno qualsiasi)
    - Molte rejection (almeno 2 rejection)
    - Tutti e tre rejection
    
    Notazione:
    - "1" = predizione corretta (non reject)
    - "0" = predizione sbagliata (non reject)
    - "?" = rejection
    """
    
    def __init__(self):
        super().__init__("Majority Voting Rejection Prediction Rate")
    
    def _compute(self, predictions: np.ndarray, y_test: np.ndarray,
                 X_test: np.ndarray = None, model = None) -> float:
        """
        Calcola il majority voting rejection prediction rate per 3 classificatori.
        
        Args:
            predictions: Array con shape [3, n_samples] contenente le predizioni
                        dei tre classificatori (possono contenere "reject")
            y_test: Labels vere
        
        Returns:
            La proporzione di campioni dove il sistema non fornisce una 
            classificazione (nessuna maggioranza di corretti o sbagliati)
        """
        # Verifica che ci siano esattamente 3 classificatori
        if predictions.shape[0] != 3:
            raise ValueError(f"Majority voting rejection prediction rate richiede esattamente 3 classificatori, "
                           f"ma ne sono stati forniti {predictions.shape[0]}")
        
        pred_1 = predictions[0, :]
        pred_2 = predictions[1, :]
        pred_3 = predictions[2, :]
        
        n_samples = len(y_test)
        
        # Contatori per i 13 pattern
        N10q = 0  # Primo corretto, secondo sbagliato, terzo reject
        N1q0 = 0  # Primo corretto, secondo reject, terzo sbagliato
        N1qq = 0  # Primo corretto, secondo e terzo reject
        N01q = 0  # Primo sbagliato, secondo corretto, terzo reject
        N0q1 = 0  # Primo sbagliato, secondo reject, terzo corretto
        N0qq = 0  # Primo sbagliato, secondo e terzo reject
        Nq10 = 0  # Primo reject, secondo corretto, terzo sbagliato
        Nq1q = 0  # Primo reject, secondo corretto, terzo reject
        Nq01 = 0  # Primo reject, secondo sbagliato, terzo corretto
        Nq0q = 0  # Primo reject, secondo sbagliato, terzo reject
        Nqq1 = 0  # Primi due reject, terzo corretto
        Nqq0 = 0  # Primi due reject, terzo sbagliato
        Nqqq = 0  # Tutti e tre reject
        
        # Per ogni sample, determina il pattern
        for i in range(n_samples):
            # Determina lo stato di ogni classificatore
            # "1" = corretto, "0" = sbagliato, "?" = reject
            state_1 = self._get_state(pred_1[i], y_test[i])
            state_2 = self._get_state(pred_2[i], y_test[i])
            state_3 = self._get_state(pred_3[i], y_test[i])
            
            pattern = state_1 + state_2 + state_3
            
            # Conta i pattern
            if pattern == "10?":
                N10q += 1
            elif pattern == "1?0":
                N1q0 += 1
            elif pattern == "1??":
                N1qq += 1
            elif pattern == "01?":
                N01q += 1
            elif pattern == "0?1":
                N0q1 += 1
            elif pattern == "0??":
                N0qq += 1
            elif pattern == "?10":
                Nq10 += 1
            elif pattern == "?1?":
                Nq1q += 1
            elif pattern == "?01":
                Nq01 += 1
            elif pattern == "?0?":
                Nq0q += 1
            elif pattern == "??1":
                Nqq1 += 1
            elif pattern == "??0":
                Nqq0 += 1
            elif pattern == "???":
                Nqqq += 1
        
        # Calcola la metrica
        Ntot = n_samples
        
        if Ntot == 0:
            return 0.0
        
        return (N10q + N1q0 + N1qq + N01q + N0q1 + N0qq + 
                Nq10 + Nq1q + Nq01 + Nq0q + Nqq1 + Nqq0 + Nqqq) / Ntot
    
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