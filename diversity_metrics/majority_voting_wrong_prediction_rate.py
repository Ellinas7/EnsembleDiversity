from .diversity_metric import diversity_metric
import numpy as np

class majority_voting_wrong_prediction_rate(diversity_metric):
    """
    Majority Voting Wrong Prediction Rate: quantifica la probabilità totale 
    che l'ensemble fornisca una classificazione errata nel contesto di 
    majority voting con 3 classificatori.
    
    Formula: (N100 + N010 + N001 + N000 + N00? + N0?0 + N?00) / Ntot
    
    Conta tutti i casi dove almeno 2 classificatori predicono erroneamente.
    
    Notazione:
    - "1" = predizione corretta (non reject)
    - "0" = predizione sbagliata (non reject)
    - "?" = rejection
    
    I 7 pattern considerati rappresentano tutte le combinazioni dove almeno 
    due classificatori forniscono predizioni sbagliate.
    """
    
    def __init__(self):
        super().__init__("Majority Voting Wrong Prediction Rate")
    
    def _compute(self, predictions: np.ndarray, y_test: np.ndarray,
                 X_test: np.ndarray = None, model = None) -> float:
        """
        Calcola il majority voting wrong prediction rate per 3 classificatori.
        
        Args:
            predictions: Array con shape [3, n_samples] contenente le predizioni
                        dei tre classificatori (possono contenere "reject")
            y_test: Labels vere
        
        Returns:
            La proporzione di campioni dove almeno 2 classificatori predicono 
            erroneamente
        """
        # Verifica che ci siano esattamente 3 classificatori
        if predictions.shape[0] != 3:
            raise ValueError(f"Majority voting wrong prediction rate richiede esattamente 3 classificatori, "
                           f"ma ne sono stati forniti {predictions.shape[0]}")
        
        pred_1 = predictions[0, :]
        pred_2 = predictions[1, :]
        pred_3 = predictions[2, :]
        
        n_samples = len(y_test)
        
        # Contatori per i 7 pattern
        N100 = 0  # Primo corretto, secondo e terzo sbagliati
        N010 = 0  # Secondo corretto, primo e terzo sbagliati
        N001 = 0  # Terzo corretto, primo e secondo sbagliati
        N000 = 0  # Tutti e tre sbagliati
        N00q = 0  # Primi due sbagliati, terzo reject (q = ?)
        N0q0 = 0  # Primo e terzo sbagliati, secondo reject
        Nq00 = 0  # Secondo e terzo sbagliati, primo reject
        
        # Per ogni sample, determina il pattern
        for i in range(n_samples):
            # Determina lo stato di ogni classificatore
            # "1" = corretto, "0" = sbagliato, "?" = reject
            state_1 = self._get_state(pred_1[i], y_test[i])
            state_2 = self._get_state(pred_2[i], y_test[i])
            state_3 = self._get_state(pred_3[i], y_test[i])
            
            pattern = state_1 + state_2 + state_3
            
            # Conta i pattern
            if pattern == "100":
                N100 += 1
            elif pattern == "010":
                N010 += 1
            elif pattern == "001":
                N001 += 1
            elif pattern == "000":
                N000 += 1
            elif pattern == "00?":
                N00q += 1
            elif pattern == "0?0":
                N0q0 += 1
            elif pattern == "?00":
                Nq00 += 1
        
        # Calcola la metrica
        Ntot = n_samples
        
        if Ntot == 0:
            return 0.0
        
        return (N100 + N010 + N001 + N000 + N00q + N0q0 + Nq00) / Ntot
    
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