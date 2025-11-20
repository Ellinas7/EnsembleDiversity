from .diversity_metric import diversity_metric
import numpy as np

class ideal_single_vote_rate(diversity_metric):
    """
    Ideal Single Vote Rate: cattura il comportamento ideale del sistema single vote.
    
    Formula: (N1?? + N?1? + N??1) / (N1?? + N0?? + N?1? + N?0? + N??1 + N??0)
    
    Questa metrica rappresenta la situazione ideale in cui un classificatore 
    altamente affidabile risponde mentre gli altri, meno certi, si astengono 
    appropriatamente. Il valore indica quanto spesso il sistema opera secondo 
    il principio per cui è stato progettato, fornendo una misura diretta 
    dell'affidabilità delle predizioni accettate in tale contesto.
    
    Numeratore: casi dove uno risponde CORRETTAMENTE e due fanno reject
    Denominatore: tutti i casi dove esattamente UNO risponde (corretto o sbagliato) 
                  e due fanno reject
    
    Notazione:
    - "1" = predizione corretta (non reject)
    - "0" = predizione sbagliata (non reject)
    - "?" = rejection
    """
    
    def __init__(self):
        super().__init__("Ideal Single Vote Rate")
    
    def _compute(self, predictions: np.ndarray, y_test: np.ndarray,
                 X_test: np.ndarray = None, model = None) -> float:
        """
        Calcola l'ideal single vote rate per 3 classificatori.
        
        Args:
            predictions: Array con shape [3, n_samples] contenente le predizioni
                        dei tre classificatori (possono contenere "reject")
            y_test: Labels vere
        
        Returns:
            Il rapporto tra "uno risponde correttamente" su "uno risponde" 
            (correttamente o erroneamente)
        """
        # Verifica che ci siano esattamente 3 classificatori
        if predictions.shape[0] != 3:
            raise ValueError(f"Ideal single vote rate richiede esattamente 3 classificatori, "
                           f"ma ne sono stati forniti {predictions.shape[0]}")
        
        pred_1 = predictions[0, :]
        pred_2 = predictions[1, :]
        pred_3 = predictions[2, :]
        
        n_samples = len(y_test)
        
        # Contatori per i 6 pattern del denominatore
        # Pattern con UNO corretto e DUE reject
        N1qq = 0  # Primo corretto, secondo e terzo reject
        Nq1q = 0  # Secondo corretto, primo e terzo reject
        Nqq1 = 0  # Terzo corretto, primo e secondo reject
        
        # Pattern con UNO sbagliato e DUE reject
        N0qq = 0  # Primo sbagliato, secondo e terzo reject
        Nq0q = 0  # Secondo sbagliato, primo e terzo reject
        Nqq0 = 0  # Terzo sbagliato, primo e secondo reject
        
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
            elif pattern == "0??":
                N0qq += 1
            elif pattern == "?0?":
                Nq0q += 1
            elif pattern == "??0":
                Nqq0 += 1
        
        # Calcola numeratore e denominatore
        numerator = N1qq + Nq1q + Nqq1
        denominator = N1qq + N0qq + Nq1q + Nq0q + Nqq1 + Nqq0
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    
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