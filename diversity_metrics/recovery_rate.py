from .diversity_metric import diversity_metric
import numpy as np

class recovery_rate(diversity_metric):
    """
    Recovery Rate: rappresenta la percentuale di risposte corrette del secondo 
    classificatore a seguito di una rejection fatta dal primo classificatore.
    
    Formula: N?1 / (N?1 + N?0 + N??)
    
    Indica la qualità del secondo classificatore sul reject del primo.
    Questa metrica è specifica per il recovery block, dove il classificatore
    secondario viene consultato solo quando il primario si astiene.
    
    Notazione:
    - N?1: primo fa reject, secondo corretto
    - N?0: primo fa reject, secondo sbagliato
    - N??: primo fa reject, secondo fa reject
    """
    
    def __init__(self):
        super().__init__("Recovery Rate")
    
    def _compute(self, predictions: np.ndarray, y_test: np.ndarray,
                 X_test: np.ndarray = None, model = None) -> float:
        """
        Calcola il recovery rate per una coppia di classificatori in recovery block.
        
        Args:
            predictions: Array con shape [2, n_samples] dove:
                        - predictions[0] = classificatore primario (D1)
                        - predictions[1] = classificatore secondario (D2)
            y_test: Labels vere
        
        Returns:
            La proporzione di risposte corrette del secondo classificatore
            quando il primo fa reject: N?1 / (N?1 + N?0 + N??)
        """
        # Verifica che ci siano esattamente 2 classificatori
        if predictions.shape[0] != 2:
            raise ValueError(f"Recovery rate richiede esattamente 2 classificatori, "
                           f"ma ne sono stati forniti {predictions.shape[0]}")
        
        pred_1 = predictions[0, :]  # Classificatore primario
        pred_2 = predictions[1, :]  # Classificatore secondario
        
        # Identifica dove il primo classificatore fa reject
        first_rejected = (pred_1 == "reject")
        
        # Consideriamo solo i casi dove il primo fa reject
        pred_2_after_reject = pred_2[first_rejected]
        y_test_after_reject = y_test[first_rejected]
        
        # Se il primo non ha mai fatto reject, return 0
        if len(pred_2_after_reject) == 0:
            return 0.0
        
        # Calcola le tre categorie quando il primo fa reject:
        # N?1: secondo corretto (non reject)
        is_correct_2 = (pred_2_after_reject == y_test_after_reject) & (pred_2_after_reject != "reject")
        N_question_1 = np.sum(is_correct_2)
        
        # N?0: secondo sbagliato (non reject)
        is_wrong_2 = (pred_2_after_reject != y_test_after_reject) & (pred_2_after_reject != "reject")
        N_question_0 = np.sum(is_wrong_2)
        
        # N??: secondo fa reject
        is_reject_2 = (pred_2_after_reject == "reject")
        N_question_question = np.sum(is_reject_2)
        
        # Denominatore: tutti i casi dove il primo fa reject
        denominator = N_question_1 + N_question_0 + N_question_question
        
        if denominator == 0:
            return 0.0
            
        return N_question_1 / denominator