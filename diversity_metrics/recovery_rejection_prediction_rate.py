from .diversity_metric import diversity_metric
from .single_rejection_rate import single_rejection_rate
from .recovery_rejection_rate import recovery_rejection_rate
import numpy as np

class recovery_rejection_prediction_rate(diversity_metric):
    """
    Recovery Rejection Prediction Rate: quantifica la probabilità complessiva 
    che il sistema non fornisca una classificazione nel contesto del recovery block.
    
    Formula: (FR + TR)/(TA + FA + FR + TR) + N??/(N?1 + N?0 + N??)
    
    La metrica è formata da due contributi distinti:
    1. Single Rejection Rate del classificatore primario (formula 1.1.3)
    2. Recovery Rejection Rate (formula 1.3.3)
       
    Questa formulazione riflette il funzionamento sequenziale del recovery block,
    dove il secondo classificatore viene consultato solo quando il primo si astiene.
    """
    
    def __init__(self):
        super().__init__("Recovery Rejection Prediction Rate")
    
    def _compute(self, predictions: np.ndarray, y_test: np.ndarray,
                 X_test: np.ndarray = None, model = None) -> float:
        """
        Calcola il recovery rejection prediction rate per recovery block.
        
        Args:
            predictions: Array con shape [2, n_samples] dove:
                        - predictions[0] = classificatore primario (D1)
                        - predictions[1] = classificatore secondario (D2)
            y_test: Labels vere
            X_test: Dati di test (necessario per single_rejection_rate)
            model: Lista/array di modelli dove model[0] è il classificatore primario
        
        Returns:
            Somma di: single_rejection_rate del primario + recovery rejection rate
        """
        # Verifica che ci siano esattamente 2 classificatori
        if predictions.shape[0] != 2:
            raise ValueError(f"Recovery rejection prediction rate richiede esattamente 2 classificatori, "
                           f"ma ne sono stati forniti {predictions.shape[0]}")
        
        if X_test is None or model is None:
            raise ValueError("Recovery rejection prediction rate richiede X_test e model")
        
        # Primo termine: Single Rejection Rate del classificatore primario
        srr = single_rejection_rate()
        first_term = srr._compute(predictions[0, :], y_test, X_test, model[0])
        
        # Secondo termine: Recovery Rejection Rate 
        rrr = recovery_rejection_rate()
        second_term = rrr._compute(predictions, y_test, X_test, model)
        
        return first_term + second_term