from .diversity_metric import diversity_metric
from .single_correct_prediction_rate import single_correct_prediction_rate
from .recovery_rate import recovery_rate
import numpy as np

class recovery_correct_prediction_rate(diversity_metric):
    """
    Recovery Correct Prediction Rate: quantifica la probabilità totale che 
    l'ensemble fornisca una classificazione corretta nel contesto del recovery block.
    
    Formula: TA/(TA + FA + FR + TR) + N?1/(N?1 + N?0 + N??)
    
    La metrica è formata da due contributi distinti:
    1. Single Correct Prediction Rate del classificatore primario (formula 1.1.1)
    2. Recovery Rate (formula 1.3.1)
       
    Questa formulazione riflette il funzionamento sequenziale del recovery block,
    dove il secondo classificatore viene consultato solo quando il primo si astiene.
    """
    
    def __init__(self):
        super().__init__("Recovery Correct Prediction Rate")
    
    def _compute(self, predictions: np.ndarray, y_test: np.ndarray,
                 X_test: np.ndarray = None, model = None) -> float:
        """
        Calcola il recovery correct prediction rate per recovery block.
        
        Args:
            predictions: Array con shape [2, n_samples] dove:
                        - predictions[0] = classificatore primario (D1)
                        - predictions[1] = classificatore secondario (D2)
            y_test: Labels vere
            X_test: Dati di test (necessario per single_correct_prediction_rate)
            model: Lista/array di modelli dove model[0] è il classificatore primario
        
        Returns:
            Somma di: single_correct_prediction_rate del primario + recovery rate
        """
        # Verifica che ci siano esattamente 2 classificatori
        if predictions.shape[0] != 2:
            raise ValueError(f"Recovery correct prediction rate richiede esattamente 2 classificatori, "
                           f"ma ne sono stati forniti {predictions.shape[0]}")
        
        if X_test is None or model is None:
            raise ValueError("Recovery correct prediction rate richiede X_test e model")
        
        # Primo termine: Single Correct Prediction Rate del classificatore primario
        scpr = single_correct_prediction_rate()
        first_term = scpr._compute(predictions[0, :], y_test, X_test, model[0])
        
        # Secondo termine: Recovery Rate 
        rr = recovery_rate()
        second_term = rr._compute(predictions, y_test, X_test, model)
        
        return first_term + second_term