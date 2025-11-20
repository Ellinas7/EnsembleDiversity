from .diversity_metric import diversity_metric
from .single_misclassification_rate import single_misclassification_rate
from .recovery_failure_rate import recovery_failure_rate
import numpy as np

class recovery_wrong_prediction_rate(diversity_metric):
    """
    Recovery Wrong Prediction Rate: quantifica la probabilità totale che 
    l'ensemble fornisca una classificazione errata nel contesto del recovery block.
    
    Formula: FA/(TA + FA + FR + TR) + N?0/(N?1 + N?0 + N??)
    
    La metrica è formata da due contributi distinti:
    1. Single Misclassification Rate del classificatore primario (formula 1.1.2)
    2. Recovery Failure Rate (formula 1.3.2)
       
    Questa formulazione riflette il funzionamento sequenziale del recovery block,
    dove il secondo classificatore viene consultato solo quando il primo si astiene.
    """
    
    def __init__(self):
        super().__init__("Recovery Wrong Prediction Rate")
    
    def _compute(self, predictions: np.ndarray, y_test: np.ndarray,
                 X_test: np.ndarray = None, model = None) -> float:
        """
        Calcola il recovery wrong prediction rate per recovery block.
        
        Args:
            predictions: Array con shape [2, n_samples] dove:
                        - predictions[0] = classificatore primario (D1)
                        - predictions[1] = classificatore secondario (D2)
            y_test: Labels vere
            X_test: Dati di test (necessario per single_misclassification_rate)
            model: Lista/array di modelli dove model[0] è il classificatore primario
        
        Returns:
            Somma di: single_misclassification_rate del primario + recovery failure rate
        """
        # Verifica che ci siano esattamente 2 classificatori
        if predictions.shape[0] != 2:
            raise ValueError(f"Recovery wrong prediction rate richiede esattamente 2 classificatori, "
                           f"ma ne sono stati forniti {predictions.shape[0]}")
        
        if X_test is None or model is None:
            raise ValueError("Recovery wrong prediction rate richiede X_test e model")
        
        # Primo termine: Single Misclassification Rate del classificatore primario
        smr = single_misclassification_rate()
        first_term = smr._compute(predictions[0, :], y_test, X_test, model[0])
        
        # Secondo termine: Recovery Failure Rate 
        rfr = recovery_failure_rate()
        second_term = rfr._compute(predictions, y_test, X_test, model)
        
        return first_term + second_term