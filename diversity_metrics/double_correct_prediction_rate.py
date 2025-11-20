from .diversity_metric import diversity_metric
import numpy as np

class double_correct_prediction_rate(diversity_metric):
    """
    Double Correct Prediction Rate: indica la probabilità che l'ensemble 
    fornisca una classificazione corretta nel contesto di voting 2 su 2.
    
    Formula: N11 / Ntot
    
    Dove N11 rappresenta i casi in cui entrambi i classificatori predicono 
    correttamente (e non si astengono).
    
    Dalla tabella delle metriche per voting 2 su 2:
    - N11: entrambi corretti e non reject
    - N01: primo sbagliato, secondo corretto
    - N?1: primo reject, secondo corretto
    - N10: primo corretto, secondo sbagliato
    - N00: entrambi sbagliati
    - N?0: primo reject, secondo sbagliato
    - N1?: primo corretto, secondo reject
    - N0?: primo sbagliato, secondo reject
    - N??: entrambi reject
    
    Questa metrica è specifica per ensemble con voting 2 su 2, dove il sistema
    accetta una predizione solo quando entrambi i classificatori concordano.
    """
    
    def __init__(self):
        super().__init__("Double Correct Prediction Rate")
    
    def _compute(self, predictions: np.ndarray, y_test: np.ndarray,
                 X_test: np.ndarray = None, model = None) -> float:
        """
        Calcola il double correct prediction rate per una coppia di classificatori.
        
        Args:
            predictions: Array con shape [2, n_samples] contenente le predizioni
                        dei due classificatori (possono contenere "reject")
            y_test: Labels vere
        
        Returns:
            La proporzione di campioni in cui entrambi i classificatori predicono
            correttamente (N11 / Ntot)
        """
        # Verifica che ci siano esattamente 2 classificatori
        if predictions.shape[0] != 2:
            raise ValueError(f"Double correct prediction rate richiede esattamente 2 classificatori, "
                            f"ma ne sono stati forniti {predictions.shape[0]}")
        
        pred_1 = predictions[0, :]
        pred_2 = predictions[1, :]
        
        # Identifica dove ci sono rejection
        is_rejected_1 = (pred_1 == "reject")
        is_rejected_2 = (pred_2 == "reject")
        
        # Identifica predizioni corrette (escludendo rejection)
        # Una predizione è corretta se: pred == y_test AND pred != "reject"
        is_correct_1 = (pred_1 == y_test) & ~is_rejected_1
        is_correct_2 = (pred_2 == y_test) & ~is_rejected_2
        
        # Conta N11: entrambi i classificatori predicono correttamente
        N11 = np.sum(is_correct_1 & is_correct_2)
        
        # Ntot è il numero totale di sample
        Ntot = len(y_test)
        
        if Ntot == 0:
            return 0.0
            
        return N11 / Ntot