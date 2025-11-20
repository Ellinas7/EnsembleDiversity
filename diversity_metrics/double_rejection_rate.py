from .diversity_metric import diversity_metric
import numpy as np

class double_rejection_rate(diversity_metric):
    """
    Double Rejection Rate: quantifica la probabilità di avere una rejection
    nel contesto di voting 2 su 2.
    
    Formula: N?? / Ntot
    
    Dove N?? rappresenta i casi in cui entrambi i classificatori si astengono
    dalla classificazione.
    
    Questa metrica caratterizza specificatamente il comportamento dell'ensemble 
    con rejection option, misurando quanto spesso entrambi i classificatori
    decidono di non rispondere sullo stesso campione.
    """
    
    def __init__(self):
        super().__init__("Double Rejection Rate")
    
    def _compute(self, predictions: np.ndarray, y_test: np.ndarray,
                 X_test: np.ndarray = None, model = None) -> float:
        """
        Calcola il double rejection rate per una coppia di classificatori.
        
        Args:
            predictions: Array con shape [2, n_samples] contenente le predizioni
                        dei due classificatori (possono contenere "reject")
            y_test: Labels vere
        
        Returns:
            La proporzione di campioni in cui entrambi i classificatori si astengono
            (N?? / Ntot)
        """
        # Verifica che ci siano esattamente 2 classificatori
        if predictions.shape[0] != 2:
            raise ValueError(f"Double rejection rate richiede esattamente 2 classificatori, "
                           f"ma ne sono stati forniti {predictions.shape[0]}")
        
        pred_1 = predictions[0, :]
        pred_2 = predictions[1, :]
        
        # Identifica dove ci sono rejection
        is_rejected_1 = (pred_1 == "reject")
        is_rejected_2 = (pred_2 == "reject")
        
        # Conta N??: entrambi i classificatori fanno reject
        N_double_reject = np.sum(is_rejected_1 & is_rejected_2)
        
        # Ntot è il numero totale di sample
        Ntot = len(y_test)
        
        if Ntot == 0:
            return 0.0
            
        return N_double_reject / Ntot