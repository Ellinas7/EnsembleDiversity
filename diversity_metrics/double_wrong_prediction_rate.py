from .diversity_metric import diversity_metric
import numpy as np

class double_wrong_prediction_rate(diversity_metric):
    """
    Double Wrong Prediction Rate: indica la probabilità che l'ensemble 
    fornisca una misclassification nel contesto di voting 2 su 2.
    
    Formula: N00 / Ntot
    
    Dove N00 rappresenta i casi in cui entrambi i classificatori predicono 
    erroneamente (e non si astengono).
    
    Questa metrica cattura le situazioni più problematiche dove entrambi i 
    classificatori commettono un errore sullo stesso campione.
    """
    
    def __init__(self):
        super().__init__("Double Wrong Prediction Rate")
    
    def _compute(self, predictions: np.ndarray, y_test: np.ndarray,
                 X_test: np.ndarray = None, model = None) -> float:
        """
        Calcola il double wrong prediction rate per una coppia di classificatori.
        
        Args:
            predictions: Array con shape [2, n_samples] contenente le predizioni
                        dei due classificatori (possono contenere "reject")
            y_test: Labels vere
        
        Returns:
            La proporzione di campioni in cui entrambi i classificatori predicono
            erroneamente (N00 / Ntot)
        """
        # Verifica che ci siano esattamente 2 classificatori
        if predictions.shape[0] != 2:
            raise ValueError(f"Double wrong prediction rate richiede esattamente 2 classificatori, "
                           f"ma ne sono stati forniti {predictions.shape[0]}")
        
        pred_1 = predictions[0, :]
        pred_2 = predictions[1, :]
        
        # Identifica dove ci sono rejection
        is_rejected_1 = (pred_1 == "reject")
        is_rejected_2 = (pred_2 == "reject")
        
        # Identifica predizioni errate (escludendo rejection)
        # Una predizione è errata se: pred != y_test AND pred != "reject"
        is_wrong_1 = (pred_1 != y_test) & ~is_rejected_1
        is_wrong_2 = (pred_2 != y_test) & ~is_rejected_2
        
        # Conta N00: entrambi i classificatori predicono erroneamente
        N00 = np.sum(is_wrong_1 & is_wrong_2)
        
        # Ntot è il numero totale di sample
        Ntot = len(y_test)
        
        if Ntot == 0:
            return 0.0
            
        return N00 / Ntot