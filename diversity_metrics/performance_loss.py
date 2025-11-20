from .diversity_metric import diversity_metric
import numpy as np

class performance_loss(diversity_metric):
    """
    Performance Loss: quantifica la perdita di performance dovuta alle 
    rejection inappropriate.
    
    Formula: FR / (TA + FR)
    
    Questa metrica calcola quanto spesso il classificatore rigetta un'istanza 
    che in verità era stata classificata correttamente per mancanza di certezza.
    
    Dalla rejection matrix:
    - TA (True Acceptance): predizioni corrette accettate
    - FA (False Acceptance): predizioni errate accettate
    - FR (False Rejection): rejection su predizioni corrette
    - TR (True Rejection): rejection su predizioni errate
    
    Nota: Questa metrica considera solo i campioni con predizione corretta (TA + FR),
    misurando quanti di questi vengono erroneamente rifiutati.
    """
    
    def __init__(self):
        super().__init__("Performance Loss")
    
    def _compute(self, predictions: np.ndarray, y_test: np.ndarray,
                 X_test: np.ndarray = None, model = None) -> float:
        """
        Calcola la performance loss dalla rejection matrix.
        
        Richiede X_test e model per ottenere le probabilità.
        """
        # Ottieni le probabilità
        probas = model.predict_proba(X_test)
        
        # Identifica dove ci sono rejection
        is_rejected = (predictions == "reject")
        
        # La predizione sottostante sarebbe corretta se argmax(probas) == y_test
        is_correct = (np.argmax(probas, axis=1) == y_test)
        
        # Calcola le 4 categorie della rejection matrix
        TA = np.sum(is_correct & ~is_rejected)   # Corrette E accettate
        FA = np.sum(~is_correct & ~is_rejected)  # Errate E accettate
        FR = np.sum(is_correct & is_rejected)    # Corrette MA rifiutate
        TR = np.sum(~is_correct & is_rejected)   # Errate E rifiutate
        
        # Calcola la metrica (solo su predizioni che sarebbero state corrette)
        total_correct = TA + FR
        
        if total_correct == 0:
            return 0.0
            
        return FR / total_correct