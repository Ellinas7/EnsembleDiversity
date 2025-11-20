from .diversity_metric import diversity_metric
import numpy as np

class miss(diversity_metric):
    """
    Miss: calcola quanto spesso il classificatore sbaglia quando decide di rispondere.
    
    Formula: FA / (TA + FA)
    
    Il miss rate rappresenta il tasso di errore condizionato dalla decisione di non
    astenersi, fornendo una misura del rischio associato alle predizioni accettate.
    
    Dalla rejection matrix:
    - TA (True Acceptance): predizioni corrette accettate
    - FA (False Acceptance): predizioni errate accettate
    - FR (False Rejection): rejection su predizioni corrette
    - TR (True Rejection): rejection su predizioni errate
    
    Nota: Questa metrica considera solo i campioni accettati (TA + FA), 
    escludendo i rifiutati (FR + TR).
    """
    
    def __init__(self):
        super().__init__("Miss")
    
    def _compute(self, predictions: np.ndarray, y_test: np.ndarray,
                 X_test: np.ndarray = None, model = None) -> float:
        """
        Calcola la metrica miss dalla rejection matrix.
        
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
        
        # Calcola la metrica (solo su campioni accettati)
        total_accepted = TA + FA
        
        if total_accepted == 0:
            return 0.0
            
        return FA / total_accepted