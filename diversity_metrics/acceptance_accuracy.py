from .diversity_metric import diversity_metric
import numpy as np

class acceptance_accuracy(diversity_metric):
    """
    Acceptance Accuracy: misura l'accuratezza del classificatore limitatamente 
    ai campioni per i quali fornisce una predizione.
    
    Formula: TA / (TA + FA)
    
    A differenza della metrica single_correct_prediction_rate, questa metrica 
    considera solamente le istanze per cui il classificatore ha effettivamente 
    fornito una risposta. Rappresenta quindi l'affidabilità del classificatore 
    quando decide di rispondere.
    
    Dalla rejection matrix:
    - TA (True Acceptance): predizioni corrette accettate
    - FA (False Acceptance): predizioni errate accettate
    - FR (False Rejection): rejection su predizioni corrette
    - TR (True Rejection): rejection su predizioni errate
    
    Nota: Questa metrica considera solo i campioni accettati (TA + FA), 
    escludendo i rifiutati (FR + TR). È il complemento a 1 della metrica Miss.
    """
    
    def __init__(self):
        super().__init__("Acceptance Accuracy")
    
    def _compute(self, predictions: np.ndarray, y_test: np.ndarray,
                 X_test: np.ndarray = None, model = None) -> float:
        """
        Calcola l'acceptance accuracy dalla rejection matrix.
        
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
            
        return TA / total_accepted