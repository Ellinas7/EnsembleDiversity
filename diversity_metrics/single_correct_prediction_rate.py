from .diversity_metric import diversity_metric
import numpy as np

class single_correct_prediction_rate(diversity_metric):
    """
    Single Correct Prediction Rate: misura la probabilità che il classificatore
    fornisca una classificazione corretta.
    
    Formula: TA / (TA + FA + FR + TR)
    
    Dalla rejection matrix:
    - TA (True Acceptance): predizioni corrette accettate
    - FA (False Acceptance): predizioni errate accettate
    - FR (False Rejection): rejection su predizioni corrette
    - TR (True Rejection): rejection su predizioni errate
    """
    
    def __init__(self):
        super().__init__("Single Correct Prediction Rate")
    
    def _compute(self, predictions: np.ndarray, y_test: np.ndarray,
                 X_test: np.ndarray = None, model = None) -> float:
        """
        Calcola il single correct prediction rate dalla rejection matrix.
        
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
        
        # Calcola la metrica
        total = TA + FA + FR + TR
        
        if total == 0:
            return 0.0
            
        return TA / total