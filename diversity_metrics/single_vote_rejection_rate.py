from .diversity_metric import diversity_metric
from .single_vote_correct_prediction_rate import single_vote_correct_prediction_rate
from .single_vote_wrong_prediction_rate import single_vote_wrong_prediction_rate
import numpy as np

class single_vote_rejection_rate(diversity_metric):
    """
    Single Vote Rejection Rate: quantifica la probabilità che l'ensemble si 
    astenga nel contesto di voting 1 su n.
    
    Formula: 1 - (singleVoteCorrectPredictionRate + singleVoteWrongPredictionRate)
    
    Questa metrica consiste nella somma di tre situazioni:
    1. Triple reject (N???): nessun classificatore fornisce una predizione
    2. Triple answer: tutti e tre rispondono ma con predizioni diverse
    3. Situazioni dove esattamente due classificatori rispondono mentre il terzo si astiene
    
    Dato che l'insieme degli esiti possibili dell'ensemble si partiziona in 
    classificazioni corrette, classificazioni errate e rejection, la metrica 
    può essere calcolata come complemento delle altre due.
    """
    
    def __init__(self):
        super().__init__("Single Vote Rejection Rate")
    
    def _compute(self, predictions: np.ndarray, y_test: np.ndarray,
                 X_test: np.ndarray = None, model = None) -> float:
        """
        Calcola il single vote rejection rate per 3 classificatori.
        
        Args:
            predictions: Array con shape [3, n_samples] contenente le predizioni
                        dei tre classificatori (possono contenere "reject")
            y_test: Labels vere
        
        Returns:
            1 - (single_vote_correct + single_vote_wrong)
        """
        # Verifica che ci siano esattamente 3 classificatori
        if predictions.shape[0] != 3:
            raise ValueError(f"Single vote rejection rate richiede esattamente 3 classificatori, "
                           f"ma ne sono stati forniti {predictions.shape[0]}")
        
        # Calcola single vote correct prediction rate (formula 1.5.1)
        svcpr = single_vote_correct_prediction_rate()
        correct_rate = svcpr._compute(predictions, y_test, X_test, model)
        
        # Calcola single vote wrong prediction rate (formula 1.5.2)
        svwpr = single_vote_wrong_prediction_rate()
        wrong_rate = svwpr._compute(predictions, y_test, X_test, model)
        
        # Formula complementare
        return 1.0 - (correct_rate + wrong_rate)