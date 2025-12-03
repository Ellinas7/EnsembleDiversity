import numpy as np
from .abstract_rejection_decorator import abstract_rejection_decorator


class percentile_threshold_rejection_decorator(abstract_rejection_decorator):
    """
    Decorator che implementa rejection con soglia basata su percentile.
    
    Rifiuta il X% dei campioni con confidenza più bassa.
    """
    
    def __init__(self, base_algorithm, rejection_percentile: float = 10.0):
        """
        Args:
            base_algorithm: Istanza di ML_algorithm da decorare
            rejection_percentile: Percentuale di campioni da rifiutare (default 10.0%)
        """
        super().__init__(base_algorithm)
        self.rejection_percentile = rejection_percentile
        self.name = f"{base_algorithm.name}_percentile_threshold_{int(rejection_percentile)}"
    
    def _calculate_threshold(self, confidences: np.ndarray) -> float:
        """Calcola la soglia come percentile delle confidenze."""
        return np.percentile(confidences, self.rejection_percentile)
