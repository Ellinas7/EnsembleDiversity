import numpy as np
from ML_algorithms.abstract_rejection_decorator import abstract_rejection_decorator


class static_threshold_rejection_decorator(abstract_rejection_decorator):
    """
    Decorator che implementa rejection con soglia statica/fissa.
    
    Rifiuta tutte le predizioni con confidenza < threshold (es. 0.9).
    Il tasso di rejection varia in base al dataset e al modello.
    """
    
    def __init__(self, base_algorithm, confidence_threshold: float = 0.9):
        """
        Args:
            base_algorithm: Istanza di ML_algorithm da decorare
            confidence_threshold: Soglia minima di confidenza (default 0.9 = 90%)
        """
        super().__init__(base_algorithm)
        self.confidence_threshold = confidence_threshold
        self.name = f"{base_algorithm.name}_static_threshold_{float(confidence_threshold)}"
    
    def _calculate_threshold(self, confidences: np.ndarray) -> float:
        """Restituisce la soglia fissa configurata"""
        return self.confidence_threshold