from classifiers.rejection_decorator import RejectionDecorator
from classifiers.classifier import Classifier
import numpy as np


class PercentileThreshold(RejectionDecorator):
    """
    Decorator che implementa rejection con soglia basata su percentile.
    
    Rifiuta il X% dei campioni con confidenza più bassa.
    """
    
    def __init__(self, base_classifier: Classifier, rejection_percentile: float = 10.0):
        """
        Args:
            base_classifier: Istanza di Classifier da decorare
            rejection_percentile: Percentuale di campioni da rifiutare (default 10.0%)
        """
        super().__init__(base_classifier)
        self.rejection_percentile = rejection_percentile
        self.name = f"{base_classifier.name}_percentile_{int(rejection_percentile)}"
    
    def _calculate_threshold(self, confidences: np.ndarray) -> float:
        """Calcola la soglia come percentile delle confidenze."""
        return np.percentile(confidences, self.rejection_percentile)
