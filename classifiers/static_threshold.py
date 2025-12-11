from classifiers.rejection_decorator import RejectionDecorator
from classifiers.classifier import Classifier
import numpy as np


class StaticThreshold(RejectionDecorator):
    """
    Decorator che implementa rejection con soglia statica/fissa.
    
    Rifiuta tutte le predizioni con confidenza < threshold.
    """
    
    def __init__(self, base_classifier: Classifier, confidence_threshold: float = 0.9):
        """
        Args:
            base_classifier: Istanza di Classifier da decorare
            confidence_threshold: Soglia minima di confidenza (default 0.9 = 90%)
        """
        super().__init__(base_classifier)
        self.confidence_threshold = confidence_threshold
        self.name = f"{base_classifier.name}_static_{confidence_threshold}"
    
    def _calculate_threshold(self, confidences: np.ndarray) -> float:
        """Restituisce la soglia fissa configurata"""
        return self.confidence_threshold
