from abc import ABC, abstractmethod
from classifiers.classifier import Classifier
import numpy as np


class RejectionDecorator(Classifier):
    """
    Decorator astratto per aggiungere rejection option a un Classifier.
    
    Implementa il pattern Decorator mantenendo l'interfaccia di Classifier
    e delegando tutte le operazioni all'oggetto wrappato, eccetto predict()
    che viene modificato per implementare la logica di rejection.
    """
    
    def __init__(self, base_classifier: Classifier):
        """
        Args:
            base_classifier: Istanza di una classe che eredita da Classifier
        """
        super().__init__()
        self.base_classifier = base_classifier
        self.name = f"{base_classifier.name}_with_rejection"

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Delega il training al classificatore base"""
        self.base_classifier.train(X_train, y_train)
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predice con rejection option.
        
        Args:
            X_test: Dati di test
            
        Returns:
            Array con predizioni: classi originali o "reject"
        """
        probas = self.predict_proba(X_test)
        max_confidences = np.max(probas, axis=1)
        
        base_predictions = self.base_classifier.predict(X_test)
        
        threshold = self._calculate_threshold(max_confidences)
        
        predictions = np.where(
            max_confidences >= threshold,
            base_predictions,
            self.rejection_label
        )
        
        return predictions
    
    @abstractmethod
    def _calculate_threshold(self, confidences: np.ndarray) -> float:
        """
        Calcola la soglia di rejection.
        
        Args:
            confidences: Array con le confidenze di ogni campione
            
        Returns:
            Soglia di confidenza da usare per il rejection
        """
        pass
    
    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """Delega al classificatore base"""
        return self.base_classifier.predict_proba(X_test)
    
    def get_confidence_scores(self, X_test: np.ndarray) -> np.ndarray:
        """
        Restituisce gli score di confidenza per ogni campione.
        
        Args:
            X_test: Dati di test
            
        Returns:
            Array con le confidenze (probabilità massime)
        """
        probas = self.predict_proba(X_test)
        return np.max(probas, axis=1)
