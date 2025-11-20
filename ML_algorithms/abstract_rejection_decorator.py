from abc import ABC, abstractmethod
import numpy as np
from typing import Optional


class abstract_rejection_decorator(ABC):
    """
    Decorator astratto per aggiungere rejection option a un ML_algorithm.
    
    Implementa il pattern Decorator mantenendo l'interfaccia di ML_algorithm
    e delegando tutte le operazioni all'oggetto wrappato, eccetto predict()
    che viene modificato per implementare la logica di rejection.
    """
    
    def __init__(self, base_algorithm):
        """
        Args:
            base_algorithm: Istanza di una classe che eredita da ML_algorithm
        """
        self.base_algorithm = base_algorithm
        self.rejection_label = "reject"  # Label per "I don't know"
        
        # Manteniamo le proprietà del modello base per compatibilità
        self.name = f"{base_algorithm.name}_with_rejection"
        self.n_estimators = base_algorithm.n_estimators
        self.random_state = base_algorithm.random_state

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Delega il training al modello base"""
        self.base_algorithm.train(X_train, y_train)
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predice con rejection option usando il Template Method pattern.
        
        La logica comune è qui, mentre il calcolo della soglia è delegato
        alle sottoclassi concrete tramite _calculate_threshold().
        
        Args:
            X_test: Dati di test
            
        Returns:
            Array con predizioni: classi originali o "reject" (I don't know)
        """
        # Ottieni probabilità e confidenze
        probas = self.predict_proba(X_test)
        max_confidences = np.max(probas, axis=1)
        
        # Predizioni del modello base
        base_predictions = self.base_algorithm.predict(X_test)
        
        # Calcola la soglia usando la strategia specifica (Template Method)
        threshold = self._calculate_threshold(max_confidences)
        
        # Applica rejection
        predictions = np.where(
            max_confidences >= threshold,
            base_predictions,
            self.rejection_label
        )
        
        return predictions
    
    def _calculate_threshold(self, confidences: np.ndarray) -> float:
        """
        Calcola la soglia di rejection usando la strategia specifica.
        
        Template Method: ogni sottoclasse concreta implementa la sua logica.
        
        Args:
            confidences: Array con le confidenze (probabilità massime) di ogni campione
            
        Returns:
            Soglia di confidenza da usare per il rejection
        """
        pass
    
    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """Delega al modello base"""
        return self.base_algorithm.model.predict_proba(X_test)
    
    def get_estimator_predictions(self, X_test: np.ndarray) -> np.ndarray:
        """Delega al modello base"""
        return self.base_algorithm.get_estimator_predictions(X_test)
    
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
    
    