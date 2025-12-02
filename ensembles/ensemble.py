from abc import ABC, abstractmethod
import numpy as np


class Ensemble(ABC):
    """Classe astratta per meta-ensemble di classificatori"""
    
    def __init__(self, classifiers: list):
        """
        Args:
            classifiers: Lista di classificatori (possono avere rejection decorator)
        """
        self.classifiers = classifiers
        self.n_classifiers = len(classifiers)
        self.rejection_label = "reject"
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Addestra tutti i classificatori"""
        for clf in self.classifiers:
            clf.train(X_train, y_train)
    
    def get_predictions(self, X_test: np.ndarray) -> np.ndarray:
        """
        Restituisce le predizioni di ogni classificatore.
        
        Returns:
            Array shape (n_classifiers, n_samples)
        """
        predictions = []
        for clf in self.classifiers:
            pred = clf.predict(X_test)
            predictions.append(pred)
        return np.array(predictions)
    
    @abstractmethod
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predizione finale dell'ensemble secondo la logica specifica.
        
        Returns:
            Array shape (n_samples,)
        """
        pass
