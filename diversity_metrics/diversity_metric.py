from abc import ABC, abstractmethod
import numpy as np

class diversity_metric(ABC):
    """Classe base astratta per metriche di diversity"""
    
    def __init__(self, name: str):
        self.name = name
    
    def calculate(self, predictions: np.ndarray, y_test: np.ndarray) -> float:
        """Calcola la metrica e stampa il risultato"""
        value = self._compute(predictions, y_test)
        print(f"{self.name}: {value:.6f}")
        return value
    
    @abstractmethod
    def _compute(self, predictions: np.ndarray, y_test: np.ndarray) -> float:
        """Metodo astratto da implementare nelle sottoclassi"""
        pass