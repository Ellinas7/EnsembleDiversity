from abc import ABC, abstractmethod
import numpy as np

class diversity_metric(ABC):
    """Classe base astratta per metriche di diversity"""
    
    def __init__(self, name: str):
        self.name = name
        self.rejection_label = -1
    
    def calculate(self, predictions: np.ndarray, y_test: np.ndarray,
                  X_test: np.ndarray = None, model = None) -> float:
        """
        Calcola la metrica e stampa il risultato.
        
        Args:
            predictions: Predizioni dei classificatori (possono contenere "reject")
            y_test: Labels vere
            X_test: Dati di test (opzionale, per metriche rejection)
            model: Modello (opzionale, per metriche rejection)
        """
        value = self._compute(predictions, y_test, X_test, model)
        print(f"{self.name}: {value:.6f}")
        return value
    
    def _get_valid_mask(self, pred_i: np.ndarray, pred_j: np.ndarray) -> np.ndarray:
        """Restituisce maschera dei campioni validi per una coppia (no reject)"""
        return (pred_i != self.rejection_label) & (pred_j != self.rejection_label)
    
    @abstractmethod
    def _compute(self, predictions: np.ndarray, y_test: np.ndarray,
                 X_test: np.ndarray = None, model = None) -> float:
        """Metodo astratto da implementare nelle sottoclassi"""
        pass