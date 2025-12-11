from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, Optional


class Classifier(ABC):
    """
    Classe base astratta per tutti i classificatori.
    
    Sia i classificatori singoli che i composite (ensemble) derivano da questa classe.
    Tutti devono implementare train() e predict().
    predict() restituisce un array di etichette che possono essere classi del problema o "reject".
    """
    
    def __init__(self):
        self.name = self.__class__.__name__
        self.rejection_label = "reject"

    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Addestra il classificatore.
        
        Args:
            X_train: Features di training
            y_train: Labels di training
        """
        pass

    @abstractmethod
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Effettua predizioni sul test set.
        
        Args:
            X_test: Dati di test
            
        Returns:
            Array di predizioni (classi o "reject")
        """
        pass
    
    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """
        Restituisce le probabilità predette per ogni classe.
        
        Args:
            X_test: Dati di test
            
        Returns:
            Array shape (n_samples, n_classes) con probabilità per ogni classe
            
        Nota: Non tutti i classificatori supportano questo metodo.
        """
        raise NotImplementedError(f"{self.name} non supporta predict_proba")
