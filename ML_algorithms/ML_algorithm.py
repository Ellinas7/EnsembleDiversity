from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics import accuracy_score, matthews_corrcoef
from typing import Tuple, Optional

class ML_algorithm(ABC):
    """Classe base astratta per algoritmi di ensemble learning"""
    
    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        """
        Args:
            n_estimators: Numero di weak learners nell'ensemble
            random_state: Seed per riproducibilità
        """
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model = None
        self.name = self.__class__.__name__

    @abstractmethod
    def _create_model(self):
        """Crea e restituisce l'istanza del modello ensemble"""
        pass

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Addestra l'ensemble.
        
        Args:
            X_train: Features di training
            y_train: Labels di training
        """
        if self.model is None:
            self.model = self._create_model()
        
        print(f"Training {self.name}...")
        self.model.fit(X_train, y_train)
        print(f"✓ {self.name} addestrato")

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Effettua predizioni sul test set.
        
        Args:
            X_test: Dati di test
            
        Returns:
            Array di predizioni
        """
        if self.model is None:
            raise ValueError("Modello non ancora addestrato. Chiamare il metodo train() prima.")
        
        return self.model.predict(X_test)
    
    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """
        Restituisce le probabilità predette per ogni classe.
        
        Args:
            X_test: Dati di test
            
        Returns:
            Array shape (n_samples, n_classes) con probabilità per ogni classe
        """
        if self.model is None:
            raise ValueError("Modello non ancora addestrato. Chiamare il metodo train() prima.")
        
        return self.model.predict_proba(X_test)
    
    def calculate_accuracy(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        """
        Calcola l'accuracy sul test set.
        
        Args:
            X_test: Dati di test
            y_test: Labels vere
            
        Returns:
            Accuracy
        """
        y_pred = self.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy {self.name}: {accuracy:.4f}")
        return accuracy
    
    def calculate_mcc(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        """
        Calcola il Matthews Correlation Coefficient sul test set.
        
        Args:
            X_test: Dati di test
            y_test: Labels vere
            
        Returns:
            MCC
        """
        y_pred = self.predict(X_test)
        mcc = matthews_corrcoef(y_test, y_pred)
        print(f"MCC {self.name}: {mcc:.4f}")
        return mcc
