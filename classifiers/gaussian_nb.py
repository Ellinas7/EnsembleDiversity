from classifiers.classifier import Classifier
from sklearn.naive_bayes import GaussianNB as GaussianNBClassifier
import numpy as np


class GaussianNB(Classifier):
    """Gaussian Naive Bayes Classifier"""
    
    def __init__(self, random_state: int = 42):
        super().__init__()
        self.random_state = random_state
        self.model = None
    
    def _create_model(self):
        return GaussianNBClassifier()

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        if self.model is None:
            self.model = self._create_model()
        
        print(f"Training {self.name}...")
        self.model.fit(X_train, y_train)
        print(f"✓ {self.name} addestrato")

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Modello non ancora addestrato. Chiama train() prima.")
        return self.model.predict(X_test)
    
    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Modello non ancora addestrato. Chiama train() prima.")
        return self.model.predict_proba(X_test)
