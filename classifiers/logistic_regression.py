from classifiers.classifier import Classifier
from sklearn.linear_model import LogisticRegression as LR
import numpy as np


class LogisticRegression(Classifier):
    """Logistic Regression Classifier"""
    
    def __init__(self, max_iter: int = 1000, random_state: int = 42, n_jobs: int = -1):
        super().__init__()
        self.max_iter = max_iter
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.model = None
    
    def _create_model(self):
        return LR(
            max_iter=self.max_iter,
            random_state=self.random_state,
            n_jobs=self.n_jobs
        )

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
