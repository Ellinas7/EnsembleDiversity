from classifiers.classifier import Classifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np


class AdaBoost(Classifier):
    """AdaBoost Classifier"""
    
    def __init__(self, n_estimators: int = 50, random_state: int = 42, 
                 max_depth: int = 1):
        super().__init__()
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.max_depth = max_depth
        self.model = None
    
    def _create_model(self):
        base_estimator = DecisionTreeClassifier(max_depth=self.max_depth)
        return AdaBoostClassifier(
            estimator=base_estimator,
            n_estimators=self.n_estimators,
            random_state=self.random_state
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
