from ML_algorithms.ML_algorithm import ML_algorithm
from logitboost import LogitBoost as LogitBoostClassifier
import numpy as np


class logitboost(ML_algorithm):
    """LogitBoost Classifier"""
    
    def __init__(self, n_estimators: int = 50, random_state: int = 42,
                 learning_rate: float = 1.0):
        super().__init__(n_estimators, random_state)
        self.learning_rate = learning_rate
    
    def _create_model(self):
        return LogitBoostClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            random_state=self.random_state
        )