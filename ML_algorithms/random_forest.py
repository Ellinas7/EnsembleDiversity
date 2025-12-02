from .ML_algorithm import ML_algorithm
from sklearn.ensemble import RandomForestClassifier
import numpy as np


class random_forest(ML_algorithm):
    """Random Forest Classifier"""
    
    def __init__(self, n_estimators: int = 100, random_state: int = 42, n_jobs: int = -1):
        super().__init__(n_estimators, random_state)
        self.n_jobs = n_jobs
    
    def _create_model(self):
        return RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=self.n_jobs
        )
