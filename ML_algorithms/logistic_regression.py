from ML_algorithms.ML_algorithm import ML_algorithm
from sklearn.linear_model import LogisticRegression as LR
import numpy as np


class logistic_regression(ML_algorithm):
    """Logistic Regression Classifier"""
    
    def __init__(self, max_iter: int = 1000, random_state: int = 42, n_jobs: int = -1):
        super().__init__(n_estimators=1, random_state=random_state)
        self.max_iter = max_iter
        self.n_jobs = n_jobs
    
    def _create_model(self):
        return LR(
            max_iter=self.max_iter,
            random_state=self.random_state,
            n_jobs=self.n_jobs
        )