from .ML_algorithm import ML_algorithm
from sklearn.ensemble import ExtraTreesClassifier
import numpy as np

class extra_trees(ML_algorithm):
    """Extra-Trees Classifier"""
    
    def __init__(self, n_estimators: int = 100, random_state: int = 42, n_jobs: int = -1):
        super().__init__(n_estimators, random_state)
        self.n_jobs = n_jobs
    
    def _create_model(self):
        return ExtraTreesClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=self.n_jobs
        )
