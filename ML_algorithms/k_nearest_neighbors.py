from ML_algorithms.ML_algorithm import ML_algorithm
from sklearn.neighbors import KNeighborsClassifier
import numpy as np


class knn(ML_algorithm):
    """K-Nearest Neighbors Classifier"""
    
    def __init__(self, n_neighbors: int = 5, n_jobs: int = -1, random_state: int = 42):
        super().__init__(n_estimators=1, random_state=random_state)
        self.n_neighbors = n_neighbors
        self.n_jobs = n_jobs
    
    def _create_model(self):
        return KNeighborsClassifier(
            n_neighbors=self.n_neighbors,
            n_jobs=self.n_jobs
        )