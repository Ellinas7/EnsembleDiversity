from ML_algorithms.ML_algorithm import ML_algorithm
from sklearn.naive_bayes import GaussianNB as GaussianNBClassifier
import numpy as np


class gaussian_nb(ML_algorithm):
    """Gaussian Naive Bayes Classifier"""
    
    def __init__(self, random_state: int = 42):
        super().__init__(n_estimators=1, random_state=random_state)
    
    def _create_model(self):
        return GaussianNBClassifier()