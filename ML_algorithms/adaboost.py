from .ML_algorithm import ML_algorithm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np

class adaboost(ML_algorithm):
    """AdaBoost Classifier"""
    
    def __init__(self, n_estimators: int = 50, random_state: int = 42, 
                 max_depth: int = 1):
        super().__init__(n_estimators, random_state)
        self.max_depth = max_depth
    
    def _create_model(self):
        base_estimator = DecisionTreeClassifier(max_depth=self.max_depth)
        return AdaBoostClassifier(
            estimator=base_estimator,
            n_estimators=self.n_estimators,
            random_state=self.random_state
        )
