from ML_algorithms.ML_algorithm import ML_algorithm
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np


class gradient_boosting_decision_trees(ML_algorithm):
    """Gradient Boosting Decision Trees (GBDT) Classifier"""
    
    def __init__(self, n_estimators: int = 100, random_state: int = 42, 
                 max_depth: int = 3, learning_rate: float = 0.1,
                 subsample: float = 1.0, loss: str = 'log_loss'):
        """
        Args:
            n_estimators: Numero di weak learners (alberi) nell'ensemble
            random_state: Seed per riproducibilità
            max_depth: Profondità massima degli alberi
            learning_rate: Tasso di apprendimento (shrinkage)
            subsample: Frazione di campioni da usare per fit di ogni albero
            loss: Funzione di perdita ('log_loss' o 'exponential')
        """
        super().__init__(n_estimators, random_state)
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.loss = loss
    
    def _create_model(self):
        return GradientBoostingClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            loss=self.loss,
            random_state=self.random_state
        )
