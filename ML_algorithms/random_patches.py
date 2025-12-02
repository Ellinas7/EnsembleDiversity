from ML_algorithms.ML_algorithm import ML_algorithm
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np

class random_patches(ML_algorithm):
    """
    Random Patches Ensemble Classifier.
    
    Combina bagging (sampling casuale di campioni) con random subspaces 
    (sampling casuale di features) per creare diversità nell'ensemble.
    """
    
    def __init__(self, 
                 n_estimators: int = 100,
                 max_samples: float = 0.5,
                 max_features: float = 0.5,
                 random_state: int = 42,
                 n_jobs: int = -1):
        """
        Args:
            n_estimators: Numero di base learners nell'ensemble
            max_samples: Frazione di campioni da campionare (0 < max_samples <= 1.0)
            max_features: Frazione di features da campionare (0 < max_features <= 1.0)
            random_state: Seed per riproducibilità
            n_jobs: Numero di core da usare (-1 = tutti)
        """
        super().__init__(n_estimators, random_state)
        self.max_samples = max_samples
        self.max_features = max_features
        self.n_jobs = n_jobs
        self.name = "RandomPatches"
    
    def _create_model(self):
        return BaggingClassifier(
            estimator=DecisionTreeClassifier(),
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            max_features=self.max_features,
            bootstrap=True,
            bootstrap_features=False,
            random_state=self.random_state,
            n_jobs=self.n_jobs
        )
