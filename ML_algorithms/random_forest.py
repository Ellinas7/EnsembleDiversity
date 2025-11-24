from .ML_algorithm import ML_algorithm
from sklearn.ensemble import RandomForestClassifier
import numpy as np


class random_forest(ML_algorithm):
    """Random Forest Classifier"""
    
    def __init__(self, n_estimators: int = 2, random_state: int = 42, n_jobs: int = -1):
        super().__init__(n_estimators, random_state)
        self.n_jobs = n_jobs
    
    def _create_model(self):
        return RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=self.n_jobs
        )
    
    def _extract_predictions(self, X_test: np.ndarray) -> np.ndarray:
        predictions = np.array([
            estimator.predict(X_test) 
            for estimator in self.model.estimators_
        ])
        return predictions
    
    def _get_estimator_confidences(self, X_test: np.ndarray) -> np.ndarray:
        """Estrae confidence di ogni singolo albero del Random Forest"""
        confidences = []
        
        for estimator in self.model.estimators_:
            probas = estimator.predict_proba(X_test)
            max_conf = np.max(probas, axis=1)
            confidences.append(max_conf)
        
        return np.array(confidences)