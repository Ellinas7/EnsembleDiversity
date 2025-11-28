from .ML_algorithm import ML_algorithm
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np

class extra_trees(ML_algorithm):
    """Extra-Trees Classifier"""
    
    def __init__(self, n_estimators: int = 2, random_state: int = 42, n_jobs: int = -1):
        super().__init__(n_estimators, random_state)
        self.n_jobs = n_jobs
    
    def _create_model(self):
        return ExtraTreesClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=self.n_jobs
        )
    """
    def _extract_predictions(self, X_test: np.ndarray) -> np.ndarray:
        predictions = np.array([
            estimator.predict(X_test) 
            for estimator in self.model.estimators_
        ])
        
        # Conversione predizioni in formato numerico se sono stringhe
        if np.issubdtype(predictions.dtype, np.number):
            # Le predizioni sono già numeri, convertiamo solo a int
            predictions_numeric = predictions.astype(int)
        else:
            # Le predizioni sono stringhe, dobbiamo convertirle
            le = LabelEncoder()
            # Fit su tutte le label uniche presenti nelle predizioni
            all_labels = np.unique(predictions.flatten())
            le.fit(all_labels)
            # Converti ogni riga di predizioni
            predictions_numeric = np.array([le.transform(pred) for pred in predictions])
        
        return predictions_numeric"""

    def _extract_predictions(self, X_test: np.ndarray) -> np.ndarray:
        predictions = np.array([
            estimator.predict(X_test) 
            for estimator in self.model.estimators_
        ])
    
        # Mappa indici alle classi originali
        classes = self.model.classes_
        predictions = np.array([[classes[int(p)] for p in row] for row in predictions])
    
        return predictions
    
    def _get_estimator_confidences(self, X_test: np.ndarray) -> np.ndarray:
        """Estrae confidence di ogni albero di Extra Trees"""
        confidences = []
        
        for estimator in self.model.estimators_:
            probas = estimator.predict_proba(X_test)
            max_conf = np.max(probas, axis=1)
            confidences.append(max_conf)
        
        return np.array(confidences)