from .ML_algorithm import ML_algorithm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np

class adaboost(ML_algorithm):
    """AdaBoost Classifier"""
    
    def __init__(self, n_estimators: int = 2, random_state: int = 42, 
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
        """Estrae confidence di ogni weak learner di AdaBoost"""
        confidences = []
        
        for estimator in self.model.estimators_:
            probas = estimator.predict_proba(X_test)
            max_conf = np.max(probas, axis=1)
            confidences.append(max_conf)
        
        return np.array(confidences)