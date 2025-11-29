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
        """
        Crea il modello Random Patches usando BaggingClassifier.
        
        Random Patches = Bagging + Random Subspaces:
        - max_samples < 1.0: campiona casualmente una frazione dei campioni (Bagging)
        - max_features < 1.0: campiona casualmente una frazione delle features (Random Subspaces)
        """
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
    
    def _extract_predictions(self, X_test: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Modello non ancora addestrato.")
        
        if hasattr(X_test, 'values'):
            X_test = X_test.values
        
        predictions = []
        for estimator, features in zip(self.model.estimators_, self.model.estimators_features_):
            X_subset = X_test[:, features]
            predictions.append(estimator.predict(X_subset))
        
        predictions = np.array(predictions)
        
        # Mappa alle classi originali
        classes = self.model.classes_
        predictions = np.array([[classes[int(p)] for p in row] for row in predictions])
        
        return predictions
    
    def _get_estimator_confidences(self, X_test: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Modello non ancora addestrato.")
        
        if hasattr(X_test, 'values'):
            X_test = X_test.values
        
        confidences = []
        for estimator, features in zip(self.model.estimators_, self.model.estimators_features_):
            X_subset = X_test[:, features]
            probas = estimator.predict_proba(X_subset)
            max_conf = np.max(probas, axis=1)
            confidences.append(max_conf)
        
        return np.array(confidences)