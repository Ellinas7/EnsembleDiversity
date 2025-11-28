from ML_algorithms.ML_algorithm import ML_algorithm
from rotation_forest import RotationForestClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np


class rotation_forest(ML_algorithm):
    """Rotation Forest Classifier"""
    
    def __init__(self, n_estimators: int = 10, random_state: int = 42, 
                 n_jobs: int = -1, max_features=None, bootstrap=True, 
                 max_depth=None, min_samples_split=2):
        """
        Args:
            n_estimators: Numero di alberi nell'ensemble (default: 10)
            random_state: Seed per riproducibilità
            n_jobs: Numero di job paralleli (-1 = tutti i core)
            max_features: Numero massimo di features per split
            bootstrap: Se usare bootstrap sampling
            max_depth: Profondità massima degli alberi
            min_samples_split: Numero minimo di campioni per split
        """
        super().__init__(n_estimators, random_state)
        self.n_jobs = n_jobs
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
    
    def _create_model(self):
        """Crea e restituisce l'istanza del modello Rotation Forest"""
        return RotationForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            max_features=self.max_features,
            bootstrap=self.bootstrap,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split
        )
    
    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """
        Restituisce le probabilità di classe predette.
        
        Questo metodo è necessario per la compatibilità con i decoratori
        di rejection (static_threshold_rejection_decorator, 
        percentile_threshold_rejection_decorator).
        
        Args:
            X_test: Dati di test
            
        Returns:
            Array shape (n_samples, n_classes) con probabilità per ogni classe
        """
        if self.model is None:
            raise ValueError("Modello non ancora addestrato. Chiamare il metodo train() prima.")
        
        # Converti in numpy array se necessario
        if hasattr(X_test, 'values'):
            X_test = X_test.values
        
        return self.model.predict_proba(X_test)
    
    """def _extract_predictions(self, X_test: np.ndarray) -> np.ndarray:
        
        Estrae le predizioni dei singoli alberi del Rotation Forest.
        
        Rotation Forest, come Random Forest, ha un attributo estimators_
        che contiene tutti gli alberi dell'ensemble.
        
        Args:
            X_test: Dati di test
            
        Returns:
            Array shape (n_estimators, n_samples) con predizioni NUMERICHE
        
        if self.model is None:
            raise ValueError("Modello non ancora addestrato.")
        
        # Converti in numpy array se necessario
        if hasattr(X_test, 'values'):
            X_test = X_test.values
        
        # Estrai predizioni di ogni singolo albero
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
        if self.model is None:
            raise ValueError("Modello non ancora addestrato.")
        
        if hasattr(X_test, 'values'):
            X_test = X_test.values
        
        predictions = np.array([
            estimator.predict(X_test) 
            for estimator in self.model.estimators_
        ])
        
        # Mappa indici alle classi originali
        classes = self.model.classes_
        predictions = np.array([[classes[int(p)] for p in row] for row in predictions])
        
        return predictions
   
    def _get_estimator_confidences(self, X_test: np.ndarray) -> np.ndarray:
        """
        Estrae le confidence dei singoli alberi del Rotation Forest.
        
        Per ogni albero, calcola predict_proba e prende il massimo
        come confidence per quel campione.
        
        Args:
            X_test: Dati di test
            
        Returns:
            Array shape (n_estimators, n_samples) con le confidence
        """
        if self.model is None:
            raise ValueError("Modello non ancora addestrato.")
        
        # Converti in numpy array se necessario
        if hasattr(X_test, 'values'):
            X_test = X_test.values
        
        confidences = []
        
        for estimator in self.model.estimators_:
            # Ottieni le probabilità dell'albero
            probas = estimator.predict_proba(X_test)
            # Prendi la probabilità massima come confidence
            max_conf = np.max(probas, axis=1)
            confidences.append(max_conf)
        
        return np.array(confidences)