from ML_algorithms.ML_algorithm import ML_algorithm
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np


class gradient_boosting_decision_trees(ML_algorithm):
    """Gradient Boosting Decision Trees (GBDT) Classifier"""
    
    def __init__(self, n_estimators: int = 2, random_state: int = 42, 
                 max_depth: int = 3, learning_rate: float = 0.1,
                 subsample: float = 1.0, loss: str = 'log_loss'):
        """
        Args:
            n_estimators: Numero di weak learners (alberi) nell'ensemble
            random_state: Seed per riproducibilità
            max_depth: Profondità massima degli alberi
            learning_rate: Tasso di apprendimento (shrinkage)
            subsample: Frazione di campioni da usare per fit di ogni albero (stochastic GB)
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
       
        Estrae predizioni degli ensemble parziali di Gradient Boosting.
        
        Per GBDT, seguiamo il pattern di staged_predict che restituisce
        le predizioni dopo ogni iterazione dell'ensemble.
        
        Returns:
            Array shape (n_estimators, n_samples) con predizioni NUMERICHE
        
        if self.model is None:
            raise ValueError("Modello non ancora addestrato.")
        
        # Converti in numpy array se necessario
        if hasattr(X_test, 'values'):
            X_test = X_test.values
        
        # staged_predict genera le predizioni per ogni stadio dell'ensemble
        # Ogni stadio corrisponde a un numero crescente di estimatori
        predictions = []
        for stage_pred in self.model.staged_predict(X_test):
            predictions.append(stage_pred)
        
        predictions = np.array(predictions)
        
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
        
        predictions = []
        for stage_pred in self.model.staged_predict(X_test):
            predictions.append(stage_pred)
        
        predictions = np.array(predictions)
        
        # staged_predict già restituisce le label originali, ma verifichiamo
        # Se sono numeriche e ci sono classi, mappiamo
        if np.issubdtype(predictions.dtype, np.number) and hasattr(self.model, 'classes_'):
            classes = self.model.classes_
            if not np.issubdtype(classes.dtype, np.number):
                predictions = np.array([[classes[int(p)] for p in row] for row in predictions])
        
        return predictions
    
    
    def _get_estimator_confidences(self, X_test: np.ndarray) -> np.ndarray:
        """
        Estrae confidence degli ensemble parziali di Gradient Boosting.
        
        Usa staged_predict_proba per ottenere le probabilità a ogni stadio
        e calcola la confidence come probabilità massima.
        
        Returns:
            Array shape (n_estimators, n_samples) con le confidence
        """
        if self.model is None:
            raise ValueError("Modello non ancora addestrato.")
        
        # Converti in numpy array se necessario
        if hasattr(X_test, 'values'):
            X_test = X_test.values
        
        # staged_predict_proba genera le probabilità per ogni stadio
        confidences = []
        for stage_proba in self.model.staged_predict_proba(X_test):
            # Confidence = probabilità massima
            max_conf = np.max(stage_proba, axis=1)
            confidences.append(max_conf)
        
        return np.array(confidences)