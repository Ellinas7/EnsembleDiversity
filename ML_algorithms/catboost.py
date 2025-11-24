from ML_algorithms.ML_algorithm import ML_algorithm
from catboost import CatBoostClassifier
import numpy as np
from sklearn.preprocessing import LabelEncoder


class catboost(ML_algorithm):
    """CatBoost Classifier con label encoding automatico"""
    
    def __init__(self, n_estimators: int = 2, random_state: int = 42, 
                 max_depth: int = 6, learning_rate: float = 0.1):
        super().__init__(n_estimators, random_state)
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.label_encoder = None
    
    def _create_model(self):
        return CatBoostClassifier(
            iterations=self.n_estimators,
            depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
            verbose=0,
            allow_writing_files=False
        )
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        if self.model is None:
            self.model = self._create_model()
        
        if hasattr(X_train, 'values'):
            X_train = X_train.values
        if hasattr(y_train, 'values'):
            y_train = y_train.values
        
        if y_train.dtype == object or y_train.dtype.name == 'category':
            print(f"  → Label encoding per CatBoost...")
            self.label_encoder = LabelEncoder()
            y_train_encoded = self.label_encoder.fit_transform(y_train)
            print(f"    Classi: {self.label_encoder.classes_}")
        else:
            self.label_encoder = None
            y_train_encoded = y_train
        
        print(f"Training {self.name}...")
        self.model.fit(X_train, y_train_encoded)
        print(f"✓ {self.name} addestrato")
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Modello non ancora addestrato. Chiama train() prima.")
        
        if hasattr(X_test, 'values'):
            X_test = X_test.values
        
        predictions_encoded = self.model.predict(X_test).flatten()
        
        if self.label_encoder is not None:
            return self.label_encoder.inverse_transform(predictions_encoded.astype(int))
        else:
            return predictions_encoded
    
    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Modello non ancora addestrato. Chiama train() prima.")
        
        if hasattr(X_test, 'values'):
            X_test = X_test.values
        
        return self.model.predict_proba(X_test)
    
    def encode_labels(self, y: np.ndarray) -> np.ndarray:
        if self.label_encoder is None:
            return y.values if hasattr(y, 'values') else y
        else:
            y_array = y.values if hasattr(y, 'values') else y
            return self.label_encoder.transform(y_array)
    
    def _extract_predictions(self, X_test: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Modello non ancora addestrato.")
        
        if hasattr(X_test, 'values'):
            X_test = X_test.values
        
        predictions = []
        for i in range(1, self.n_estimators + 1):
            pred = self.model.predict(X_test, ntree_end=i)
            predictions.append(pred)
        
        return np.array(predictions).astype(int)
    
    def _get_estimator_confidences(self, X_test: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Modello non ancora addestrato.")
        
        if hasattr(X_test, 'values'):
            X_test = X_test.values
        
        confidences = []
        for i in range(1, self.n_estimators + 1):
            proba = self.model.predict_proba(X_test, ntree_end=i)
            max_conf = np.max(proba, axis=1)
            confidences.append(max_conf)
        
        return np.array(confidences)