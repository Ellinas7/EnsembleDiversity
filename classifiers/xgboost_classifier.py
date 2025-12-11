from classifiers.classifier import Classifier
import xgboost as xgb
import numpy as np
from sklearn.preprocessing import LabelEncoder


class XGBoost(Classifier):
    """XGBoost Classifier con label encoding automatico"""
    
    def __init__(self, n_estimators: int = 100, random_state: int = 42, 
                 max_depth: int = 6, learning_rate: float = 0.3, n_jobs: int = -1):
        super().__init__()
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_jobs = n_jobs
        self.model = None
        self.label_encoder = None
    
    def _create_model(self):
        return xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            eval_metric='logloss'
        )

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        if self.model is None:
            self.model = self._create_model()
        
        if hasattr(X_train, 'values'):
            X_train = X_train.values
        if hasattr(y_train, 'values'):
            y_train = y_train.values
        
        if y_train.dtype == object or y_train.dtype.name == 'category':
            print(f"  → Label encoding per XGBoost...")
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
        
        predictions_encoded = self.model.predict(X_test)
        
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
