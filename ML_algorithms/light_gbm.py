from ML_algorithms.ML_algorithm import ML_algorithm
import lightgbm as lgb
import numpy as np
from sklearn.preprocessing import LabelEncoder


class light_gbm(ML_algorithm):
    """LightGBM Classifier con label encoding automatico"""
    
    def __init__(self, n_estimators: int = 100, random_state: int = 42, 
                 max_depth: int = -1, learning_rate: float = 0.1, n_jobs: int = -1,
                 num_leaves: int = 31):
        super().__init__(n_estimators, random_state)
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_jobs = n_jobs
        self.num_leaves = num_leaves
        self.label_encoder = None
    
    def _create_model(self):
        return lgb.LGBMClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            verbose=-1
        )
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Addestra LightGBM con label encoding automatico.
        """
        if self.model is None:
            self.model = self._create_model()
        
        if hasattr(X_train, 'values'):
            X_train = X_train.values
        if hasattr(y_train, 'values'):
            y_train = y_train.values
        
        if y_train.dtype == object or y_train.dtype.name == 'category':
            print(f"  → Label encoding per LightGBM...")
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
        """
        Effettua predizioni sul test set.
        """
        if self.model is None:
            raise ValueError("Modello non ancora addestrato. Chiama train() prima.")
        
        if hasattr(X_test, 'values'):
            X_test = X_test.values
        
        predictions = self.model.predict(X_test)
        
        if self.label_encoder is not None:
            predictions = self.label_encoder.inverse_transform(predictions.astype(int))
        
        return predictions
    
    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """
        Restituisce le probabilità predette per ogni classe.
        """
        if self.model is None:
            raise ValueError("Modello non ancora addestrato. Chiama train() prima.")
        
        if hasattr(X_test, 'values'):
            X_test = X_test.values
        
        return self.model.predict_proba(X_test)
