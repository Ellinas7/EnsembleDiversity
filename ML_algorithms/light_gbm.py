from ML_algorithms.ML_algorithm import ML_algorithm
import lightgbm as lgb
import numpy as np
from sklearn.preprocessing import LabelEncoder


class light_gbm(ML_algorithm):
    """LightGBM Classifier con label encoding automatico (stessa strategia di XGBoost)"""
    
    def __init__(self, n_estimators: int = 2, random_state: int = 42, 
                 max_depth: int = -1, learning_rate: float = 0.1, n_jobs: int = -1,
                 num_leaves: int = 31):
        super().__init__(n_estimators, random_state)
        self.max_depth = max_depth  # -1 = nessun limite (default LightGBM)
        self.learning_rate = learning_rate
        self.n_jobs = n_jobs
        self.num_leaves = num_leaves  # Parametro specifico di LightGBM
        self.label_encoder = None  # Sarà creato se necessario
    
    def _create_model(self):
        return lgb.LGBMClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            verbose=-1  # Silenzioso per evitare troppo output
        )
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Addestra LightGBM con label encoding automatico (stessa logica di XGBoost).
        
        Args:
            X_train: Features di training
            y_train: Labels di training (possono essere stringhe o numeriche)
        """
        if self.model is None:
            self.model = self._create_model()
        
        # Converti in numpy array se necessario
        if hasattr(X_train, 'values'):
            X_train = X_train.values
        if hasattr(y_train, 'values'):
            y_train = y_train.values
        
        # Label encoding se necessario (stesso pattern di XGBoost)
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
        
        Args:
            X_test: Dati di test
            
        Returns:
            Array di predizioni nelle LABEL ORIGINALI (con inverse_transform se necessario)
        """
        if self.model is None:
            raise ValueError("Modello non ancora addestrato. Chiama train() prima.")
        
        # Converti in numpy array se necessario
        if hasattr(X_test, 'values'):
            X_test = X_test.values
        
        # Predizione
        predictions = self.model.predict(X_test)
        
        # Inverse transform se necessario
        if self.label_encoder is not None:
            predictions = self.label_encoder.inverse_transform(predictions.astype(int))
        
        return predictions
    
    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """
        Restituisce le probabilità predette per ogni classe.
        
        Args:
            X_test: Dati di test
            
        Returns:
            Array con probabilità per ogni classe
        """
        if self.model is None:
            raise ValueError("Modello non ancora addestrato. Chiama train() prima.")
        
        if hasattr(X_test, 'values'):
            X_test = X_test.values
        
        return self.model.predict_proba(X_test)
    
    def _extract_predictions(self, X_test: np.ndarray) -> np.ndarray:
        """
        Estrae le predizioni di ogni singolo boosting round (albero) del LightGBM.
        Questo è più complesso in LightGBM rispetto a Random Forest.
        
        Args:
            X_test: Dati di test
            
        Returns:
            Array di shape (n_estimators, n_samples) con predizioni di ogni boosting round
        """
        if self.model is None:
            raise ValueError("Modello non ancora addestrato.")
        
        # Converti in numpy array se necessario
        if hasattr(X_test, 'values'):
            X_test = X_test.values
        
        # LightGBM usa predict con num_iteration per ottenere predizioni ad ogni step
        predictions = []
        
        for i in range(1, self.n_estimators + 1):
            # Predizione usando solo i primi i alberi
            pred = self.model.predict(X_test, num_iteration=i)
            
            # Inverse transform se necessario
            if self.label_encoder is not None:
                pred = self.label_encoder.inverse_transform(pred.astype(int))
            
            predictions.append(pred)
        
        return np.array(predictions)
    
    def _get_estimator_confidences(self, X_test: np.ndarray) -> np.ndarray:
        """
        Estrae confidence di ogni boosting round del LightGBM.
        
        Args:
            X_test: Dati di test
            
        Returns:
            Array di shape (n_estimators, n_samples) con confidence di ogni boosting round
        """
        if self.model is None:
            raise ValueError("Modello non ancora addestrato.")
        
        # Converti in numpy array se necessario
        if hasattr(X_test, 'values'):
            X_test = X_test.values
        
        confidences = []
        
        for i in range(1, self.n_estimators + 1):
            # Probabilità usando solo i primi i alberi
            probas = self.model.predict_proba(X_test, num_iteration=i)
            max_conf = np.max(probas, axis=1)
            confidences.append(max_conf)
        
        return np.array(confidences)