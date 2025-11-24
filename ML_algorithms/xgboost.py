from ML_algorithms.ML_algorithm import ML_algorithm
import xgboost as xgb
import numpy as np
from sklearn.preprocessing import LabelEncoder


class xgboost(ML_algorithm):
    """XGBoost Classifier con label encoding automatico"""
    
    def __init__(self, n_estimators: int = 2, random_state: int = 42, 
                 max_depth: int = 6, learning_rate: float = 0.3, n_jobs: int = -1):
        super().__init__(n_estimators, random_state)
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_jobs = n_jobs
        self.label_encoder = None  # Sarà creato se necessario
    
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
        """
        Addestra XGBoost con label encoding automatico.
        
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
        
        # Label encoding se necessario
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
        
        predictions_encoded = self.model.predict(X_test)
        
        # Riconverti nelle label originali se era stato fatto encoding
        if self.label_encoder is not None:
            return self.label_encoder.inverse_transform(predictions_encoded.astype(int))
        else:
            return predictions_encoded
    
    def encode_labels(self, y: np.ndarray) -> np.ndarray:
        """
        Helper per convertire label in formato numerico usando lo stesso encoding del training.
        Utile per convertire y_test prima di calcolare le metriche di diversity.
        
        Args:
            y: Labels da convertire (nel formato originale)
            
        Returns:
            Labels in formato numerico
        """
        if self.label_encoder is None:
            # Nessun encoding necessario
            return y.values if hasattr(y, 'values') else y
        else:
            # Converti usando il label encoder del training
            y_array = y.values if hasattr(y, 'values') else y
            return self.label_encoder.transform(y_array)
        
    def _extract_predictions(self, X_test: np.ndarray) -> np.ndarray:
        """
        Estrae predizioni degli ensemble parziali di XGBoost.
        
        IMPORTANTE: Seguendo il pattern dei notebook, questo metodo restituisce
        predizioni in formato NUMERICO (encoded), non nelle label originali.
        Questo è necessario per calcolare correttamente le metriche di diversity.
        
        Pattern esatto dai notebook:
        for i in range(xgb_model.n_estimators):
            pred = xgb_model.predict(X_test, iteration_range=(0, i+1))
            predictions_xgb.append(pred)
        predictions_xgb_numeric = np.array(predictions_xgb).astype(int)
        
        Returns:
            Array shape (n_estimators, n_samples) con predizioni NUMERICHE
        """
        if self.model is None:
            raise ValueError("Modello non ancora addestrato.")
        
        # Converti in numpy array se necessario
        if hasattr(X_test, 'values'):
            X_test = X_test.values
        
        predictions = []
        
        # Pattern ESATTO dai notebook
        for i in range(self.model.n_estimators):
            pred = self.model.predict(X_test, iteration_range=(0, i+1))
            predictions.append(pred)
        
        # Le predizioni sono già in formato numerico (encoded)
        predictions_numeric = np.array(predictions).astype(int)
        
        return predictions_numeric
    
    def _get_estimator_confidences(self, X_test: np.ndarray) -> np.ndarray:
        """
        Estrae le confidence degli ensemble PARZIALI di XGBoost.
        
        Per ogni iterazione i, calcola predict_proba con iteration_range=(0, i+1)
        e prende la probabilità massima come confidence.
        
        Returns:
            Array shape (n_estimators, n_samples) con le confidence
        """
        if self.model is None:
            raise ValueError("Modello non ancora addestrato.")
        
        # Converti in numpy array se necessario
        if hasattr(X_test, 'values'):
            X_test = X_test.values
        
        confidences = []
        
        for i in range(self.model.n_estimators):
            # Ottieni probabilità dell'ensemble parziale (primi i+1 alberi)
            probas = self.model.predict_proba(X_test, iteration_range=(0, i+1))
            # Prendi la probabilità massima come confidence
            max_conf = np.max(probas, axis=1)
            confidences.append(max_conf)
        
        return np.array(confidences)