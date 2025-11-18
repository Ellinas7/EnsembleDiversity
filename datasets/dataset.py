import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from typing import Tuple, Optional


class dataset:
    """Classe per la gestione dei dataset"""
    
    def __init__(self, file_path: str, dataset_name: str = None, 
                 test_size: float = 0.2, random_state: int = 42, 
                 stratify: bool = False):
        """
        Args:
            file_path: Path completo del file CSV
            dataset_name: Nome del dataset (opzionale, per il logging)
            test_size: Percentuale del test set (default 0.2 = 20%)
            random_state: Seed per riproducibilità
            stratify: Se True, mantiene la distribuzione delle classi nello split
        """
        self.file_path = file_path
        self.dataset_name = dataset_name or "Dataset"
        self.test_size = test_size
        self.random_state = random_state
        self.stratify = stratify
        
        # Questi saranno popolati dopo prepare()
        self._X_train = None
        self._X_test = None
        self._y_train = None
        self._y_test = None
        self._df = None
    
    def _load_data(self) -> pd.DataFrame:
        """Carica il dataset dal file CSV"""
        return pd.read_csv(self.file_path)
    
    def _handle_missing_values(self, X: pd.DataFrame) -> pd.DataFrame:
        """Gestisce valori mancanti usando l'imputer con mediana"""
        n_missing = X.isnull().sum().sum()
        if n_missing > 0:
            print(f"  → Valori mancanti rilevati: {n_missing}")
            print(f"  → Applicando imputer con mediana...")
            imputer = SimpleImputer(strategy='median')
            X_imputed = imputer.fit_transform(X)
            X = pd.DataFrame(X_imputed, columns=X.columns, index=X.index)
            print(f"  ✓ Valori mancanti gestiti")
        return X
    
    def _split_data(self):
        """Separa features e target, gestisce NaN e fa lo split train/test"""
        # Separazione X e y
        X = self._df.drop('multilabel', axis=1)
        y = self._df['multilabel']
        
        # Gestione valori mancanti
        X = self._handle_missing_values(X)
        
        # Split train/test
        stratify_param = y if self.stratify else None
        self._X_train, self._X_test, self._y_train, self._y_test = train_test_split(
            X, y, 
            test_size=self.test_size, 
            random_state=self.random_state,
            stratify=stratify_param
        )
    
    def preprocess(self):
        """Prepara il dataset: carica, preprocessa e splitta"""
        print(f"\n{'='*70}")
        print(f"PREPARAZIONE DATASET - {self.dataset_name}")
        print(f"{'='*70}")
        print(f"File: {self.file_path}")
        
        # Caricamento
        self._df = self._load_data()
        print(f"✓ Dataset caricato - Shape: {self._df.shape}")
        print(f"  Colonne: {self._df.shape[1]}, Campioni: {self._df.shape[0]}")
        
        # Verifica presenza colonna target
        if 'multilabel' not in self._df.columns:
            raise ValueError("Colonna 'multilabel' non trovata nel dataset!")
        
        classi = self._df['multilabel'].unique()
        print(f"  Classi target: {sorted(classi)}")
        
        # Split
        self._split_data()
        print(f"✓ Split completato ({int(self.test_size*100)}% test)")
        print(f"  Train: {self._X_train.shape[0]} campioni")
        print(f"  Test: {self._X_test.shape[0]} campioni")
        
        if self.stratify:
            print(f"  Stratify: attivo")
        
        print(f"{'='*70}\n")
    
    @property
    def X_train(self) -> pd.DataFrame:
        if self._X_train is None:
            raise ValueError("Dataset non preparato. Chiama prepare() prima.")
        return self._X_train
    
    @property
    def X_test(self) -> pd.DataFrame:
        if self._X_test is None:
            raise ValueError("Dataset non preparato. Chiama prepare() prima.")
        return self._X_test
    
    @property
    def y_train(self) -> pd.Series:
        if self._y_train is None:
            raise ValueError("Dataset non preparato. Chiama prepare() prima.")
        return self._y_train
    
    @property
    def y_test(self) -> pd.Series:
        if self._y_test is None:
            raise ValueError("Dataset non preparato. Chiama prepare() prima.")
        return self._y_test
    
    @property
    def data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Restituisce tutti i dati in una volta"""
        return self.X_train, self.X_test, self.y_train, self.y_test