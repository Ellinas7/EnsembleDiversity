from ML_algorithms.ML_algorithm import ML_algorithm
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np


class decision_tree(ML_algorithm):
    """Decision Tree Classifier"""
    
    def __init__(self, max_depth: int = None, min_samples_split: int = 2,
                 min_samples_leaf: int = 1, criterion: str = 'gini',
                 random_state: int = 42):
        """
        Args:
            max_depth: Profondità massima dell'albero (None = illimitata)
            min_samples_split: Minimo numero di campioni per fare uno split
            min_samples_leaf: Minimo numero di campioni in una foglia
            criterion: Funzione per misurare la qualità dello split ('gini' o 'entropy')
            random_state: Seed per riproducibilità
        """
        # Decision Tree è un singolo classificatore, quindi n_estimators=1
        super().__init__(n_estimators=1, random_state=random_state)
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
    
    def _create_model(self):
        return DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            criterion=self.criterion,
            random_state=self.random_state
        )
    
    def _extract_predictions(self, X_test: np.ndarray) -> np.ndarray:
        """
        Estrae le predizioni del singolo Decision Tree.
        
        Returns:
            Array shape (1, n_samples) con le predizioni
        """
        predictions = self.model.predict(X_test)
        
        # Conversione predizioni in formato numerico se sono stringhe
        if np.issubdtype(predictions.dtype, np.number):
            predictions_numeric = predictions.astype(int)
        else:
            # Le predizioni sono stringhe, dobbiamo convertirle
            le = LabelEncoder()
            all_labels = np.unique(predictions)
            le.fit(all_labels)
            predictions_numeric = le.transform(predictions)
        
        # Ritorna come array 2D (1, n_samples) per compatibilità
        return predictions_numeric.reshape(1, -1)