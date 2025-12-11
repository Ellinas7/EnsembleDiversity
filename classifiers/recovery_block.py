from classifiers.classifier import Classifier
import numpy as np


class RecoveryBlock(Classifier):
    """
    Recovery Block: consulta D1, se fa reject passa a D2.
    L'ordine dei classificatori nella lista è importante.
    
    Composite pattern: contiene una lista di Classifier.
    """
    
    def __init__(self, classifiers: list):
        super().__init__()
        if len(classifiers) != 2:
            raise ValueError("RecoveryBlock richiede esattamente 2 classificatori")
        self.classifiers = classifiers
        self.name = f"RecoveryBlock({classifiers[0].name} -> {classifiers[1].name})"
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Addestra tutti i classificatori interni"""
        print(f"Training {self.name}...")
        for clf in self.classifiers:
            clf.train(X_train, y_train)
        print(f"✓ {self.name} addestrato")
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predizione: usa D1, se fa reject passa a D2.
        """
        pred_primary = self.classifiers[0].predict(X_test)    # D1
        pred_secondary = self.classifiers[1].predict(X_test)  # D2
        
        n_samples = len(pred_primary)
        final_predictions = np.empty(n_samples, dtype=object)
        
        for i in range(n_samples):
            if pred_primary[i] != self.rejection_label:
                # D1 risponde, usa la sua predizione
                final_predictions[i] = pred_primary[i]
            else:
                # D1 fa reject, passa a D2
                final_predictions[i] = pred_secondary[i]
        
        return final_predictions
    
    def get_base_predictions(self, X_test: np.ndarray) -> np.ndarray:
        """
        Restituisce le predizioni dei singoli classificatori base.
        Utile per calcolare le metriche di coppia.
        
        Returns:
            Array shape (2, n_samples)
        """
        pred_1 = self.classifiers[0].predict(X_test)
        pred_2 = self.classifiers[1].predict(X_test)
        return np.array([pred_1, pred_2])
