from classifiers.classifier import Classifier
import numpy as np


class Voting2outof2(Classifier):
    """
    Voting 2 su 2: accetta una predizione solo quando entrambi 
    i classificatori concordano (include reject unanime).
    
    Composite pattern: contiene una lista di Classifier.
    """
    
    def __init__(self, classifiers: list):
        super().__init__()
        if len(classifiers) != 2:
            raise ValueError("Voting2of2 richiede esattamente 2 classificatori")
        self.classifiers = classifiers
        self.name = f"Voting2of2({classifiers[0].name}, {classifiers[1].name})"
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Addestra tutti i classificatori interni"""
        print(f"Training {self.name}...")
        for clf in self.classifiers:
            clf.train(X_train, y_train)
        print(f"✓ {self.name} addestrato")
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predizione: accetta solo se entrambi concordano.
        """
        pred_1 = self.classifiers[0].predict(X_test)
        pred_2 = self.classifiers[1].predict(X_test)
        
        n_samples = len(pred_1)
        final_predictions = np.full(n_samples, self.rejection_label, dtype=object)
        
        for i in range(n_samples):
            if pred_1[i] == pred_2[i]:
                # Concordano (può essere classe o reject)
                final_predictions[i] = pred_1[i]
            # Altrimenti rimane reject (non concordano)
        
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
