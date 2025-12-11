from classifiers.classifier import Classifier
import numpy as np


class Voting1outofN(Classifier):
    """
    Voting 1 su N: accetta solo se esattamente 1 classificatore risponde
    (non reject) e gli altri 2 fanno reject.
    
    Composite pattern: contiene una lista di Classifier.
    """
    
    def __init__(self, classifiers: list):
        super().__init__()
        if len(classifiers) != 3:
            raise ValueError("Voting1ofN richiede esattamente 3 classificatori")
        self.classifiers = classifiers
        self.name = f"Voting1ofN({classifiers[0].name}, {classifiers[1].name}, {classifiers[2].name})"
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Addestra tutti i classificatori interni"""
        print(f"Training {self.name}...")
        for clf in self.classifiers:
            clf.train(X_train, y_train)
        print(f"✓ {self.name} addestrato")
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predizione: accetta solo se esattamente 1 risponde.
        """
        predictions = self.get_base_predictions(X_test)
        n_samples = predictions.shape[1]
        final_predictions = np.empty(n_samples, dtype=object)
        
        for i in range(n_samples):
            # Conta quanti classificatori rispondono (non reject)
            responses = [predictions[j, i] for j in range(3)
                         if predictions[j, i] != self.rejection_label]
            
            if len(responses) == 1:
                # Esattamente 1 risponde, accetta la sua predizione
                final_predictions[i] = responses[0]
            else:
                # 0 o 2+ rispondono → reject
                final_predictions[i] = self.rejection_label
        
        return final_predictions
    
    def get_base_predictions(self, X_test: np.ndarray) -> np.ndarray:
        """
        Restituisce le predizioni dei singoli classificatori base.
        Utile per calcolare le metriche di terna.
        
        Returns:
            Array shape (3, n_samples)
        """
        predictions = []
        for clf in self.classifiers:
            predictions.append(clf.predict(X_test))
        return np.array(predictions)
