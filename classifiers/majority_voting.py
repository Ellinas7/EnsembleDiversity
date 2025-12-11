from classifiers.classifier import Classifier
import numpy as np
from collections import Counter


class MajorityVoting(Classifier):
    """
    Majority Voting: accetta la classe/reject che ottiene almeno 2 voti su 3.
    Il reject conta come voto.
    
    Composite pattern: contiene una lista di Classifier.
    """
    
    def __init__(self, classifiers: list):
        super().__init__()
        if len(classifiers) != 3:
            raise ValueError("MajorityVoting richiede esattamente 3 classificatori")
        self.classifiers = classifiers
        self.name = f"MajorityVoting({classifiers[0].name}, {classifiers[1].name}, {classifiers[2].name})"
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Addestra tutti i classificatori interni"""
        print(f"Training {self.name}...")
        for clf in self.classifiers:
            clf.train(X_train, y_train)
        print(f"✓ {self.name} addestrato")
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predizione: accetta la classe con almeno 2 voti su 3.
        """
        predictions = self.get_base_predictions(X_test)
        n_samples = predictions.shape[1]
        final_predictions = np.empty(n_samples, dtype=object)
        
        for i in range(n_samples):
            votes = [predictions[j, i] for j in range(3)]
            counter = Counter(votes)
            most_common_class, count = counter.most_common(1)[0]
            
            if count >= 2:
                # Maggioranza trovata (classe o reject)
                final_predictions[i] = most_common_class
            else:
                # Nessuna maggioranza (3 voti diversi)
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
