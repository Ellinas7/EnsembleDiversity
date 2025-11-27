import numpy as np
from ML_algorithms.abstract_rejection_decorator import abstract_rejection_decorator


class percentile_threshold_rejection_decorator(abstract_rejection_decorator):
    """
    Decorator che implementa rejection con soglia basata su percentile.
    
    Rifiuta il X% dei campioni con confidenza più bassa.
    Garantisce un tasso di rejection costante indipendentemente dal dataset.
    """
    
    def __init__(self, base_algorithm, rejection_percentile: float = 10.0):
        """
        Args:
            base_algorithm: Istanza di ML_algorithm da decorare
            rejection_percentile: Percentuale di campioni da rifiutare (default 10.0%)
        """
        super().__init__(base_algorithm)
        self.rejection_percentile = rejection_percentile
        self.name = f"{base_algorithm.name}_percentile_threshold_{int(rejection_percentile)}"
    
    def _calculate_threshold(self, confidences: np.ndarray) -> float:
        """
        Calcola la soglia come percentile delle confidenze.
        
        Il percentile è calcolato dinamicamente sui dati di test:
        - rejection_percentile=10 → rifiuta il 10% con confidenza più bassa
        - rejection_percentile=20 → rifiuta il 20% con confidenza più bassa
        """
        return np.percentile(confidences, self.rejection_percentile)
    
    def get_percentile_value(self, X_test: np.ndarray) -> float:
        """
        Metodo di utility per ottenere il valore del percentile effettivamente usato.
        
        Utile per analisi e debugging, mostra quale soglia viene calcolata
        dinamicamente per un dato test set.
        
        Args:
            X_test: Dati di test
            
        Returns:
            Valore del percentile usato come soglia di confidenza
        """
        confidences = self.get_confidence_scores(X_test)
        return self._calculate_threshold(confidences)