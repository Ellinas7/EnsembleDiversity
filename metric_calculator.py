from ML_algorithms.static_threshold_rejection_decorator import static_threshold_rejection_decorator
from datasets.dataset import dataset
from ML_algorithms.ML_algorithm import ML_algorithm
from diversity_metrics.diversity_metric import diversity_metric
import numpy as np
import warnings


class metric_calculator:
    """Calcola metriche di diversity su un ensemble addestrato"""
    warnings.filterwarnings("ignore")
    
    def calculate(self, ds: dataset, algorithm: ML_algorithm, 
                  metric: diversity_metric) -> float:
        """
        Addestra l'algoritmo sul dataset e calcola la metrica.
        
        Args:
            ds: Dataset già preprocessato
            algorithm: Algoritmo ML (con o senza rejection)
            metric: Metrica di diversity da calcolare
            
        Returns:
            Valore della metrica
        """
        X_train, X_test, y_train, y_test = ds.data
        
        # Addestra
        algorithm.train(X_train, y_train)
        
        # Estrai predizioni dei singoli estimatori
        predictions = algorithm.get_estimator_predictions(X_test)
        
        # Converti y_test in numpy array
        y_test_array = y_test.values if hasattr(y_test, 'values') else y_test

        # Se predizioni numeriche e y_test stringhe, mappa le predizioni
        if np.issubdtype(predictions.dtype, np.number) and not np.issubdtype(y_test_array.dtype, np.number):
            y_train_array = y_train.values if hasattr(y_train, 'values') else y_train
            classes = np.sort(np.unique(y_train_array))  # sklearn usa ordine alfabetico
            predictions = np.array([[classes[int(p)] for p in row] for row in predictions])

        # Allinea i tipi: converti entrambi a stringa per confronti consistenti
        predictions = predictions.astype(str)
        y_test_array = y_test_array.astype(str)
        
        # Calcola metrica (gestisce sia metriche classiche che rejection)
        try:
            return metric._compute(predictions, y_test_array, X_test, algorithm)
        except TypeError:
            return metric._compute(predictions, y_test_array)
    




if __name__ == "__main__":
    from datasets.dataset import dataset
    from ML_algorithms.random_forest import random_forest
    from ML_algorithms.extra_trees import extra_trees
    from ML_algorithms.percentile_threshold_rejection_decorator import percentile_threshold_rejection_decorator
    from diversity_metrics.Q_statistic import Q_statistic
    from diversity_metrics.disagreement_measure import disagreement_measure
    from diversity_metrics.double_correct_prediction_rate import double_correct_prediction_rate
    from diversity_metrics.majority_voting_correct_prediction_rate import majority_voting_correct_prediction_rate
    
    # Dataset
    ds = dataset("/Users/matteopascuzzo/Desktop/Datasets/Error Detection/arancino_all_scikit.csv",
                 dataset_name="arancino_all_scikit", stratify=True)
    ds.preprocess()
    
    calc = metric_calculator()
    
    # Esperimento
    rf = random_forest(n_estimators=2)
    rf_with_statitic_rej = static_threshold_rejection_decorator(rf, confidence_threshold=0.6)
    result = calc.calculate(ds, rf, Q_statistic())
    print(f"Q-statistic: {result}\n")