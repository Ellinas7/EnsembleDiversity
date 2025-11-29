# experiment_logger.py
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
from metric_calculator import metric_calculator

# Import dataset
from datasets.dataset import dataset

# Import algoritmi ML
from ML_algorithms.ML_algorithm import ML_algorithm
from ML_algorithms.decision_tree import decision_tree
from ML_algorithms.random_forest import random_forest
from ML_algorithms.xgboost import xgboost
from ML_algorithms.adaboost import adaboost
from ML_algorithms.extra_trees import extra_trees
from ML_algorithms.gradient_boosting_decision_trees import gradient_boosting_decision_trees
from ML_algorithms.light_gbm import light_gbm
from ML_algorithms.catboost import catboost
from ML_algorithms.rotation_forest import rotation_forest
from ML_algorithms.random_rotation_forest import random_rotation_forest
from ML_algorithms.random_patches import random_patches

# Import decoratori rejection
from ML_algorithms.static_threshold_rejection_decorator import static_threshold_rejection_decorator
from ML_algorithms.percentile_threshold_rejection_decorator import percentile_threshold_rejection_decorator

# Import metriche classiche
from diversity_metrics.Q_statistic import Q_statistic
from diversity_metrics.disagreement_measure import disagreement_measure
from diversity_metrics.double_fault_measure import double_fault_measure
from diversity_metrics.entropy_measure import entropy_measure
from diversity_metrics.kohavi_wolpert_variance import kohavi_wolpert_variance
from diversity_metrics.generalized_diversity import generalized_diversity
from diversity_metrics.coincident_failure_diversity import coincident_failure_diversity

# Import metriche singolo classificatore
from diversity_metrics.single_correct_prediction_rate import single_correct_prediction_rate
from diversity_metrics.single_misclassification_rate import single_misclassification_rate
from diversity_metrics.single_rejection_rate import single_rejection_rate
from diversity_metrics.hit import hit
from diversity_metrics.miss import miss
from diversity_metrics.acceptance_accuracy import acceptance_accuracy
from diversity_metrics.performance_loss import performance_loss

# Import metriche voting 2/2
from diversity_metrics.double_correct_prediction_rate import double_correct_prediction_rate
from diversity_metrics.double_wrong_prediction_rate import double_wrong_prediction_rate
from diversity_metrics.double_rejection_rate import double_rejection_rate

# Import metriche recovery block
from diversity_metrics.recovery_rate import recovery_rate
from diversity_metrics.recovery_failure_rate import recovery_failure_rate
from diversity_metrics.recovery_rejection_rate import recovery_rejection_rate
from diversity_metrics.recovery_correct_prediction_rate import recovery_correct_prediction_rate
from diversity_metrics.recovery_wrong_prediction_rate import recovery_wrong_prediction_rate
from diversity_metrics.recovery_rejection_prediction_rate import recovery_rejection_prediction_rate

# Import metriche majority voting
from diversity_metrics.majority_voting_correct_prediction_rate import majority_voting_correct_prediction_rate
from diversity_metrics.majority_voting_wrong_prediction_rate import majority_voting_wrong_prediction_rate
from diversity_metrics.majority_voting_rejection_prediction_rate import majority_voting_rejection_prediction_rate

# Import metriche single vote
from diversity_metrics.single_vote_correct_prediction_rate import single_vote_correct_prediction_rate
from diversity_metrics.single_vote_wrong_prediction_rate import single_vote_wrong_prediction_rate
from diversity_metrics.single_vote_rejection_rate import single_vote_rejection_rate
from diversity_metrics.ideal_single_vote_rate import ideal_single_vote_rate


class ExperimentLogger:
    """Gestisce l'esecuzione di esperimenti e il salvataggio nel Megafile.csv"""
    
    COLUMNS = [
        "experiment_number", "experiment_name", "dataset_name", "classification_strategy",
        "q_statistic", "disagreement_measure", "double_fault_measure", "entropy_measure",
        "kohavi_wolpert_variance", "generalized_diversity", "coincident_failure_diversity",
        "single_correct_prediction_rate", "single_misclassification_rate", "single_rejection_rate",
        "hit", "miss", "acceptance_accuracy", "performance_loss",
        "double_correct_prediction_rate", "double_wrong_prediction_rate", "double_rejection_rate",
        "recovery_rate", "recovery_failure_rate", "recovery_rejection_rate",
        "recovery_correct_prediction_rate", "recovery_wrong_prediction_rate", "recovery_rejection_prediction_rate",
        "majority_voting_correct_prediction_rate", "majority_voting_wrong_prediction_rate", 
        "majority_voting_rejection_prediction_rate",
        "single_vote_correct_prediction_rate", "single_vote_wrong_prediction_rate",
        "single_vote_rejection_rate", "ideal_single_vote_rate"
    ]

    DATASETS = {
        'arancino_all_scikit': '/Users/matteopascuzzo/Desktop/Datasets/Error Detection/arancino_all_scikit.csv',
        'mafaulda': '/Users/matteopascuzzo/Desktop/Datasets/Error Detection/MAFAULDA.csv',
        'mechfailure_electriccomponent_scikit': '/Users/matteopascuzzo/Desktop/Datasets/Error Detection/MechFailure_ElectricComponent_scikit.csv',
        'scaniatrucks_aps_scikit': '/Users/matteopascuzzo/Desktop/Datasets/Error Detection/ScaniaTrucks_APS_scikit.csv',
        'backblaze_2017_5percrate_scikit': '/Users/matteopascuzzo/Desktop/Datasets/Error Detection/BackBlaze_2017_5PercRate_scikit.csv',
        'backblaze_2023': '/Users/matteopascuzzo/Desktop/Datasets/Error Detection/BackBlaze_2023.csv',
        'baidu_smartdataset_15perc_scikit': '/Users/matteopascuzzo/Desktop/Datasets/Error Detection/Baidu_SMART Dataset_15Perc_scikit.csv',
        'baiot_mirai': '/Users/matteopascuzzo/Desktop/Datasets/Error Detection/BAIoT_mirai.csv',
        'full_iot_ids_dataset_scikit': '/Users/matteopascuzzo/Desktop/Datasets/Error Detection/Full_IoT_IDS_Dataset_scikit.csv',
        'iscx_meta': '/Users/matteopascuzzo/Desktop/Datasets/Error Detection/ISCX_Meta.csv'
    }

    # Gruppi di metriche

    METRICHE_CLASSICHE = [
        'q_statistic',
        'disagreement_measure',
        'double_fault_measure',
        'entropy_measure',
        'kohavi_wolpert_variance',
        'generalized_diversity',
        'coincident_failure_diversity'
    ]

    METRICHE_DOPPIO = [
        'double_correct_prediction_rate',
        'double_wrong_prediction_rate',
        'double_rejection_rate',
        'recovery_rate',
        'recovery_failure_rate',
        'recovery_rejection_rate',
        'recovery_correct_prediction_rate',
        'recovery_wrong_prediction_rate',
        'recovery_rejection_prediction_rate'
    ]

    METRICHE_TERNA = [
        'majority_voting_correct_prediction_rate',
        'majority_voting_wrong_prediction_rate',
        'majority_voting_rejection_prediction_rate',
        'single_vote_correct_prediction_rate',
        'single_vote_wrong_prediction_rate',
        'single_vote_rejection_rate',
        'ideal_single_vote_rate'
    ]

    # METRICHE_SINGOLO = [
    #     'single_correct_prediction_rate',
    #     'single_misclassification_rate',
    #     'single_rejection_rate',
    #     'hit',
    #     'miss',
    #     'acceptance_accuracy',
    #     'performance_loss'
    # ]

    METRIC_GROUPS = {
        'classiche': METRICHE_CLASSICHE,
        'doppio': METRICHE_DOPPIO,
        'terna': METRICHE_TERNA,
        # 'singolo': METRICHE_SINGOLO
    }

    METRIC_CLASSES = {
        'q_statistic': Q_statistic,
        'disagreement_measure': disagreement_measure,
        'double_fault_measure': double_fault_measure,
        'entropy_measure': entropy_measure,
        'kohavi_wolpert_variance': kohavi_wolpert_variance,
        'generalized_diversity': generalized_diversity,
        'coincident_failure_diversity': coincident_failure_diversity,
        'single_correct_prediction_rate': single_correct_prediction_rate,
        'single_misclassification_rate': single_misclassification_rate,
        'single_rejection_rate': single_rejection_rate,
        'hit': hit,
        'miss': miss,
        'acceptance_accuracy': acceptance_accuracy,
        'performance_loss': performance_loss,
        'double_correct_prediction_rate': double_correct_prediction_rate,
        'double_wrong_prediction_rate': double_wrong_prediction_rate,
        'double_rejection_rate': double_rejection_rate,
        'recovery_rate': recovery_rate,
        'recovery_failure_rate': recovery_failure_rate,
        'recovery_rejection_rate': recovery_rejection_rate,
        'recovery_correct_prediction_rate': recovery_correct_prediction_rate,
        'recovery_wrong_prediction_rate': recovery_wrong_prediction_rate,
        'recovery_rejection_prediction_rate': recovery_rejection_prediction_rate,
        'majority_voting_correct_prediction_rate': majority_voting_correct_prediction_rate,
        'majority_voting_wrong_prediction_rate': majority_voting_wrong_prediction_rate,
        'majority_voting_rejection_prediction_rate': majority_voting_rejection_prediction_rate,
        'single_vote_correct_prediction_rate': single_vote_correct_prediction_rate,
        'single_vote_wrong_prediction_rate': single_vote_wrong_prediction_rate,
        'single_vote_rejection_rate': single_vote_rejection_rate,
        'ideal_single_vote_rate': ideal_single_vote_rate
    }
    
    def __init__(self, megafile_path: str = "/Users/matteopascuzzo/Desktop/Megafile.csv"):
        self.megafile_path = Path(megafile_path)
        self.calc = metric_calculator()
        self._load_or_create_megafile()
    
    def _load_or_create_megafile(self):
        """Carica il Megafile esistente o ne crea uno nuovo"""
        if self.megafile_path.exists():
            self.df = pd.read_csv(self.megafile_path)
        else:
            self.df = pd.DataFrame(columns=self.COLUMNS)
            self.df.to_csv(self.megafile_path, index=False)
    
    def _get_next_experiment_number(self) -> int:
        """Restituisce il prossimo numero di esperimento"""
        if len(self.df) == 0:
            return 1
        return int(self.df["experiment_number"].max()) + 1
    
    def run_experiment(self, ds: dataset, algorithm: ML_algorithm, 
                   metric_groups: list) -> dict:
        """
        Esegue un esperimento calcolando le metriche dei gruppi specificati.
        """
        X_train, X_test, y_train, y_test = ds.data
        
        # Addestra una sola volta
        algorithm.train(X_train, y_train)
        
        # Estrai predizioni
        predictions = algorithm.get_estimator_predictions(X_test)
        y_test_array = y_test.values if hasattr(y_test, 'values') else y_test
        predictions = predictions.astype(str)
        y_test_array = y_test_array.astype(str)

        # Genera nome esperimento automaticamente
        experiment_name = f"{ds.dataset_name}_{algorithm.name}"
        
        # Inizializza risultati
        results = {
            "experiment_number": self._get_next_experiment_number(),
            "experiment_name": experiment_name,
            "dataset_name": ds.dataset_name,
            "classification_strategy": algorithm.name
        }

        # Inizializza tutte le metriche a "non calcolata"
        for col in self.COLUMNS[4:]:
            results[col] = "non calcolata"
        
        # Raccogli tutte le metriche da calcolare
        metrics_to_compute = []
        for group in metric_groups:
            if group not in self.METRIC_GROUPS:
                raise ValueError(f"Gruppo '{group}' non trovato. Disponibili: {list(self.METRIC_GROUPS.keys())}")
            metrics_to_compute.extend(self.METRIC_GROUPS[group])
        
        # Calcola le metriche richieste
        for metric_name in metrics_to_compute:
            metric_class = self.METRIC_CLASSES[metric_name]
            
            if metric_name in self.METRICHE_CLASSICHE:
                results[metric_name] = self._safe_compute(metric_class(), predictions, y_test_array)
            
            elif metric_name in self.METRICHE_DOPPIO:
                pred_pair = predictions[:2, :]
                if metric_name in ['recovery_correct_prediction_rate', 'recovery_wrong_prediction_rate', 
                                'recovery_rejection_prediction_rate']:
                    models_pair = [self._get_single_model(algorithm, i) for i in range(2)]
                    results[metric_name] = self._safe_compute(metric_class(), pred_pair, y_test_array, X_test, models_pair)
                else:
                    results[metric_name] = self._safe_compute(metric_class(), pred_pair, y_test_array)
            
            elif metric_name in self.METRICHE_TERNA:
                pred_triple = predictions[:3, :]
                results[metric_name] = self._safe_compute(metric_class(), pred_triple, y_test_array)
        
        return results
    
    def _get_single_model(self, algorithm, index: int):
        """Estrae un singolo modello dall'ensemble per le metriche che richiedono predict_proba"""
        base_algo = algorithm.base_algorithm if hasattr(algorithm, 'base_algorithm') else algorithm
        
        if hasattr(base_algo.model, 'estimators_'):
            return base_algo.model.estimators_[index]
        else:
            # Per modelli boosting, restituiamo il modello completo
            return base_algo.model
    
    def _safe_compute(self, metric, predictions, y_test, X_test=None, model=None):
        """Esegue il calcolo in modo sicuro, restituendo None in caso di errore"""
        try:
            if X_test is not None and model is not None:
                return metric._compute(predictions, y_test, X_test, model)
            else:
                return metric._compute(predictions, y_test)
        except Exception as e:
            print(f"Errore calcolo {metric.name}: {e}")
            return None
    
    def save_experiment(self, results: dict):
        """Salva i risultati di un esperimento nel Megafile"""
        new_row = pd.DataFrame([results])
        self.df = pd.concat([self.df, new_row], ignore_index=True)
        self.df.to_csv(self.megafile_path, index=False)
        print(f"✓ Esperimento #{results['experiment_number']} salvato in {self.megafile_path}")
    
    def run_and_save(self, ds, algorithm: ML_algorithm, 
                 metric_groups: list, stratify: bool = True) -> dict:
        """
        Esegue un esperimento e lo salva automaticamente.
        
        Args:
            ds: Nome del dataset (stringa) oppure oggetto dataset già preparato
            algorithm: Algoritmo ML
            metric_groups: Lista di gruppi di metriche (es. ['classiche', 'doppio', 'terna'])
            stratify: Se True, usa stratify nello split (solo se ds è stringa)
        """
        # Se ds è una stringa, carica il dataset
        if isinstance(ds, str):
            if ds not in self.DATASETS:
                raise ValueError(f"Dataset '{ds}' non trovato. Disponibili: {list(self.DATASETS.keys())}")
            ds = dataset(self.DATASETS[ds], dataset_name=ds, stratify=stratify)
            ds.preprocess()
    
        results = self.run_experiment(ds, algorithm, metric_groups)
        self.save_experiment(results)
        return results
    
    def delete_experiment(self, experiment_number: int):
        """
        Cancella un esperimento e rinumera i successivi.
        
        Args:
            experiment_number: Numero dell'esperimento da cancellare
        """
        if experiment_number not in self.df["experiment_number"].values:
            raise ValueError(f"Esperimento #{experiment_number} non trovato")
        
        # Cancella la riga
        self.df = self.df[self.df["experiment_number"] != experiment_number]
        
        # Rinumera tutti gli esperimenti da 1 a N
        self.df = self.df.reset_index(drop=True)
        self.df["experiment_number"] = range(1, len(self.df) + 1)
        
        # Salva
        self.df.to_csv(self.megafile_path, index=False)
        print(f"✓ Esperimento #{experiment_number} cancellato e numerazione aggiornata")


if __name__ == "__main__":
    
    algo = rotation_forest(n_estimators=2)
    algo_with_rej = static_threshold_rejection_decorator(algo, confidence_threshold=0.9)
        
    logger = ExperimentLogger()
    results = logger.run_and_save('arancino_all_scikit', algo_with_rej, ['classiche', 'doppio'])
        
    print("\nRisultati:")
    for k, v in results.items():
        print(f"  {k}: {v}")
    

"""
logger = ExperimentLogger()
logger.delete_experiment(1)
"""