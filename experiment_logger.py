# experiment_logger.py

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, List, Dict

from datasets.dataset import dataset
from metric_calculator import metric_calculator

# Import Ensemble
from ensembles.ensemble import Ensemble
from ensembles.voting_2of2_ensemble import Voting2of2Ensemble
from ensembles.recovery_block_ensemble import RecoveryBlockEnsemble
from ensembles.majority_voting_ensemble import MajorityVotingEnsemble
from ensembles.voting_1ofn_ensemble import Voting1ofNEnsemble

# Import rejection decorator
from rejection_techniques.abstract_rejection_decorator import abstract_rejection_decorator


class ExperimentLogger:
    """Gestisce l'esecuzione di esperimenti e il salvataggio nel Megafile.csv"""
    
    COLUMNS = [
        "experiment_number","experiment_name", "dataset_name", "ensemble_type", "rejection_strategy",
        # Metriche classiche
        "q_statistic", "disagreement_measure", "double_fault_measure", "entropy_measure",
        "kohavi_wolpert_variance", "generalized_diversity", "coincident_failure_diversity",
        # Metriche singolo classificatore
        "single_correct_prediction_rate", "single_misclassification_rate", "single_rejection_rate",
        "hit", "miss", "acceptance_accuracy", "performance_loss",
        # Metriche voting 2/2
        "double_correct_prediction_rate", "double_wrong_prediction_rate", "double_rejection_rate",
        # Metriche recovery block
        "recovery_rate", "recovery_failure_rate", "recovery_rejection_rate",
        "recovery_correct_prediction_rate", "recovery_wrong_prediction_rate", 
        "recovery_rejection_prediction_rate",
        # Metriche majority voting
        "majority_voting_correct_prediction_rate", "majority_voting_wrong_prediction_rate",
        "majority_voting_rejection_prediction_rate",
        # Metriche voting 1/n
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
    
    def _get_ensemble_type(self, model) -> str:
        """Determina il tipo di ensemble/classificatore"""
        if isinstance(model, Voting2of2Ensemble):
            return "Voting2of2"
        elif isinstance(model, RecoveryBlockEnsemble):
            return "RecoveryBlock"
        elif isinstance(model, MajorityVotingEnsemble):
            return "MajorityVoting"
        elif isinstance(model, Voting1ofNEnsemble):
            return "Voting1ofN"
        elif isinstance(model, abstract_rejection_decorator):
            return "Single"
        else:
            return "Unknown"
        
    def _get_next_experiment_number(self) -> int:
        """Restituisce il prossimo numero di esperimento"""
        if len(self.df) == 0:
            return 1
        return int(self.df["experiment_number"].max()) + 1
    
    def _get_experiment_name(self, model) -> str:
        """Genera nome esperimento dai nomi dei classificatori"""
        if isinstance(model, Ensemble):
            base_names = []
            for clf in model.classifiers:
                if isinstance(clf, abstract_rejection_decorator):
                    base_names.append(clf.base_algorithm.name)
                else:
                    base_names.append(clf.name)
            return "+".join(base_names)
        elif isinstance(model, abstract_rejection_decorator):
            return model.base_algorithm.name
        else:
            return model.name
    
    def _get_rejection_strategy(self, model) -> str:
        """Estrae la strategia di rejection dal modello"""
        if isinstance(model, Ensemble):
            # Prendi dal primo classificatore (assumendo stessa strategia)
            clf = model.classifiers[0]
            if isinstance(clf, abstract_rejection_decorator):
                # Estrai tipo e parametro dal nome
                name = clf.name
                base_name = clf.base_algorithm.name
                # Rimuovi il nome base per ottenere la strategia
                strategy = name.replace(base_name + "_", "")
                return strategy
            return "none"
        elif isinstance(model, abstract_rejection_decorator):
            name = model.name
            base_name = model.base_algorithm.name
            strategy = name.replace(base_name + "_", "")
            return strategy
        return "none"
    
    def _get_applicable_metrics(self, model) -> set:
        """Restituisce i nomi delle metriche applicabili per questo modello"""
        return set(self.calc.get_metrics_for_model(model).keys())
    
    def run_experiment(self, ds: dataset, 
                       model: Union[Ensemble, abstract_rejection_decorator]) -> Dict:
        """
        Esegue un esperimento calcolando automaticamente le metriche appropriate.
        
        Args:
            ds: Dataset preparato
            model: Ensemble o singolo classificatore con rejection
        
        Returns:
            Dizionario con risultati
        """
        # Calcola metriche
        metric_results = self.calc.calculate(ds, model)
        
        # Ottieni metriche applicabili
        applicable_metrics = self._get_applicable_metrics(model)
        
        # Costruisci risultato
        results = {
            "experiment_number": self._get_next_experiment_number(),
            "experiment_name": self._get_experiment_name(model),
            "dataset_name": ds.dataset_name,
            "ensemble_type": self._get_ensemble_type(model),
            "rejection_strategy": self._get_rejection_strategy(model)
        }
        
        # Popola metriche
        for col in self.COLUMNS[5:]:
            if col in metric_results:
                results[col] = metric_results[col]
            elif col in applicable_metrics:
                # Metrica applicabile ma calcolo fallito
                results[col] = None
            else:
                # Metrica non prevista per questo tipo di ensemble
                results[col] = "non prevista"
        
        return results
    
    def save_experiment(self, results: Dict):
        """Salva i risultati di un esperimento nel Megafile"""
        new_row = pd.DataFrame([results])
        self.df = pd.concat([self.df, new_row], ignore_index=True)
        self.df.to_csv(self.megafile_path, index=False)
        print(f"✓ Esperimento '{results['experiment_name']}' salvato in {self.megafile_path}")
    
    def run_and_save(self, ds_name: str, 
                     model: Union[Ensemble, abstract_rejection_decorator],
                     stratify: bool = True) -> Dict:
        """
        Esegue un esperimento e lo salva automaticamente.
        
        Args:
            ds_name: Nome del dataset (chiave in DATASETS)
            model: Ensemble o singolo classificatore con rejection
            stratify: Se True, usa stratify nello split
        
        Returns:
            Dizionario con risultati
        """
        if ds_name not in self.DATASETS:
            raise ValueError(f"Dataset '{ds_name}' non trovato. Disponibili: {list(self.DATASETS.keys())}")
        
        ds = dataset(self.DATASETS[ds_name], dataset_name=ds_name, stratify=stratify)
        ds.preprocess()
        
        results = self.run_experiment(ds, model)
        self.save_experiment(results)
        return results
    
    def delete_experiment(self, experiment_number: int):
        """Cancella un esperimento e rinumera i successivi."""
        if experiment_number not in self.df["experiment_number"].values:
            raise ValueError(f"Esperimento #{experiment_number} non trovato")
        
        self.df = self.df[self.df["experiment_number"] != experiment_number]
        self.df = self.df.reset_index(drop=True)
        self.df["experiment_number"] = range(1, len(self.df) + 1)
        self.df.to_csv(self.megafile_path, index=False)
        print(f"✓ Esperimento #{experiment_number} cancellato e numerazione aggiornata")


if __name__ == "__main__":
    from ML_algorithms.random_forest import random_forest
    from ML_algorithms.xgboost import xgboost
    from ML_algorithms.rotation_forest import rotation_forest
    from rejection_techniques.static_threshold_rejection_decorator import static_threshold_rejection_decorator
    
    
    D1 = random_forest()
    D2 = rotation_forest()
    
    rf_rej = static_threshold_rejection_decorator(D1, confidence_threshold=0.9)
    rotf_rej = static_threshold_rejection_decorator(D2, confidence_threshold=0.9)
    
    ensemble = Voting2of2Ensemble([rf_rej, rotf_rej])
    
    logger = ExperimentLogger()
    results = logger.run_and_save('arancino_all_scikit', ensemble)
    
    print("\nRisultati:")
    for k, v in results.items():
        if v is not None and v != "non prevista":
            print(f"  {k}: {v}")

    """
    logger = ExperimentLogger()
    logger.delete_experiment(1)
    """