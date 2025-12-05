import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import time

warnings.filterwarnings("ignore")

# Dataset
from datasets.dataset import dataset

# Metric calculator
from metric_calculator import metric_calculator

# Ensemble
from ensembles.voting_1ofn_ensemble import Voting1ofNEnsemble

# Algoritmi
from ML_algorithms.random_forest import random_forest
from ML_algorithms.extra_trees import extra_trees
from ML_algorithms.xgboost import xgboost
from ML_algorithms.light_gbm import light_gbm
from ML_algorithms.catboost import catboost
from ML_algorithms.adaboost import adaboost
from ML_algorithms.rotation_forest import rotation_forest
from ML_algorithms.gaussian_nb import gaussian_nb
from ML_algorithms.k_nearest_neighbors import knn
from ML_algorithms.logistic_regression import logistic_regression

# Rejection techniques
from rejection_techniques.static_threshold_rejection_decorator import static_threshold_rejection_decorator
from rejection_techniques.percentile_threshold_rejection_decorator import percentile_threshold_rejection_decorator


# === CONFIGURAZIONE ===

DATASETS = {
    'arancino_all_scikit': '/Users/matteopascuzzo/Desktop/Datasets/Error Detection/arancino_all_scikit.csv',
    'mafaulda': '/Users/matteopascuzzo/Desktop/Datasets/Error Detection/MAFAULDA.csv',
    'mechfailure_electriccomponent_scikit': '/Users/matteopascuzzo/Desktop/Datasets/Error Detection/MechFailure_ElectricComponent_scikit.csv',
    'scaniatrucks_aps_scikit': '/Users/matteopascuzzo/Desktop/Datasets/Error Detection/ScaniaTrucks_APS_scikit.csv',
    'backblaze_2017_5percrate_scikit': '/Users/matteopascuzzo/Desktop/Datasets/HW_Failure/BackBlaze_2017_5PercRate_scikit.csv',
    'backblaze_2023': '/Users/matteopascuzzo/Desktop/Datasets/HW_Failure/BackBlaze_2023.csv',
    'baidu_smartdataset_15perc_scikit': '/Users/matteopascuzzo/Desktop/Datasets/HW_Failure/Baidu_SMART Dataset_15Perc_scikit.csv',
    'baiot_mirai': '/Users/matteopascuzzo/Desktop/Datasets/NIDS/BAIoT_mirai.csv',
    'full_iot_ids_dataset_scikit': '/Users/matteopascuzzo/Desktop/Datasets/NIDS/Full_IoT_IDS_Dataset_scikit.csv',
    'iscx_meta': '/Users/matteopascuzzo/Desktop/Datasets/NIDS/ISCX_Meta.csv'
}

OUTPUT_FILE = "/Users/matteopascuzzo/Desktop/Voting1ofN.csv"

# Mapping nome -> classe algoritmo
ALGORITHMS = {
    "random_forest": random_forest,
    "extra_trees": extra_trees,
    "xgboost": xgboost,
    "light_gbm": light_gbm,
    "catboost": catboost,
    "adaboost": adaboost,
    "rotation_forest": rotation_forest,
    "gaussian_nb": gaussian_nb,
    "knn": knn,
    "logistic_regression": logistic_regression
}

# 20 terne
TRIPLES = [
    ("random_forest", "gaussian_nb", "knn"),
    ("random_forest", "gaussian_nb", "logistic_regression"),
    ("random_forest", "knn", "logistic_regression"),
    ("random_forest", "gaussian_nb", "rotation_forest"),
    ("random_forest", "knn", "rotation_forest"),
    ("extra_trees", "gaussian_nb", "knn"),
    ("extra_trees", "gaussian_nb", "logistic_regression"),
    ("xgboost", "gaussian_nb", "knn"),
    ("xgboost", "gaussian_nb", "logistic_regression"),
    ("xgboost", "knn", "logistic_regression"),
    ("xgboost", "gaussian_nb", "rotation_forest"),
    ("light_gbm", "gaussian_nb", "knn"),
    ("light_gbm", "knn", "logistic_regression"),
    ("catboost", "gaussian_nb", "knn"),
    ("catboost", "gaussian_nb", "logistic_regression"),
    ("adaboost", "gaussian_nb", "knn"),
    ("adaboost", "knn", "logistic_regression"),
    ("rotation_forest", "gaussian_nb", "knn"),
    ("rotation_forest", "gaussian_nb", "logistic_regression"),
    ("rotation_forest", "knn", "logistic_regression")
]

REJECTION_TECHNIQUES = {
    "static_threshold_0.9": lambda algo: static_threshold_rejection_decorator(algo, confidence_threshold=0.9),
    "percentile_threshold_10": lambda algo: percentile_threshold_rejection_decorator(algo, rejection_percentile=10.0)
}

# Colonne output
COLUMNS = [
    "experiment_number", "experiment_name", "dataset_name", "ensemble_type", "rejection_strategy",
    "classifier_1", "classifier_2", "classifier_3",
    "q_statistic", "disagreement_measure", "double_fault_measure", "entropy_measure",
    "kohavi_wolpert_variance", "generalized_diversity", "coincident_failure_diversity",
    "single_vote_correct_prediction_rate", "single_vote_wrong_prediction_rate", 
    "single_vote_rejection_rate", "ideal_single_vote_rate"
]


# === FUNZIONI ===

def load_or_create_file() -> pd.DataFrame:
    path = Path(OUTPUT_FILE)
    if path.exists():
        return pd.read_csv(path)
    else:
        df = pd.DataFrame(columns=COLUMNS)
        df.to_csv(path, index=False)
        return df


def get_next_experiment_number(df: pd.DataFrame) -> int:
    if len(df) == 0:
        return 1
    return int(df["experiment_number"].max()) + 1


def get_algorithms_needed() -> set:
    """Restituisce il set di algoritmi necessari per le terne"""
    needed = set()
    for algo1, algo2, algo3 in TRIPLES:
        needed.add(algo1)
        needed.add(algo2)
        needed.add(algo3)
    return needed


def train_base_classifiers(X_train, y_train) -> dict:
    """Addestra tutti i classificatori base necessari una sola volta"""
    needed = get_algorithms_needed()
    trained = {}
    
    for algo_name in needed:
        print(f"  Training {algo_name}...")
        algo = ALGORITHMS[algo_name]()
        algo.train(X_train, y_train)
        trained[algo_name] = algo
    
    return trained


def run_experiment(ds, trained_classifiers: dict, triple: tuple, rejection_name: str, 
                   rejection_func, calc: metric_calculator) -> dict:
    """Esegue un singolo esperimento"""
    algo1_name, algo2_name, algo3_name = triple
    
    # Applica rejection decorator ai classificatori già addestrati
    clf1 = rejection_func(trained_classifiers[algo1_name])
    clf2 = rejection_func(trained_classifiers[algo2_name])
    clf3 = rejection_func(trained_classifiers[algo3_name])
    
    # Crea ensemble
    ensemble = Voting1ofNEnsemble([clf1, clf2, clf3])
    
    # Calcola metriche (skip_training=True perché già addestrati)
    metric_results = calc.calculate(ds, ensemble, skip_training=True)
    
    # Costruisci risultato
    experiment_name = f"{algo1_name}+{algo2_name}+{algo3_name}"
    
    results = {
        "experiment_name": experiment_name,
        "dataset_name": ds.dataset_name,
        "ensemble_type": "Voting1ofN",
        "rejection_strategy": rejection_name,
        "classifier_1": algo1_name,
        "classifier_2": algo2_name,
        "classifier_3": algo3_name
    }
    
    # Aggiungi metriche
    for col in COLUMNS[8:]:
        results[col] = metric_results.get(col, None)
    
    return results


def main():
    print("=" * 70)
    print("VOTING 1 OF N - EXPERIMENT RUNNER")
    print("=" * 70)
    print(f"Dataset: {len(DATASETS)}")
    print(f"Terne: {len(TRIPLES)}")
    print(f"Rejection techniques: {len(REJECTION_TECHNIQUES)}")
    print(f"Totale esperimenti: {len(DATASETS) * len(TRIPLES) * len(REJECTION_TECHNIQUES)}")
    print("=" * 70)
    
    calc = metric_calculator()
    df = load_or_create_file()
    
    total_start = time.time()
    experiment_count = 0
    
    for ds_name, ds_path in DATASETS.items():
        print(f"\n{'='*70}")
        print(f"DATASET: {ds_name}")
        print(f"{'='*70}")
        
        # Carica e prepara dataset
        ds = dataset(ds_path, dataset_name=ds_name, stratify=True)
        ds.preprocess()
        X_train, X_test, y_train, y_test = ds.data
        
        # Addestra tutti i classificatori base una sola volta
        print("\n--- Training classificatori base ---")
        trained_classifiers = train_base_classifiers(X_train, y_train)
        print("--- Training completato ---\n")
        
        # Itera su terne e rejection techniques
        for triple in TRIPLES:
            for rejection_name, rejection_func in REJECTION_TECHNIQUES.items():
                experiment_count += 1
                triple_name = f"{triple[0]}+{triple[1]}+{triple[2]}"
                print(f"[{experiment_count}] {triple_name} | {rejection_name}")
                
                try:
                    results = run_experiment(ds, trained_classifiers, triple, 
                                           rejection_name, rejection_func, calc)
                    
                    # Aggiungi numero esperimento
                    results["experiment_number"] = get_next_experiment_number(df)
                    
                    # Salva
                    new_row = pd.DataFrame([results])
                    df = pd.concat([df, new_row], ignore_index=True)
                    df.to_csv(OUTPUT_FILE, index=False)
                    
                    print(f"    ✓ Salvato (exp #{results['experiment_number']})")
                    
                except Exception as e:
                    print(f"    ✗ Errore: {e}")
    
    total_time = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"COMPLETATO")
    print(f"Esperimenti eseguiti: {experiment_count}")
    print(f"Tempo totale: {total_time/60:.1f} minuti")
    print(f"Output: {OUTPUT_FILE}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()