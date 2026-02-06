"""
Script per eseguire gli esperimenti sulle configurazioni selezionate.

Genera 16 CSV di output:
- 4 per coppie static (2 gruppi * 2 ensemble)
- 4 per coppie percentile (2 gruppi * 2 ensemble)
- 4 per triple static (2 gruppi * 2 ensemble)
- 4 per triple percentile (2 gruppi * 2 ensemble)

Ogni classificatore viene addestrato UNA SOLA VOLTA per dataset.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import time

warnings.filterwarnings("ignore")

# Import dal framework
from datasets.dataset import dataset

# Classificatori singoli
from classifiers.random_forest import RandomForest
from classifiers.extra_trees import ExtraTrees
from classifiers.random_patches import RandomPatches
from classifiers.xgboost_classifier import XGBoost
from classifiers.lightgbm_classifier import LightGBM
from classifiers.catboost_classifier import CatBoost
from classifiers.adaboost import AdaBoost
from classifiers.rotation_forest import RotationForest
from classifiers.random_rotation_forest import RandomRotationForest
from classifiers.gaussian_nb import GaussianNB
from classifiers.knn import KNN
from classifiers.logistic_regression import LogisticRegression

# Rejection decorators
from classifiers.static_threshold import StaticThreshold
from classifiers.percentile_threshold import PercentileThreshold

# Ensemble
from classifiers.voting_2outof2 import Voting2outof2
from classifiers.recovery_block import RecoveryBlock
from classifiers.majority_voting import MajorityVoting
from classifiers.voting_1outofN import Voting1outofN


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

# Mappa nomi classificatori -> classi
CLASSIFIER_MAP = {
    'random_forest': RandomForest,
    'extra_trees': ExtraTrees,
    'random_patches': RandomPatches,
    'xgboost': XGBoost,
    'light_gbm': LightGBM,
    'catboost': CatBoost,
    'adaboost': AdaBoost,
    'rotation_forest': RotationForest,
    'random_rotation_forest': RandomRotationForest,
    'gaussian_nb': GaussianNB,
    'knn': KNN,
    'logistic_regression': LogisticRegression
}

# Directory output
OUTPUT_DIR = Path("/Users/matteopascuzzo/Desktop/Results")


# === CONFIGURAZIONI SELEZIONATE (dalle 8 tabelle del Capitolo 4) ===

# Coppie Static - Gruppo 1: dw (basso), kw (alto), dr (basso)
COPPIE_STATIC_GRUPPO1 = [
    ('logistic_regression', 'random_forest'),
    ('extra_trees', 'knn'),
    ('knn', 'random_forest'),
    ('random_patches', 'xgboost'),
    ('random_forest', 'rotation_forest'),
    ('gaussian_nb', 'random_forest'),
    ('knn', 'light_gbm'),
    ('logistic_regression', 'xgboost'),
    ('knn', 'xgboost'),
    ('extra_trees', 'gaussian_nb'),
]

# Coppie Static - Gruppo 2: dc (alto), dw (basso)
COPPIE_STATIC_GRUPPO2 = [
    ('rotation_forest', 'xgboost'),
    ('random_forest', 'rotation_forest'),
    ('catboost', 'rotation_forest'),
    ('knn', 'xgboost'),
    ('knn', 'light_gbm'),
    ('knn', 'random_forest'),
    ('extra_trees', 'knn'),
    ('knn', 'random_rotation_forest'),
    ('random_patches', 'xgboost'),
    ('gaussian_nb', 'xgboost'),
]

# Coppie Percentile - Gruppo 1: dc (alto), dw (basso)
COPPIE_PERCENTILE_GRUPPO1 = [
    ('knn', 'random_rotation_forest'),
    ('knn', 'random_forest'),
    ('extra_trees', 'knn'),
    ('random_forest', 'rotation_forest'),
    ('knn', 'xgboost'),
    ('knn', 'light_gbm'),
    ('catboost', 'rotation_forest'),
    ('random_patches', 'xgboost'),
    ('rotation_forest', 'xgboost'),
    ('logistic_regression', 'random_forest'),
]

# Coppie Percentile - Gruppo 2: rc (alto), rw (basso), cfd (alto), gd (alto)
COPPIE_PERCENTILE_GRUPPO2 = [
    ('knn', 'xgboost'),
    ('knn', 'random_forest'),
    ('extra_trees', 'knn'),
    ('knn', 'light_gbm'),
    ('knn', 'random_rotation_forest'),
    ('random_forest', 'rotation_forest'),
    ('random_patches', 'xgboost'),
    ('rotation_forest', 'xgboost'),
    ('catboost', 'rotation_forest'),
    ('logistic_regression', 'random_forest'),
]

# Triple Static - Gruppo 1: mvc (alto), mvr (basso)
TRIPLE_STATIC_GRUPPO1 = [
    ('gaussian_nb', 'knn', 'xgboost'),
    ('gaussian_nb', 'knn', 'light_gbm'),
    ('gaussian_nb', 'knn', 'random_forest'),
    ('knn', 'random_forest', 'rotation_forest'),
    ('extra_trees', 'gaussian_nb', 'knn'),
    ('knn', 'logistic_regression', 'xgboost'),
    ('gaussian_nb', 'rotation_forest', 'xgboost'),
    ('knn', 'light_gbm', 'logistic_regression'),
    ('catboost', 'gaussian_nb', 'knn'),
    ('knn', 'logistic_regression', 'random_forest'),
]

# Triple Static - Gruppo 2: df (basso), cfd (alto), svw (basso)
TRIPLE_STATIC_GRUPPO2 = [
    ('knn', 'random_forest', 'rotation_forest'),
    ('adaboost', 'gaussian_nb', 'knn'),
    ('gaussian_nb', 'random_forest', 'rotation_forest'),
    ('adaboost', 'knn', 'logistic_regression'),
    ('gaussian_nb', 'rotation_forest', 'xgboost'),
    ('gaussian_nb', 'knn', 'random_forest'),
    ('extra_trees', 'gaussian_nb', 'knn'),
    ('gaussian_nb', 'knn', 'xgboost'),
    ('knn', 'logistic_regression', 'random_forest'),
    ('gaussian_nb', 'knn', 'light_gbm'),
]

# Triple Percentile - Gruppo 1: mvc (alto), mvw (basso)
TRIPLE_PERCENTILE_GRUPPO1 = [
    ('knn', 'logistic_regression', 'random_forest'),
    ('gaussian_nb', 'knn', 'random_forest'),
    ('knn', 'random_forest', 'rotation_forest'),
    ('extra_trees', 'gaussian_nb', 'knn'),
    ('knn', 'logistic_regression', 'rotation_forest'),
    ('gaussian_nb', 'knn', 'rotation_forest'),
    ('gaussian_nb', 'knn', 'xgboost'),
    ('knn', 'logistic_regression', 'xgboost'),
    ('gaussian_nb', 'knn', 'light_gbm'),
    ('knn', 'light_gbm', 'logistic_regression'),
]

# Triple Percentile - Gruppo 2: isvr (alto), svw (basso)
TRIPLE_PERCENTILE_GRUPPO2 = [
    ('knn', 'random_forest', 'rotation_forest'),
    ('knn', 'logistic_regression', 'xgboost'),
    ('knn', 'light_gbm', 'logistic_regression'),
    ('adaboost', 'gaussian_nb', 'knn'),
    ('adaboost', 'knn', 'logistic_regression'),
    ('gaussian_nb', 'knn', 'xgboost'),
    ('knn', 'logistic_regression', 'rotation_forest'),
    ('extra_trees', 'gaussian_nb', 'logistic_regression'),
    ('gaussian_nb', 'knn', 'light_gbm'),
    ('catboost', 'gaussian_nb', 'knn'),
]


# === FUNZIONI METRICHE ===

def calculate_fundamental_metrics(predictions: np.ndarray, y_test: np.ndarray) -> dict:
    """Calcola correct_rate, misclassification_rate, rejection_rate"""
    predictions = np.array(predictions).astype(str)
    y_test = np.array(y_test).astype(str)
    n = len(y_test)
    
    correct = (predictions == y_test)
    rejected = (predictions == "reject")
    wrong = ~correct & ~rejected
    
    return {
        'correct_rate': np.sum(correct) / n,
        'misclassification_rate': np.sum(wrong) / n,
        'rejection_rate': np.sum(rejected) / n
    }


# === FASE 1: ADDESTRAMENTO CLASSIFICATORI ===

def get_all_classifier_names():
    """Estrae tutti i nomi dei classificatori usati nelle configurazioni"""
    classifier_names = set()
    
    # Da tutte le coppie
    for coppia in (COPPIE_STATIC_GRUPPO1 + COPPIE_STATIC_GRUPPO2 + 
                   COPPIE_PERCENTILE_GRUPPO1 + COPPIE_PERCENTILE_GRUPPO2):
        classifier_names.update(coppia)
    
    # Da tutte le triple
    for tripla in (TRIPLE_STATIC_GRUPPO1 + TRIPLE_STATIC_GRUPPO2 + 
                   TRIPLE_PERCENTILE_GRUPPO1 + TRIPLE_PERCENTILE_GRUPPO2):
        classifier_names.update(tripla)
    
    return classifier_names


def train_all_classifiers():
    """
    Addestra tutti i classificatori per tutti i dataset UNA SOLA VOLTA.
    
    Returns:
        trained_models: dict {dataset_name: {classifier_name: trained_model}}
        dataset_data: dict {dataset_name: (X_train, X_test, y_train, y_test)}
    """
    print("="*70)
    print("FASE 1: ADDESTRAMENTO CLASSIFICATORI BASE")
    print("="*70)
    
    classifier_names = get_all_classifier_names()
    
    print(f"\nClassificatori da addestrare: {sorted(classifier_names)}")
    print(f"Dataset: {len(DATASETS)}")
    print(f"Totale addestramenti: {len(classifier_names)} × {len(DATASETS)} = {len(classifier_names) * len(DATASETS)}")
    
    trained_models = {}
    dataset_data = {}
    
    for ds_name, ds_path in DATASETS.items():
        print(f"\n{'='*50}")
        print(f"DATASET: {ds_name}")
        print(f"{'='*50}")
        
        # Carica dataset
        try:
            ds = dataset(ds_path, dataset_name=ds_name)
            ds.preprocess()
            X_train, X_test, y_train, y_test = ds.data
            dataset_data[ds_name] = (X_train, X_test, y_train, y_test)
        except Exception as e:
            print(f"  ✗ Errore caricamento dataset: {e}")
            continue
        
        trained_models[ds_name] = {}
        
        for clf_name in sorted(classifier_names):
            if clf_name not in CLASSIFIER_MAP:
                print(f"  ⚠ Classificatore '{clf_name}' non trovato, skip")
                continue
            
            try:
                clf = CLASSIFIER_MAP[clf_name]()
                clf.train(X_train, y_train)
                trained_models[ds_name][clf_name] = clf
                print(f"  ✓ {clf_name}")
            except Exception as e:
                print(f"  ✗ {clf_name}: {e}")
    
    total_trained = sum(len(models) for models in trained_models.values())
    print(f"\n{'='*70}")
    print(f"ADDESTRAMENTO COMPLETATO: {total_trained} modelli")
    print(f"{'='*70}")
    
    return trained_models, dataset_data


# === FASE 2: PROCESSAMENTO CONFIGURAZIONI ===

def wrap_with_rejection(trained_clf, rejection_type: str):
    """Wrappa un classificatore GIÀ ADDESTRATO con un rejection decorator."""
    if rejection_type == 'static':
        return StaticThreshold(trained_clf, confidence_threshold=0.9)
    elif rejection_type == 'percentile':
        return PercentileThreshold(trained_clf, rejection_percentile=10.0)
    else:
        raise ValueError(f"Rejection type sconosciuto: {rejection_type}")


def process_coppie(trained_models: dict, dataset_data: dict, output_dir: Path,
                   coppie: list, rejection_type: str, gruppo_name: str):
    """
    Processa le coppie con Voting2su2 e RecoveryBlock.
    
    Args:
        coppie: lista di tuple (clf1_name, clf2_name)
        rejection_type: 'static' o 'percentile'
        gruppo_name: es. 'CoppieStaticGruppo1'
    """
    print(f"\n{'='*70}")
    print(f"PROCESSING {gruppo_name} - {rejection_type}")
    print(f"{'='*70}")
    
    results_voting = []
    results_recovery = []
    
    for clf1_name, clf2_name in coppie:
        experiment_name = f"{clf1_name}+{clf2_name}"
        
        for ds_name in dataset_data.keys():
            if ds_name not in trained_models:
                continue
            
            if clf1_name not in trained_models[ds_name] or clf2_name not in trained_models[ds_name]:
                print(f"  ⚠ Skip {experiment_name} su {ds_name} - classificatore mancante")
                continue
            
            X_train, X_test, y_train, y_test = dataset_data[ds_name]
            
            # Recupera modelli già addestrati e wrappa con rejection
            clf1_base = trained_models[ds_name][clf1_name]
            clf2_base = trained_models[ds_name][clf2_name]
            
            clf1 = wrap_with_rejection(clf1_base, rejection_type)
            clf2 = wrap_with_rejection(clf2_base, rejection_type)
            
            # Predizioni base
            pred_clf1 = clf1.predict(X_test)
            pred_clf2 = clf2.predict(X_test)
            
            metrics_clf1 = calculate_fundamental_metrics(pred_clf1, y_test)
            metrics_clf2 = calculate_fundamental_metrics(pred_clf2, y_test)
            
            # === VOTING 2 SU 2 ===
            ensemble_voting = Voting2outof2([clf1, clf2])
            pred_voting = ensemble_voting.predict(X_test)
            metrics_voting = calculate_fundamental_metrics(pred_voting, y_test)
            
            results_voting.append({
                'experiment_name': experiment_name,
                'dataset_name': ds_name,
                'classifier_1': clf1_name,
                'classifier_2': clf2_name,
                'ensemble_correct_rate': metrics_voting['correct_rate'],
                'ensemble_misclassification_rate': metrics_voting['misclassification_rate'],
                'ensemble_rejection_rate': metrics_voting['rejection_rate'],
                'classifier_1_correct_rate': metrics_clf1['correct_rate'],
                'classifier_1_misclassification_rate': metrics_clf1['misclassification_rate'],
                'classifier_1_rejection_rate': metrics_clf1['rejection_rate'],
                'classifier_2_correct_rate': metrics_clf2['correct_rate'],
                'classifier_2_misclassification_rate': metrics_clf2['misclassification_rate'],
                'classifier_2_rejection_rate': metrics_clf2['rejection_rate'],
            })
            
            # === RECOVERY BLOCK ===
            # Testiamo entrambi gli ordini e prendiamo clf1 -> clf2
            ensemble_recovery = RecoveryBlock([clf1, clf2])
            pred_recovery = ensemble_recovery.predict(X_test)
            metrics_recovery = calculate_fundamental_metrics(pred_recovery, y_test)
            
            results_recovery.append({
                'experiment_name': experiment_name,
                'dataset_name': ds_name,
                'classifier_1': clf1_name,
                'classifier_2': clf2_name,
                'ensemble_correct_rate': metrics_recovery['correct_rate'],
                'ensemble_misclassification_rate': metrics_recovery['misclassification_rate'],
                'ensemble_rejection_rate': metrics_recovery['rejection_rate'],
                'classifier_1_correct_rate': metrics_clf1['correct_rate'],
                'classifier_1_misclassification_rate': metrics_clf1['misclassification_rate'],
                'classifier_1_rejection_rate': metrics_clf1['rejection_rate'],
                'classifier_2_correct_rate': metrics_clf2['correct_rate'],
                'classifier_2_misclassification_rate': metrics_clf2['misclassification_rate'],
                'classifier_2_rejection_rate': metrics_clf2['rejection_rate'],
            })
    
    # Salva CSV
    if results_voting:
        df_voting = pd.DataFrame(results_voting)
        filename_voting = f"{gruppo_name}_Voting2su2.csv"
        df_voting.to_csv(output_dir / filename_voting, index=False)
        print(f"✓ {filename_voting} ({len(results_voting)} righe)")
    
    if results_recovery:
        df_recovery = pd.DataFrame(results_recovery)
        filename_recovery = f"{gruppo_name}_RecoveryBlock.csv"
        df_recovery.to_csv(output_dir / filename_recovery, index=False)
        print(f"✓ {filename_recovery} ({len(results_recovery)} righe)")


def process_triple(trained_models: dict, dataset_data: dict, output_dir: Path,
                   triple: list, rejection_type: str, gruppo_name: str):
    """
    Processa le triple con MajorityVoting e Voting1suN.
    
    Args:
        triple: lista di tuple (clf1_name, clf2_name, clf3_name)
        rejection_type: 'static' o 'percentile'
        gruppo_name: es. 'TripleStaticGruppo1'
    """
    print(f"\n{'='*70}")
    print(f"PROCESSING {gruppo_name} - {rejection_type}")
    print(f"{'='*70}")
    
    results_majority = []
    results_1ofn = []
    
    for clf1_name, clf2_name, clf3_name in triple:
        experiment_name = f"{clf1_name}+{clf2_name}+{clf3_name}"
        
        for ds_name in dataset_data.keys():
            if ds_name not in trained_models:
                continue
            
            if (clf1_name not in trained_models[ds_name] or 
                clf2_name not in trained_models[ds_name] or
                clf3_name not in trained_models[ds_name]):
                print(f"  ⚠ Skip {experiment_name} su {ds_name} - classificatore mancante")
                continue
            
            X_train, X_test, y_train, y_test = dataset_data[ds_name]
            
            # Recupera modelli già addestrati e wrappa con rejection
            clf1_base = trained_models[ds_name][clf1_name]
            clf2_base = trained_models[ds_name][clf2_name]
            clf3_base = trained_models[ds_name][clf3_name]
            
            clf1 = wrap_with_rejection(clf1_base, rejection_type)
            clf2 = wrap_with_rejection(clf2_base, rejection_type)
            clf3 = wrap_with_rejection(clf3_base, rejection_type)
            
            # Predizioni base
            pred_clf1 = clf1.predict(X_test)
            pred_clf2 = clf2.predict(X_test)
            pred_clf3 = clf3.predict(X_test)
            
            metrics_clf1 = calculate_fundamental_metrics(pred_clf1, y_test)
            metrics_clf2 = calculate_fundamental_metrics(pred_clf2, y_test)
            metrics_clf3 = calculate_fundamental_metrics(pred_clf3, y_test)
            
            # === MAJORITY VOTING ===
            ensemble_majority = MajorityVoting([clf1, clf2, clf3])
            pred_majority = ensemble_majority.predict(X_test)
            metrics_majority = calculate_fundamental_metrics(pred_majority, y_test)
            
            results_majority.append({
                'experiment_name': experiment_name,
                'dataset_name': ds_name,
                'classifier_1': clf1_name,
                'classifier_2': clf2_name,
                'classifier_3': clf3_name,
                'ensemble_correct_rate': metrics_majority['correct_rate'],
                'ensemble_misclassification_rate': metrics_majority['misclassification_rate'],
                'ensemble_rejection_rate': metrics_majority['rejection_rate'],
                'classifier_1_correct_rate': metrics_clf1['correct_rate'],
                'classifier_1_misclassification_rate': metrics_clf1['misclassification_rate'],
                'classifier_1_rejection_rate': metrics_clf1['rejection_rate'],
                'classifier_2_correct_rate': metrics_clf2['correct_rate'],
                'classifier_2_misclassification_rate': metrics_clf2['misclassification_rate'],
                'classifier_2_rejection_rate': metrics_clf2['rejection_rate'],
                'classifier_3_correct_rate': metrics_clf3['correct_rate'],
                'classifier_3_misclassification_rate': metrics_clf3['misclassification_rate'],
                'classifier_3_rejection_rate': metrics_clf3['rejection_rate'],
            })
            
            # === VOTING 1 SU N ===
            ensemble_1ofn = Voting1outofN([clf1, clf2, clf3])
            pred_1ofn = ensemble_1ofn.predict(X_test)
            metrics_1ofn = calculate_fundamental_metrics(pred_1ofn, y_test)
            
            results_1ofn.append({
                'experiment_name': experiment_name,
                'dataset_name': ds_name,
                'classifier_1': clf1_name,
                'classifier_2': clf2_name,
                'classifier_3': clf3_name,
                'ensemble_correct_rate': metrics_1ofn['correct_rate'],
                'ensemble_misclassification_rate': metrics_1ofn['misclassification_rate'],
                'ensemble_rejection_rate': metrics_1ofn['rejection_rate'],
                'classifier_1_correct_rate': metrics_clf1['correct_rate'],
                'classifier_1_misclassification_rate': metrics_clf1['misclassification_rate'],
                'classifier_1_rejection_rate': metrics_clf1['rejection_rate'],
                'classifier_2_correct_rate': metrics_clf2['correct_rate'],
                'classifier_2_misclassification_rate': metrics_clf2['misclassification_rate'],
                'classifier_2_rejection_rate': metrics_clf2['rejection_rate'],
                'classifier_3_correct_rate': metrics_clf3['correct_rate'],
                'classifier_3_misclassification_rate': metrics_clf3['misclassification_rate'],
                'classifier_3_rejection_rate': metrics_clf3['rejection_rate'],
            })
    
    # Salva CSV
    if results_majority:
        df_majority = pd.DataFrame(results_majority)
        filename_majority = f"{gruppo_name}_MajorityVoting.csv"
        df_majority.to_csv(output_dir / filename_majority, index=False)
        print(f"✓ {filename_majority} ({len(results_majority)} righe)")
    
    if results_1ofn:
        df_1ofn = pd.DataFrame(results_1ofn)
        filename_1ofn = f"{gruppo_name}_Voting1suN.csv"
        df_1ofn.to_csv(output_dir / filename_1ofn, index=False)
        print(f"✓ {filename_1ofn} ({len(results_1ofn)} righe)")


# === MAIN ===

def main():
    start_time = time.time()
    
    print("="*70)
    print("ESPERIMENTI SULLE CONFIGURAZIONI SELEZIONATE")
    print("="*70)
    print(f"Output: 16 file CSV")
    print(f"Directory: {OUTPUT_DIR}")
    
    # Crea directory output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # FASE 1: Addestra tutti i classificatori una sola volta
    trained_models, dataset_data = train_all_classifiers()
    
    # FASE 2: Processa le configurazioni
    
    # Coppie Static
    process_coppie(trained_models, dataset_data, OUTPUT_DIR,
                   COPPIE_STATIC_GRUPPO1, 'static', 'CoppieStaticGruppo1')
    process_coppie(trained_models, dataset_data, OUTPUT_DIR,
                   COPPIE_STATIC_GRUPPO2, 'static', 'CoppieStaticGruppo2')
    
    # Coppie Percentile
    process_coppie(trained_models, dataset_data, OUTPUT_DIR,
                   COPPIE_PERCENTILE_GRUPPO1, 'percentile', 'CoppiePercentileGruppo1')
    process_coppie(trained_models, dataset_data, OUTPUT_DIR,
                   COPPIE_PERCENTILE_GRUPPO2, 'percentile', 'CoppiePercentileGruppo2')
    
    # Triple Static
    process_triple(trained_models, dataset_data, OUTPUT_DIR,
                   TRIPLE_STATIC_GRUPPO1, 'static', 'TripleStaticGruppo1')
    process_triple(trained_models, dataset_data, OUTPUT_DIR,
                   TRIPLE_STATIC_GRUPPO2, 'static', 'TripleStaticGruppo2')
    
    # Triple Percentile
    process_triple(trained_models, dataset_data, OUTPUT_DIR,
                   TRIPLE_PERCENTILE_GRUPPO1, 'percentile', 'TriplePercentileGruppo1')
    process_triple(trained_models, dataset_data, OUTPUT_DIR,
                   TRIPLE_PERCENTILE_GRUPPO2, 'percentile', 'TriplePercentileGruppo2')
    
    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"COMPLETATO in {elapsed:.1f} secondi")
    print(f"{'='*70}")
    print(f"\nFile generati in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()