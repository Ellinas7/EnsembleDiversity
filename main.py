"""
Script OTTIMIZZATO per generare gli 8 CSV con le nuove metriche fondamentali.

Ottimizzazione: ogni classificatore base viene addestrato UNA SOLA VOLTA per dataset.
I RejectionDecorator usano solo predict_proba() del modello già addestrato.

Output: 8 CSV (4 tipi ensemble × 2 rejection strategies)
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
    'lightgbm': LightGBM,
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

# Path CSV input
INPUT_DIR = Path("/Users/matteopascuzzo/Desktop")


# === FUNZIONI METRICHE ===

def calculate_fundamental_metrics(predictions: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Calcola le 3 metriche fondamentali: correct_rate, misclassification_rate, rejection_rate
    """
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


def calculate_double_metrics(pred_clf1: np.ndarray, pred_clf2: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Calcola le metriche double_* per una coppia di classificatori.
    """
    pred_clf1 = np.array(pred_clf1).astype(str)
    pred_clf2 = np.array(pred_clf2).astype(str)
    y_test = np.array(y_test).astype(str)
    n = len(y_test)
    
    is_correct_1 = (pred_clf1 == y_test)
    is_correct_2 = (pred_clf2 == y_test)
    is_rejected_1 = (pred_clf1 == "reject")
    is_rejected_2 = (pred_clf2 == "reject")
    is_wrong_1 = ~is_correct_1 & ~is_rejected_1
    is_wrong_2 = ~is_correct_2 & ~is_rejected_2
    
    N11 = np.sum(is_correct_1 & is_correct_2)
    N00 = np.sum(is_wrong_1 & is_wrong_2)
    Nqq = np.sum(is_rejected_1 & is_rejected_2)
    
    return {
        'double_correct_prediction_rate': N11 / n,
        'double_wrong_prediction_rate': N00 / n,
        'double_rejection_rate': Nqq / n
    }


# === FASE 1: ADDESTRAMENTO CLASSIFICATORI ===

def get_all_classifier_names_from_csvs():
    """Estrae tutti i nomi dei classificatori usati nei CSV"""
    classifier_names = set()
    
    # Da Voting2outof2
    df = pd.read_csv(INPUT_DIR / "Voting2outof2.csv")
    for _, row in df.iterrows():
        names = row['experiment_name'].split('+')
        classifier_names.update(names)
    
    # Da RecoveryBlock
    df = pd.read_csv(INPUT_DIR / "RecoveryBlock.csv")
    classifier_names.update(df['primary_classifier'].unique())
    classifier_names.update(df['secondary_classifier'].unique())
    
    # Da MajorityVoting
    df = pd.read_csv(INPUT_DIR / "MajorityVoting.csv")
    classifier_names.update(df['classifier_1'].unique())
    classifier_names.update(df['classifier_2'].unique())
    classifier_names.update(df['classifier_3'].unique())
    
    # Da Voting1outofN
    df = pd.read_csv(INPUT_DIR / "Voting1outofN.csv")
    classifier_names.update(df['classifier_1'].unique())
    classifier_names.update(df['classifier_2'].unique())
    classifier_names.update(df['classifier_3'].unique())
    
    return classifier_names


def get_all_dataset_names_from_csvs():
    """Estrae tutti i nomi dei dataset usati nei CSV"""
    dataset_names = set()
    
    for csv_file in ["Voting2outof2.csv", "RecoveryBlock.csv", "MajorityVoting.csv", "Voting1outofN.csv"]:
        df = pd.read_csv(INPUT_DIR / csv_file)
        dataset_names.update(df['dataset_name'].unique())
    
    return dataset_names


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
    
    classifier_names = get_all_classifier_names_from_csvs()
    dataset_names = get_all_dataset_names_from_csvs()
    
    print(f"\nClassificatori da addestrare: {sorted(classifier_names)}")
    print(f"Dataset: {sorted(dataset_names)}")
    print(f"Totale addestramenti: {len(classifier_names)} × {len(dataset_names)} = {len(classifier_names) * len(dataset_names)}")
    
    trained_models = {}
    dataset_data = {}
    
    for ds_name in sorted(dataset_names):
        print(f"\n{'='*50}")
        print(f"DATASET: {ds_name}")
        print(f"{'='*50}")
        
        if ds_name not in DATASETS:
            print(f"  ⚠ Dataset non trovato in DATASETS, skip")
            continue
        
        # Carica dataset
        try:
            ds = dataset(DATASETS[ds_name], dataset_name=ds_name)
            ds.preprocess()
            X_train, X_test, y_train, y_test = ds.data
            dataset_data[ds_name] = (X_train, X_test, y_train, y_test)
        except Exception as e:
            print(f"  ✗ Errore caricamento dataset: {e}")
            continue
        
        trained_models[ds_name] = {}
        
        for clf_name in sorted(classifier_names):
            clf_name_lower = clf_name.lower().replace(' ', '_').replace('-', '_')
            
            if clf_name_lower not in CLASSIFIER_MAP:
                print(f"  ⚠ Classificatore '{clf_name}' non trovato, skip")
                continue
            
            try:
                # Crea e addestra il classificatore
                clf = CLASSIFIER_MAP[clf_name_lower]()
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


# === FASE 2: PROCESSAMENTO CSV ===

def wrap_with_rejection(trained_clf, rejection_type: str):
    """
    Wrappa un classificatore GIÀ ADDESTRATO con un rejection decorator.
    NON riaddestra il modello.
    """
    if rejection_type == 'static':
        return StaticThreshold(trained_clf, confidence_threshold=0.9)
    elif rejection_type == 'percentile':
        return PercentileThreshold(trained_clf, rejection_percentile=10.0)
    else:
        raise ValueError(f"Rejection type sconosciuto: {rejection_type}")


def process_voting2of2(trained_models: dict, dataset_data: dict, output_dir: Path):
    """Processa Voting2outof2.csv"""
    print("\n" + "="*70)
    print("FASE 2A: PROCESSING VOTING 2 OF 2")
    print("="*70)
    
    df = pd.read_csv(INPUT_DIR / "Voting2outof2.csv")
    
    results_static = []
    results_percentile = []
    
    for idx, row in df.iterrows():
        ds_name = row['dataset_name']
        
        if ds_name not in trained_models or ds_name not in dataset_data:
            continue
        
        X_train, X_test, y_train, y_test = dataset_data[ds_name]
        
        # Estrai nomi classificatori
        clf_names = row['experiment_name'].split('+')
        if len(clf_names) != 2:
            continue
        
        clf1_name, clf2_name = clf_names
        
        if clf1_name not in trained_models[ds_name] or clf2_name not in trained_models[ds_name]:
            continue
        
        # Determina rejection type
        rejection_type = 'static' if 'static' in row['rejection_strategy'] else 'percentile'
        
        # Recupera modelli già addestrati e wrappa con rejection
        clf1_base = trained_models[ds_name][clf1_name]
        clf2_base = trained_models[ds_name][clf2_name]
        
        clf1 = wrap_with_rejection(clf1_base, rejection_type)
        clf2 = wrap_with_rejection(clf2_base, rejection_type)
        
        # Predizioni (NO training!)
        pred_clf1 = clf1.predict(X_test)
        pred_clf2 = clf2.predict(X_test)
        
        # Ensemble prediction
        ensemble = Voting2outof2([clf1, clf2])
        pred_ensemble = ensemble.predict(X_test)
        
        # Calcola metriche
        metrics_ensemble = calculate_fundamental_metrics(pred_ensemble, y_test)
        metrics_clf1 = calculate_fundamental_metrics(pred_clf1, y_test)
        metrics_clf2 = calculate_fundamental_metrics(pred_clf2, y_test)
        
        # Costruisci riga risultato
        result_row = {
            'experiment_number': row['experiment_number'],
            'experiment_name': row['experiment_name'],
            'dataset_name': ds_name,
            'ensemble_type': row['ensemble_type'],
            'classifier_1': clf1_name,
            'classifier_2': clf2_name,
            'ensemble_correct_rate': metrics_ensemble['correct_rate'],
            'ensemble_misclassification_rate': metrics_ensemble['misclassification_rate'],
            'ensemble_rejection_rate': metrics_ensemble['rejection_rate'],
            'classifier_1_correct_rate': metrics_clf1['correct_rate'],
            'classifier_1_misclassification_rate': metrics_clf1['misclassification_rate'],
            'classifier_1_rejection_rate': metrics_clf1['rejection_rate'],
            'classifier_2_correct_rate': metrics_clf2['correct_rate'],
            'classifier_2_misclassification_rate': metrics_clf2['misclassification_rate'],
            'classifier_2_rejection_rate': metrics_clf2['rejection_rate'],
            'q_statistic': row['q_statistic'],
            'disagreement_measure': row['disagreement_measure'],
            'double_fault_measure': row['double_fault_measure'],
            'entropy_measure': row['entropy_measure'],
            'kohavi_wolpert_variance': row['kohavi_wolpert_variance'],
            'generalized_diversity': row['generalized_diversity'],
            'coincident_failure_diversity': row['coincident_failure_diversity'],
            'double_correct_prediction_rate': row['double_correct_prediction_rate'],
            'double_wrong_prediction_rate': row['double_wrong_prediction_rate'],
            'double_rejection_rate': row['double_rejection_rate']
        }
        
        if rejection_type == 'static':
            results_static.append(result_row)
        else:
            results_percentile.append(result_row)
    
    # Salva CSV
    if results_static:
        df_static = pd.DataFrame(results_static)
        df_static.to_csv(output_dir / "Voting2outof2_Static.csv", index=False)
        print(f"✓ Voting2outof2_Static.csv ({len(results_static)} righe)")
    
    if results_percentile:
        df_percentile = pd.DataFrame(results_percentile)
        df_percentile.to_csv(output_dir / "Voting2outof2_Percentile.csv", index=False)
        print(f"✓ Voting2outof2_Percentile.csv ({len(results_percentile)} righe)")


def process_recovery_block(trained_models: dict, dataset_data: dict, output_dir: Path):
    """Processa RecoveryBlock.csv"""
    print("\n" + "="*70)
    print("FASE 2B: PROCESSING RECOVERY BLOCK")
    print("="*70)
    
    df = pd.read_csv(INPUT_DIR / "RecoveryBlock.csv")
    
    results_static = []
    results_percentile = []
    
    for idx, row in df.iterrows():
        ds_name = row['dataset_name']
        
        if ds_name not in trained_models or ds_name not in dataset_data:
            continue
        
        X_train, X_test, y_train, y_test = dataset_data[ds_name]
        
        clf1_name = row['primary_classifier']
        clf2_name = row['secondary_classifier']
        
        if clf1_name not in trained_models[ds_name] or clf2_name not in trained_models[ds_name]:
            continue
        
        rejection_type = 'static' if 'static' in row['rejection_strategy'] else 'percentile'
        
        clf1_base = trained_models[ds_name][clf1_name]
        clf2_base = trained_models[ds_name][clf2_name]
        
        clf1 = wrap_with_rejection(clf1_base, rejection_type)
        clf2 = wrap_with_rejection(clf2_base, rejection_type)
        
        pred_clf1 = clf1.predict(X_test)
        pred_clf2 = clf2.predict(X_test)
        
        ensemble = RecoveryBlock([clf1, clf2])
        pred_ensemble = ensemble.predict(X_test)
        
        metrics_ensemble = calculate_fundamental_metrics(pred_ensemble, y_test)
        metrics_clf1 = calculate_fundamental_metrics(pred_clf1, y_test)
        metrics_clf2 = calculate_fundamental_metrics(pred_clf2, y_test)
        double_metrics = calculate_double_metrics(pred_clf1, pred_clf2, y_test)
        
        result_row = {
            'experiment_number': row['experiment_number'],
            'experiment_name': row['experiment_name'],
            'dataset_name': ds_name,
            'ensemble_type': row['ensemble_type'],
            'classifier_1': clf1_name,
            'classifier_2': clf2_name,
            'ensemble_correct_rate': metrics_ensemble['correct_rate'],
            'ensemble_misclassification_rate': metrics_ensemble['misclassification_rate'],
            'ensemble_rejection_rate': metrics_ensemble['rejection_rate'],
            'classifier_1_correct_rate': metrics_clf1['correct_rate'],
            'classifier_1_misclassification_rate': metrics_clf1['misclassification_rate'],
            'classifier_1_rejection_rate': metrics_clf1['rejection_rate'],
            'classifier_2_correct_rate': metrics_clf2['correct_rate'],
            'classifier_2_misclassification_rate': metrics_clf2['misclassification_rate'],
            'classifier_2_rejection_rate': metrics_clf2['rejection_rate'],
            'q_statistic': row['q_statistic'],
            'disagreement_measure': row['disagreement_measure'],
            'double_fault_measure': row['double_fault_measure'],
            'entropy_measure': row['entropy_measure'],
            'kohavi_wolpert_variance': row['kohavi_wolpert_variance'],
            'generalized_diversity': row['generalized_diversity'],
            'coincident_failure_diversity': row['coincident_failure_diversity'],
            'double_correct_prediction_rate': double_metrics['double_correct_prediction_rate'],
            'double_wrong_prediction_rate': double_metrics['double_wrong_prediction_rate'],
            'double_rejection_rate': double_metrics['double_rejection_rate'],
            'recovery_rate': row['recovery_rate'],
            'recovery_failure_rate': row['recovery_failure_rate'],
            'recovery_rejection_rate': row['recovery_rejection_rate'],
            'recovery_correct_prediction_rate': row['recovery_correct_prediction_rate'],
            'recovery_wrong_prediction_rate': row['recovery_wrong_prediction_rate'],
            'recovery_rejection_prediction_rate': row['recovery_rejection_prediction_rate']
        }
        
        if rejection_type == 'static':
            results_static.append(result_row)
        else:
            results_percentile.append(result_row)
    
    if results_static:
        df_static = pd.DataFrame(results_static)
        df_static.to_csv(output_dir / "RecoveryBlock_Static.csv", index=False)
        print(f"✓ RecoveryBlock_Static.csv ({len(results_static)} righe)")
    
    if results_percentile:
        df_percentile = pd.DataFrame(results_percentile)
        df_percentile.to_csv(output_dir / "RecoveryBlock_Percentile.csv", index=False)
        print(f"✓ RecoveryBlock_Percentile.csv ({len(results_percentile)} righe)")


def process_majority_voting(trained_models: dict, dataset_data: dict, output_dir: Path):
    """Processa MajorityVoting.csv"""
    print("\n" + "="*70)
    print("FASE 2C: PROCESSING MAJORITY VOTING")
    print("="*70)
    
    df = pd.read_csv(INPUT_DIR / "MajorityVoting.csv")
    
    results_static = []
    results_percentile = []
    
    for idx, row in df.iterrows():
        ds_name = row['dataset_name']
        
        if ds_name not in trained_models or ds_name not in dataset_data:
            continue
        
        X_train, X_test, y_train, y_test = dataset_data[ds_name]
        
        clf1_name = row['classifier_1']
        clf2_name = row['classifier_2']
        clf3_name = row['classifier_3']
        
        if (clf1_name not in trained_models[ds_name] or 
            clf2_name not in trained_models[ds_name] or
            clf3_name not in trained_models[ds_name]):
            continue
        
        rejection_type = 'static' if 'static' in row['rejection_strategy'] else 'percentile'
        
        clf1_base = trained_models[ds_name][clf1_name]
        clf2_base = trained_models[ds_name][clf2_name]
        clf3_base = trained_models[ds_name][clf3_name]
        
        clf1 = wrap_with_rejection(clf1_base, rejection_type)
        clf2 = wrap_with_rejection(clf2_base, rejection_type)
        clf3 = wrap_with_rejection(clf3_base, rejection_type)
        
        pred_clf1 = clf1.predict(X_test)
        pred_clf2 = clf2.predict(X_test)
        pred_clf3 = clf3.predict(X_test)
        
        ensemble = MajorityVoting([clf1, clf2, clf3])
        pred_ensemble = ensemble.predict(X_test)
        
        metrics_ensemble = calculate_fundamental_metrics(pred_ensemble, y_test)
        metrics_clf1 = calculate_fundamental_metrics(pred_clf1, y_test)
        metrics_clf2 = calculate_fundamental_metrics(pred_clf2, y_test)
        metrics_clf3 = calculate_fundamental_metrics(pred_clf3, y_test)
        
        result_row = {
            'experiment_number': row['experiment_number'],
            'experiment_name': row['experiment_name'],
            'dataset_name': ds_name,
            'ensemble_type': row['ensemble_type'],
            'classifier_1': clf1_name,
            'classifier_2': clf2_name,
            'classifier_3': clf3_name,
            'ensemble_correct_rate': metrics_ensemble['correct_rate'],
            'ensemble_misclassification_rate': metrics_ensemble['misclassification_rate'],
            'ensemble_rejection_rate': metrics_ensemble['rejection_rate'],
            'classifier_1_correct_rate': metrics_clf1['correct_rate'],
            'classifier_1_misclassification_rate': metrics_clf1['misclassification_rate'],
            'classifier_1_rejection_rate': metrics_clf1['rejection_rate'],
            'classifier_2_correct_rate': metrics_clf2['correct_rate'],
            'classifier_2_misclassification_rate': metrics_clf2['misclassification_rate'],
            'classifier_2_rejection_rate': metrics_clf2['rejection_rate'],
            'classifier_3_correct_rate': metrics_clf3['correct_rate'],
            'classifier_3_misclassification_rate': metrics_clf3['misclassification_rate'],
            'classifier_3_rejection_rate': metrics_clf3['rejection_rate'],
            'q_statistic': row['q_statistic'],
            'disagreement_measure': row['disagreement_measure'],
            'double_fault_measure': row['double_fault_measure'],
            'entropy_measure': row['entropy_measure'],
            'kohavi_wolpert_variance': row['kohavi_wolpert_variance'],
            'generalized_diversity': row['generalized_diversity'],
            'coincident_failure_diversity': row['coincident_failure_diversity'],
            'majority_voting_correct_prediction_rate': row['majority_voting_correct_prediction_rate'],
            'majority_voting_wrong_prediction_rate': row['majority_voting_wrong_prediction_rate'],
            'majority_voting_rejection_prediction_rate': row['majority_voting_rejection_prediction_rate']
        }
        
        if rejection_type == 'static':
            results_static.append(result_row)
        else:
            results_percentile.append(result_row)
    
    if results_static:
        df_static = pd.DataFrame(results_static)
        df_static.to_csv(output_dir / "MajorityVoting_Static.csv", index=False)
        print(f"✓ MajorityVoting_Static.csv ({len(results_static)} righe)")
    
    if results_percentile:
        df_percentile = pd.DataFrame(results_percentile)
        df_percentile.to_csv(output_dir / "MajorityVoting_Percentile.csv", index=False)
        print(f"✓ MajorityVoting_Percentile.csv ({len(results_percentile)} righe)")


def process_voting_1ofn(trained_models: dict, dataset_data: dict, output_dir: Path):
    """Processa Voting1outofN.csv"""
    print("\n" + "="*70)
    print("FASE 2D: PROCESSING VOTING 1 OF N")
    print("="*70)
    
    df = pd.read_csv(INPUT_DIR / "Voting1outofN.csv")
    
    results_static = []
    results_percentile = []
    
    for idx, row in df.iterrows():
        ds_name = row['dataset_name']
        
        if ds_name not in trained_models or ds_name not in dataset_data:
            continue
        
        X_train, X_test, y_train, y_test = dataset_data[ds_name]
        
        clf1_name = row['classifier_1']
        clf2_name = row['classifier_2']
        clf3_name = row['classifier_3']
        
        if (clf1_name not in trained_models[ds_name] or 
            clf2_name not in trained_models[ds_name] or
            clf3_name not in trained_models[ds_name]):
            continue
        
        rejection_type = 'static' if 'static' in row['rejection_strategy'] else 'percentile'
        
        clf1_base = trained_models[ds_name][clf1_name]
        clf2_base = trained_models[ds_name][clf2_name]
        clf3_base = trained_models[ds_name][clf3_name]
        
        clf1 = wrap_with_rejection(clf1_base, rejection_type)
        clf2 = wrap_with_rejection(clf2_base, rejection_type)
        clf3 = wrap_with_rejection(clf3_base, rejection_type)
        
        pred_clf1 = clf1.predict(X_test)
        pred_clf2 = clf2.predict(X_test)
        pred_clf3 = clf3.predict(X_test)
        
        ensemble = Voting1outofN([clf1, clf2, clf3])
        pred_ensemble = ensemble.predict(X_test)
        
        metrics_ensemble = calculate_fundamental_metrics(pred_ensemble, y_test)
        metrics_clf1 = calculate_fundamental_metrics(pred_clf1, y_test)
        metrics_clf2 = calculate_fundamental_metrics(pred_clf2, y_test)
        metrics_clf3 = calculate_fundamental_metrics(pred_clf3, y_test)
        
        result_row = {
            'experiment_number': row['experiment_number'],
            'experiment_name': row['experiment_name'],
            'dataset_name': ds_name,
            'ensemble_type': row['ensemble_type'],
            'classifier_1': clf1_name,
            'classifier_2': clf2_name,
            'classifier_3': clf3_name,
            'ensemble_correct_rate': metrics_ensemble['correct_rate'],
            'ensemble_misclassification_rate': metrics_ensemble['misclassification_rate'],
            'ensemble_rejection_rate': metrics_ensemble['rejection_rate'],
            'classifier_1_correct_rate': metrics_clf1['correct_rate'],
            'classifier_1_misclassification_rate': metrics_clf1['misclassification_rate'],
            'classifier_1_rejection_rate': metrics_clf1['rejection_rate'],
            'classifier_2_correct_rate': metrics_clf2['correct_rate'],
            'classifier_2_misclassification_rate': metrics_clf2['misclassification_rate'],
            'classifier_2_rejection_rate': metrics_clf2['rejection_rate'],
            'classifier_3_correct_rate': metrics_clf3['correct_rate'],
            'classifier_3_misclassification_rate': metrics_clf3['misclassification_rate'],
            'classifier_3_rejection_rate': metrics_clf3['rejection_rate'],
            'q_statistic': row['q_statistic'],
            'disagreement_measure': row['disagreement_measure'],
            'double_fault_measure': row['double_fault_measure'],
            'entropy_measure': row['entropy_measure'],
            'kohavi_wolpert_variance': row['kohavi_wolpert_variance'],
            'generalized_diversity': row['generalized_diversity'],
            'coincident_failure_diversity': row['coincident_failure_diversity'],
            'single_vote_correct_prediction_rate': row['single_vote_correct_prediction_rate'],
            'single_vote_wrong_prediction_rate': row['single_vote_wrong_prediction_rate'],
            'single_vote_rejection_rate': row['single_vote_rejection_rate'],
            'ideal_single_vote_rate': row['ideal_single_vote_rate']
        }
        
        if rejection_type == 'static':
            results_static.append(result_row)
        else:
            results_percentile.append(result_row)
    
    if results_static:
        df_static = pd.DataFrame(results_static)
        df_static.to_csv(output_dir / "Voting1outofN_Static.csv", index=False)
        print(f"✓ Voting1outofN_Static.csv ({len(results_static)} righe)")
    
    if results_percentile:
        df_percentile = pd.DataFrame(results_percentile)
        df_percentile.to_csv(output_dir / "Voting1outofN_Percentile.csv", index=False)
        print(f"✓ Voting1outofN_Percentile.csv ({len(results_percentile)} righe)")


# === MAIN ===

def main():
    start_time = time.time()
    
    print("="*70)
    print("GENERAZIONE CSV CON METRICHE FONDAMENTALI (OTTIMIZZATO)")
    print("="*70)
    
    # Crea directory output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # FASE 1: Addestra tutti i classificatori una sola volta
    trained_models, dataset_data = train_all_classifiers()
    
    # FASE 2: Processa i CSV usando i modelli già addestrati
    process_voting2of2(trained_models, dataset_data, OUTPUT_DIR)
    process_recovery_block(trained_models, dataset_data, OUTPUT_DIR)
    process_majority_voting(trained_models, dataset_data, OUTPUT_DIR)
    process_voting_1ofn(trained_models, dataset_data, OUTPUT_DIR)
    
    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"COMPLETATO in {elapsed:.1f} secondi")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()