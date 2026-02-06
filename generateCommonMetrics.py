"""
Script per generare 4 CSV con metriche classiche e metriche ensemble.

Output:
- Coppie_Static.csv: metriche classiche + double_* + recovery_*
- Coppie_Percentile.csv: stesso, con rejection percentile
- Triple_Static.csv: metriche classiche + majority_voting_* + single_vote_* + ideal_single_vote_rate
- Triple_Percentile.csv: stesso, con rejection percentile

Ottimizzazione: ogni classificatore viene addestrato UNA SOLA VOLTA per dataset.
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

OUTPUT_DIR = Path("/Users/matteopascuzzo/Desktop/Results")

# Combinazioni estratte dai file esistenti (40 coppie con ordine per recovery_*)
COPPIE = [
    ('random_forest', 'gaussian_nb'),
    ('gaussian_nb', 'random_forest'),
    ('random_forest', 'knn'),
    ('knn', 'random_forest'),
    ('random_forest', 'logistic_regression'),
    ('logistic_regression', 'random_forest'),
    ('random_forest', 'rotation_forest'),
    ('rotation_forest', 'random_forest'),
    ('extra_trees', 'gaussian_nb'),
    ('gaussian_nb', 'extra_trees'),
    ('extra_trees', 'knn'),
    ('knn', 'extra_trees'),
    ('xgboost', 'gaussian_nb'),
    ('gaussian_nb', 'xgboost'),
    ('xgboost', 'knn'),
    ('knn', 'xgboost'),
    ('xgboost', 'logistic_regression'),
    ('logistic_regression', 'xgboost'),
    ('xgboost', 'rotation_forest'),
    ('rotation_forest', 'xgboost'),
    ('light_gbm', 'gaussian_nb'),
    ('gaussian_nb', 'light_gbm'),
    ('light_gbm', 'knn'),
    ('knn', 'light_gbm'),
    ('catboost', 'gaussian_nb'),
    ('gaussian_nb', 'catboost'),
    ('catboost', 'logistic_regression'),
    ('logistic_regression', 'catboost'),
    ('catboost', 'rotation_forest'),
    ('rotation_forest', 'catboost'),
    ('adaboost', 'gaussian_nb'),
    ('gaussian_nb', 'adaboost'),
    ('adaboost', 'knn'),
    ('knn', 'adaboost'),
    ('random_rotation_forest', 'gaussian_nb'),
    ('gaussian_nb', 'random_rotation_forest'),
    ('random_rotation_forest', 'knn'),
    ('knn', 'random_rotation_forest'),
    ('random_patches', 'xgboost'),
    ('xgboost', 'random_patches'),
]

TRIPLE = [
    ('random_forest', 'gaussian_nb', 'knn'),
    ('random_forest', 'gaussian_nb', 'logistic_regression'),
    ('random_forest', 'knn', 'logistic_regression'),
    ('random_forest', 'gaussian_nb', 'rotation_forest'),
    ('random_forest', 'knn', 'rotation_forest'),
    ('extra_trees', 'gaussian_nb', 'knn'),
    ('extra_trees', 'gaussian_nb', 'logistic_regression'),
    ('xgboost', 'gaussian_nb', 'knn'),
    ('xgboost', 'gaussian_nb', 'logistic_regression'),
    ('xgboost', 'knn', 'logistic_regression'),
    ('xgboost', 'gaussian_nb', 'rotation_forest'),
    ('light_gbm', 'gaussian_nb', 'knn'),
    ('light_gbm', 'knn', 'logistic_regression'),
    ('catboost', 'gaussian_nb', 'knn'),
    ('catboost', 'gaussian_nb', 'logistic_regression'),
    ('adaboost', 'gaussian_nb', 'knn'),
    ('adaboost', 'knn', 'logistic_regression'),
    ('rotation_forest', 'gaussian_nb', 'knn'),
    ('rotation_forest', 'gaussian_nb', 'logistic_regression'),
    ('rotation_forest', 'knn', 'logistic_regression'),
]


# === FUNZIONI METRICHE ===

def wrap_with_rejection(trained_clf, rejection_type: str):
    """Wrappa un classificatore con rejection decorator."""
    if rejection_type == 'static':
        return StaticThreshold(trained_clf, confidence_threshold=0.9)
    else:
        return PercentileThreshold(trained_clf, rejection_percentile=10.0)


def get_prediction_states(predictions: np.ndarray, y_test: np.ndarray):
    """
    Restituisce array di stati: '1' (corretto), '0' (sbagliato), '?' (reject)
    """
    predictions = np.array(predictions).astype(str)
    y_test = np.array(y_test).astype(str)
    
    states = np.empty(len(y_test), dtype='U1')
    for i in range(len(y_test)):
        if predictions[i] == 'reject':
            states[i] = '?'
        elif predictions[i] == y_test[i]:
            states[i] = '1'
        else:
            states[i] = '0'
    return states


# === METRICHE CLASSICHE (DIVERSITY) ===

def calc_pairwise_diversity(pred1: np.ndarray, pred2: np.ndarray, y_test: np.ndarray):
    """Calcola metriche di diversity pairwise."""
    pred1 = np.array(pred1).astype(str)
    pred2 = np.array(pred2).astype(str)
    y_test = np.array(y_test).astype(str)
    
    # Maschera per campioni validi (no reject)
    valid = (pred1 != 'reject') & (pred2 != 'reject')
    
    if np.sum(valid) == 0:
        return {'q_statistic': 0, 'disagreement_measure': 0, 'double_fault_measure': 0}
    
    pred1_v = pred1[valid]
    pred2_v = pred2[valid]
    y_v = y_test[valid]
    
    correct1 = (pred1_v == y_v)
    correct2 = (pred2_v == y_v)
    
    N11 = np.sum(correct1 & correct2)
    N00 = np.sum(~correct1 & ~correct2)
    N10 = np.sum(correct1 & ~correct2)
    N01 = np.sum(~correct1 & correct2)
    total = N11 + N00 + N10 + N01
    
    # Q-statistic
    denom_q = N11*N00 + N01*N10
    q = (N11*N00 - N01*N10) / denom_q if denom_q != 0 else 0
    
    # Disagreement
    dis = (N01 + N10) / total if total != 0 else 0
    
    # Double fault
    df = N00 / total if total != 0 else 0
    
    return {'q_statistic': q, 'disagreement_measure': dis, 'double_fault_measure': df}


def calc_nonpairwise_diversity(predictions_list: list, y_test: np.ndarray):
    """Calcola metriche di diversity non-pairwise."""
    L = len(predictions_list)
    n_samples = len(y_test)
    y_test = np.array(y_test).astype(str)
    
    # Converti predizioni
    preds = [np.array(p).astype(str) for p in predictions_list]
    
    # Per ogni sample, conta quanti classificatori (validi) sbagliano
    p_counts = np.zeros(L + 1)  # p[i] = proporzione con i errori
    
    for i in range(n_samples):
        valid_mask = np.array([preds[j][i] != 'reject' for j in range(L)])
        n_valid = np.sum(valid_mask)
        
        if n_valid < 2:
            p_counts[0] += 1
            continue
        
        n_wrong = sum(1 for j in range(L) if valid_mask[j] and preds[j][i] != y_test[i])
        p_counts[n_wrong] += 1
    
    p = p_counts / n_samples
    
    # Entropy
    entropy_sum = 0
    for i in range(n_samples):
        valid_mask = np.array([preds[j][i] != 'reject' for j in range(L)])
        n_valid = np.sum(valid_mask)
        if n_valid < 2:
            continue
        l_x = sum(1 for j in range(L) if valid_mask[j] and preds[j][i] == y_test[i])
        entropy_sum += min(l_x, n_valid - l_x)
    
    denom_e = n_samples * (L - np.ceil(L / 2))
    entropy = entropy_sum / denom_e if denom_e != 0 else 0
    
    # Kohavi-Wolpert variance
    kw_sum = 0
    for i in range(n_samples):
        valid_mask = np.array([preds[j][i] != 'reject' for j in range(L)])
        n_valid = np.sum(valid_mask)
        if n_valid < 2:
            continue
        l_x = sum(1 for j in range(L) if valid_mask[j] and preds[j][i] != y_test[i])
        kw_sum += l_x * (n_valid - l_x)
    
    kw = kw_sum / (n_samples * L**2) if n_samples > 0 else 0
    
    # Generalized Diversity
    p1 = sum((i / L) * p[i] for i in range(1, L + 1))
    p2 = sum((i / L) * ((i - 1) / (L - 1)) * p[i] for i in range(2, L + 1))
    gd = 1 - (p2 / p1) if p1 != 0 else 0
    
    # Coincident Failure Diversity
    if p[0] == 1.0:
        cfd = 0
    else:
        cfd_sum = sum(((L - i) / (L - 1)) * p[i] for i in range(1, L + 1))
        cfd = cfd_sum / (1 - p[0]) if (1 - p[0]) != 0 else 0
    
    return {
        'entropy_measure': entropy,
        'kohavi_wolpert_variance': kw,
        'generalized_diversity': gd,
        'coincident_failure_diversity': cfd
    }


def calc_classic_metrics_pair(pred1, pred2, y_test):
    """Calcola tutte le metriche classiche per una coppia."""
    pairwise = calc_pairwise_diversity(pred1, pred2, y_test)
    nonpairwise = calc_nonpairwise_diversity([pred1, pred2], y_test)
    return {**pairwise, **nonpairwise}


def calc_classic_metrics_triple(pred1, pred2, pred3, y_test):
    """Calcola tutte le metriche classiche per una tripla."""
    # Media delle metriche pairwise
    pw12 = calc_pairwise_diversity(pred1, pred2, y_test)
    pw13 = calc_pairwise_diversity(pred1, pred3, y_test)
    pw23 = calc_pairwise_diversity(pred2, pred3, y_test)
    
    pairwise_avg = {
        'q_statistic': np.mean([pw12['q_statistic'], pw13['q_statistic'], pw23['q_statistic']]),
        'disagreement_measure': np.mean([pw12['disagreement_measure'], pw13['disagreement_measure'], pw23['disagreement_measure']]),
        'double_fault_measure': np.mean([pw12['double_fault_measure'], pw13['double_fault_measure'], pw23['double_fault_measure']])
    }
    
    nonpairwise = calc_nonpairwise_diversity([pred1, pred2, pred3], y_test)
    return {**pairwise_avg, **nonpairwise}


# === METRICHE COPPIA ===

def calc_double_metrics(pred1: np.ndarray, pred2: np.ndarray, y_test: np.ndarray):
    """Calcola double_correct, double_wrong, double_rejection."""
    states1 = get_prediction_states(pred1, y_test)
    states2 = get_prediction_states(pred2, y_test)
    n = len(y_test)
    
    N11 = np.sum((states1 == '1') & (states2 == '1'))
    N00 = np.sum((states1 == '0') & (states2 == '0'))
    Nqq = np.sum((states1 == '?') & (states2 == '?'))
    
    return {
        'double_correct_prediction_rate': N11 / n,
        'double_wrong_prediction_rate': N00 / n,
        'double_rejection_rate': Nqq / n
    }


def calc_recovery_metrics(pred1: np.ndarray, pred2: np.ndarray, y_test: np.ndarray):
    """Calcola le metriche recovery_* per recovery block (pred1 = primario)."""
    states1 = get_prediction_states(pred1, y_test)
    states2 = get_prediction_states(pred2, y_test)
    n = len(y_test)
    
    # Casi dove il primario fa reject
    first_rejected = (states1 == '?')
    
    if np.sum(first_rejected) == 0:
        return {
            'recovery_rate': 0,
            'recovery_failure_rate': 0,
            'recovery_rejection_rate': 0,
            'recovery_correct_prediction_rate': 0,
            'recovery_wrong_prediction_rate': 0,
            'recovery_rejection_prediction_rate': 0
        }
    
    states2_after_reject = states2[first_rejected]
    
    Nq1 = np.sum(states2_after_reject == '1')
    Nq0 = np.sum(states2_after_reject == '0')
    Nqq = np.sum(states2_after_reject == '?')
    
    denom = Nq1 + Nq0 + Nqq
    
    recovery_rate = Nq1 / denom if denom > 0 else 0
    recovery_failure_rate = Nq0 / denom if denom > 0 else 0
    recovery_rejection_rate = Nqq / denom if denom > 0 else 0
    
    # Metriche del primario
    TA = np.sum(states1 == '1')
    FA = np.sum(states1 == '0')
    rejection_primary = np.sum(states1 == '?')
    
    total_primary = TA + FA + rejection_primary
    single_correct = TA / total_primary if total_primary > 0 else 0
    single_wrong = FA / total_primary if total_primary > 0 else 0
    single_rejection = rejection_primary / total_primary if total_primary > 0 else 0
    
    return {
        'recovery_rate': recovery_rate,
        'recovery_failure_rate': recovery_failure_rate,
        'recovery_rejection_rate': recovery_rejection_rate,
        'recovery_correct_prediction_rate': single_correct + recovery_rate * single_rejection,
        'recovery_wrong_prediction_rate': single_wrong + recovery_failure_rate * single_rejection,
        'recovery_rejection_prediction_rate': single_rejection * recovery_rejection_rate
    }


# === METRICHE TRIPLE ===

def calc_majority_voting_metrics(pred1, pred2, pred3, y_test):
    """Calcola majority_voting_* metrics."""
    states1 = get_prediction_states(pred1, y_test)
    states2 = get_prediction_states(pred2, y_test)
    states3 = get_prediction_states(pred3, y_test)
    n = len(y_test)
    
    correct_count = 0
    wrong_count = 0
    
    # Pattern con almeno 2 corretti
    correct_patterns = ['111', '110', '11?', '101', '1?1', '011', '?11']
    # Pattern con almeno 2 sbagliati
    wrong_patterns = ['100', '010', '001', '000', '00?', '0?0', '?00']
    
    for i in range(n):
        pattern = states1[i] + states2[i] + states3[i]
        if pattern in correct_patterns:
            correct_count += 1
        elif pattern in wrong_patterns:
            wrong_count += 1
    
    rejection_count = n - correct_count - wrong_count
    
    return {
        'majority_voting_correct_prediction_rate': correct_count / n,
        'majority_voting_wrong_prediction_rate': wrong_count / n,
        'majority_voting_rejection_prediction_rate': rejection_count / n
    }


def calc_single_vote_metrics(pred1, pred2, pred3, y_test):
    """Calcola single_vote_* metrics e ideal_single_vote_rate."""
    states1 = get_prediction_states(pred1, y_test)
    states2 = get_prediction_states(pred2, y_test)
    states3 = get_prediction_states(pred3, y_test)
    n = len(y_test)
    
    # Pattern: uno risponde correttamente, altri due reject
    correct_patterns = ['1??', '?1?', '??1']
    # Pattern: uno risponde sbagliato, altri due reject
    wrong_patterns = ['0??', '?0?', '??0']
    
    correct_count = 0
    wrong_count = 0
    
    for i in range(n):
        pattern = states1[i] + states2[i] + states3[i]
        if pattern in correct_patterns:
            correct_count += 1
        elif pattern in wrong_patterns:
            wrong_count += 1
    
    rejection_count = n - correct_count - wrong_count
    
    # Ideal single vote rate
    denom_ideal = correct_count + wrong_count
    ideal_rate = correct_count / denom_ideal if denom_ideal > 0 else 0
    
    return {
        'single_vote_correct_prediction_rate': correct_count / n,
        'single_vote_wrong_prediction_rate': wrong_count / n,
        'single_vote_rejection_rate': rejection_count / n,
        'ideal_single_vote_rate': ideal_rate
    }


# === TRAINING ===

def train_all_classifiers(dataset_names: set, classifier_names: set):
    """Addestra tutti i classificatori per tutti i dataset UNA SOLA VOLTA."""
    print("=" * 70)
    print("FASE 1: ADDESTRAMENTO CLASSIFICATORI")
    print("=" * 70)
    
    trained_models = {}
    dataset_data = {}
    
    for ds_name in sorted(dataset_names):
        print(f"\n{'='*50}")
        print(f"DATASET: {ds_name}")
        print(f"{'='*50}")
        
        if ds_name not in DATASETS:
            print(f"  ⚠ Dataset non trovato, skip")
            continue
        
        try:
            ds = dataset(DATASETS[ds_name], dataset_name=ds_name)
            ds.preprocess()
            X_train, X_test, y_train, y_test = ds.data
            dataset_data[ds_name] = (X_train, X_test, y_train, y_test)
        except Exception as e:
            print(f"  ✗ Errore caricamento: {e}")
            continue
        
        trained_models[ds_name] = {}
        
        for clf_name in sorted(classifier_names):
            if clf_name not in CLASSIFIER_MAP:
                print(f"  ⚠ '{clf_name}' non trovato, skip")
                continue
            
            try:
                clf = CLASSIFIER_MAP[clf_name]()
                clf.train(X_train, y_train)
                trained_models[ds_name][clf_name] = clf
                print(f"  ✓ {clf_name}")
            except Exception as e:
                print(f"  ✗ {clf_name}: {e}")
    
    return trained_models, dataset_data


# === PROCESSING ===

def process_coppie(trained_models, dataset_data, rejection_type):
    """Processa tutte le coppie per un tipo di rejection."""
    results = []
    
    for ds_name in sorted(dataset_data.keys()):
        X_train, X_test, y_train, y_test = dataset_data[ds_name]
        
        # Cache predizioni con rejection per questo dataset
        predictions_cache = {}
        
        for clf_name, clf_base in trained_models[ds_name].items():
            clf_wrapped = wrap_with_rejection(clf_base, rejection_type)
            predictions_cache[clf_name] = clf_wrapped.predict(X_test)
        
        for clf1_name, clf2_name in COPPIE:
            if clf1_name not in predictions_cache or clf2_name not in predictions_cache:
                continue
            
            pred1 = predictions_cache[clf1_name]
            pred2 = predictions_cache[clf2_name]
            
            # Metriche classiche
            classic = calc_classic_metrics_pair(pred1, pred2, y_test)
            
            # Metriche double_*
            double = calc_double_metrics(pred1, pred2, y_test)
            
            # Metriche recovery_* (pred1 = primario)
            recovery = calc_recovery_metrics(pred1, pred2, y_test)
            
            result = {
                'dataset_name': ds_name,
                'classifier_1': clf1_name,
                'classifier_2': clf2_name,
                **classic,
                **double,
                **recovery
            }
            results.append(result)
    
    return pd.DataFrame(results)


def process_triple(trained_models, dataset_data, rejection_type):
    """Processa tutte le triple per un tipo di rejection."""
    results = []
    
    for ds_name in sorted(dataset_data.keys()):
        X_train, X_test, y_train, y_test = dataset_data[ds_name]
        
        # Cache predizioni con rejection per questo dataset
        predictions_cache = {}
        
        for clf_name, clf_base in trained_models[ds_name].items():
            clf_wrapped = wrap_with_rejection(clf_base, rejection_type)
            predictions_cache[clf_name] = clf_wrapped.predict(X_test)
        
        for clf1_name, clf2_name, clf3_name in TRIPLE:
            if (clf1_name not in predictions_cache or 
                clf2_name not in predictions_cache or 
                clf3_name not in predictions_cache):
                continue
            
            pred1 = predictions_cache[clf1_name]
            pred2 = predictions_cache[clf2_name]
            pred3 = predictions_cache[clf3_name]
            
            # Metriche classiche
            classic = calc_classic_metrics_triple(pred1, pred2, pred3, y_test)
            
            # Metriche majority_voting_*
            majority = calc_majority_voting_metrics(pred1, pred2, pred3, y_test)
            
            # Metriche single_vote_* + ideal_single_vote_rate
            single_vote = calc_single_vote_metrics(pred1, pred2, pred3, y_test)
            
            result = {
                'dataset_name': ds_name,
                'classifier_1': clf1_name,
                'classifier_2': clf2_name,
                'classifier_3': clf3_name,
                **classic,
                **majority,
                **single_vote
            }
            results.append(result)
    
    return pd.DataFrame(results)


# === MAIN ===

def main():
    start_time = time.time()
    
    print("=" * 70)
    print("GENERAZIONE CSV METRICHE")
    print("=" * 70)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Estrai tutti i classificatori necessari
    clf_names = set()
    for c1, c2 in COPPIE:
        clf_names.add(c1)
        clf_names.add(c2)
    for c1, c2, c3 in TRIPLE:
        clf_names.add(c1)
        clf_names.add(c2)
        clf_names.add(c3)
    
    dataset_names = set(DATASETS.keys())
    
    # FASE 1: Addestramento
    trained_models, dataset_data = train_all_classifiers(dataset_names, clf_names)
    
    # FASE 2: Generazione CSV
    print("\n" + "=" * 70)
    print("FASE 2: GENERAZIONE CSV")
    print("=" * 70)
    
    # Coppie Static
    print("\nProcessing Coppie_Static...")
    df = process_coppie(trained_models, dataset_data, 'static')
    df.to_csv(OUTPUT_DIR / "Coppie_Static.csv", index=False)
    print(f"✓ Coppie_Static.csv ({len(df)} righe)")
    
    # Coppie Percentile
    print("\nProcessing Coppie_Percentile...")
    df = process_coppie(trained_models, dataset_data, 'percentile')
    df.to_csv(OUTPUT_DIR / "Coppie_Percentile.csv", index=False)
    print(f"✓ Coppie_Percentile.csv ({len(df)} righe)")
    
    # Triple Static
    print("\nProcessing Triple_Static...")
    df = process_triple(trained_models, dataset_data, 'static')
    df.to_csv(OUTPUT_DIR / "Triple_Static.csv", index=False)
    print(f"✓ Triple_Static.csv ({len(df)} righe)")
    
    # Triple Percentile
    print("\nProcessing Triple_Percentile...")
    df = process_triple(trained_models, dataset_data, 'percentile')
    df.to_csv(OUTPUT_DIR / "Triple_Percentile.csv", index=False)
    print(f"✓ Triple_Percentile.csv ({len(df)} righe)")
    
    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"COMPLETATO in {elapsed:.1f} secondi")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()