import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import time
from sklearn.metrics import accuracy_score, matthews_corrcoef

warnings.filterwarnings("ignore")

# Dataset
from datasets.dataset import dataset

# Algoritmo
from ML_algorithms.decision_tree import decision_tree

# Rejection techniques
from rejection_techniques.static_threshold_rejection_decorator import static_threshold_rejection_decorator
from rejection_techniques.percentile_threshold_rejection_decorator import percentile_threshold_rejection_decorator

# Metriche singolo classificatore
from diversity_metrics.single_correct_prediction_rate import single_correct_prediction_rate
from diversity_metrics.single_misclassification_rate import single_misclassification_rate
from diversity_metrics.single_rejection_rate import single_rejection_rate
from diversity_metrics.hit import hit
from diversity_metrics.miss import miss
from diversity_metrics.acceptance_accuracy import acceptance_accuracy
from diversity_metrics.performance_loss import performance_loss


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

OUTPUT_FILE = "/Users/matteopascuzzo/Desktop/SingleClassifierBenchmark.csv"

# Colonne output
COLUMNS = [
    "experiment_number", "dataset_name", "classifier", "rejection_strategy",
    "accuracy", "mcc",
    "single_correct_prediction_rate", "single_misclassification_rate", "single_rejection_rate",
    "hit", "miss", "acceptance_accuracy", "performance_loss"
]

# Metriche singolo classificatore
SINGLE_METRICS = {
    'single_correct_prediction_rate': single_correct_prediction_rate,
    'single_misclassification_rate': single_misclassification_rate,
    'single_rejection_rate': single_rejection_rate,
    'hit': hit,
    'miss': miss,
    'acceptance_accuracy': acceptance_accuracy,
    'performance_loss': performance_loss
}


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

def calculate_single_metrics(model, X_test, y_test) -> dict:
    """Calcola le 7 metriche per singolo classificatore con rejection"""
    predictions = model.predict(X_test)
    
    # Converti in array numpy di stringhe
    predictions = np.array(predictions).astype(str)
    y_test_array = y_test.values if hasattr(y_test, 'values') else y_test
    y_test_array = np.array(y_test_array).astype(str)
    
    results = {}
    for metric_name, metric_class in SINGLE_METRICS.items():
        try:
            metric_instance = metric_class()
            value = metric_instance._compute(predictions, y_test_array, X_test, model)
            results[metric_name] = value
        except Exception as e:
            print(f"    Errore calcolo {metric_name}: {e}")
            results[metric_name] = None
    
    return results

def run_baseline_experiment(model, X_test, y_test, ds_name: str) -> dict:
    """Esperimento senza rejection: solo accuracy e MCC"""
    predictions = model.predict(X_test)
    
    y_test_array = y_test.values if hasattr(y_test, 'values') else y_test
    
    acc = accuracy_score(y_test_array, predictions)
    mcc = matthews_corrcoef(y_test_array, predictions)
    
    return {
        "dataset_name": ds_name,
        "classifier": "decision_tree",
        "rejection_strategy": "none",
        "accuracy": acc,
        "mcc": mcc,
        "single_correct_prediction_rate": None,
        "single_misclassification_rate": None,
        "single_rejection_rate": None,
        "hit": None,
        "miss": None,
        "acceptance_accuracy": None,
        "performance_loss": None
    }

def run_rejection_experiment(base_model, X_test, y_test, ds_name: str, 
                             rejection_name: str, rejection_func) -> dict:
    """Esperimento con rejection: calcola le 7 metriche"""
    # Applica rejection al modello base
    model_with_rejection = rejection_func(base_model)
    
    # Calcola metriche
    metrics = calculate_single_metrics(model_with_rejection, X_test, y_test)
    
    return {
        "dataset_name": ds_name,
        "classifier": "decision_tree",
        "rejection_strategy": rejection_name,
        "accuracy": None,
        "mcc": None,
        **metrics
    }

def main():
    print("=" * 70)
    print("SINGLE CLASSIFIER BENCHMARK - DECISION TREE")
    print("=" * 70)
    print(f"Dataset: {len(DATASETS)}")
    print(f"Esperimenti per dataset: 3 (none, static, percentile)")
    print(f"Totale esperimenti: {len(DATASETS) * 3}")
    print("=" * 70)
    
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
        
        # Addestra Decision Tree una sola volta
        print("\n--- Training Decision Tree ---")
        dt = decision_tree(random_state=42)
        dt.train(X_train, y_train)
        print("--- Training completato ---\n")
        
        # 1. Esperimento baseline (senza rejection)
        experiment_count += 1
        print(f"[{experiment_count}] decision_tree | none")
        
        try:
            results = run_baseline_experiment(dt, X_test, y_test, ds_name)
            results["experiment_number"] = get_next_experiment_number(df)
            
            new_row = pd.DataFrame([results])
            df = pd.concat([df, new_row], ignore_index=True)
            df.to_csv(OUTPUT_FILE, index=False)
            
            print(f"    ✓ Salvato (exp #{results['experiment_number']}) - Acc: {results['accuracy']:.4f}, MCC: {results['mcc']:.4f}")
        except Exception as e:
            print(f"    ✗ Errore: {e}")
        
        # 2. Esperimento con static_threshold_0.9
        experiment_count += 1
        print(f"[{experiment_count}] decision_tree | static_threshold_0.9")
        
        try:
            results = run_rejection_experiment(
                dt, X_test, y_test, ds_name,
                "static_threshold_0.9",
                lambda algo: static_threshold_rejection_decorator(algo, confidence_threshold=0.9)
            )
            results["experiment_number"] = get_next_experiment_number(df)
            
            new_row = pd.DataFrame([results])
            df = pd.concat([df, new_row], ignore_index=True)
            df.to_csv(OUTPUT_FILE, index=False)
            
            print(f"    ✓ Salvato (exp #{results['experiment_number']})")
        except Exception as e:
            print(f"    ✗ Errore: {e}")
        
        # 3. Esperimento con percentile_threshold_10
        experiment_count += 1
        print(f"[{experiment_count}] decision_tree | percentile_threshold_10")
        
        try:
            results = run_rejection_experiment(
                dt, X_test, y_test, ds_name,
                "percentile_threshold_10",
                lambda algo: percentile_threshold_rejection_decorator(algo, rejection_percentile=10.0)
            )
            results["experiment_number"] = get_next_experiment_number(df)
            
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