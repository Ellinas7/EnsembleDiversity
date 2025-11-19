import numpy as np
from sklearn.preprocessing import LabelEncoder

# Import dataset
from datasets.dataset import dataset

# Import ML algorithms
from ML_algorithms.random_forest import random_forest
from ML_algorithms.xgboost import xgboost
from ML_algorithms.adaboost import adaboost
from ML_algorithms.extra_trees import extra_trees

# Import diversity metrics
from diversity_metrics.disagreement_measure import disagreement_measure
from diversity_metrics.double_fault_measure import double_fault_measure
from diversity_metrics.Q_statistic import Q_statistic
from diversity_metrics.entropy_measure import entropy_measure
from diversity_metrics.kohavi_wolpert_variance import kohavi_wolpert_variance
from diversity_metrics.generalized_diversity import generalized_diversity
from diversity_metrics.coincident_failure_diversity import coincident_failure_diversity


def calculate_all_diversity_metrics(model, model_name, X_test, y_test):
    """
    Calcola tutte le metriche di diversity per un modello.
    
    Args:
        model: Modello addestrato
        model_name: Nome del modello per il logging
        X_test: Dati di test
        y_test: Labels di test
    """
    print(f"\n{'='*70}")
    print(f"METRICHE DI DIVERSITY - {model_name}")
    print(f"{'='*70}")
    
    # Converti X_test in numpy array per evitare warning sklearn
    X_test_array = X_test.values if hasattr(X_test, 'values') else X_test
    
    # Estrai predizioni dei singoli estimatori
    predictions = model.get_estimator_predictions(X_test_array)
    print(f"Predizioni estratte: {predictions.shape[0]} estimatori, {predictions.shape[1]} campioni\n")
    
    # Per le metriche di diversity, y_test deve essere numerico
    # Se il modello è XGBoost con label_encoder, usa encode_labels()
    if hasattr(model, 'encode_labels'):
        y_test_numeric = model.encode_labels(y_test)
    else:
        # Per Random Forest e AdaBoost, convertiamo manualmente se necessario
        y_test_array = y_test.values if hasattr(y_test, 'values') else y_test
        if y_test_array.dtype == object or str(y_test_array.dtype).startswith('str'):
            # Label stringhe - convertiamo usando mapping
            le = LabelEncoder()
            y_test_numeric = le.fit_transform(y_test_array)
        else:
            y_test_numeric = y_test_array
    
    # Crea istanze di tutte le metriche
    metrics = [
        disagreement_measure(),
        double_fault_measure(),
        Q_statistic(),
        entropy_measure(),
        kohavi_wolpert_variance(),
        generalized_diversity(),
        coincident_failure_diversity()
    ]
    
    # Calcola tutte le metriche
    results = {}
    for metric in metrics:
        value = metric.calculate(predictions, y_test_numeric)
        results[metric.name] = value
    
    print(f"{'='*70}\n")
    return results


# ============================================================================
# PREPARAZIONE DATASET
# ============================================================================

# Crea un'istanza del dataset
ds = dataset(
    file_path='/Users/matteopascuzzo/MachineLearningTesi/Datasets/HW_Failure/BackBlaze_2017_5PercRate_scikit.csv',  # <-- INSERISCI IL TUO PATH QUI
    dataset_name="MyDataset",
    test_size=0.2,
    random_state=42,
    stratify=False  # Mantiene la distribuzione delle classi
)

# Prepara il dataset (carica, gestisce NaN, split train/test)
ds.preprocess()

# Ottieni i dati
X_train, X_test, y_train, y_test = ds.data


# ============================================================================
# RANDOM FOREST
# ============================================================================

print(f"\n{'#'*70}")
print("RANDOM FOREST")
print(f"{'#'*70}")

# Crea l'istanza di random_forest
rf = random_forest(n_estimators=2, random_state=42, n_jobs=-1)

# Addestra il modello
rf.train(X_train, y_train)

# Valuta il modello
rf.calculate_accuracy(X_test, y_test)
rf.calculate_mcc(X_test, y_test)

# Calcola tutte le metriche di diversity
rf_diversity = calculate_all_diversity_metrics(rf, "Random Forest", X_test, y_test)


# ============================================================================
# XGBOOST
# ============================================================================

print(f"\n{'#'*70}")
print("XGBOOST")
print(f"{'#'*70}")

# Crea l'istanza di xgboost
xgb_model = xgboost(
    n_estimators=2, 
    random_state=42, 
    max_depth=6,
    learning_rate=0.3,
    n_jobs=-1
)

# Addestra il modello (label encoding automatico!)
xgb_model.train(X_train, y_train)

# Valuta il modello (predict restituisce label originali!)
xgb_model.calculate_accuracy(X_test, y_test)
xgb_model.calculate_mcc(X_test, y_test)

# Calcola tutte le metriche di diversity
# La funzione usa encode_labels() per convertire y_test automaticamente
xgb_diversity = calculate_all_diversity_metrics(xgb_model, "XGBoost", X_test, y_test)


# ============================================================================
# ADABOOST
# ============================================================================

print(f"\n{'#'*70}")
print("ADABOOST")
print(f"{'#'*70}")

# Crea l'istanza di adaboost
ada = adaboost(
    n_estimators=2, 
    random_state=42,
    max_depth=1
)

# Addestra il modello
ada.train(X_train, y_train)

# Valuta il modello
ada.calculate_accuracy(X_test, y_test)
ada.calculate_mcc(X_test, y_test)

# Calcola tutte le metriche di diversity
ada_diversity = calculate_all_diversity_metrics(ada, "AdaBoost", X_test, y_test)


# ============================================================================
# EXTRA TREES
# ============================================================================

print(f"\n{'#'*70}")
print("EXTRA TREES")
print(f"{'#'*70}")

# Crea l'istanza di extra_trees
et = extra_trees(
    n_estimators=2, 
    random_state=42,
    n_jobs=-1
)

# Addestra il modello
et.train(X_train, y_train)

# Valuta il modello
et.calculate_accuracy(X_test, y_test)
et.calculate_mcc(X_test, y_test)

# Calcola tutte le metriche di diversity
et_diversity = calculate_all_diversity_metrics(et, "Extra Trees", X_test, y_test)


# ============================================================================
# RIEPILOGO COMPARATIVO
# ============================================================================

print(f"\n{'='*70}")
print("RIEPILOGO COMPARATIVO")
print(f"{'='*70}\n")

print("METRICHE DI DIVERSITY:")
print(f"{'Metrica':<35} {'Random Forest':>15} {'XGBoost':>15} {'AdaBoost':>15} {'Extra Trees':>15}")
print(f"{'-'*95}")

for metric_name in rf_diversity.keys():
    rf_value = rf_diversity[metric_name]
    xgb_value = xgb_diversity[metric_name]
    ada_value = ada_diversity[metric_name]
    et_value = et_diversity[metric_name]
    print(f"{metric_name:<35} {rf_value:>15.6f} {xgb_value:>15.6f} {ada_value:>15.6f} {et_value:>15.6f}")

print(f"{'='*95}\n")