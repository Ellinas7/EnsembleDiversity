# app.py

import streamlit as st
import pandas as pd
from pathlib import Path

from datasets.dataset import dataset
from metric_calculator import metric_calculator

# Ensemble
from ensembles.voting_2of2_ensemble import Voting2of2Ensemble
from ensembles.recovery_block_ensemble import RecoveryBlockEnsemble
from ensembles.majority_voting_ensemble import MajorityVotingEnsemble
from ensembles.voting_1ofn_ensemble import Voting1ofNEnsemble

# Algoritmi
from ML_algorithms.random_forest import random_forest
from ML_algorithms.extra_trees import extra_trees
from ML_algorithms.adaboost import adaboost
from ML_algorithms.xgboost import xgboost
from ML_algorithms.light_gbm import light_gbm
from ML_algorithms.catboost import catboost
from ML_algorithms.gradient_boosting_decision_trees import gradient_boosting_decision_trees
from ML_algorithms.random_patches import random_patches
from ML_algorithms.rotation_forest import rotation_forest
from ML_algorithms.random_rotation_forest import random_rotation_forest
from ML_algorithms.gaussian_nb import gaussian_nb
from ML_algorithms.k_nearest_neighbors import knn
from ML_algorithms.logistic_regression import logistic_regression

# Rejection techniques
from rejection_techniques.static_threshold_rejection_decorator import static_threshold_rejection_decorator
from rejection_techniques.percentile_threshold_rejection_decorator import percentile_threshold_rejection_decorator

# Configurazione pagina
st.set_page_config(page_title="Ensemble Experiment Runner", layout="wide")
st.title("🧪 Ensemble Experiment Runner")

# Dizionari
ALGORITHMS = {
    "Random Forest": random_forest,
    "Extra Trees": extra_trees,
    "AdaBoost": adaboost,
    "XGBoost": xgboost,
    "LightGBM": light_gbm,
    "CatBoost": catboost,
    "Gradient Boosting": gradient_boosting_decision_trees,
    "Random Patches": random_patches,
    "Rotation Forest": rotation_forest,
    "Random Rotation Forest": random_rotation_forest,
    "Gaussian Naive Bayes": gaussian_nb,
    "K-Nearest Neighbors": knn,
    "Logistic Regression": logistic_regression
}

ENSEMBLES = {
    "Voting 2 out of 2": Voting2of2Ensemble,
    "Recovery Block": RecoveryBlockEnsemble,
    "Majority Voting": MajorityVotingEnsemble,
    "Voting 1 out of N": Voting1ofNEnsemble
}

REJECTION_TECHNIQUES = {
    "Static Threshold (0.9)": lambda algo: static_threshold_rejection_decorator(algo, confidence_threshold=0.9),
    "Percentile Threshold (10%)": lambda algo: percentile_threshold_rejection_decorator(algo, rejection_percentile=10.0)
}

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

FILE_PATHS = {
    "Voting 2 out of 2": "/Users/matteopascuzzo/Desktop/Voting2outof2.csv",
    "Recovery Block": "/Users/matteopascuzzo/Desktop/RecoveryBlock.csv",
    "Majority Voting": "/Users/matteopascuzzo/Desktop/MajorityVoting.csv",
    "Voting 1 out of N": "/Users/matteopascuzzo/Desktop/Voting1outofN.csv"
}

# Colonne per ogni tipo di ensemble
BASE_COLUMNS = ["experiment_number", "experiment_name", "dataset_name", "ensemble_type", "rejection_strategy"]

CLASSIC_METRIC_COLUMNS = [
    "q_statistic", "disagreement_measure", "double_fault_measure", "entropy_measure",
    "kohavi_wolpert_variance", "generalized_diversity", "coincident_failure_diversity"
]

ENSEMBLE_SPECIFIC_COLUMNS = {
    "Voting 2 out of 2": ["double_correct_prediction_rate", "double_wrong_prediction_rate", "double_rejection_rate"],
    "Recovery Block": ["recovery_rate", "recovery_failure_rate", "recovery_rejection_rate",
                       "recovery_correct_prediction_rate", "recovery_wrong_prediction_rate", 
                       "recovery_rejection_prediction_rate"],
    "Majority Voting": ["majority_voting_correct_prediction_rate", "majority_voting_wrong_prediction_rate",
                        "majority_voting_rejection_prediction_rate"],
    "Voting 1 out of N": ["single_vote_correct_prediction_rate", "single_vote_wrong_prediction_rate",
                          "single_vote_rejection_rate", "ideal_single_vote_rate"]
}


def get_columns_for_ensemble(ensemble_type: str) -> list:
    """Restituisce le colonne per un tipo di ensemble"""
    return BASE_COLUMNS + CLASSIC_METRIC_COLUMNS + ENSEMBLE_SPECIFIC_COLUMNS[ensemble_type]


def load_or_create_file(ensemble_type: str) -> pd.DataFrame:
    """Carica o crea il file CSV per un tipo di ensemble"""
    file_path = Path(FILE_PATHS[ensemble_type])
    columns = get_columns_for_ensemble(ensemble_type)
    
    if file_path.exists():
        return pd.read_csv(file_path)
    else:
        df = pd.DataFrame(columns=columns)
        df.to_csv(file_path, index=False)
        return df


def get_next_experiment_number(df: pd.DataFrame) -> int:
    """Restituisce il prossimo numero di esperimento"""
    if len(df) == 0:
        return 1
    return int(df["experiment_number"].max()) + 1


def get_experiment_name(classifiers: list) -> str:
    """Genera nome esperimento dai nomi dei classificatori"""
    base_names = []
    for clf in classifiers:
        base_names.append(clf.base_algorithm.name)
    return "+".join(base_names)


def get_rejection_strategy(rejection_name: str) -> str:
    """Estrae la strategia di rejection dal nome"""
    if "Static" in rejection_name:
        return "static_threshold_0.9"
    else:
        return "percentile_threshold_10"


# Sidebar
st.sidebar.header("⚙️ Configurazione")

# 1. Dataset
st.sidebar.subheader("1. Dataset")
dataset_name = st.sidebar.selectbox("Seleziona dataset", list(DATASETS.keys()))

# 2. Tipo di Ensemble
st.sidebar.subheader("2. Tipo di Ensemble")
ensemble_type = st.sidebar.selectbox("Seleziona ensemble", list(ENSEMBLES.keys()))

# 3. Classificatori
st.sidebar.subheader("3. Classificatori")
selected_algos = st.sidebar.multiselect("Seleziona classificatori", list(ALGORITHMS.keys()))

# 4. Tecnica di Rejection
st.sidebar.subheader("4. Rejection Technique")
selected_rejections = st.sidebar.multiselect(
    "Seleziona tecniche", 
    list(REJECTION_TECHNIQUES.keys()),
    default=[list(REJECTION_TECHNIQUES.keys())[0]]
)

# 5. Gruppi Metriche
st.sidebar.subheader("5. Gruppi Metriche")
use_classiche = st.sidebar.checkbox("Classiche", value=True, disabled=True)
use_specifiche = st.sidebar.checkbox(f"Specifiche {ensemble_type}", value=True)

# Info pannello principale
st.subheader("📋 Configurazione Esperimento")
col1, col2 = st.columns(2)
with col1:
    st.write(f"**Dataset:** {dataset_name}")
    st.write(f"**Ensemble:** {ensemble_type}")
    st.write(f"**Classificatori:** {', '.join(selected_algos) if selected_algos else 'Nessuno'}")
with col2:
    st.write(f"**File output:** {Path(FILE_PATHS[ensemble_type]).name}")

# Bottone esegui
if st.sidebar.button("🚀 Esegui Esperimento", type="primary"):
    if not selected_algos:
        st.error("❌ Seleziona almeno un classificatore!")
    elif not selected_rejections:
        st.error("❌ Seleziona almeno una tecnica di rejection!")
    else:
        with st.spinner("Esecuzione in corso..."):
            try:
                # Prepara dataset
                ds = dataset(DATASETS[dataset_name], dataset_name=dataset_name, stratify=True)
                ds.preprocess()
                
                # Crea e addestra classificatori base UNA SOLA VOLTA
                base_classifiers = []
                for algo_name in selected_algos:
                    algo = ALGORITHMS[algo_name]()
                    base_classifiers.append(algo)
                
                # Addestra tutti i classificatori base
                X_train, X_test, y_train, y_test = ds.data
                for clf in base_classifiers:
                    clf.train(X_train, y_train)
                
                calc = metric_calculator()
                df = load_or_create_file(ensemble_type)
                
                # Per ogni tecnica di rejection selezionata
                for rejection_name in selected_rejections:
                    rejection_func = REJECTION_TECHNIQUES[rejection_name]
                    
                    # Crea decorator sui classificatori già addestrati
                    classifiers_with_rejection = []
                    for base_clf in base_classifiers:
                        clf_with_rejection = rejection_func(base_clf)
                        classifiers_with_rejection.append(clf_with_rejection)
                    
                    # Crea ensemble
                    ensemble_class = ENSEMBLES[ensemble_type]
                    ensemble = ensemble_class(classifiers_with_rejection)
                    
                    # Calcola metriche (skip training perché già fatto)
                    metric_results = calc.calculate(ds, ensemble, skip_training=True)
                    
                    # Costruisci risultato
                    results = {
                        "experiment_number": get_next_experiment_number(df),
                        "experiment_name": get_experiment_name(classifiers_with_rejection),
                        "dataset_name": dataset_name,
                        "ensemble_type": ensemble_type,
                        "rejection_strategy": get_rejection_strategy(rejection_name)
                    }
                    
                    # Aggiungi metriche
                    columns = get_columns_for_ensemble(ensemble_type)
                    for col in columns[5:]:
                        results[col] = metric_results.get(col, None)
                    
                    # Salva
                    new_row = pd.DataFrame([results])
                    df = pd.concat([df, new_row], ignore_index=True)
                    df.to_csv(FILE_PATHS[ensemble_type], index=False)
                    
                    st.success(f"✅ Esperimento #{results['experiment_number']} ({rejection_name}) salvato")
                
                # Mostra risultati ultimo esperimento
                st.subheader("📊 Risultati ultimo esperimento")
                for k, v in results.items():
                    if v is not None and k not in BASE_COLUMNS:
                        st.write(f"**{k}:** {v:.6f}" if isinstance(v, float) else f"**{k}:** {v}")
                
            except Exception as e:
                st.error(f"❌ Errore: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

# Sezione visualizzazione esperimenti
st.sidebar.markdown("---")
st.sidebar.subheader("📂 Visualizza Esperimenti")

if st.sidebar.button("Mostra esperimenti correnti"):
    df = load_or_create_file(ensemble_type)
    if len(df) > 0:
        st.subheader(f"Esperimenti in {Path(FILE_PATHS[ensemble_type]).name}")
        st.dataframe(df)
    else:
        st.info("Nessun esperimento presente in questo file.")