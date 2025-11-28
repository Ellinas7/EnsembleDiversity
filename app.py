import streamlit as st
import numpy as np
import pandas as pd
import tempfile
import os

# Dataset
from datasets.dataset import dataset

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

# Rejection decorators
from ML_algorithms.static_threshold_rejection_decorator import static_threshold_rejection_decorator
from ML_algorithms.percentile_threshold_rejection_decorator import percentile_threshold_rejection_decorator

# Metriche classiche
from diversity_metrics.Q_statistic import Q_statistic
from diversity_metrics.disagreement_measure import disagreement_measure
from diversity_metrics.double_fault_measure import double_fault_measure
from diversity_metrics.entropy_measure import entropy_measure
from diversity_metrics.kohavi_wolpert_variance import kohavi_wolpert_variance
from diversity_metrics.generalized_diversity import generalized_diversity
from diversity_metrics.coincident_failure_diversity import coincident_failure_diversity

# Metriche rejection singolo
from diversity_metrics.single_correct_prediction_rate import single_correct_prediction_rate
from diversity_metrics.single_misclassification_rate import single_misclassification_rate
from diversity_metrics.single_rejection_rate import single_rejection_rate
from diversity_metrics.acceptance_accuracy import acceptance_accuracy
from diversity_metrics.miss import miss
from diversity_metrics.hit import hit
from diversity_metrics.performance_loss import performance_loss

# Metriche doppio (voting 2/2)
from diversity_metrics.double_correct_prediction_rate import double_correct_prediction_rate
from diversity_metrics.double_wrong_prediction_rate import double_wrong_prediction_rate
from diversity_metrics.double_rejection_rate import double_rejection_rate

# Metriche recovery block
from diversity_metrics.recovery_rate import recovery_rate
from diversity_metrics.recovery_failure_rate import recovery_failure_rate
from diversity_metrics.recovery_rejection_rate import recovery_rejection_rate
from diversity_metrics.recovery_correct_prediction_rate import recovery_correct_prediction_rate
from diversity_metrics.recovery_wrong_prediction_rate import recovery_wrong_prediction_rate
from diversity_metrics.recovery_rejection_prediction_rate import recovery_rejection_prediction_rate

# Metriche majority voting
from diversity_metrics.majority_voting_correct_prediction_rate import majority_voting_correct_prediction_rate
from diversity_metrics.majority_voting_wrong_prediction_rate import majority_voting_wrong_prediction_rate
from diversity_metrics.majority_voting_rejection_prediction_rate import majority_voting_rejection_prediction_rate

# Metriche single vote
from diversity_metrics.single_vote_correct_prediction_rate import single_vote_correct_prediction_rate
from diversity_metrics.single_vote_wrong_prediction_rate import single_vote_wrong_prediction_rate
from diversity_metrics.single_vote_rejection_rate import single_vote_rejection_rate
from diversity_metrics.ideal_single_vote_rate import ideal_single_vote_rate

from metric_calculator import metric_calculator

# Configurazione pagina
st.set_page_config(page_title="Diversity Metrics Calculator", layout="wide")
st.title("🎯 Diversity & Fault Tolerance Calculator")

# Dizionari per mapping
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
    "Random Rotation Forest": random_rotation_forest
}

METRICS = {
    "Classiche": {
        "Q-statistic": Q_statistic,
        "Disagreement": disagreement_measure,
        "Double Fault": double_fault_measure,
        "Entropy": entropy_measure,
        "Kohavi-Wolpert Variance": kohavi_wolpert_variance,
        "Generalized Diversity": generalized_diversity,
        "Coincident Failure Diversity": coincident_failure_diversity
    },
    "Single Classifier (Rejection)": {
        "Single Correct Prediction Rate": single_correct_prediction_rate,
        "Single Misclassification Rate": single_misclassification_rate,
        "Single Rejection Rate": single_rejection_rate,
        "Acceptance Accuracy": acceptance_accuracy,
        "Miss": miss,
        "Hit": hit,
        "Performance Loss": performance_loss
    },
    "Voting 2 su 2": {
        "Double Correct Prediction Rate": double_correct_prediction_rate,
        "Double Wrong Prediction Rate": double_wrong_prediction_rate,
        "Double Rejection Rate": double_rejection_rate
    },
    "Recovery Block": {
        "Recovery Rate": recovery_rate,
        "Recovery Failure Rate": recovery_failure_rate,
        "Recovery Rejection Rate": recovery_rejection_rate,
        "Recovery Correct Prediction Rate": recovery_correct_prediction_rate,
        "Recovery Wrong Prediction Rate": recovery_wrong_prediction_rate,
        "Recovery Rejection Prediction Rate": recovery_rejection_prediction_rate
    },
    "Majority Voting (3 classificatori)": {
        "Majority Voting Correct Prediction Rate": majority_voting_correct_prediction_rate,
        "Majority Voting Wrong Prediction Rate": majority_voting_wrong_prediction_rate,
        "Majority Voting Rejection Prediction Rate": majority_voting_rejection_prediction_rate
    },
    "Single Vote (3 classificatori)": {
        "Single Vote Correct Prediction Rate": single_vote_correct_prediction_rate,
        "Single Vote Wrong Prediction Rate": single_vote_wrong_prediction_rate,
        "Single Vote Rejection Rate": single_vote_rejection_rate,
        "Ideal Single Vote Rate": ideal_single_vote_rate
    }
}

# Sidebar per configurazione
st.sidebar.header("⚙️ Configurazione")

# 1. Upload dataset
st.sidebar.subheader("1. Dataset")
uploaded_file = st.sidebar.file_uploader("Carica CSV", type="csv")

if uploaded_file:
    # Salva file temporaneo
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name
    
    # Preview dataset
    df_preview = pd.read_csv(tmp_path)
    st.subheader("📊 Preview Dataset")
    st.dataframe(df_preview.head(10))
    st.write(f"Shape: {df_preview.shape}")
    
    # Parametri dataset
    test_size = st.sidebar.slider("Test size", 0.1, 0.5, 0.2)
    stratify = st.sidebar.checkbox("Stratify", value=True)
    
    # 2. Algoritmo
    st.sidebar.subheader("2. Algoritmo")
    algo_name = st.sidebar.selectbox("Seleziona algoritmo", list(ALGORITHMS.keys()))
    n_estimators = st.sidebar.number_input("N. estimators", min_value=1, max_value=100, value=2)
    
    # 3. Rejection Option
    st.sidebar.subheader("3. Rejection Option")
    rejection_type = st.sidebar.selectbox(
        "Tipo rejection",
        ["Nessuna", "Static Threshold", "Percentile Threshold"]
    )
    
    if rejection_type == "Static Threshold":
        confidence_threshold = st.sidebar.slider("Confidence threshold", 0.1, 0.99, 0.6)
    elif rejection_type == "Percentile Threshold":
        rejection_percentile = st.sidebar.slider("Rejection percentile", 1.0, 50.0, 10.0)
    
    # 4. Metrica
    st.sidebar.subheader("4. Metrica")
    metric_category = st.sidebar.selectbox("Categoria", list(METRICS.keys()))
    metric_name = st.sidebar.selectbox("Metrica", list(METRICS[metric_category].keys()))
    
    # Calcola
    if st.sidebar.button("🚀 Calcola", type="primary"):
        with st.spinner("Calcolo in corso..."):
            try:
                # Prepara dataset
                ds = dataset(tmp_path, dataset_name=uploaded_file.name, 
                            test_size=test_size, stratify=stratify)
                ds.preprocess()
                
                # Crea algoritmo
                algo_class = ALGORITHMS[algo_name]
                if algo_name == "Decision Tree":
                    algo = algo_class()
                else:
                    algo = algo_class(n_estimators=n_estimators)
                
                # Applica rejection se richiesto
                if rejection_type == "Static Threshold":
                    algo = static_threshold_rejection_decorator(algo, confidence_threshold=confidence_threshold)
                elif rejection_type == "Percentile Threshold":
                    algo = percentile_threshold_rejection_decorator(algo, rejection_percentile=rejection_percentile)
                
                # Crea metrica
                metric = METRICS[metric_category][metric_name]()
                
                # Calcola
                calc = metric_calculator()
                result = calc.calculate(ds, algo, metric)
                
                # Mostra risultato
                st.success("✅ Calcolo completato!")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Algoritmo", algo_name)
                with col2:
                    st.metric("Metrica", metric_name)
                with col3:
                    st.metric("Risultato", f"{result:.6f}")
                
                # Dettagli
                with st.expander("📋 Dettagli configurazione"):
                    st.write(f"**Dataset:** {uploaded_file.name}")
                    st.write(f"**Test size:** {test_size}")
                    st.write(f"**Stratify:** {stratify}")
                    st.write(f"**N. estimators:** {n_estimators}")
                    st.write(f"**Rejection:** {rejection_type}")
                    if rejection_type == "Static Threshold":
                        st.write(f"**Confidence threshold:** {confidence_threshold}")
                    elif rejection_type == "Percentile Threshold":
                        st.write(f"**Rejection percentile:** {rejection_percentile}%")
                
            except Exception as e:
                st.error(f"❌ Errore: {str(e)}")
            finally:
                # Cleanup
                os.unlink(tmp_path)
else:
    st.info("👈 Carica un dataset CSV per iniziare")
    st.markdown("""
    ### Come usare:
    1. Carica un file CSV con colonna target `multilabel`
    2. Seleziona un algoritmo di ML
    3. Scegli se applicare rejection option
    4. Seleziona una metrica da calcolare
    5. Clicca **Calcola**
    """)